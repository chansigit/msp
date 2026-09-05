"""Public downstream APIs and numerical logging remain auditable."""

import warnings
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import msp.annotate as annotate
import msp.deg_logging as deg_logging
import msp.evidence as evidence
import msp.report as report


def test_deg_warning_summary_preserves_other_warnings_and_results(monkeypatch, caplog):
    result = object()

    def rank(*args, **kwargs):
        for _ in range(8):
            warnings.warn_explicit(
                "divide by zero encountered in log2",
                RuntimeWarning,
                "/env/scanpy/tools/_rank_genes_groups.py",
                400,
                module="scanpy.tools._rank_genes_groups",
            )
        warnings.warn("different numerical issue", RuntimeWarning, stacklevel=2)
        return result

    monkeypatch.setattr(deg_logging.sc.tl, "rank_genes_groups", rank)
    with pytest.warns(RuntimeWarning, match="different numerical issue") as recorded:
        assert deg_logging.rank_genes_groups(None) is result
    assert len(recorded) == 1
    assert "8 Scanpy log-fold-change warnings" in caplog.text
    assert "non-finite values remain" in caplog.text


def test_deg_errors_propagate(monkeypatch):
    def rank(*args, **kwargs):
        raise ValueError("invalid input")

    monkeypatch.setattr(deg_logging.sc.tl, "rank_genes_groups", rank)
    with pytest.raises(ValueError, match="invalid input"):
        deg_logging.rank_genes_groups(None)


def test_prior_columns_and_cluster_context():
    obs = pd.DataFrame(
        {
            "batch": ["A", "A", "B", "B"],
            "source_identity": ["p1", "p1", "p2", "p2"],
            "prior": ["T", "B", "T", "B"],
            "boolean_label": ["yes", "no", "yes", "no"],
            "numeric": [1, 2, 3, 4],
            "msp_old_label": ["a", "b", "a", "b"],
            "unlabelled": [None, "B", None, "T"],
            annotate.BASE_KEY: ["0", "1", "0", "1"],
            annotate.PARENT_KEY: ["0"] * 4,
            "_msp_action": ["keep", "keep", None, "flag"],
            "doublet_score": [0.1, 0.2, 0.3, 0.4],
        },
        index=[f"c{i}" for i in range(4)],
    )
    ad = anndata.AnnData(np.ones((4, 2)), obs=obs)
    assert evidence.prior_label_columns(ad, "batch") == ["prior"]
    tables = SimpleNamespace(markers_text=lambda key, cluster: "cached marker evidence")
    context = annotate._cluster_context(
        ad,
        "0",
        "batch",
        ["prior", "unlabelled"],
        {"0": ["1"]},
        np.array([True, False, False, False]),
        tables,
    )
    for text in [
        "n=2",
        "1 (50.0%)",
        "siblings under parent 0",
        "1:2",
        "samples: 2/2",
        "cached marker evidence",
        "1 missing",
        "unlabelled in this cluster",
        "NOT ground truth",
    ]:
        assert text in context
    assert "unknown cluster" in annotate._cluster_context(ad, "absent", "batch", [], {}, np.zeros(4, bool))


@pytest.mark.parametrize("remove", [False, True])
def test_public_subcluster_real_graph_keeps_full_sizes(remove):
    rng = np.random.default_rng(13)
    ad = anndata.AnnData(np.log1p(rng.poisson(3, (12, 4))).astype(float))
    ad.obs["cluster"] = pd.Categorical(["0"] * 12)
    graph = sparse.block_diag([np.ones((6, 6)) - np.eye(6)] * 2, format="csr")
    ad.obsp["connectivities"] = graph
    ad.uns["neighbors"] = {"connectivities_key": "connectivities"}
    count, message = evidence.subcluster_once(ad, "cluster", "0", 0.1, "split", np.full(12, remove))
    assert count == 2
    assert message.count("(n=6)") == 2
    assert ("no DE" in message) == remove
    assert "split" in ad.obs


def test_public_subcluster_unsplit_removes_temporary_column():
    ad = anndata.AnnData(np.ones((6, 2)))
    ad.obs["cluster"] = pd.Categorical(["0"] * 6)
    ad.obsp["connectivities"] = sparse.csr_matrix(np.ones((6, 6)) - np.eye(6))
    ad.uns["neighbors"] = {"connectivities_key": "connectivities"}
    count, message = evidence.subcluster_once(ad, "cluster", "0", 0.01, "split", np.zeros(6, bool))
    assert count == 0
    assert "did not split" in message
    assert "split" not in ad.obs


def test_public_helpers_keep_legacy_behavior(tmp_path):
    entries = {"0": {"merge_target": "1"}, "1": {"merge_target": None}}
    assert evidence.components(entries) == {"0": ["0", "1"], "1": ["0", "1"]}
    assert report.csv_table is report._csv_table
    assert report.img is report._img


def test_subcluster_skips_deg_with_singleton_after_removal():
    ad = anndata.AnnData(np.ones((12, 2)))
    ad.obs["cluster"] = pd.Categorical(["0"] * 12)
    ad.obsp["connectivities"] = sparse.block_diag([np.ones((6, 6)) - np.eye(6)] * 2, format="csr")
    ad.uns["neighbors"] = {"connectivities_key": "connectivities"}
    mask = np.zeros(12, bool)
    mask[:5] = True
    count, message = evidence.subcluster_once(ad, "cluster", "0", 0.1, "split", mask)
    assert count == 2
    assert message.count("(n=6)") == 2
    assert "too few non-removed cells" in message
