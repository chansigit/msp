"""Regression checks for resume metadata and small-data output contracts."""

import runpy
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

import msp.integrate as integrate
import msp.resources as resources
from msp.inspect import DegTables, _load_paga_neighbors, _load_removal_mask


@pytest.mark.parametrize("harmony", [{}, {"theta": [1, 2]}])
@pytest.mark.parametrize("metadata_change", [None, "different", "missing"])
def test_cli_resume_after_h5ad_roundtrip(tmp_path, monkeypatch, harmony, metadata_change):
    """Identical metadata resumes; changed or incomplete metadata recomputes."""
    data = ad.AnnData(np.ones((3, 2)))
    data.uns["msp"] = {
        "batch_col": "batch",
        "species": "",
        "resolutions": [0.3, 1.0, 2.0],
        "n_top_genes": 2000,
        "n_pcs_requested": 50,
        "n_neighbors": 15,
        "harmony": harmony,
        "inputs": ["a.h5ad", "b.h5ad"],
        "n_batches": 2,
    }
    if metadata_change == "different":
        data.uns["msp"]["n_pcs_requested"] = 25
    elif metadata_change == "missing":
        del data.uns["msp"]["n_pcs_requested"]
    data.write_h5ad(tmp_path / "integrated.h5ad")
    (tmp_path / "report.html").write_text("complete report")

    calls, opened = [], []

    def recompute(*args, **kwargs):
        calls.append(kwargs)
        return None, {"n_cells": 3}

    read_h5ad = sc.read_h5ad

    def tracked_read(*args, **kwargs):
        result = read_h5ad(*args, **kwargs)
        opened.append(result)
        return result

    monkeypatch.setattr(integrate, "run_multi_sample_pipeline", recompute)
    monkeypatch.setattr(sc, "read_h5ad", tracked_read)
    argv = ["msp", "a.h5ad", "b.h5ad", "--batch-col", "batch", "--outdir", str(tmp_path)]
    if harmony:
        argv += ["--harmony", "theta=1,2"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_module("msp.__main__", run_name="__main__")

    assert len(calls) == (0 if metadata_change is None else 1)
    assert opened and all(not item.file.is_open for item in opened)


def test_removal_mask_preserves_cell_ids_and_alignment(tmp_path):
    data = ad.AnnData(np.ones((4, 2)), obs=pd.DataFrame(index=["NA", "002", "001", "unlisted"]))
    pd.DataFrame(
        {
            "cell": ["001", "002", "NA"],
            "recommend_removal": [True, False, True],
        }
    ).to_csv(tmp_path / "preannotation_removal.csv", index=False)

    np.testing.assert_array_equal(_load_removal_mask(tmp_path, data), [True, False, True, False])


def test_missing_removal_file_preserves_legacy_behavior(tmp_path):
    data = ad.AnnData(np.ones((2, 2)))
    np.testing.assert_array_equal(_load_removal_mask(tmp_path, data), [False, False])


@pytest.mark.parametrize("contents", [None, "\n", "cluster,neighbor,rank,connectivity\n"])
def test_empty_paga_neighbors(tmp_path, contents):
    if contents is not None:
        (tmp_path / "paga_neighbors_k.csv").write_text(contents)
    assert _load_paga_neighbors(tmp_path, "k") == {}


def test_paga_reader_preserves_ids_and_rank_order(tmp_path):
    (tmp_path / "paga_neighbors_k.csv").write_text("cluster,neighbor,rank,connectivity\n001,003,2,0.2\n001,002,1,0.8\n")
    assert _load_paga_neighbors(tmp_path, "k") == {"001": ["002", "003"]}


def test_malformed_paga_schema_is_not_silently_ignored(tmp_path):
    (tmp_path / "paga_neighbors_k.csv").write_text("wrong_column\nvalue\n")
    with pytest.raises(KeyError):
        _load_paga_neighbors(tmp_path, "k")


@pytest.mark.parametrize("second_size,connected", [(2, True), (12, True), (12, False)])
def test_cluster_annotations_preserve_csv_contract(tmp_path, monkeypatch, second_size, connected):
    """Run real Wilcoxon tests with a controlled two-cluster neighbor graph."""
    rng = np.random.default_rng(4)
    data = ad.AnnData(
        np.log1p(rng.poisson(2, (12 + second_size, 8))).astype(float),
        obs=pd.DataFrame(
            {"k": pd.Categorical(["0"] * 12 + ["1"] * second_size)},
            index=[f"c{i}" for i in range(12 + second_size)],
        ),
    )
    data.raw = data.copy()

    def paga(subset, groups):
        edge = 1 if connected else 0
        subset.uns["paga"] = {"connectivities": sparse.csr_matrix([[0, edge], [edge, 0]])}

    # Graph construction is unrelated to the DEG serialization regression.
    monkeypatch.setattr(sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(sc.tl, "paga", paga)
    monkeypatch.setattr(resources, "available_cpus", lambda: 1)
    integrate._cluster_annotations(data, np.zeros(data.n_obs, dtype=bool), ["k"], [1.0], tmp_path)

    global_df = pd.read_csv(tmp_path / "deg_global_k.csv", dtype={"group": str})
    expected_groups = {"0"} if second_size < 10 else {"0", "1"}
    assert set(global_df["group"]) == expected_groups
    assert set(global_df.columns) == {
        "group",
        "names",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "pct1",
        "pct2",
    }
    assert global_df.groupby("group").size().to_dict() == dict.fromkeys(expected_groups, data.n_vars)
    assert data.n_obs == 12 + second_size
    neighbors = pd.read_csv(tmp_path / "paga_neighbors_k.csv")
    assert list(neighbors.columns) == ["cluster", "neighbor", "rank", "connectivity"]
    if connected:
        local_df = pd.read_csv(tmp_path / "deg_local_k.csv", dtype={"group": str})
        assert set(local_df["group"]) == expected_groups
    else:
        assert neighbors.empty
        assert _load_paga_neighbors(tmp_path, "k") == {}


def test_deg_lookup_limits_each_view_after_filtering(tmp_path):
    rows = pd.DataFrame(
        {
            "group": ["0"] * 4,
            "names": ["G0", "G1", "G2", "G3"],
            "logfoldchanges": [0.1, 2.0, 3.0, 4.0],
            "pvals_adj": [0.01] * 4,
            "pct1": [0.8] * 4,
            "pct2": [0.1] * 4,
        }
    )
    for view in ("global", "local"):
        rows.to_csv(tmp_path / f"deg_{view}_k.csv", index=False)
    tables = DegTables(tmp_path, base_key="k")
    try:
        result = tables.lookup(cluster="0", top_n=2, min_logfc=1)
        assert "4 row(s) of 6 passing" in result
        assert result.count("G1 #2") == result.count("G2 #3") == 2
        assert "G0 #1" not in result and "G3 #4" not in result
    finally:
        tables.conn.close()
