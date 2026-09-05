"""The core-and-satellite statistics in msp.integrate, on synthetic inputs
built around each rule's thresholds."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

import msp.resources as resources
from msp.integrate import (
    BIG_SIBLING_FRAC,
    BIG_SIBLING_N,
    MIN_N_FOR_TEST,
    STRESS_HIT_THRESHOLD,
    _build_removal_mask,
    _cell_level_outliers,
    _cluster_annotations,
    _is_stress_gene,
    _leiden_cluster_qc_violins,
    _minor_sibling_qc,
    _select_fractal_markers,
    _stress_hits,
)

# ---------------------------------------------------------------- minor-sibling QC

# (subcluster, n_cells, mt centre, doublet centre, inherited _qc_action)
FRAGMENTS = [
    ("c0_0", 40, 5.0, 0.05, "keep"),  # parent 0 core
    ("c0_1", 8, 35.0, 0.60, "keep"),  # dirty: mt and doublet shifted above both floors
    ("c0_2", MIN_N_FOR_TEST - 1, 5.0, 0.05, "drop"),  # too small for stats, majority upstream drop
    ("c0_3", int(BIG_SIBLING_FRAC * 40), 35.0, 0.60, "keep"),  # exactly 25% of the core: big, skipped
    ("c0_4", 6, 5.0, 0.05, "keep"),  # clean minor sibling
    ("c0_5", 8, 8.0, 0.05, "keep"),  # mt shifted, but median below the 20% floor
    ("c1_0", 3300, 5.0, 0.05, "keep"),  # parent 1 core
    ("c1_1", BIG_SIBLING_N, 5.0, 0.05, "keep"),  # 24% of its core, but 800 cells outright: big
]


def fragment_dataset():
    rng = np.random.default_rng(7)
    frames, rows = [], []
    for sub, n, mt, dbl, action in FRAGMENTS:
        parent, rank = sub[1:].split("_")
        rows.append({"subcluster": sub, "parent": parent, "rank": int(rank), "n_cells": n})
        frames.append(
            pd.DataFrame(
                {
                    "standissect_product": sub,
                    "pct_counts_mt": np.clip(rng.normal(mt, 1.0, n), 0, None),
                    "doublet_score": np.clip(rng.normal(dbl, 0.02, n), 0, 1),
                    "_qc_action": action,
                }
            )
        )
    obs = pd.concat(frames, ignore_index=True)
    obs.index = [f"cell{i}" for i in range(len(obs))]
    obs["standissect_product"] = obs["standissect_product"].astype("category")
    data = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    return data, SimpleNamespace(fragments=pd.DataFrame(rows))


def test_minor_sibling_qc_applies_every_rule_at_its_threshold(tmp_path):
    data, res = fragment_dataset()
    df = _minor_sibling_qc(data, res, tmp_path).set_index("subcluster")
    assert set(df.index) == {s for s, *_ in FRAGMENTS if not s.endswith("_0")}  # cores are never rows

    assert df.loc["c0_3", "status"] == "big_sibling_skip" and df.loc["c1_1", "status"] == "big_sibling_skip"
    assert df.loc[["c0_3", "c1_1"], "recommend_removal"].isna().all()  # no verdict for big siblings

    assert df.loc["c0_2", "status"] == "insufficient_data"
    assert df.loc["c0_2", "pct_drop_upstream"] == 100.0 and df.loc["c0_2", "recommend_removal"] == True  # noqa: E712

    tested = df[df["status"] == "tested"]
    assert set(tested.index) == {"c0_1", "c0_4", "c0_5"}
    assert df.loc["c0_1", "n_hits"] == 2 and df.loc["c0_1", "recommend_removal"] == True  # noqa: E712
    assert df.loc["c0_1", "mt_significant"] and df.loc["c0_1", "doublet_significant"]
    assert df.loc["c0_4", "n_hits"] == 0 and df.loc["c0_4", "recommend_removal"] == False  # noqa: E712
    # shifted but below the absolute floor: the statistical hit alone does not count
    assert df.loc["c0_5", "mt_median"] < 20 and not df.loc["c0_5", "mt_significant"]
    assert df.loc["c0_5", "recommend_removal"] == False  # noqa: E712

    written = pd.read_csv(tmp_path / "minor_sibling_qc.csv").set_index("subcluster")
    assert written["recommend_removal"].astype(str).to_dict() == df["recommend_removal"].astype(str).to_dict()
    for col in ("core_n_cells", "frac_of_core", "pct_drop_upstream", "doublet_median", "mt_median"):
        assert col in written


def test_minor_sibling_qc_without_inherited_actions_skips_the_drop_rule(tmp_path):
    data, res = fragment_dataset()
    del data.obs["_qc_action"]
    df = _minor_sibling_qc(data, res, tmp_path).set_index("subcluster")
    assert "pct_drop_upstream" not in df
    assert df.loc["c0_2", "recommend_removal"] == False  # noqa: E712


# ---------------------------------------------------------------- cell-level outliers


def outlier_dataset():
    doublet = np.array([0.05] * 18 + [0.45, 0.90] + [0.60] * 19 + [0.95])
    obs = pd.DataFrame(
        {"msp_leiden_r1.0": pd.Categorical(["0"] * 20 + ["1"] * 20), "doublet_score": doublet},
        index=[f"cell{i}" for i in range(40)],
    )
    return ad.AnnData(np.zeros((40, 1), dtype=np.float32), obs=obs)


def test_cell_level_outliers_need_both_the_mad_and_the_floor_gate(tmp_path):
    data = outlier_dataset()
    df = _cell_level_outliers(data, ["msp_leiden_r1.0"], [1.0], tmp_path)
    flagged = set(df.index[df["recommend_removal"]])
    # cluster 0: MAD is 0, so 0.45 clears the MAD gate but not the 0.5 floor; 0.90 clears both.
    # cluster 1: 0.95 clears both; the 0.60 cells sit at the median.
    assert flagged == {"cell19", "cell39"}
    written = pd.read_csv(tmp_path / "cell_outliers.csv", dtype={"cell": str}).set_index("cell")
    assert set(written.index[written["recommend_removal"]]) == flagged
    summary = pd.read_csv(tmp_path / "cell_outlier_summary.csv", dtype={"cluster": str}).set_index("cluster")
    assert summary["n_doublet_score_outlier"].to_dict() == {"0": 1, "1": 1}
    assert summary["n_recommend_removal"].to_dict() == {"0": 1, "1": 1}
    assert summary["pct_recommend_removal"].to_dict() == {"0": 5.0, "1": 5.0}
    assert summary["n_cells"].sum() == 40


def test_cell_level_outliers_return_none_without_metrics_or_target_resolutions(tmp_path):
    data = outlier_dataset()
    assert _cell_level_outliers(data, ["msp_leiden_r0.3"], [0.3], tmp_path) is None
    del data.obs["doublet_score"]
    assert _cell_level_outliers(data, ["msp_leiden_r1.0"], [1.0], tmp_path) is None
    assert not (tmp_path / "cell_outliers.csv").exists()


def test_leiden_cutoff_violins_draw_one_file_per_metric_and_resolution(tmp_path):
    data = outlier_dataset()
    _leiden_cluster_qc_violins(data, ["msp_leiden_r1.0"], [1.0], tmp_path)
    (png,) = tmp_path.glob("leiden_qc_violin_*.png")
    assert png.name == "leiden_qc_violin_doublet_score_msp_leiden_r1.0.png"
    assert png.read_bytes().startswith(b"\x89PNG")


# ---------------------------------------------------------------- removal union


def removal_dataset():
    obs = pd.DataFrame(
        {
            "standissect_product": pd.Categorical(["c0_0", "c0_1", "c0_1", "c1_0"]),
            "_qc_action": pd.Categorical([None, "keep", "drop", "keep"]),
        },
        index=["a", "b", "c", "d"],
    )
    return ad.AnnData(np.zeros((4, 1), dtype=np.float32), obs=obs)


def test_removal_mask_unions_fragments_cells_and_inherited_drops_by_cell_id(tmp_path):
    data = removal_dataset()
    fragments = pd.DataFrame({"subcluster": ["c0_1", "c1_0"], "recommend_removal": [True, False]})
    cells = pd.DataFrame({"recommend_removal": [True, False]}, index=["d", "a"])  # partial, reordered
    mask = _build_removal_mask(data, fragments, cells, tmp_path)
    np.testing.assert_array_equal(mask, [False, True, True, True])
    written = pd.read_csv(tmp_path / "preannotation_removal.csv", dtype={"cell": str})
    assert written["cell"].tolist() == ["a", "b", "c", "d"]
    assert written["recommend_removal"].tolist() == [False, True, True, True]


def test_removal_mask_with_no_tables_keeps_only_inherited_drops(tmp_path):
    data = removal_dataset()
    np.testing.assert_array_equal(_build_removal_mask(data, None, None, tmp_path), [False, False, True, False])
    del data.obs["_qc_action"]
    del data.obs["standissect_product"]
    np.testing.assert_array_equal(_build_removal_mask(data, None, None, tmp_path), [False] * 4)


# ---------------------------------------------------------------- stress signature


def test_stress_gene_matching_is_case_insensitive_and_covers_mitochondrial_prefix():
    assert (
        _is_stress_gene("hspa1a") and _is_stress_gene("FOS") and _is_stress_gene("mt-Co1") and _is_stress_gene("MT-ND1")
    )
    assert not _is_stress_gene("ACTB") and not _is_stress_gene("MTOR")
    assert _stress_hits(["Fos", "ACTB", "mt-co1"]) == ["Fos", "mt-co1"]


def two_cluster_graph(monkeypatch):
    def paga(subset, groups):
        subset.uns["paga"] = {"connectivities": sparse.csr_matrix([[0, 1], [1, 0]])}

    monkeypatch.setattr(sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(sc.tl, "paga", paga)
    monkeypatch.setattr(resources, "available_cpus", lambda: 1)


@pytest.mark.parametrize("stress_names", [True, False])
def test_stress_clusters_flag_either_view_and_merge_the_verdict(tmp_path, monkeypatch, stress_names):
    rng = np.random.default_rng(1)
    genes = ["HSPA1A", "hspb1", "FOS", "mt-Co1", "JUN", "EGR1", "ATF3", "IER2"] if stress_names else list("ABCDEFGH")
    data = ad.AnnData(
        np.log1p(rng.poisson(2, (24, 8))).astype(float),
        obs=pd.DataFrame({"k": pd.Categorical(["0"] * 12 + ["1"] * 12)}, index=[f"c{i}" for i in range(24)]),
        var=pd.DataFrame(index=genes),
    )
    data.raw = data.copy()
    two_cluster_graph(monkeypatch)
    _cluster_annotations(data, np.zeros(data.n_obs, dtype=bool), ["k"], [1.0], tmp_path)
    stress = pd.read_csv(tmp_path / "stress_clusters.csv", dtype={"cluster": str})
    assert set(stress["view"]) == {"global", "local"} and set(stress["cluster"]) == {"0", "1"}
    if stress_names:
        # every displayed top gene is a stress gene, so both views trip the threshold
        assert (stress["n_hits"] == 8).all() and (stress["n_hits"] > STRESS_HIT_THRESHOLD).all()
        assert stress["stress"].all() and stress["recommend_removal"].all()
        assert all("mt-Co1" in h.split("|") for h in stress["hit_genes"])
    else:
        assert (stress["n_hits"] == 0).all() and not stress["stress"].any()
        assert not stress["recommend_removal"].any()


def test_stress_verdict_is_merged_across_views():
    """The per-(key, cluster) recommend_removal is the OR of its views."""
    rows = pd.DataFrame(
        {
            "key": ["k", "k", "k", "k"],
            "cluster": ["0", "0", "1", "1"],
            "view": ["global", "local", "global", "local"],
            "stress": [False, True, False, False],
        }
    )
    overall = rows.groupby(["key", "cluster"])["stress"].any().reset_index()
    merged = rows.merge(overall.rename(columns={"stress": "recommend_removal"}), on=["key", "cluster"])
    assert merged.set_index(["cluster", "view"])["recommend_removal"].to_dict() == {
        ("0", "global"): True,
        ("0", "local"): True,
        ("1", "global"): False,
        ("1", "local"): False,
    }


# ---------------------------------------------------------------- fractal markers


def test_fractal_marker_selection_filters_orders_and_deduplicates():
    de = pd.DataFrame(
        {
            "group": ["10", "10", "10", "10", "2", "2", "2"],
            "names": ["SHARED", "RPL3", "LOW", "NEG", "SHARED", "P2A", "P2B"],
            "logfoldchanges": [3.0, 5.0, 1.0, -2.0, 4.0, 2.5, 2.0],
            "pvals_adj": [1e-5, 1e-9, 0.2, 1e-5, 1e-5, 1e-3, 1e-3],
        }
    )
    rows, gene_parent, markers = _select_fractal_markers(de, ribo={"RPL3"}, top_n=2)
    # parent 2 first (numeric order), its top-2 by logFC, then parent 10 minus the duplicate
    assert markers == ["SHARED", "P2A"]
    assert "RPL3" not in markers and "NEG" not in markers
    assert gene_parent["SHARED"] == "2"  # the strip colour follows the lowest parent, like the row order
    by_parent = {p: [r["gene"] for r in rows if r["parent"] == p] for p in ("2", "10")}
    assert by_parent == {"2": ["SHARED", "P2A"], "10": ["SHARED"]}
    assert [r["rank"] for r in rows if r["parent"] == "2"] == [1, 2]


def test_fractal_marker_selection_handles_no_passing_genes():
    de = pd.DataFrame({"group": ["0"], "names": ["X"], "logfoldchanges": [-1.0], "pvals_adj": [0.5]})
    assert _select_fractal_markers(de, ribo=set(), top_n=5) == ([], {}, [])
