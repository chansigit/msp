"""One real integration run on the synthetic two-sample dataset: no scanpy
monkeypatching, every artifact of the integrate stage written for real."""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from synthetic_data import CELLS_PER_POPULATION, N_POPULATIONS, SAMPLES, write_samples

from msp import generate_report, run_multi_sample_pipeline
from msp.evidence import DegTables, load_paga_neighbors, load_removal_mask
from msp.steps import begin_step, step_pending

N_CELLS = len(SAMPLES) * N_POPULATIONS * CELLS_PER_POPULATION


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    inputs = write_samples(root)
    out = root / "msp_out"
    begin_step(out, "annotate")  # an older interrupted annotation is superseded by a fresh upstream run
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("msp")
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)  # a bare interpreter leaves the family at WARNING
    try:
        data, summary = run_multi_sample_pipeline(
            inputs,
            batch_col="sample_id",
            outdir=out,
            species="human",
            resolutions=(0.3, 1.0, 2.0),
            n_top_genes=30,
            n_pcs=10,
            n_neighbors=10,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    LOG_RECORDS.extend(records)
    return out, data, summary


LOG_RECORDS: list[logging.LogRecord] = []


def test_pipeline_progress_goes_through_the_msp_logger(integrated):
    messages = [r.getMessage() for r in LOG_RECORDS]
    assert any(m.startswith("== integrating:") for m in messages)
    assert any(m.startswith("== wrote ") and m.endswith("integrated.h5ad") for m in messages)
    assert {r.name.split(".")[0] for r in LOG_RECORDS} == {"msp"}
    assert all(r.levelno >= logging.INFO for r in LOG_RECORDS)


def test_pipeline_writes_every_integrate_artifact(integrated):
    out, data, summary = integrated
    assert summary["n_cells"] == N_CELLS and summary["n_samples"] == 2
    assert not any(step_pending(out, step) for step in ("integrate", "inspect", "annotate"))
    for name in (
        "integrated.h5ad",
        "integration_summary.csv",
        "per_sample_qc.csv",
        "cluster_qc_standissect_product.csv",
        "cluster_qc_msp_leiden_r1.0.csv",
        "cluster_qc_msp_leiden_r2.0.csv",
        "fragments_msp_leiden_r0.3.csv",
        "minor_sibling_qc.csv",
        "cell_outliers.csv",
        "cell_outlier_summary.csv",
        "preannotation_removal.csv",
        "deg_global_msp_leiden_r1.0.csv",
        "deg_global_msp_leiden_r2.0.csv",
        "paga_neighbors_msp_leiden_r1.0.csv",
        "stress_clusters.csv",
        "figures/umap_sample_id.png",
        "figures/umap_msp_leiden_r1.0.png",
        "figures/umap__qc_action.png",
        "figures/standissect_product.png",
        "figures/umap_preannotation_removal.png",
        "figures/qc_umap_pct_counts_mt.png",
        "figures/qc_violin_doublet_score.png",
        "figures/leiden_qc_violin_doublet_score_msp_leiden_r1.0.png",
    ):
        assert (out / name).is_file(), name


def test_integrated_object_contract(integrated):
    out, data, _ = integrated
    stored = ad.read_h5ad(out / "integrated.h5ad")
    assert stored.shape == data.shape == (N_CELLS, 60)
    np.testing.assert_array_equal(stored.layers["counts"].sum(axis=1), data.layers["counts"].sum(axis=1))
    assert stored.X.max() < 10  # log1p-normalized, not counts
    for key in ("msp_leiden_r0.3", "msp_leiden_r1.0", "msp_leiden_r2.0", "standissect_product"):
        assert key in stored.obs and str(stored.obs[key].dtype) == "category"
    assert stored.obsm["X_pca"].shape == stored.obsm["X_pca_harmony"].shape == (N_CELLS, 10)
    assert stored.obsm["X_umap"].shape == (N_CELLS, 2)
    assert stored.var["highly_variable"].sum() == 30
    # inherited columns ride along; sample-local leiden labels are prefixed
    assert set(stored.obs["leiden_r1.0"].astype(str).str[:2]) == {"A:", "B:"}
    assert stored.obs["_qc_action"].astype(str).value_counts()["drop"] == 2
    meta = stored.uns["msp"]
    assert meta["batch_col"] == "sample_id" and meta["species"] == "human" and meta["n_batches"] == 2
    assert list(meta["resolutions"]) == [0.3, 1.0, 2.0] and meta["n_pcs"] == 10
    assert meta["harmony"] == {}  # harmonypy defaults, recorded as no overrides


def test_populations_are_recovered_and_evidence_tables_agree(integrated):
    out, data, _ = integrated
    # the three planted populations dominate three r1.0 clusters
    crosstab = pd.crosstab(data.obs["msp_leiden_r1.0"], data.obs["population"])
    assert (crosstab.max(axis=1) / crosstab.sum(axis=1) > 0.9).all()
    assert data.obs["msp_leiden_r1.0"].nunique() >= 3
    # removal mask on disk matches the union rule and excludes the inherited drops
    mask = load_removal_mask(out, data)
    assert mask[(data.obs["_qc_action"].astype(str) == "drop").to_numpy()].all()
    deg = pd.read_csv(out / "deg_global_msp_leiden_r1.0.csv", dtype={"group": str})
    assert set(deg.columns) >= {"group", "names", "logfoldchanges", "pvals_adj", "pct1", "pct2"}
    assert deg.groupby("group").size().max() <= 50
    paga = load_paga_neighbors(out, "msp_leiden_r1.0")
    assert paga and all(len(v) <= 3 for v in paga.values())
    with DegTables(out, base_key="msp_leiden_r1.0") as tables:
        assert "msp_leiden_r1.0" in tables.keys and "msp_leiden_r2.0" in tables.keys
        assert "precomputed DEG" in tables.lookup(cluster=tables.clusters("msp_leiden_r1.0")[0])
        assert "cluster_qc_msp_leiden_r1_0" in tables.extra_tables and "cell_outliers" not in tables.extra_tables


def test_report_renders_every_integrate_section(integrated):
    out, _, _ = integrated
    text = Path(generate_report(out)).read_text()
    for heading in (
        "1. Sample Summary",
        "2. UMAPs (integrated space)",
        "3. Per-cluster QC (standissect clusters)",
        "4. Leiden Cluster QC",
        "5. Cluster Annotations",
    ):
        assert heading in text, heading
    assert "Integration QC Inspection" not in text and "Cell Type Annotation" not in text
    assert "Incomplete steps" not in text
