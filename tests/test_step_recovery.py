"""Exercise invalidation, interrupted reruns, and report-only recovery."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import msp.annotate as annotate
import msp.inspect as inspect
import msp.integrate as integrate
import msp.steps as steps
from msp.report import generate_report
from msp.steps import begin_step, complete_step, step_pending


def integrated_data():
    data = ad.AnnData(
        np.ones((20, 3)),
        obs=pd.DataFrame(
            {
                "batch": ["A"] * 10 + ["B"] * 10,
                "msp_leiden_r1.0": pd.Categorical(["0"] * 20),
                "msp_leiden_r2.0": pd.Categorical(["0"] * 10 + ["1"] * 10),
                "_msp_action": pd.Categorical(["keep"] * 20),
                "_msp_verdict": pd.Categorical(["real"] * 20),
            },
            index=[f"cell{i}" for i in range(20)],
        ),
    )
    data.layers["counts"] = data.X.copy()
    data.obsm["X_umap"] = np.arange(40).reshape(20, 2).astype(float)
    data.uns["msp"] = {
        "batch_col": "batch",
        "species": "",
        "resolutions": [0.3, 1.0, 2.0],
        "n_top_genes": 2000,
        "n_pcs_requested": 50,
        "n_neighbors": 15,
        "harmony": {},
        "inputs": ["input.h5ad"],
        "n_batches": 2,
    }
    return data


def inspection_proposal():
    return {
        "cluster_key": "msp_leiden_r1.0",
        "clusters": [{"cluster": "0", "action": "keep", "verdict": "real"}],
    }


def annotation_proposal(label):
    return {
        "cluster_key": "msp_leiden_r2.0",
        "merged_groups": [],
        "overall": label,
        "clusters": [
            {
                "cluster_id": c,
                "coarse_label": label,
                "fine_label": f"{label}-{c}",
                "action": "keep",
                "merge_target": None,
                "evidence": {},
            }
            for c in ("0", "1")
        ],
    }


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def completed_run(tmp_path):
    data = integrated_data()
    data.write_h5ad(tmp_path / "integrated.h5ad")
    data.write_h5ad(tmp_path / "annotated.h5ad")
    write_json(tmp_path / "inspection_proposal.json", inspection_proposal())
    write_json(tmp_path / "annotation_proposal.json", annotation_proposal("OLD_LABEL"))
    (tmp_path / "inspection_notes.md").write_text("old inspection notes")
    (tmp_path / "annotation_notes.md").write_text("old annotation notes")
    (tmp_path / "annotation_removed.csv").write_text("cell,annotate_remove\n")
    (tmp_path / "integration_summary.csv").write_text("n_cells,20\n")
    (tmp_path / "deg_global_msp_leiden_r1.0.csv").write_text("group,names,logfoldchanges\n")
    (tmp_path / "report_context.txt").write_text("test context")
    (tmp_path / "sample_decisions.csv").write_text("sample,decision\nA,include\n")
    (tmp_path / "caller-input.txt").write_text("leave intact")
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ("inspect_umap_action.png", "annotation_umap_coarse.png", "umap_batch.png"):
        (figures / name).write_bytes(b"test image bytes")
    generate_report(tmp_path)
    return tmp_path


def install_agents(monkeypatch, calls):
    """Replace model work and plotting, keeping the public data-writing paths."""

    async def inspect_agent(data, outdir, *args):
        calls.append("inspect")
        assert "_msp_action" not in data.obs and "_msp_verdict" not in data.obs
        proposal = inspection_proposal()
        write_json(Path(outdir) / "inspection_proposal.json", proposal)
        return proposal

    async def annotate_agent(data, outdir, *args):
        calls.append("annotate")
        proposal = annotation_proposal("NEW_LABEL")
        write_json(Path(outdir) / "annotation_proposal.json", proposal)
        return proposal

    def plot_inspection(data, figdir):
        path = Path(figdir)
        path.mkdir(exist_ok=True)
        (path / "inspect_umap_action.png").write_bytes(b"new inspection")

    def plot_annotation(full, kept, figdir):
        path = Path(figdir)
        path.mkdir(exist_ok=True)
        for suffix in ("coarse", "fine", "removed"):
            (path / f"annotation_umap_{suffix}.png").write_bytes(b"new annotation")

    monkeypatch.setattr(inspect, "_run_agent", inspect_agent)
    monkeypatch.setattr(annotate, "_run_agent", annotate_agent)
    monkeypatch.setattr(inspect, "_plot_verdicts", plot_inspection)
    monkeypatch.setattr(annotate, "_plot", plot_annotation)


def run_cli(outdir, monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "msp",
            "input.h5ad",
            "--batch-col",
            "batch",
            "--outdir",
            str(outdir),
            "--annotate",
            "--model",
            "test-model",
            *extra,
        ],
    )
    runpy.run_module("msp.__main__", run_name="__main__")


def forbid_integration(*args, **kwargs):
    pytest.fail("completed integration was unexpectedly repeated")


def test_integration_rerun_invalidates_all_outputs(completed_run, monkeypatch):
    root = completed_run
    (root / "figures" / "umap_msp_leiden_r9.0.png").write_bytes(b"stale resolution")
    (root / "deg_local_msp_leiden_r9.0.csv").write_text("stale DE")
    data = integrated_data()

    # Fail immediately after invalidation, before any new numerical work.
    def fail_normalize(*args, **kwargs):
        raise RuntimeError("interrupted integration")

    monkeypatch.setattr(integrate.sc.pp, "normalize_total", fail_normalize)
    with pytest.raises(RuntimeError, match="interrupted integration"):
        integrate.integrate_adata(data, "batch", root)
    assert step_pending(root, "integrate")
    for name in (
        "integrated.h5ad",
        "annotated.h5ad",
        "inspection_proposal.json",
        "annotation_proposal.json",
        "deg_local_msp_leiden_r9.0.csv",
    ):
        assert not (root / name).exists()
        assert len(list((root / ".msp-history").glob(f"*/{name}"))) == 1
    assert not (root / "figures" / "umap_msp_leiden_r9.0.png").exists()
    assert not any(c.startswith("msp_leiden_r") for c in data.obs)
    assert "_msp_action" not in data.obs
    for name in ("sample_decisions.csv", "report_context.txt", "caller-input.txt"):
        assert (root / name).is_file()
    # Interrupted computations may have left invalid CSVs: report must not read them.
    (root / "deg_global_broken.csv").write_text("invalid partial table")
    report = Path(generate_report(root)).read_text()
    assert "Incomplete steps: integrate, inspect, annotate" in report
    assert "OLD_LABEL" not in report
    with pytest.raises(RuntimeError, match="integrate is incomplete"):
        inspect.inspect_clusters(root)
    with pytest.raises(RuntimeError, match="integrate is incomplete"):
        annotate.annotate_clusters(root)


def test_failed_inspection_blocks_annotation_and_recovers_via_cli(completed_run, monkeypatch):
    root = completed_run
    old_integrated = (root / "integrated.h5ad").read_bytes()
    calls = []
    install_agents(monkeypatch, calls)

    async def fail_after_submission(data, outdir, *args):
        write_json(Path(outdir) / "inspection_proposal.json", inspection_proposal())
        raise RuntimeError("interrupted inspection")

    monkeypatch.setattr(inspect, "_run_agent", fail_after_submission)
    with pytest.raises(RuntimeError, match="interrupted inspection"):
        inspect.inspect_clusters(root, model="test-model")
    assert step_pending(root, "inspect") and not step_pending(root, "integrate")
    assert (root / "integrated.h5ad").read_bytes() == old_integrated
    assert not (root / "annotated.h5ad").exists()
    snapshot = next((root / ".msp-history").glob("inspect-*/integrated.h5ad"))
    assert snapshot.read_bytes() == old_integrated
    with pytest.raises(RuntimeError, match="inspect is incomplete"):
        annotate.annotate_clusters(root, model="test-model")
    report = Path(generate_report(root)).read_text()
    assert "Sample Summary" in report and "OLD_LABEL" not in report
    assert "Incomplete steps: inspect, annotate" in report

    install_agents(monkeypatch, calls)
    monkeypatch.setattr(integrate, "run_multi_sample_pipeline", forbid_integration)
    run_cli(root, monkeypatch)
    assert calls == ["inspect", "annotate"]
    assert not step_pending(root, "annotate")
    assert "NEW_LABEL" in (root / "report.html").read_text()
    assert set(ad.read_h5ad(root / "annotated.h5ad").obs["msp_ann_coarse"]) == {"NEW_LABEL"}
    assert snapshot.read_bytes() == old_integrated
    assert (root / "integrated.h5ad").read_bytes() != old_integrated


def test_failed_annotation_files_do_not_satisfy_resume(completed_run, monkeypatch):
    calls = []
    install_agents(monkeypatch, calls)

    async def fail_after_writing_files(data, outdir, *args):
        write_json(Path(outdir) / "annotation_proposal.json", annotation_proposal("PARTIAL_LABEL"))
        data.write_h5ad(Path(outdir) / "annotated.h5ad")
        raise RuntimeError("interrupted annotation")

    monkeypatch.setattr(annotate, "_run_agent", fail_after_writing_files)
    with pytest.raises(RuntimeError, match="interrupted annotation"):
        annotate.annotate_clusters(completed_run, model="test-model")
    assert step_pending(completed_run, "annotate")
    report = Path(generate_report(completed_run)).read_text()
    assert "PARTIAL_LABEL" not in report and "OLD_LABEL" not in report
    install_agents(monkeypatch, calls)
    monkeypatch.setattr(integrate, "run_multi_sample_pipeline", forbid_integration)
    run_cli(completed_run, monkeypatch)
    assert calls == ["annotate"]
    assert not step_pending(completed_run, "annotate")


def test_report_failure_does_not_repeat_completed_agent(completed_run, monkeypatch):
    calls = []
    install_agents(monkeypatch, calls)

    def fail_report(*args, **kwargs):
        raise OSError("report storage unavailable")

    monkeypatch.setattr(annotate, "generate_report", fail_report)
    with pytest.raises(OSError, match="report storage unavailable"):
        annotate.annotate_clusters(completed_run, model="test-model")
    assert not step_pending(completed_run, "annotate")
    assert not (completed_run / "report.html").exists()
    monkeypatch.setattr(integrate, "run_multi_sample_pipeline", forbid_integration)
    run_cli(completed_run, monkeypatch)
    assert calls == ["annotate"]
    assert "NEW_LABEL" in (completed_run / "report.html").read_text()


def test_successful_inspection_alone_invalidates_annotation(completed_run, monkeypatch):
    calls = []
    install_agents(monkeypatch, calls)
    inspect.inspect_clusters(completed_run, model="test-model")
    assert calls == ["inspect"]
    assert not (completed_run / "annotated.h5ad").exists()
    assert not (completed_run / "annotation_proposal.json").exists()
    assert "OLD_LABEL" not in (completed_run / "report.html").read_text()
    assert not step_pending(completed_run, "inspect")


def test_cli_changed_integration_recomputes_both_agents(completed_run, monkeypatch):
    calls = []
    install_agents(monkeypatch, calls)

    def recompute(inputs, batch_col, outdir, **kwargs):
        calls.append("integrate")
        begin_step(outdir, "integrate")
        data = integrated_data()
        data.obs_names = [f"NEW_{name}" for name in data.obs_names]
        data.write_h5ad(Path(outdir) / "integrated.h5ad")
        (Path(outdir) / "deg_global_msp_leiden_r1.0.csv").write_text("group,names,logfoldchanges\n")
        complete_step(outdir, "integrate")
        return data, {"n_cells": data.n_obs}

    monkeypatch.setattr(integrate, "run_multi_sample_pipeline", recompute)
    run_cli(completed_run, monkeypatch, "--n-pcs", "25")
    assert calls == ["integrate", "inspect", "annotate"]
    result = ad.read_h5ad(completed_run / "annotated.h5ad")
    assert all(name.startswith("NEW_") for name in result.obs_names)
    assert not step_pending(completed_run, "annotate")


def test_report_only_never_clears_pending(completed_run, monkeypatch):
    begin_step(completed_run, "annotate")
    marker = completed_run / ".msp-state" / "annotate.pending"
    before = marker.stat().st_mtime_ns
    monkeypatch.setattr(sys, "argv", ["msp.report", str(completed_run)])
    runpy.run_module("msp.report", run_name="__main__")
    assert marker.stat().st_mtime_ns == before
    assert "Incomplete steps: annotate" in (completed_run / "report.html").read_text()


def test_interrupted_archive_preserves_files_and_blocks_resume(completed_run, monkeypatch):
    original = {
        name: (completed_run / name).read_bytes() for name in ("report.html", "integrated.h5ad", "annotated.h5ad")
    }
    replace = steps.os.replace
    moves = []

    def fail_second_move(src, dst):
        moves.append(src)
        if len(moves) == 2:
            raise OSError("archive interrupted")
        replace(src, dst)

    monkeypatch.setattr(steps.os, "replace", fail_second_move)
    with pytest.raises(OSError, match="archive interrupted"):
        begin_step(completed_run, "integrate")
    assert moves[0].name == "report.html"
    assert step_pending(completed_run, "annotate")
    monkeypatch.setattr(steps.os, "replace", replace)
    begin_step(completed_run, "integrate")
    for name, contents in original.items():
        assert not (completed_run / name).exists()
        archived = list((completed_run / ".msp-history").glob(f"*/{name}"))
        assert len(archived) == 1 and archived[0].read_bytes() == contents


def test_completed_integration_allows_external_annotation_report(completed_run, monkeypatch):
    """Exercise the real integration entry/exit and the ZMIP-style report path."""
    data = integrated_data()
    # A new upstream run supersedes an older interrupted annotation, too.
    begin_step(completed_run, "annotate")

    monkeypatch.setattr(integrate.sc.pp, "normalize_total", lambda *args, **kwargs: None)
    monkeypatch.setattr(integrate.sc.pp, "log1p", lambda *args, **kwargs: None)

    def hvg(subset, **kwargs):
        subset.var["highly_variable"] = True

    def leiden(subset, key_added, **kwargs):
        subset.obs[key_added] = pd.Categorical(["0"] * subset.n_obs)

    def umap(subset):
        subset.obsm["X_umap"] = np.arange(subset.n_obs * 2).reshape(-1, 2).astype(float)

    monkeypatch.setattr(integrate.sc.pp, "highly_variable_genes", hvg)
    monkeypatch.setattr(integrate.sc.pp, "scale", lambda *args, **kwargs: None)
    monkeypatch.setattr(integrate.sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(integrate.sc.tl, "leiden", leiden)
    monkeypatch.setattr(integrate.sc.tl, "umap", umap)
    monkeypatch.setattr(
        integrate,
        "PCA",
        lambda **kwargs: SimpleNamespace(fit_transform=lambda x: np.zeros((len(x), kwargs["n_components"]))),
    )
    data.obs["batch"] = "A"  # One batch takes the existing Harmony skip branch.
    # Isolate numerical libraries from this filesystem recovery test.
    monkeypatch.setitem(sys.modules, "harmonypy", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace(mps=None)),
    )
    partition = SimpleNamespace(
        labels=pd.DataFrame({"subcluster": ["c0_0"] * data.n_obs}, index=data.obs_names),
        fragments=pd.DataFrame({"subcluster": ["c0_0"]}),
        overlap=pd.DataFrame(),
    )
    monkeypatch.setitem(
        sys.modules, "standissect_lite", SimpleNamespace(dissect_partition=lambda *args, **kwargs: partition)
    )
    for name in (
        "_minor_sibling_qc",
        "_cell_level_outliers",
        "_leiden_cluster_qc_violins",
        "_preannotation_removal_umap",
        "_cluster_annotations",
        "save_single_umap",
        "_qc_outputs",
        "_fractal_marker_heatmap",
    ):
        monkeypatch.setattr(integrate, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(integrate, "_build_removal_mask", lambda *args: np.zeros(data.n_obs, dtype=bool))

    result, summary = integrate.integrate_adata(data, "batch", completed_run)
    assert summary["n_cells"] == data.n_obs
    assert not step_pending(completed_run, "integrate")
    assert not step_pending(completed_run, "annotate")
    assert not (completed_run / "annotation_proposal.json").exists()
    assert "_msp_action" not in ad.read_h5ad(completed_run / "integrated.h5ad").obs

    # External callers already write these public files; no new API is required.
    write_json(completed_run / "annotation_proposal.json", annotation_proposal("EXTERNAL_LABEL"))
    result.write_h5ad(completed_run / "annotated.h5ad")
    assert "EXTERNAL_LABEL" in Path(generate_report(completed_run)).read_text()
