"""Regressions for inherited metadata, evidence retrieval, and host validation."""

import asyncio
import copy
import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import msp.annotate as annotate
import msp.harness as harness
import msp.inspect as inspect
from msp.integrate import _mwu_greater, _qc_outputs, load_and_merge
from msp.report import generate_report
from msp.steps import step_pending


def data_with_clusters(labels=("0", "1")):
    data = ad.AnnData(
        np.ones((len(labels), 2)),
        obs=pd.DataFrame(
            {
                "batch": pd.Categorical(["A"] * len(labels)),
                annotate.BASE_KEY: pd.Categorical(labels),
                annotate.PARENT_KEY: pd.Categorical(["0"] * len(labels)),
                "doublet_score": np.linspace(0, 1, len(labels)),
            },
            index=[f"cell{i}" for i in range(len(labels))],
        ),
    )
    data.obsm["X_umap"] = np.arange(len(labels) * 2).reshape(-1, 2).astype(float)
    data.uns["msp"] = {"batch_col": "batch"}
    return data


def inspection_entry(cluster="0"):
    return {
        "cluster": cluster,
        "verdict": "real",
        "action": "keep",
        "confidence": "medium",
        "tests": dict.fromkeys(
            ("markers", "qc", "composition", "geometry", "stability"),
            "Evidence unavailable in this synthetic fixture; retain for review.",
        ),
        "rationale": "Synthetic fixture for host validation.",
    }


def annotation_entry(cluster="0", **updates):
    return {
        "cluster_id": cluster,
        "coarse_label": "Immune",
        "fine_label": f"Type {cluster}",
        "merge_target": None,
        "action": "keep",
        "confidence": "low",
        "evidence": dict.fromkeys(("distinctness", "markers", "merge"), "Insufficient evidence; defer."),
        "rationale": "Retain this uncertain population for review.",
        **updates,
    }


def test_merge_preserves_partial_metadata_counts_and_h5ad_roundtrip(tmp_path):
    first = data_with_clusters()
    first.obs["_qc_action"] = pd.Categorical(["drop", "keep"])
    first.obs["_ann_coarse"] = pd.Categorical(["T", "B"])
    first.obs["predicted_doublet"] = [True, False]
    first.layers["counts"] = np.array([[2, 3], [4, 5]], dtype=np.int32)
    second = data_with_clusters()
    second.obs_names = ["other0", "other1"]
    second.obs["batch"] = "B"
    del second.obs["doublet_score"]
    second.layers["counts"] = np.array([[6, 7], [8, 9]], dtype=np.int32)
    paths = [tmp_path / "first.h5ad", tmp_path / "second.h5ad"]
    first.write_h5ad(paths[0])
    second.write_h5ad(paths[1])

    merged = load_and_merge(paths, "batch")
    merged.write_h5ad(tmp_path / "merged.h5ad")
    actual = ad.read_h5ad(tmp_path / "merged.h5ad")
    np.testing.assert_array_equal(actual.layers["counts"], [[2, 3], [4, 5], [6, 7], [8, 9]])
    for col in ("_qc_action", "_ann_coarse", "doublet_score", "predicted_doublet"):
        assert actual.obs[col].iloc[:2].tolist() == first.obs[col].tolist()
        assert actual.obs[col].iloc[2:].isna().all()
    assert actual.obs_names.tolist() == ["cell0", "cell1", "other0", "other1"]
    assert "nan|nan" in inspect._qc_table(actual, "batch", "batch")


@pytest.mark.parametrize("mismatch", ["genes", "barcodes"])
def test_merge_still_rejects_incompatible_inputs(tmp_path, mismatch):
    first, second = data_with_clusters(), data_with_clusters()
    for data in (first, second):
        data.layers["counts"] = data.X.copy()
    if mismatch == "genes":
        second.var_names = ["different", "genes"]
    paths = [tmp_path / "a.h5ad", tmp_path / "b.h5ad"]
    first.write_h5ad(paths[0])
    second.write_h5ad(paths[1])
    with pytest.raises(ValueError, match="var axis differs|duplicated barcodes"):
        load_and_merge(paths, "batch")


@pytest.mark.parametrize("actions", [["keep", None], [None, None]])
def test_missing_qc_actions_render_and_remain_missing_in_summaries(tmp_path, actions):
    data = data_with_clusters()
    del data.obs["doublet_score"]
    data.obs["batch"] = pd.Categorical(["A", "B"])
    data.obs["_qc_action"] = pd.Categorical(actions)
    figures = tmp_path / "figures"
    figures.mkdir()
    _qc_outputs(data, "batch", annotate.BASE_KEY, tmp_path, figures)
    assert (figures / "umap__qc_action.png").read_bytes().startswith(b"\x89PNG")
    summary = pd.read_csv(tmp_path / "per_sample_qc.csv", index_col=0)
    assert pd.isna(summary.loc["B", "pct_drop"])
    assert pd.isna(data.obs["_qc_action"].iloc[1])
    if actions[0] == "keep":
        assert summary.loc["A", "pct_drop"] == 0


def test_minor_fragment_qc_uses_only_available_measurements():
    assert _mwu_greater([0.8] * 6 + [np.nan], [0.1] * 10 + [np.nan]) is True
    assert _mwu_greater([np.nan] * 20, [0.1] * 10) is None
    assert _mwu_greater([0.8] * 6, [0.1] * 3 + [np.nan] * 10) is None


def test_filtered_deg_cache_falls_back_once_then_reuses_complete_table(tmp_path, monkeypatch):
    data = data_with_clusters()
    frame = pd.DataFrame(
        {
            "group": ["0"] * 60,
            "names": [f"G{i}" for i in range(60)],
            "logfoldchanges": [0.1] * 50 + [3.0] * 10,
            "pvals_adj": [0.01] * 60,
            "pct1": [0.9] * 60,
            "pct2": [0.1] * 60,
        }
    )
    frame.iloc[:50].to_csv(tmp_path / f"deg_global_{annotate.BASE_KEY}.csv", index=False)
    calls = []

    def compute(*args):
        calls.append(args)
        return frame.copy()

    monkeypatch.setattr(inspect, "_deg_frame", compute)
    cache = inspect.DegCache(data, tmp_path, np.zeros(2, dtype=bool))
    cache.table(annotate.BASE_KEY, "0", "rest", 5)
    assert not calls
    text = cache.table(annotate.BASE_KEY, "0", "rest", 5, min_logfc=1)
    assert "G50 " in text and "G54 " in text and "G0 " not in text
    assert "5 gene(s) of 10 passing" in text
    assert "G59 " in cache.table(annotate.BASE_KEY, "0", "rest", 15, min_logfc=1)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("5,1", ("5,1",)),
        ('"5,0","5,1"', ("5,0", "5,1")),
        ("1,2", ("1", "2")),
        ("rest", "rest"),
    ],
)
def test_reference_preserves_subcluster_ids(reference, expected):
    assert inspect._parse_reference(reference, ["1", "2", "5", "5,0", "5,1"]) == expected


def test_ambiguous_reference_requests_quoted_ids():
    with pytest.raises(ValueError, match="CSV-quote"):
        inspect._parse_reference("5,0,5,1", ["0", "1", "5", "5,0", "5,1"])
    with pytest.raises(ValueError, match="unknown reference"):
        inspect._parse_reference("missing", ["0"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", "certain"),
        ("tests", None),
        ("tests", dict.fromkeys(("markers", "qc", "composition", "geometry", "stability"))),
        ("rationale", " "),
        ("cluster", "unknown"),
        ("cluster", []),
    ],
)
def test_inspection_rejects_invalid_evidence(field, value):
    entry = inspection_entry()
    entry[field] = value
    assert inspect._validate_proposal({"clusters": [entry]}, ["0"], data_with_clusters().obs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("metric", []),
        ("op", {}),
        ("value", float("nan")),
        ("value", float("inf")),
        ("value", True),
        ("value", None),
        ("reason", None),
        ("note", ""),
    ],
)
def test_cell_rules_return_errors_instead_of_raising(field, value):
    rule = {
        "cluster": "0",
        "metric": "doublet_score",
        "op": ">",
        "value": 0.5,
        "action": "drop",
        "reason": "doublet",
        "note": "High score with mixed markers.",
    }
    rule[field] = value
    assert inspect._validate_proposal(
        {"clusters": [inspection_entry()], "cell_actions": [rule]}, ["0"], data_with_clusters().obs
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence", {"distinctness": None, "markers": None, "merge": None}),
        ("rationale", None),
        ("rationale", []),
        ("merge_target", {}),
        ("confidence", []),
    ],
)
def test_annotation_rejects_empty_or_malformed_evidence(field, value):
    entry = annotation_entry(**{field: value})
    assert annotate._validate_cluster(entry, ["0"])
    assert annotate._validate_final({"0": entry}, ["0"])


def test_merge_and_drop_defenses_remain_enforced():
    entries = {c: annotation_entry(c) for c in ("0", "1")}
    assert not annotate._validate_final(entries, ["0", "1"])
    entries["1"]["fine_label"] = entries["0"]["fine_label"]
    assert any("not merged" in p for p in annotate._validate_final(entries, ["0", "1"]))
    entries["1"]["merge_target"] = "0"
    assert not annotate._validate_final(entries, ["0", "1"])
    entries["0"].update(action="remove", remove_reason="other")
    assert any("action=remove" in p for p in annotate._validate_final(entries, ["0", "1"]))
    data = data_with_clusters()
    proposal = {
        "clusters": [inspection_entry("0"), inspection_entry("1")],
        "cell_actions": [{"cluster": "0", "action": "flag", "metric": "doublet_score", "op": ">=", "value": 0}],
    }
    proposal["clusters"][0]["action"] = "drop"
    inspect._apply_proposal(data, annotate.BASE_KEY, proposal)
    assert data.obs["_msp_action"].tolist() == ["drop", "keep"]


def test_inspection_tool_allows_correction_and_subcluster_reference(tmp_path, monkeypatch):
    data = data_with_clusters(("5,0", "5,1"))
    proposal = {"clusters": [inspection_entry(c) for c in ("5,0", "5,1")]}
    refs = []
    frame = pd.DataFrame({"names": ["G"], "logfoldchanges": [2.0], "pvals_adj": [0.01], "pct1": [0.9], "pct2": [0.1]})

    def compute(data, key, cluster, reference, mask):
        refs.append(reference)
        return frame

    async def run_agent(**kwargs):
        tools = {t.name: t.handler for t in kwargs["tools"]}
        malformed = copy.deepcopy(proposal)
        malformed["clusters"][0]["tests"] = None
        error = await tools["submit_inspection"]({"proposal_json": json.dumps(malformed)})
        assert error["is_error"] and not (tmp_path / "inspection_proposal.json").exists()
        assert (await tools["submit_inspection"]({"proposal_json": None}))["is_error"]
        result = await tools["check_deg"]({"cluster": "5,0", "reference": "5,1"})
        assert not result.get("is_error")
        result = await tools["submit_inspection"]({"proposal_json": json.dumps(proposal)})
        return SimpleNamespace(submitted=result["_submitted"], transcript_text="Inspection notes")

    monkeypatch.setattr(inspect, "_deg_frame", compute)
    monkeypatch.setattr(harness, "run_agent", run_agent)
    result = asyncio.run(
        inspect._run_agent(
            data,
            tmp_path,
            annotate.BASE_KEY,
            [],
            "batch",
            None,
            "English",
            "test-model",
            None,
            10,
            np.zeros(2, dtype=bool),
        )
    )
    assert len(result["clusters"]) == 2
    assert refs == [("5,1",)]


def test_all_removed_annotation_delivers_empty_h5ad_and_real_plots(tmp_path, monkeypatch):
    data = data_with_clusters()
    data.write_h5ad(tmp_path / "integrated.h5ad")

    async def run_agent(**kwargs):
        tools = {t.name: t.handler for t in kwargs["tools"]}
        assert (await tools["finalize_annotation"]({"overall": "Too early"}))["is_error"]
        for c in ("0", "1"):
            entry = annotation_entry(c, action="remove", remove_reason="doublet")
            bad = {**entry, "rationale": None}
            assert (await tools["submit_cluster"]({"cluster_json": json.dumps(bad)}))["is_error"]
            assert not (await tools["submit_cluster"]({"cluster_json": json.dumps(entry)})).get("is_error")
        result = await tools["finalize_annotation"]({"overall": "All cells removed in synthetic test."})
        return SimpleNamespace(submitted=result["_submitted"], transcript_text="All removal sources retained.")

    monkeypatch.setattr(harness, "run_agent", run_agent)
    annotate.annotate_clusters(tmp_path, model="test-model")
    kept = ad.read_h5ad(tmp_path / "annotated.h5ad")
    assert kept.shape == (0, 2) and "msp_ann_action" in kept.obs
    archive = pd.read_csv(tmp_path / "annotation_removed.csv")
    assert archive["cell"].tolist() == data.obs_names.tolist()
    assert archive["annotate_remove"].all()
    assert not step_pending(tmp_path, "annotate")
    assert ad.read_h5ad(tmp_path / "integrated.h5ad").n_obs == 2
    for suffix in ("coarse", "fine", "removed"):
        assert (tmp_path / "figures" / f"annotation_umap_{suffix}.png").read_bytes().startswith(b"\x89PNG")
    assert "2 cells removed" in (tmp_path / "report.html").read_text()


def test_report_renders_inspection_evidence_and_escapes_model_text(tmp_path):
    entry = inspection_entry()
    entry["tests"]["markers"] = "<script>not executable</script>"
    proposal = {
        "clusters": [entry],
        "overall": "Review required",
        "cell_actions": [
            {
                "cluster": "0",
                "metric": "doublet_score",
                "op": ">",
                "value": 0.5,
                "action": "drop",
                "reason": "doublet",
                "note": "mixed markers",
            }
        ],
    }
    (tmp_path / "inspection_proposal.json").write_text(json.dumps(proposal))
    (tmp_path / "inspection_notes.md").write_text("<b>Detailed inspection notes</b>")
    report = generate_report(tmp_path)
    from pathlib import Path

    text = Path(report).read_text()
    assert 'href="#inspection"' in text and "Five-test evidence" in text
    assert "Cell-level action rules" in text and "mixed markers" in text
    assert "&lt;script&gt;not executable&lt;/script&gt;" in text
    assert "&lt;b&gt;Detailed inspection notes&lt;/b&gt;" in text
    state = tmp_path / ".msp-state"
    state.mkdir()
    (state / "inspect.pending").touch()
    assert "Five-test evidence" not in Path(generate_report(tmp_path)).read_text()
