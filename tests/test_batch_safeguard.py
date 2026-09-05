"""Batch composition may request review, but never proves invalid cells."""

import asyncio
import copy
import json
from types import SimpleNamespace

import anndata as ad
import harness_bridge
import numpy as np
import pandas as pd
import pytest

from msp import annotate, inspect


def proposal():
    return {
        "cluster_key": annotate.BASE_KEY,
        "clusters": [
            {
                "cluster": "0",
                "verdict": "artifact-batch",
                "action": "drop",
                "confidence": "low",
                "tests": dict.fromkeys(
                    ["markers", "qc", "composition", "geometry", "stability"], "Unresolved sample/condition confounding"
                ),
                "rationale": "One sample dominates this cluster",
            },
            {
                "cluster": "1",
                "verdict": "artifact-doublet",
                "action": "drop",
                "confidence": "low",
                "tests": dict.fromkeys(
                    ["markers", "qc", "composition", "geometry", "stability"], "Independent evidence needs review"
                ),
                "rationale": "Existing low-confidence policy is unchanged",
            },
        ],
    }


def data():
    obs = pd.DataFrame(
        {
            annotate.BASE_KEY: pd.Categorical(["0", "0", "1"]),
            annotate.PARENT_KEY: pd.Categorical(["0", "0", "1"]),
            "batch": pd.Categorical(["a", "a", "b"]),
            "doublet_score": [0.1, 0.9, 0.2],
        },
        index=["c0", "c1", "c2"],
    )
    obj = ad.AnnData(np.ones((3, 2)), obs=obs)
    obj.layers["counts"] = obj.X.copy()
    obj.uns["msp"] = {"batch_col": "batch"}
    obj.obsm["X_umap"] = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return obj


def test_guard_is_auditable_idempotent_and_does_not_override_other_low_confidence():
    p = proposal()
    original = copy.deepcopy(p)
    assert inspect._guard_batch_actions(p) is p
    entry = p["clusters"][0]
    assert entry["action"] == "flag" and entry["requested_action"] == "drop"
    assert entry["host_adjustment"]["policy"] == "batch_verdict_non_destructive_v1"
    assert entry["tests"] == original["clusters"][0]["tests"]
    assert p["clusters"][1] == original["clusters"][1]
    snapshot = copy.deepcopy(p)
    assert inspect._guard_batch_actions(p) == snapshot


def test_saved_proposal_apply_cannot_bypass_guard_and_keeps_independent_cell_qc():
    obj = data()
    p = proposal()
    p["cell_actions"] = [
        {
            "cluster": "0",
            "metric": "doublet_score",
            "op": ">",
            "value": 0.8,
            "action": "drop",
            "reason": "doublet",
            "note": "Independent cell-level QC",
        }
    ]
    before = obj.layers["counts"].copy()
    inspect._apply_proposal(obj, annotate.BASE_KEY, p)
    assert obj.obs["_msp_action"].tolist() == ["flag", "drop", "drop"]
    assert p["clusters"][0]["action"] == "flag"
    np.testing.assert_array_equal(obj.layers["counts"], before)


def test_submit_normalizes_once_without_an_agent_retry(tmp_path, monkeypatch):
    obj = data()
    calls = []

    async def run_agent(**kwargs):
        calls.append(1)
        submit = next(t.handler for t in kwargs["tools"] if t.name == "submit_inspection")
        result = await submit({"proposal_json": json.dumps(proposal())})
        assert not result.get("is_error")
        return SimpleNamespace(submitted=result["_submitted"], transcript_text="")

    monkeypatch.setattr(harness_bridge, "run_agent", run_agent)
    result = asyncio.run(
        inspect._run_agent(
            obj, tmp_path, annotate.BASE_KEY, [], "batch", None, "English", "test", None, 1, np.zeros(3, bool)
        )
    )
    saved = json.loads((tmp_path / "inspection_proposal.json").read_text())
    assert calls == [1]
    assert saved == result
    assert saved["clusters"][0]["requested_action"] == "drop"
    assert saved["clusters"][0]["action"] == "flag"


def test_inspection_persists_guarded_saved_proposal_and_report(tmp_path, monkeypatch):
    obj = data()
    obj.write_h5ad(tmp_path / "integrated.h5ad")

    async def saved_proposal(*args, **kwargs):
        return proposal()

    monkeypatch.setattr(inspect, "_run_agent", saved_proposal)
    monkeypatch.setattr(inspect, "load_removal_mask", lambda *a: np.zeros(3, bool))
    monkeypatch.setattr(inspect, "_plot_verdicts", lambda *a: None)
    result = inspect.inspect_clusters(tmp_path, cluster_key=annotate.BASE_KEY, model="test")
    saved = json.loads((tmp_path / "inspection_proposal.json").read_text())
    kept = ad.read_h5ad(tmp_path / "integrated.h5ad")
    assert result == saved
    assert kept.obs["_msp_action"].tolist() == ["flag", "flag", "drop"]
    content = (tmp_path / "report.html").read_text()
    assert "Host safeguards: retained for review" in content
    assert "requested_action" in content and "<td>flag</td>" in content
    assert "Sample/batch composition alone does not establish invalid cells" in content


def test_legacy_batch_drop_annotation_stops_before_agent_or_file_mutation(tmp_path, monkeypatch):
    obj = data()
    obj.obs["_msp_action"] = ["drop", "drop", "drop"]
    obj.obs["_msp_verdict"] = ["artifact-batch", "artifact-batch", "artifact-doublet"]
    obj.write_h5ad(tmp_path / "integrated.h5ad")
    (tmp_path / "inspection_proposal.json").write_text(json.dumps(proposal()))
    before = (tmp_path / "integrated.h5ad").read_bytes()
    monkeypatch.setattr(annotate, "load_removal_mask", lambda *a: np.zeros(3, bool))
    monkeypatch.setattr(annotate, "begin_step", lambda *a: pytest.fail("must stop before invalidating output"))
    with pytest.raises(ValueError, match="reapplying the saved inspection"):
        annotate.annotate_clusters(tmp_path, model="test")
    assert (tmp_path / "integrated.h5ad").read_bytes() == before
    assert not (tmp_path / "annotation_proposal.json").exists()


def test_independent_cell_drop_inside_flagged_batch_cluster_reaches_annotation(tmp_path, monkeypatch):
    obj = data()
    p = proposal()
    p["cell_actions"] = [
        {
            "cluster": "0",
            "metric": "doublet_score",
            "op": ">",
            "value": 0.8,
            "action": "drop",
            "reason": "doublet",
            "note": "Independent QC",
        }
    ]
    inspect._apply_proposal(obj, annotate.BASE_KEY, p)
    obj.write_h5ad(tmp_path / "integrated.h5ad")
    (tmp_path / "inspection_proposal.json").write_text(json.dumps(p))
    monkeypatch.setattr(annotate, "load_removal_mask", lambda *a: np.zeros(3, bool))

    def reached(*args):
        raise RuntimeError("reached annotation step")

    monkeypatch.setattr(annotate, "begin_step", reached)
    with pytest.raises(RuntimeError, match="reached annotation step"):
        annotate.annotate_clusters(tmp_path, model="test")
