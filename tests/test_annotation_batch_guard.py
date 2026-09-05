"""Batch-only requests remain auditable and cannot delete cells."""

import asyncio
import copy
import json
from types import SimpleNamespace

import harness_bridge
import numpy as np
import pytest
from test_evidence_contracts import annotation_entry, data_with_clusters

import msp.annotate as A
from msp.report import _section_annotation


@pytest.mark.parametrize("reason", ["batch", "other"])
def test_submission_normalizes_before_merge_validation_and_persists(tmp_path, monkeypatch, reason):
    data = data_with_clusters()
    original = annotation_entry(
        "0", action="remove", remove_reason=reason, merge_target="1", rationale="batch artifact with ambient RNA"
    )

    async def run_agent(**kwargs):
        tools = {tool.name: tool.handler for tool in kwargs["tools"]}
        response = await tools["submit_cluster"]({"cluster_json": json.dumps(original)})
        assert "host retained" in response["content"][0]["text"]
        await tools["submit_cluster"]({"cluster_json": json.dumps(annotation_entry("1"))})
        # As a retained member, cluster 0 must now agree with the merge target.
        response = await tools["finalize_annotation"]({"overall": "review"})
        assert response["is_error"] and "disagrees on fine_label" in response["content"][0]["text"]
        corrected = {**original, "fine_label": "Type 1"}
        await tools["submit_cluster"]({"cluster_json": json.dumps(corrected)})
        response = await tools["finalize_annotation"]({"overall": "review"})
        return SimpleNamespace(submitted=response["_submitted"], transcript_text="")

    monkeypatch.setattr(harness_bridge, "run_agent", run_agent)
    proposal = asyncio.run(
        A._run_agent(
            data,
            str(tmp_path),
            ["0", "1"],
            "batch",
            "mouse",
            [],
            {},
            np.zeros(2, bool),
            "English",
            "test-model",
            "low",
            20,
        )
    )
    saved = json.loads((tmp_path / "annotation_proposal.json").read_text())
    assert saved == proposal
    entry = saved["clusters"][0]
    assert entry["action"] == "keep" and entry["remove_reason"] is None
    assert entry["requested_action"] == "remove" and entry["requested_remove_reason"] == reason
    assert entry["review_required"] is True
    assert entry["rationale"] == original["rationale"] and entry["evidence"] == original["evidence"]
    assert entry["coarse_label"] == original["coarse_label"]
    assert saved["merged_groups"] == ["0+1"]
    report = _section_annotation(str(tmp_path), [])
    assert "Host policy adjustments" in report and f"requested remove ({reason}); applied keep" in report
    assert "review required" in report and original["rationale"] in report


@pytest.mark.parametrize("reason", ["doublet", "low-quality", "ambient", "stress", "other"])
def test_other_removal_reasons_unchanged(reason):
    entry = annotation_entry("0", action="remove", remove_reason=reason)
    original = copy.deepcopy(entry)
    assert A._guard_batch_annotation(entry) == original


def test_saved_unguarded_proposal_cannot_bypass_host(tmp_path):
    data = data_with_clusters()
    proposal = {"clusters": [annotation_entry("0", action="remove", remove_reason="batch"), annotation_entry("1")]}
    path = tmp_path / "old-proposal.json"
    path.write_text(json.dumps(proposal))
    with pytest.raises(ValueError, match="unguarded batch-only"):
        A._apply(data, json.loads(path.read_text()), np.zeros(2, bool), {"preannotation": np.zeros(2, bool)})
    assert "msp_ann_action" not in data.obs


def test_guard_preserves_independent_removal_sources_and_review():
    data = data_with_clusters(("0", "0", "1"))
    entry = A._guard_batch_annotation(annotation_entry("0", action="remove", remove_reason="batch"))
    proposal = {"clusters": [entry, annotation_entry("1")]}
    pre = np.array([True, False, False])
    removed = A._apply(data, proposal, pre, {"preannotation": pre})
    assert removed["cell"].tolist() == ["cell0"]
    assert not removed["annotate_remove"].any()
    assert removed["preannotation"].all() and removed["remove_reason"].isna().all()
    assert data.obs["msp_ann_action"].tolist() == ["remove", "keep", "keep"]
    assert data.obs["msp_ann_review"].tolist() == [True, True, False]


def test_normalization_does_not_silently_repair_invalid_merge_target():
    entries = {
        "0": A._guard_batch_annotation(annotation_entry("0", action="remove", remove_reason="batch", merge_target="1")),
        "1": annotation_entry("1", action="remove", remove_reason="doublet"),
    }
    problems = A._validate_final(entries, ["0", "1"])
    assert any("action=remove" in problem for problem in problems)


@pytest.mark.parametrize(
    "fields",
    [
        {"fine_label": "Mixed stromal batch artifact", "rationale": "batch artifact with ambient RNA"},
        {"rationale": "batch artifact with ambient RNA"},
        {"fine_label": "Sample-artefact"},
        {"rationale": "批次伪影"},
    ],
)
def test_other_cannot_disguise_explicit_batch_artifact(fields, tmp_path):
    original = annotation_entry("0", action="remove", remove_reason="other", **fields)
    data = data_with_clusters()
    proposal = {"clusters": [copy.deepcopy(original), annotation_entry("1")]}
    with pytest.raises(ValueError, match="unguarded batch-only"):
        A._apply(data, proposal, np.zeros(2, bool), {"preannotation": np.zeros(2, bool)})
    assert "msp_ann_action" not in data.obs
    guarded = A._guard_batch_annotation(copy.deepcopy(original))
    assert guarded["requested_remove_reason"] == "other"
    assert guarded["action"] == "keep" and guarded["review_required"] is True
    for field in ("fine_label", "rationale", "evidence"):
        assert guarded[field] == original[field]
    assert A._guard_batch_annotation(copy.deepcopy(guarded)) == guarded
    proposal["clusters"][0] = guarded
    removed = A._apply(data, proposal, np.zeros(2, bool), {"preannotation": np.zeros(2, bool)})
    assert removed.empty and data.obs["msp_ann_review"].tolist() == [True, False]
    (tmp_path / "annotation_proposal.json").write_text(json.dumps(proposal))
    assert "requested remove (other); applied keep" in _section_annotation(str(tmp_path), [])


@pytest.mark.parametrize("reason", ["doublet", "low-quality", "ambient", "stress"])
def test_explicit_independent_qc_reason_not_overridden_by_batch_text(reason):
    original = annotation_entry(
        "0",
        action="remove",
        remove_reason=reason,
        rationale="Independent numeric QC evidence; batch artifact is a secondary concern",
    )
    assert A._guard_batch_annotation(copy.deepcopy(original)) == original


@pytest.mark.parametrize("action", ["keep", "remove"])
def test_general_batch_mentions_are_not_artifact_classification(action):
    original = annotation_entry(
        "0",
        action=action,
        remove_reason="other" if action == "remove" else None,
        rationale="Mixed markers observed across each batch and sample",
    )
    assert A._guard_batch_annotation(copy.deepcopy(original)) == original


def test_non_removal_with_explicit_batch_artifact_is_unchanged():
    original = annotation_entry("0", rationale="batch artifact with ambient RNA")
    assert A._guard_batch_annotation(copy.deepcopy(original)) == original
