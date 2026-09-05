"""Recover host annotation state without replaying every evidence payload."""

import asyncio
import copy
import json
from types import SimpleNamespace

import anndata as ad
import harness_bridge
import numpy as np
import pandas as pd
import pytest

from msp import annotate


def entry(cluster, long=False):
    return {
        "cluster_id": cluster,
        "coarse_label": "Immune",
        "fine_label": "Type " + cluster,
        "merge_target": None,
        "action": "keep",
        "confidence": "low",
        "evidence": dict.fromkeys(
            ("distinctness", "markers", "merge"), "细胞证据🙂" * 10000 if long else "Evidence retained"
        ),
        "rationale": "Retain for review.",
    }


def payload(result):
    assert not result.get("is_error"), result
    text = result["content"][0]["text"]
    assert len(text.encode()) <= 16 * 1024
    return json.loads(text)


def test_status_pages_cover_77_clusters_without_serializing_long_evidence():
    clusters = list(map(str, range(77)))
    entries = {c: entry(c, long=True) for c in clusters[:65]}
    before = copy.deepcopy(entries)
    accepted, pending, offset = [], [], 0
    while True:
        result = payload(annotate._annotation_status(entries, clusters, offset=offset))
        assert result["submitted_count"] == 65 and result["pending_count"] == 12
        assert len(result["submitted"]) <= 8 and len(result["pending_ids"]) <= 8
        assert "细胞证据" not in json.dumps(result, ensure_ascii=False)
        accepted.extend(row["cluster_id"] for row in result["submitted"])
        pending.extend(result["pending_ids"])
        offset = result["next_offset"]
        if offset is None:
            break
    assert accepted == clusters[:65] and pending == clusters[65:]
    assert entries == before


def test_single_entry_pages_reconstruct_complete_unicode_evidence_exactly():
    original = entry("0", long=True)
    offset, pieces = 0, []
    while True:
        result = payload(annotate._annotation_status({"0": original}, ["0"], "0", offset))
        pieces.append(result["entry_json"])
        next_offset = result["next_offset"]
        if next_offset is None:
            break
        assert next_offset > offset
        offset = next_offset
    assert len(pieces) > 1
    assert json.loads("".join(pieces)) == original


def test_summary_shortens_labels_but_detail_preserves_them():
    original = entry("0")
    original["coarse_label"] = "长标签" * 10000
    result = payload(annotate._annotation_status({"0": original}, ["0"]))
    assert len(result["submitted"][0]["coarse_label"]) == 97
    assert result["submitted"][0]["coarse_label"].endswith("…")


@pytest.mark.parametrize("cluster,offset", [("missing", 0), ("", -1), ("0", True), ("0", 10**9), ("", 10**9)])
def test_invalid_status_queries_are_bounded_errors(cluster, offset):
    result = annotate._annotation_status({"0": entry("0")}, ["0"], cluster, offset)
    assert result["is_error"]
    assert len(result["content"][0]["text"].encode()) < 1024


def test_unsubmitted_status_does_not_invent_an_annotation():
    assert payload(annotate._annotation_status({}, ["0"], "0"))["submitted"] is False


def test_real_handlers_recover_submissions_and_keep_finalize_contract(tmp_path, monkeypatch):
    clusters = list(map(str, range(77)))
    obj = ad.AnnData(
        np.ones((77, 2)),
        obs=pd.DataFrame(
            {
                "batch": pd.Categorical(["S1"] * 77),
                annotate.BASE_KEY: pd.Categorical(clusters),
                annotate.PARENT_KEY: pd.Categorical(["0"] * 77),
            },
            index=["cell" + c for c in clusters],
        ),
    )
    responses = []

    async def run_agent(**kwargs):
        assert "Recovery rule" in kwargs["system_prompt"]
        assert "at most FOUR pending clusters" in kwargs["system_prompt"]
        handlers = {spec.name: spec.handler for spec in kwargs["tools"]}
        status = handlers["annotation_status"]
        assert payload(await status({"cluster": "", "offset": 0}))["pending_count"] == 77
        for c in clusters:
            result = await handlers["submit_cluster"]({"cluster_json": json.dumps(entry(c))})
            assert not result.get("is_error"), result
        # A reset creates a new model view, not new host handlers: recover
        # accepted labels without resubmitting or querying cluster_context.
        summary = payload(await status({"cluster": "", "offset": 0}))
        responses.append(summary)
        assert summary["submitted_count"] == 77 and summary["pending_count"] == 0
        detail = payload(await status({"cluster": "37", "offset": 0}))
        assert json.loads(detail["entry_json"]) == entry("37")
        result = await handlers["finalize_annotation"]({"overall": "77 synthetic clusters covered"})
        assert not result.get("is_error"), result
        return SimpleNamespace(submitted=result["_submitted"], transcript_text="")

    monkeypatch.setattr(harness_bridge, "run_agent", run_agent)
    result = asyncio.run(
        annotate._run_agent(
            obj, tmp_path, clusters, "batch", None, [], {}, np.zeros(77, bool), "English", "test", None, 2
        )
    )
    saved = json.loads((tmp_path / "annotation_proposal.json").read_text())
    assert saved == result and len(result["clusters"]) == 77
    assert len(responses) == 1
