import asyncio
import json
from types import SimpleNamespace

import anndata as ad
import harness_bridge
import numpy as np
import pandas as pd
import pytest
from test_annotation_status import entry, payload

from msp import annotate, checkpoint


def test_interrupted_annotation_restores_only_validated_submissions(tmp_path, monkeypatch):
    obj = ad.AnnData(
        np.ones((2, 2)),
        obs=pd.DataFrame(
            {
                "batch": pd.Categorical(["S1", "S1"]),
                annotate.BASE_KEY: pd.Categorical(["0", "1"]),
                annotate.PARENT_KEY: pd.Categorical(["0", "0"]),
            },
            index=["001", "002"],
        ),
    )
    calls = []

    async def agent(**kwargs):
        handlers = {tool.name: tool.handler for tool in kwargs["tools"]}
        status = payload(await handlers["annotation_status"]({}))
        calls.append(status)
        if len(calls) == 1:
            assert not (await handlers["submit_cluster"]({"cluster_json": json.dumps(entry("0"))})).get("is_error")
            raise RuntimeError("simulated process loss")
        assert status["submitted_count"] == 1 and status["pending_ids"] == ["1"]
        await handlers["submit_cluster"]({"cluster_json": json.dumps(entry("1"))})
        result = await handlers["finalize_annotation"]({"overall": "restored"})
        return SimpleNamespace(submitted=result["_submitted"], transcript_text="")

    monkeypatch.setattr(harness_bridge, "run_agent", agent)

    def run(data=obj):
        return asyncio.run(
            annotate._run_agent(
                data.copy(), tmp_path, ["0", "1"], "batch", None, [], {}, np.zeros(2, bool), "English", "test", None, 2
            )
        )

    with pytest.raises(RuntimeError, match="process loss"):
        run()
    result = run()
    assert run() == result and len(calls) == 2  # crash after final submission needs no new model call
    changed = obj.copy()
    changed.X[0, 0] = 2
    with pytest.raises(ValueError, match="input, evidence"):
        run(changed)
    path = tmp_path / ".msp-state/annotate-progress.json"
    saved = json.loads(path.read_text())
    saved["progress"]["entries"]["0"]["action"] = "invented"
    checkpoint.write_json(path, saved)
    with pytest.raises(ValueError, match="invalid annotation checkpoint entry"):
        run()


def test_failed_atomic_write_keeps_last_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    checkpoint.save(path, "identity", {"entries": {"0": "accepted"}})

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(checkpoint.os, "replace", fail)
    with pytest.raises(OSError):
        checkpoint.save(path, "identity", {"entries": {"1": "new"}})
    assert checkpoint.load(path, "identity") == {"entries": {"0": "accepted"}}


def test_inspection_recovers_accepted_proposal_after_agent_disconnect(tmp_path, monkeypatch):
    from msp import inspect

    obj = ad.AnnData(np.ones((2, 2)), obs=pd.DataFrame({"cluster": pd.Categorical(["0", "0"]), "batch": "A"}, index=["01", "02"]))
    calls = []

    async def agent(**kwargs):
        calls.append(True)
        tool = next(t for t in kwargs["tools"] if t.name == "submit_inspection")
        proposal = {
            "clusters": [
                dict(
                    cluster="0",
                    verdict="artifact-batch",
                    action="drop",
                    confidence="low",
                    tests=dict.fromkeys(("markers", "qc", "composition", "geometry", "stability"), "reviewed"),
                    rationale="batch composition",
                )
            ]
        }
        result = await tool.handler({"proposal_json": json.dumps(proposal)})
        assert not result.get("is_error"), result
        raise RuntimeError("agent disconnected after accepted submission")

    monkeypatch.setattr(harness_bridge, "run_agent", agent)
    monkeypatch.setattr(inspect, "_system_prompt", lambda *a, **k: "test")

    def run():
        return asyncio.run(
            inspect._run_agent(
                obj.copy(), tmp_path, "cluster", [], "batch", None, "English", "test", None, 2, np.zeros(2, bool)
            )
        )

    with pytest.raises(RuntimeError, match="agent disconnected"):
        run()
    proposal = run()
    assert len(calls) == 1 and proposal["clusters"][0]["action"] == "flag"
