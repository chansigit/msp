"""Optional ZMIP annotations remain visible through the shared report renderer."""

import json

import pytest

from msp.report import _section_annotation


def proposal(root, *, foreign=True, reassign=True, key="msp_leiden_r2.0"):
    entry = {
        "cluster_id": "7",
        "action": "reassign" if reassign else "keep",
        "evidence": {"distinctness": "distinct", "markers": "markers", "merge": "none"},
    }
    if reassign:
        entry["reassign_to"] = "B <cell>"
    if foreign:
        entry["evidence"]["foreign"] = "<signal> 1.343"
    (root / "annotation_proposal.json").write_text(json.dumps({"cluster_key": key, "clusters": [entry]}))


def test_ordinary_msp_preserves_columns(tmp_path):
    proposal(tmp_path, foreign=False, reassign=False)
    result = _section_annotation(str(tmp_path), [])
    assert "<th>action</th><th>remove_reason</th>" in result
    assert "<th>1. distinctness</th><th>2. markers</th><th>3. merge</th>" in result
    assert "reassign_to" not in result
    assert "Foreign" not in result


def test_reassignment_ledger_and_foreign_render(tmp_path):
    proposal(tmp_path)
    (tmp_path / "annotation_reassigned.csv").write_text("cell,reassign_to\na,B <cell>\nb,B <cell>\nc,Stromal\n")
    (tmp_path / "annotation_removed.csv").write_text("cell,remove_reason\nx,doublet\n")
    (tmp_path / "foreign_signal_msp_leiden_r2.0.csv").write_text("cluster,Stromal_p90\n7,1.343\n")
    result = _section_annotation(str(tmp_path), [])
    assert "<th>action</th><th>reassign_to</th><th>remove_reason</th>" in result
    assert "<td>B &lt;cell&gt;</td><td>2</td>" in result
    assert "3 cells reassigned." in result
    assert "<th>3. foreign</th><th>4. merge</th>" in result
    assert "&lt;signal&gt; 1.343" in result
    assert "not probabilities" in result
    assert "<th>Stromal_p90</th>" in result


@pytest.mark.parametrize(
    "content,expected",
    [
        (None, "ledger unavailable"),
        ("cell,wrong\na,B\n", "ledger is malformed"),
        ("cell,reassign_to\na,\n", "ledger is malformed"),
        ("cell,reassign_to\n", "0 cells reassigned."),
    ],
)
def test_missing_malformed_and_empty_ledgers_are_distinct(tmp_path, content, expected):
    proposal(tmp_path)
    if content is not None:
        (tmp_path / "annotation_reassigned.csv").write_text(content)
    result = _section_annotation(str(tmp_path), [])
    assert expected in result
    if content is None or "malformed" in expected:
        assert "0 cells reassigned." not in result


def test_dynamic_clusters_do_not_borrow_old_foreign_measurements(tmp_path):
    proposal(tmp_path, key="zmip_sub2")
    (tmp_path / "foreign_signal_msp_leiden_r2.0.csv").write_text("cluster,obsolete\n7,123\n")
    result = _section_annotation(str(tmp_path), [])
    assert "No computed foreign signal table for the current clustering." in result
    assert "obsolete" not in result
    assert "&lt;signal&gt; 1.343" in result


def test_duplicate_cell_ids_do_not_inflate_reassignment_counts(tmp_path):
    proposal(tmp_path)
    (tmp_path / "annotation_reassigned.csv").write_text("cell,reassign_to\na,B cell\na,Stromal\n")
    result = _section_annotation(str(tmp_path), [])
    assert "ledger is malformed; counts unavailable" in result
    assert "2 cells reassigned." not in result
    assert "<th>n_cells</th>" not in result
