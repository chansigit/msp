"""--design-context: persisted like --report-context, injected into both agent prompts, no-op when absent."""

from msp import annotate, inspect
from msp.__main__ import build_parser
from msp.report import design_block, design_context, write_design_context

DESIGN = "sample P1: subtissue=Immune (CD45+), mouse.id=3_8_M\nsample P2: subtissue=Epithelial, mouse.id=3_9_M"


def prompts(outdir):
    return (
        inspect._system_prompt(str(outdir), "msp_leiden_r1.0", ["0", "1"], "sample", "mouse", "English"),
        annotate._system_prompt(str(outdir), ["0", "1"], "sample", "mouse", [], "English"),
    )


def test_cli_option_and_persistence(tmp_path):
    args = build_parser().parse_args(["--from-h5ad", "x.h5ad", "--batch-col", "s", "--outdir", str(tmp_path), "--design-context", DESIGN])
    write_design_context(str(tmp_path), args.design_context)
    assert (tmp_path / "design_context.txt").read_text() == DESIGN + "\n"
    assert design_context(str(tmp_path)) == DESIGN
    # per-lineage sub-directories (zmip) inherit the parent's file, as report_context does
    (tmp_path / "lineage").mkdir()
    assert design_context(str(tmp_path / "lineage")) == DESIGN


def test_absent_is_noop(tmp_path):
    assert build_parser().parse_args(["--from-h5ad", "x", "--batch-col", "s", "--outdir", "o"]).design_context is None
    write_design_context(str(tmp_path), None)
    assert not (tmp_path / "design_context.txt").exists()
    assert design_block(str(tmp_path)) == ""
    for prompt in prompts(tmp_path):
        assert "Study design" not in prompt


def test_prompts_carry_the_block_verbatim(tmp_path):
    write_design_context(str(tmp_path), DESIGN)
    for prompt in prompts(tmp_path):
        assert "Study design (from the caller — treat as ground truth about how samples were produced):" in prompt
        assert f"<<<\n{DESIGN}\n>>>" in prompt
