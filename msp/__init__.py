"""msp (multi-sample-pipeline): integrate osp per-sample outputs (harmony)
→ multi-resolution leiden + UMAP → cluster QC / DEG tables → self-contained
HTML report, with two optional Claude-agent steps that run afterwards:

    integrate  (msp.integrate)  propose-only: nothing deleted, nothing named
    inspect    (msp.inspect)    per-cluster five-test QC verdicts, proposals only
    annotate   (msp.annotate)   coarse/fine cell identity on msp_leiden_r2.0,
                                explicit merges, REAL removal → annotated.h5ad

Entry points:
    from msp import run_multi_sample_pipeline, generate_report

    run_multi_sample_pipeline(["A/clustered.h5ad", "B/clustered.h5ad"],
                              batch_col="project", outdir="msp_out")
    generate_report("msp_out")

Command line:
    python -m msp A/clustered.h5ad B/clustered.h5ad --batch-col project --outdir msp_out
    python -m msp ... --inspect --annotate --model claude-sonnet-5   # full chain
    python -m msp.inspect  msp_out          # QC inspection agent only
    python -m msp.annotate msp_out          # annotation agent only (after inspect)
    python -m msp.report   msp_out          # rebuild the report only

msp.inspect / msp.annotate are intentionally not imported here — they depend
on the optional claude-agent-sdk (`pip install "msp[agent]"`); use
`from msp.inspect import inspect_clusters` / `from msp.annotate import
annotate_clusters` when needed.
"""

from .integrate import integrate_adata, load_and_merge, run_multi_sample_pipeline
from .plots import save_single_umap
from .report import generate_report

__all__ = ["integrate_adata", "load_and_merge", "run_multi_sample_pipeline", "generate_report",
           "save_single_umap"]
