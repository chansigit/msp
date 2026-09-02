"""python -m msp: integrate osp per-sample outputs (concat → harmony → leiden →
UMAP → QC/DEG tables → HTML report), end to end.

With --inspect, the per-cluster QC inspection agent (msp.inspect) runs
afterwards; with --annotate, the cell-type annotation agent (msp.annotate)
runs after that (both need the optional claude-agent-sdk). Each step is
skipped when its contract file already exists, so re-running the same
command resumes where it stopped; --force redoes everything.
"""

import argparse
import os
import sys

from .integrate import run_multi_sample_pipeline
from .report import generate_report

parser = argparse.ArgumentParser(prog="msp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("inputs", nargs="+", help="per-sample clustered.h5ad files (osp outputs)")
parser.add_argument("--batch-col", required=True, help="obs column naming the sample/batch")
parser.add_argument("--outdir", required=True)
parser.add_argument("--species", default=None, help="stored in uns['msp']; context for the agents")
parser.add_argument("--resolutions", type=float, nargs="+", default=[0.3, 1.0, 2.0],
                    help="leiden resolutions; 1.0 and 2.0 must be present for inspect/annotate")
parser.add_argument("--n-top-genes", type=int, default=2000)
parser.add_argument("--inspect", action="store_true",
                    help="after integration, run the per-cluster QC inspection agent (msp.inspect)")
parser.add_argument("--annotate", action="store_true",
                    help="after inspection, run the cell-type annotation agent (msp.annotate); "
                         "implies --inspect")
parser.add_argument("--language", default="English", help='agent prose language (default "English")')
parser.add_argument("--model", default=None, help='model for the agents, e.g. "claude-sonnet-5"')
parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="reasoning effort for the agents (models that support it)")
parser.add_argument("--max-turns", type=int, default=None,
                    help="agent turn budget (defaults: inspect 100, annotate 200)")
parser.add_argument("--force", action="store_true", help="redo steps whose outputs already exist")
args = parser.parse_args()

if args.annotate:
    args.inspect = True
if args.inspect and not {1.0, 2.0} <= set(args.resolutions):
    sys.exit("--inspect/--annotate need leiden resolutions 1.0 and 2.0 (see --resolutions)")

out = args.outdir


def _done(*names):
    return all(os.path.exists(os.path.join(out, n)) for n in names)


if args.force or not _done("integrated.h5ad", "report.html"):
    _, summary = run_multi_sample_pipeline(
        args.inputs, batch_col=args.batch_col, outdir=out,
        species=args.species, resolutions=tuple(args.resolutions), n_top_genes=args.n_top_genes,
    )
    print(summary)
    print(f"report: {generate_report(out)}")
else:
    print(f"[resume] integration already done in {out} (integrated.h5ad + report.html) — skipping")

agent_kw = dict(species=args.species, language=args.language, model=args.model, effort=args.effort)

if args.inspect:
    if args.force or not _done("inspection_proposal.json"):
        from .inspect import inspect_clusters

        inspect_clusters(out, max_turns=args.max_turns or 100, **agent_kw)
    else:
        print(f"[resume] inspection_proposal.json exists in {out} — skipping inspect")

if args.annotate:
    if args.force or not _done("annotation_proposal.json", "annotated.h5ad"):
        from .annotate import annotate_clusters

        annotate_clusters(out, max_turns=args.max_turns or 200, **agent_kw)
    else:
        print(f"[resume] annotation_proposal.json + annotated.h5ad exist in {out} — skipping annotate")
