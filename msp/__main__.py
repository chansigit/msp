"""python -m msp: multi-sample integration (concat → harmony → leiden → UMAP → report)."""

import argparse

from .integrate import run_multi_sample_pipeline
from .report import generate_report

parser = argparse.ArgumentParser(prog="msp", description=__doc__)
parser.add_argument("inputs", nargs="+", help="per-sample clustered.h5ad files (osp outputs)")
parser.add_argument("--batch-col", required=True, help="obs column naming the sample/batch")
parser.add_argument("--outdir", required=True)
parser.add_argument("--species", default=None)
parser.add_argument("--resolutions", type=float, nargs="+", default=[0.3, 1.0, 2.0])
args = parser.parse_args()

_, summary = run_multi_sample_pipeline(
    args.inputs, batch_col=args.batch_col, outdir=args.outdir,
    species=args.species, resolutions=tuple(args.resolutions),
)
print(summary)
print(f"report: {generate_report(args.outdir)}")
