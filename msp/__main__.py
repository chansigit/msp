"""python -m msp: integrate osp per-sample outputs (concat → harmony → leiden →
UMAP → QC/DEG tables → HTML report), end to end.

With --inspect, the per-cluster QC inspection agent (msp.inspect) runs
afterwards; with --annotate, the cell-type annotation agent (msp.annotate)
runs after that (both need the optional agent dependencies). Each step is
skipped when its contract file already exists, so re-running the same
command resumes where it stopped; --force redoes everything.
"""

import argparse
import os
import sys

from .integrate import integrate_adata, run_multi_sample_pipeline
from .report import generate_report, write_report_context

parser = argparse.ArgumentParser(prog="msp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("inputs", nargs="*", help="per-sample clustered.h5ad files (osp outputs)")
parser.add_argument("--from-h5ad", default=None, metavar="H5AD",
                    help="instead of per-sample inputs: one already-merged h5ad with layers['counts'] "
                         "(e.g. a previous round's annotated_zmip.h5ad) — re-integrated from scratch via "
                         "integrate_adata; prior obs columns ride along as annotation evidence")
parser.add_argument("--batch-col", required=True, help="obs column naming the sample/batch")
parser.add_argument("--outdir", required=True)
parser.add_argument("--species", default=None, help="stored in uns['msp']; context for the agents")
parser.add_argument("--resolutions", type=float, nargs="+", default=[0.3, 1.0, 2.0],
                    help="leiden resolutions; 1.0 and 2.0 must be present for inspect/annotate")
parser.add_argument("--n-top-genes", type=int, default=2000)
parser.add_argument("--n-pcs", type=int, default=50)
parser.add_argument("--n-neighbors", type=int, default=15)
parser.add_argument("--harmony", action="append", default=[], metavar="KEY=VALUE",
                    help="harmonypy.run_harmony override, repeatable: e.g. --harmony theta=1 "
                         "--harmony lamb=-1 --harmony max_iter_harmony=20 --harmony sigma=0.2 "
                         "(defaults: theta=2, lamb=1, sigma=0.1, nclust=min(N/30,100), "
                         "max_iter_harmony=10, max_iter_kmeans=20)")
parser.add_argument("--inspect", action="store_true",
                    help="after integration, run the per-cluster QC inspection agent (msp.inspect)")
parser.add_argument("--annotate", action="store_true",
                    help="after inspection, run the cell-type annotation agent (msp.annotate); "
                         "implies --inspect")
parser.add_argument("--language", default="English", help='agent prose language (default "English")')
parser.add_argument("--harness", choices=["deepseek", "openai", "claude"], default=None,
                    help="agent runtime backend (default: HARNESS env, then openai)")
parser.add_argument("--model", default=None, help='model id for the selected HARNESS backend')
parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="reasoning effort for the agents (models that support it)")
parser.add_argument("--max-turns", type=int, default=None,
                    help="agent turn budget (defaults: inspect 100, annotate 200)")
parser.add_argument("--report-context", default=None, metavar="TEXT",
                    help='where this run sits, for report titles (e.g. "round 2 · fu2022-meniscus"); '
                         "persisted in <outdir>/report_context.txt so later report refreshes keep it")
parser.add_argument("--force", action="store_true", help="redo steps whose outputs already exist")
args = parser.parse_args()

if args.harness:
    os.environ["HARNESS"] = args.harness

if bool(args.inputs) == bool(args.from_h5ad):
    sys.exit("give either per-sample inputs or --from-h5ad, not both / neither")
if args.annotate:
    args.inspect = True
if args.inspect and not {1.0, 2.0} <= set(args.resolutions):
    sys.exit("--inspect/--annotate need leiden resolutions 1.0 and 2.0 (see --resolutions)")

out = args.outdir
write_report_context(out, args.report_context)


def _parse_kv(items):
    """KEY=VALUE → {key: number|list|str}; comma-separated values become lists."""
    def conv(v):
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v
    out = {}
    for it in items:
        if "=" not in it:
            sys.exit(f"--harmony expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = [conv(x) for x in v.split(",")] if "," in v else conv(v)
    return out


harmony_kwargs = _parse_kv(args.harmony)


def _done(*names):
    return all(os.path.exists(os.path.join(out, n)) for n in names)


def _integration_matches():
    """Refuse to resume an integration produced with different core inputs."""
    path = os.path.join(out, "integrated.h5ad")
    if not os.path.exists(path):
        return False
    try:
        import scanpy as sc

        meta = sc.read_h5ad(path, backed="r").uns.get("msp", {})
        expected_harmony = ("skipped: single batch"
                            if meta.get("n_batches") == 1 else harmony_kwargs)
        expected = {
            "batch_col": args.batch_col,
            "species": args.species or "",
            "resolutions": list(args.resolutions),
            "n_top_genes": args.n_top_genes,
            "n_pcs_requested": args.n_pcs,
            "n_neighbors": args.n_neighbors,
            "harmony": expected_harmony,
            "inputs": [str(p) for p in (args.inputs or [args.from_h5ad])],
        }
        return all(meta.get(k) == v for k, v in expected.items())
    except Exception:
        return False


if args.force or not _done("integrated.h5ad", "report.html") or not _integration_matches():
    kw = dict(species=args.species, resolutions=tuple(args.resolutions), n_top_genes=args.n_top_genes,
              n_pcs=args.n_pcs, n_neighbors=args.n_neighbors, harmony_kwargs=harmony_kwargs)
    if args.from_h5ad:
        import scanpy as sc

        ad = sc.read_h5ad(args.from_h5ad)
        _, summary = integrate_adata(ad, args.batch_col, out, inputs=[args.from_h5ad], **kw)
    else:
        _, summary = run_multi_sample_pipeline(args.inputs, batch_col=args.batch_col, outdir=out, **kw)
    print(summary)
    print(f"report: {generate_report(out)}")
else:
    print(f"[resume] integration already done in {out} (integrated.h5ad + report.html) — skipping")

if args.inspect:
    from .harness import backend_name, default_model

    args.model = args.model or default_model()
    print(f"[agent] harness={backend_name()} model={args.model}")
agent_kw = dict(species=args.species, language=args.language, model=args.model, effort=args.effort)

if args.inspect:
    inspection_applied = False
    if _done("integrated.h5ad", "inspection_proposal.json", "report.html",
             "figures/inspect_umap_action.png"):
        try:
            import scanpy as sc

            inspected = sc.read_h5ad(os.path.join(out, "integrated.h5ad"), backed="r")
            inspection_applied = "_msp_action" in inspected.obs and "_msp_verdict" in inspected.obs
        except Exception:
            inspection_applied = False
    if args.force or not inspection_applied:
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
