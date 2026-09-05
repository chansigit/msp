"""
msp.inspect — per-cluster inspection of an msp integration directory with an
agent (harness_bridge: HARNESS=claude|deepseek|openai), mirroring osp.annotate's
architecture.

Every integrated cluster goes through five tests (the R11 battery):

  (a) markers     — own specific positive markers; ribo/mito/stress modules
                    with flat logFC → noise partition;
  (b) QC axis     — separation from neighbors along QC metrics → technical;
  (c) composition — multi-sample presence; single-sample dominance → batch
                    artifact (unless metadata explains it);
  (d) geometry    — sits between two populations with intermediate
                    signatures → doublet candidate;
  (e) stability   — persists across resolutions; one-resolution splinters
                    are not actionable.

Evidence on disk (deg_global_*/deg_local_* at both leiden resolutions,
cluster/per-sample QC tables, the standissect-lite bundle, figures) plus
live MCP tools (check_genes, check_qc_scores, check_stability, check_deg,
subcluster). The agent submits a
structured proposal; the host validates it, writes
inspection_proposal.json / inspection_notes.md, maps it onto
obs["_msp_action"] (keep/flag/drop) in integrated.h5ad, renders the
verdict UMAP and refreshes report.html.

Verdicts are PROPOSALS only — msp deletes nothing; execution is a later,
separate step. Cells flagged/dropped per-sample (obs["_qc_action"]) are in
here on purpose: their cross-sample clustering is exactly the evidence
this step weighs — check_genes/check_qc_scores/check_stability/subcluster's
own splitting still see them. Only DEG (check_deg, and subcluster's
built-in sibling DE) excludes preannotation_removal.csv's cells, matching
the precomputed deg_global_*/deg_local_* CSVs exactly.

Usage:
    python -m msp.inspect <msp_outdir> [--species human] [--model ...]
"""

import argparse
import asyncio
import glob
import json
import logging
import operator
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from harness_bridge import default_model

from .agent_tools import DEG_FILTER_ARGS, deg_filters, shared_tools, text_result
from .evidence import (
    QC_COLS,
    DegCache,
    DegTables,
    cluster_order,
    file_inventory,
    load_removal_mask,
    parse_reference,
    qc_table,
    stability_table,
)
from .log import configure, ensure
from .report import generate_report
from .steps import begin_step, complete_step, require_upstream_ready

log = logging.getLogger(__name__)

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def _detect_primary_key(outdir):
    """Cluster Annotations computes DEG for both r1.0 and r2.0 (deg_global_*/
    deg_local_*.csv); default to r1.0 when both exist (matches the old
    single-resolution 'primary_key' convention), else whichever is present."""
    paths = sorted(glob.glob(os.path.join(outdir, "deg_global_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no deg_global_*.csv in {outdir} — run the msp pipeline first")
    keys = [os.path.basename(p)[len("deg_global_") : -len(".csv")] for p in paths]
    return "msp_leiden_r1.0" if "msp_leiden_r1.0" in keys else keys[0]


def _subcluster_once(ad, key, cluster, resolution, new_key, remove_mask):
    """Split one cluster; sizes reported are the FULL split (removed cells
    included, so counts stay honest), but the built-in sibling DE excludes
    remove_mask cells — same DEG-only exclusion as check_deg / the
    precomputed deg_global_*/deg_local_* CSVs."""
    parent_mask = (ad.obs[key].astype(str) == cluster).values
    sc.tl.leiden(
        ad, restrict_to=(key, [cluster]), resolution=resolution, key_added=new_key, flavor="igraph", n_iterations=2
    )
    sub_labels = ad.obs[new_key][parent_mask].astype(str)
    subs = cluster_order(sub_labels)
    if len(subs) < 2:
        del ad.obs[new_key]
        return 0, f"cluster {cluster} did not split at resolution {resolution}; try higher"
    sub = ad[parent_mask].copy()
    sub.obs["_sub"] = pd.Categorical(sub_labels.values)
    sub_clean = sub[~remove_mask[parent_mask]]
    top = None
    if sub_clean.obs["_sub"].nunique(dropna=True) >= 2:
        sc.tl.rank_genes_groups(sub_clean, "_sub", method="wilcoxon", use_raw=False)
        top = sc.get.rank_genes_groups_df(sub_clean, group=None).groupby("group", observed=True).head(10)
    sizes = sub_labels.value_counts()
    lines = [f"cluster {cluster} split into {len(subs)} subclusters at resolution {resolution}:"]
    for s in subs:
        genes = ", ".join(top.loc[top["group"] == s, "names"]) if top is not None else ""
        note = "" if genes else " (no DE — too few non-removed cells)"
        lines.append(f"  {s} (n={int(sizes[s])}) top genes vs siblings: {genes}{note}")
    return len(subs), "\n".join(lines)


_PROPOSAL_SCHEMA_DOC = """{
  "clusters": [
    {"cluster": "<id>", "verdict": "real|artifact-doublet|artifact-lowquality|artifact-batch|artifact-ambient|ambiguous",
     "action": "keep|flag|drop", "confidence": "high|medium|low",
     "tests": {"markers": "<test a finding>", "qc": "<test b>", "composition": "<test c>",
               "geometry": "<test d>", "stability": "<test e>"},
     "rationale": "<one or two sentences tying the tests to the verdict>"}
    // must cover EVERY cluster of the current clustering (incl. subcluster ids like "5,0")
  ],
  "cell_actions": [
    // optional finer-than-cluster records: only cells of that cluster matching metric op value
    {"cluster": "<id>", "metric": "<numeric obs column>", "op": ">|>=|<|<=", "value": 0.3,
     "action": "drop|flag", "reason": "doublet|ambient|debris|low-quality|other", "note": "<free text>"}
  ],
  "overall": "<overall assessment of the integration>"
}"""

_VERDICTS = ("real", "artifact-doublet", "artifact-lowquality", "artifact-batch", "artifact-ambient", "ambiguous")


def _validate_proposal(proposal, clusters, obs):
    problems = []
    if not isinstance(proposal, dict):
        return [f"proposal must be a JSON object, got {type(proposal).__name__}"]
    entries = proposal.get("clusters")
    if not isinstance(entries, list) or not entries:
        problems.append('missing "clusters" list')
        entries = []
    seen = set()
    for e in entries:
        if not isinstance(e, dict):
            problems.append(f"cluster entry must be an object: {e!r}")
            continue
        missing = [k for k in ("cluster", "verdict", "action", "confidence", "tests", "rationale") if k not in e]
        if missing:
            problems.append(f"cluster entry missing {missing}: {e}")
            continue
        if e["verdict"] not in _VERDICTS:
            problems.append(f"verdict must be one of {_VERDICTS}: {e}")
        if e["action"] not in ("keep", "flag", "drop"):
            problems.append(f"action must be keep|flag|drop: {e}")
        if e["confidence"] not in ("high", "medium", "low"):
            problems.append(f"confidence must be high|medium|low: {e}")
        if not isinstance(e["tests"], dict) or not all(
            isinstance(e["tests"].get(k), str) and e["tests"][k].strip()
            for k in ("markers", "qc", "composition", "geometry", "stability")
        ):
            problems.append(
                f"tests must provide non-empty text for markers/qc/composition/geometry/stability "
                f"(explain unavailable evidence explicitly): {e}"
            )
        if not isinstance(e["rationale"], str) or not e["rationale"].strip():
            problems.append(f"rationale must be non-empty text: {e}")
        cluster = str(e.get("cluster"))
        if cluster not in clusters:
            problems.append(f"unknown cluster entry: {cluster!r}")
        if cluster in seen:
            problems.append(f"duplicate cluster entry: {cluster!r}")
        seen.add(cluster)
    covered = {str(e.get("cluster")) for e in entries if isinstance(e, dict)}
    missed = [c for c in clusters if c not in covered]
    if missed:
        problems.append(f"clusters without a verdict: {missed}")
    cell_actions = proposal.get("cell_actions", [])
    if not isinstance(cell_actions, list):
        problems.append('"cell_actions" must be a list when present')
        cell_actions = []
    for a in cell_actions:
        if not isinstance(a, dict):
            problems.append(f"cell_action must be an object: {a!r}")
            continue
        if str(a.get("cluster")) not in clusters:
            problems.append(f"cell_action cluster {a.get('cluster')!r} is not a current cluster id: {a}")
        metric = a.get("metric")
        if not isinstance(metric, str) or metric not in obs.columns or not pd.api.types.is_numeric_dtype(obs[metric]):
            problems.append(f'cell_action "metric" must be a numeric obs column: {a}')
        if not isinstance(a.get("op"), str) or a["op"] not in _OPS:
            problems.append(f'cell_action "op" must be one of {sorted(_OPS)}: {a}')
        try:
            if isinstance(a.get("value"), bool) or not np.isfinite(float(a.get("value"))):
                raise ValueError("not a finite number")
        except (TypeError, ValueError, OverflowError):
            problems.append(f'cell_action "value" must be finite and numeric: {a}')
        if a.get("action") not in ("drop", "flag"):
            problems.append(f'cell_action "action" must be drop|flag: {a}')
        if a.get("reason") not in ("doublet", "ambient", "debris", "low-quality", "other"):
            problems.append(f'cell_action "reason" must be doublet|ambient|debris|low-quality|other: {a}')
        if not isinstance(a.get("note"), str) or not a["note"].strip():
            problems.append(f'cell_action "note" must be non-empty text: {a}')
    return problems


def _apply_proposal(ad, key, proposal):
    """obs["_msp_action"] in keep/flag/drop: cluster actions first, then
    cell_actions refine; within each pass flags before drops so drop wins."""
    lab = ad.obs[key].astype(str)
    action = np.array(["keep"] * ad.n_obs, dtype=object)
    for verb in ("flag", "drop"):
        for e in proposal["clusters"]:
            if e["action"] == verb:
                action[(lab == str(e["cluster"])).values] = verb
        for a in proposal.get("cell_actions", []):
            if a["action"] == verb:
                mask = (lab == str(a["cluster"])).values
                mask &= _OPS[a["op"]](ad.obs[a["metric"]].to_numpy(dtype=float), float(a["value"]))
                action[mask] = verb
    ad.obs["_msp_action"] = pd.Categorical(action, categories=["keep", "flag", "drop"])
    ad.obs["_msp_verdict"] = lab.map({str(e["cluster"]): e["verdict"] for e in proposal["clusters"]}).astype("category")


def _plot_verdicts(ad, figdir):
    from .plots import UMAP_DPI, umap_axes

    os.makedirs(figdir, exist_ok=True)
    xy = np.asarray(ad.obsm["X_umap"])
    act = ad.obs["_msp_action"].astype(str).values
    base = 120000 / ad.n_obs
    fig, ax = umap_axes(ad)
    for name, color, size in (
        ("keep", "#d3d3d3", base),
        ("flag", "#b8860b", 1.5 * base),
        ("drop", "#8b0000", 1.5 * base),
    ):
        m = act == name
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=size, c=color, linewidths=0, label=f"{name} (n={int(m.sum())})")
    ax.set_title("UMAP: inspection action (proposal)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(os.path.join(figdir, "inspect_umap_action.png"), dpi=UMAP_DPI)
    plt.close(fig)


def _system_prompt(outdir, cluster_key, clusters, batch_col, species, language, n_batches=None):
    context = (
        f"Context — species: {species}."
        if species
        else "No species context was provided — infer cautiously and say so."
    )
    context += f" Sample/batch column: {batch_col!r}."
    if n_batches is not None and n_batches < 2:
        context += (
            " THIS DATASET HAS A SINGLE SAMPLE (harmony was skipped): test (c) composition carries no "
            "information — n_samples=1 and share=1.00 are expected for every cluster, never evidence of "
            "a batch artifact, and the verdict artifact-batch is unavailable; decide on (a), (b), (d), (e)."
        )
    return f"""You are a single-cell RNA-seq integration QC expert. The working directory is an \
msp (multi-sample pipeline) integration output. Task: put EVERY integrated cluster \
({cluster_key}, {len(clusters)} clusters: {clusters}) through the five-test battery and submit \
a verdict + action per cluster.
{context}

The five tests (all five must be answered for every cluster):
(a) markers — own specific positive markers? ribo/mito/stress modules with flat logFC → noise;
(b) QC axis — separated from neighbors mainly along QC metrics (mt/doublet/contamination/depth) → technical;
(c) composition — cells from several samples, or dominated by one sample → batch artifact unless \
biology explains it; clusters rich in per-sample flag/drop cells that CO-CLUSTER across samples \
are the point of this integration — judge whether the cross-sample agreement confirms the artifact \
(e.g. shared doublet signature) or rescues the cells (a real rare population every sample flagged);
(d) geometry — between two populations with intermediate signatures → doublet candidate;
(e) stability — persists across resolutions (check_stability); one-resolution splinters are not actionable.

Standard verdict mapping: passes a+c+e → real/keep; fails a and trips b → artifact-lowquality; \
trips d → artifact-doublet; single-sample + no biological explanation → artifact-batch; \
unclear → ambiguous/flag (defer, never force). "drop" proposes removal, "flag" requests review — \
NOTHING is executed here; deletion is a later separate step, so be precise, not timid.

All relevant files (paths relative to the working directory — Read exactly these, no guessing):
{file_inventory(outdir)}

What they are:
- deg_global_{{key}}.csv / deg_local_{{key}}.csv, for key in msp_leiden_r1.0 and msp_leiden_r2.0: \
precomputed DEG at both resolutions (global = one-vs-rest; local = vs the cluster's 3 nearest \
PAGA neighbors pooled; pct1/pct2 = expressing fraction in/out). They are TOO LARGE to Read whole — \
query them with deg_lookup (a cluster's ranked markers, or every cluster a gene marks) and deg_sql \
(one SELECT for anything else). These only cover the ORIGINAL clustering, though — once you \
subcluster, use check_deg for DEG on the refined ids;
- stress_clusters.csv: (key, cluster) pairs whose deg_global/deg_local top genes already trip the \
dissociation-stress/mitochondrial signature check — a head start, not a substitute for your own judgment;
- cluster_qc_*.csv, per_sample_qc.csv, cell_outlier_summary.csv: QC + composition tables (the \
latter is the same cluster-median-MAD-plus-floor doublet/ambient outlier test as check_qc_scores \
gives you live, precomputed at both resolutions);
- fragments_*.csv, minor_sibling_qc.csv: standissect-lite's cartesian product (leiden × UMAP-side \
clustering) and its per-fragment QC verdict; rows/fragments recommend_removal are candidate stray \
fragments of a parent cluster — detection only, judge each one yourself with the five tests \
(obs["standissect_product"] holds the per-cell fragment labels, so subcluster on a parent \
reproduces them);
- figures/standissect_product.png: the fragments on the UMAP; figures/umap_preannotation_removal.png: \
every cell already proposed for removal (standissect fragments ∪ cell-level outliers ∪ inherited \
osp _qc_action=="drop") BEFORE the precomputed deg_global/deg_local tables were computed — those \
tables already exclude these cells, so don't re-flag them for the same reasons found here;
- figures/umap_*.png: sample mixing, clusterings at three resolutions, inherited annotation; \
figures/qc_umap_*.png, figures/qc_violin_*.png: QC metrics on the UMAP / per-cluster violins, \
plus inherited keep/flag/drop.

Mandatory workflow:
1. Figures BEFORE conclusions: sample-mixing UMAP, the three resolution UMAPs, qc_umap_metrics, \
qc_umap_qc_action, umap_preannotation_removal.
2. deg_lookup(cluster=<id>) for every cluster of {cluster_key} (batch several calls in one turn; the \
CSVs themselves exceed the Read limit), then Read the QC/composition tables and the standissect \
diagnosis + drift tables.
3. Verify markers with check_genes (batch dozens of genes per call); QC/composition with \
check_qc_scores (one call, no arguments); stability with check_stability per suspicious cluster.
4. If a cluster is heterogeneous, split it with subcluster (ids like "5,0") — all tools and the \
final submission follow the refined clustering automatically. subcluster's own reply only DEs \
the new subclusters against each other (their immediate siblings); for a real global or local \
view on a subcluster (or on any custom comparison you want — a merged group of cluster ids, a \
specific pair), call check_deg — the precomputed deg_global_*/deg_local_* CSVs only cover the \
ORIGINAL clustering and cannot see ids you created.
5. Finish by calling submit_inspection — conclusions only in the submitted JSON.

Efficiency: parallel Reads in one turn; batch genes; check_qc_scores once; get the JSON right \
first try (schema in the tool description).

Principles: output language {language} (gene symbols excepted). Weak evidence → low confidence \
+ ambiguous/flag, never a forced guess. Distinguish genuine expression from ambient \
contamination (decontX evidence exists for that)."""


async def _run_agent(
    ad, outdir, cluster_key, other_keys, batch_col, species, language, model, effort, max_turns, remove_mask
):
    from harness_bridge import ToolSpec, run_agent

    state = {"key": cluster_key, "n_sub": 0}
    deg = DegCache(ad, outdir, remove_mask, label="inspect")
    tables = DegTables(outdir, base_key=cluster_key)
    log.info(f"== precomputed DEG tables loaded: {tables.n_rows} rows for keys {tables.keys}")

    def current_clusters():
        return cluster_order(ad.obs[state["key"]].astype(str))

    async def check_qc_scores(args):
        return text_result(qc_table(ad, state["key"], batch_col))

    async def check_stability(args):
        return text_result(stability_table(ad, str(args["cluster"]), state["key"], other_keys))

    async def check_deg(args):
        c = str(args["cluster"])
        if c not in current_clusters():
            return text_result(f"unknown cluster {c!r}; current: {current_clusters()}", is_error=True)
        reference = str(args.get("reference") or "rest").strip() or "rest"
        try:
            ref = parse_reference(reference, current_clusters())
            if ref != "rest" and c in ref:
                raise ValueError("reference must exclude the target cluster")
        except ValueError as exc:
            return text_result(exc, is_error=True)
        return text_result(deg.table(state["key"], c, reference, int(args.get("top_n") or 20), **deg_filters(args)))

    async def subcluster(args):
        c = str(args["cluster"])
        if c not in current_clusters():
            return text_result(f"unknown cluster {c!r}; current: {current_clusters()}", is_error=True)
        new_key = f"inspect_sub{state['n_sub'] + 1}"
        n, text = _subcluster_once(ad, state["key"], c, float(args["resolution"]), new_key, remove_mask)
        if n >= 2:
            state["n_sub"] += 1
            state["key"] = new_key
            text += "\n(working clustering refined; all tools and the submission now use the new ids)"
        return text_result(text)

    async def submit_inspection(args):
        try:
            proposal = json.loads(args["proposal_json"])
        except (json.JSONDecodeError, TypeError) as e:
            return text_result(f"JSON parse error, fix and resubmit: {e}", is_error=True)
        problems = _validate_proposal(proposal, current_clusters(), ad.obs)
        if problems:
            return text_result("validation failed, fix and resubmit:\n- " + "\n- ".join(problems), is_error=True)
        proposal["cluster_key"] = state["key"]
        path = os.path.join(outdir, "inspection_proposal.json")
        with open(path, "w") as fh:
            json.dump(proposal, fh, ensure_ascii=False, indent=2)
        return {**text_result(f"saved to {path}"), "_submitted": proposal}

    tools = shared_tools(
        tables,
        ad,
        lambda: state["key"],
        "Per-cluster mean expression and expressing-cell fraction for the given genes "
        "(case-insensitive). Use to verify markers.",
    ) + [
        ToolSpec(
            "check_qc_scores",
            "Per-cluster QC (median|p90) + composition (n_samples, dominant-sample share, "
            "inherited flag/drop fractions). No arguments.",
            {},
            check_qc_scores,
        ),
        ToolSpec(
            "check_stability",
            "How one cluster decomposes across the other clustering resolutions — test (e). "
            "A one-resolution splinter dissolves elsewhere.",
            {"cluster": str},
            check_stability,
        ),
        ToolSpec(
            "check_deg",
            "On-demand DEG (wilcoxon) for the CURRENT working clustering, including any subcluster "
            "splits already made — use this once a cluster you're investigating has an id the "
            "precomputed deg_global_*/deg_local_* CSVs never saw. reference='rest' (default) is "
            "one-vs-rest against every other current cluster, same semantics as deg_global_*. Pass "
            "a comma-separated list of other cluster ids as reference instead for a pooled-group "
            "comparison. A single exact ID such as 5,1 is accepted; CSV-quote pooled subcluster IDs "
            '(e.g. "5,0","5,1"). Such pooled comparisons use the same '
            "semantics as deg_local_*. Thresholds (0/empty = off): min_logfc, max_padj, min_pct1, "
            "max_pct2 — ask for exactly the gene list you need (e.g. min_logfc=1, max_padj=1e-10). "
            "Results are cached per (cluster, reference); one-vs-rest on the base clustering is answered "
            "from the precomputed table.",
            {"cluster": str, "reference": str, **DEG_FILTER_ARGS},
            check_deg,
        ),
        ToolSpec(
            "subcluster",
            "Split one heterogeneous cluster with leiden restrict_to at the given resolution "
            '(0.3-1.0 typical). New ids look like "5,0"; all tools and the final submission '
            "follow the refined clustering.",
            {"cluster": str, "resolution": float},
            subcluster,
        ),
        ToolSpec(
            "submit_inspection",
            "Submit the final verdicts (mandatory; the run completes only after validation "
            "passes). proposal_json is a JSON string with this schema:\n" + _PROPOSAL_SCHEMA_DOC,
            {"proposal_json": str},
            submit_inspection,
        ),
    ]
    try:
        result = await run_agent(
            tools=tools,
            submit_tool="submit_inspection",
            prompt="Inspect this msp integration directory following the workflow in the system "
            "prompt exactly, and finish by submitting via submit_inspection.",
            system_prompt=_system_prompt(
                outdir,
                cluster_key,
                cluster_order(ad.obs[cluster_key].astype(str)),
                batch_col,
                species,
                language,
                n_batches=int(ad.obs[batch_col].nunique()),
            ),
            cwd=os.path.abspath(outdir),
            model=model,
            effort=effort,
            max_turns=max_turns,
            allowed_builtin=("read", "glob", "grep"),
            label="inspect",
            max_buffer_size=50_000_000,  # figure Reads exceed the 1MB default pipe buffer
        )
    finally:
        tables.close()
    if result.transcript_text:
        with open(os.path.join(outdir, "inspection_notes.md"), "w") as fh:
            fh.write(result.transcript_text)
    return result.submitted


def inspect_clusters(
    outdir, species=None, language="English", cluster_key=None, model=None, effort=None, max_turns=100
):
    """Run the per-cluster inspection agent on an msp output directory.

    Writes inspection_proposal.json + inspection_notes.md, maps the accepted
    proposal onto obs["_msp_action"]/obs["_msp_verdict"] in integrated.h5ad,
    renders the verdict UMAP, refreshes report.html. Returns the proposal.
    """
    ensure()
    require_upstream_ready(outdir, "inspect")
    ad = sc.read_h5ad(os.path.join(outdir, "integrated.h5ad"))
    cluster_key = cluster_key or _detect_primary_key(outdir)
    msp_meta = ad.uns.get("msp", {})
    batch_col = msp_meta.get("batch_col")
    if not batch_col:
        raise ValueError("integrated.h5ad lacks uns['msp']['batch_col'] — not an msp output?")
    other_keys = [k for k in ad.obs.columns if k.startswith("msp_leiden_r") and k != cluster_key]
    species = species or (msp_meta.get("species") or None)

    remove_mask = load_removal_mask(outdir, ad)
    log.info(
        f"== {int(remove_mask.sum())}/{ad.n_obs} cells already recommend_removal "
        "(pre-annotation filtering) — excluded from check_deg / subcluster DE",
    )

    begin_step(outdir, "inspect")
    # Do not present the previous inspection's verdicts as fresh evidence.
    for key in ("_msp_action", "_msp_verdict"):
        if key in ad.obs:
            del ad.obs[key]
    proposal = asyncio.run(
        _run_agent(
            ad,
            outdir,
            cluster_key,
            other_keys,
            batch_col,
            species,
            language,
            model or default_model(),
            effort,
            max_turns,
            remove_mask,
        )
    )
    _apply_proposal(ad, proposal["cluster_key"], proposal)
    _plot_verdicts(ad, os.path.join(outdir, "figures"))
    tmp = os.path.join(outdir, "integrated.tmp.h5ad")
    ad.write_h5ad(tmp)
    os.replace(tmp, os.path.join(outdir, "integrated.h5ad"))
    complete_step(outdir, "inspect")
    log.info(f"== report refreshed: {generate_report(outdir)}")
    return proposal


def main(argv=None):
    configure()
    parser = argparse.ArgumentParser(prog="msp.inspect", description=__doc__)
    parser.add_argument("outdir", help="msp integration output directory")
    parser.add_argument("--species", default=None, help="defaults to uns['msp']['species']")
    parser.add_argument("--language", default="English")
    parser.add_argument("--cluster-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args(argv)

    proposal = inspect_clusters(
        args.outdir,
        species=args.species,
        language=args.language,
        cluster_key=args.cluster_key,
        model=args.model,
        effort=args.effort,
        max_turns=args.max_turns,
    )
    for e in proposal["clusters"]:
        print(f"cluster {e['cluster']}: {e['verdict']} -> {e['action']} [{e['confidence']}]")


__all__ = [
    "QC_COLS",
    "DegCache",
    "DegTables",
    "inspect_clusters",
    "main",
]


if __name__ == "__main__":
    main()
