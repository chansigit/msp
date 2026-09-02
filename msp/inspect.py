"""
msp.inspect — per-cluster inspection of an msp integration directory with a
Claude agent (claude-agent-sdk), mirroring osp.annotate's architecture.

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
import operator
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from .report import generate_report

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}

QC_COLS = ("pct_counts_mt", "n_genes_by_counts", "total_counts", "doublet_score",
           "decontX_contamination", "dissociation_score", "pct_counts_malat1")


def _detect_primary_key(outdir):
    """Cluster Annotations computes DEG for both r1.0 and r2.0 (deg_global_*/
    deg_local_*.csv); default to r1.0 when both exist (matches the old
    single-resolution 'primary_key' convention), else whichever is present."""
    paths = sorted(glob.glob(os.path.join(outdir, "deg_global_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no deg_global_*.csv in {outdir} — run the msp pipeline first")
    keys = [os.path.basename(p)[len("deg_global_"):-len(".csv")] for p in paths]
    return "msp_leiden_r1.0" if "msp_leiden_r1.0" in keys else keys[0]


def _load_removal_mask(outdir, ad):
    """Cells already proposed for removal before Cluster Annotations ran
    (msp.integrate._build_removal_mask / the Pre-annotation filtering
    UMAP) — excluded from every live DEG test here too (check_deg,
    subcluster's built-in sibling DE), matching the precomputed
    deg_global_*/deg_local_* CSVs exactly. Missing file (older msp output,
    predates this artifact) → nothing excluded. Boolean numpy array,
    aligned to ad.obs_names order."""
    path = os.path.join(outdir, "preannotation_removal.csv")
    if not os.path.exists(path):
        return np.zeros(ad.n_obs, dtype=bool)
    df = pd.read_csv(path).set_index("cell")
    return df["recommend_removal"].reindex(ad.obs_names).fillna(False).to_numpy()


def _cluster_order(labels):
    seen = pd.unique(labels)
    try:
        return sorted(seen, key=lambda x: float(str(x).split(",")[0]))
    except ValueError:
        return sorted(seen)


def _gene_table(ad, genes, cluster_key):
    upper = {g.upper(): g for g in ad.var_names}
    found = {q: upper[q.upper()] for q in genes if q.upper() in upper}
    missing = [q for q in genes if q.upper() not in upper]
    if not found:
        return f"none of these genes are in var_names: {genes}"
    idx = [ad.var_names.get_loc(g) for g in found.values()]
    X = ad.X[:, idx]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    cl = ad.obs[cluster_key].astype(str)
    cols = {}
    for c in _cluster_order(cl):
        m = (cl == c).values
        mean = X[m].mean(axis=0)
        pct = 100 * (X[m] > 0).mean(axis=0)
        cols[c] = [f"{mn:.2f}|{p:.0f}%" for mn, p in zip(mean, pct)]
    df = pd.DataFrame(cols, index=list(found.values()))
    out = "mean lognorm expr | pct expressing, per cluster:\n" + df.to_string()
    if missing:
        out += f"\nnot found in var_names: {missing}"
    return out


def _qc_table(ad, cluster_key, batch_col):
    """Per-cluster QC (median|p90) + composition: n_samples, dominant-sample
    share, inherited flag/drop fractions — tests (b) and (c) in one view."""
    cols = [c for c in QC_COLS if c in ad.obs]
    cl = ad.obs[cluster_key].astype(str)
    rows = {}
    for c in _cluster_order(cl):
        m = (cl == c).values
        sub = ad.obs.loc[m]
        row = [int(m.sum()), int(sub[batch_col].nunique()),
               f"{sub[batch_col].value_counts(normalize=True).iloc[0]:.2f}"]
        if "_qc_action" in ad.obs:
            act = sub["_qc_action"].astype(str)
            row += [f"{(act == 'flag').mean():.2f}", f"{(act == 'drop').mean():.2f}"]
        row += [f"{sub[col].median():.3g}|{sub[col].quantile(0.9):.3g}" for col in cols]
        rows[c] = row
    index = ["n_cells", "n_samples", "max_sample_share"]
    if "_qc_action" in ad.obs:
        index += ["frac_flag_inherited", "frac_drop_inherited"]
    index += cols
    df = pd.DataFrame(rows, index=index).T
    df.index.name = cluster_key
    return ("per-cluster QC (median|p90) and composition "
            "(frac_* inherited from per-sample annotation):\n" + df.to_string())


def _stability_table(ad, cluster, cluster_key, other_keys):
    """How one cluster decomposes across the other clustering resolutions —
    test (e). A cluster that dissolves at neighboring resolutions is a
    one-resolution splinter."""
    m = (ad.obs[cluster_key].astype(str) == cluster).values
    if not m.any():
        return f"unknown cluster {cluster!r}"
    lines = [f"cluster {cluster} (n={int(m.sum())}) across resolutions:"]
    for k in other_keys:
        vc = ad.obs.loc[m, k].astype(str).value_counts()
        top = ", ".join(f"{i}:{v}" for i, v in vc.head(6).items())
        lines.append(f"  {k}: {top}")
        main = vc.index[0]
        rev = int((ad.obs[k].astype(str) == main).sum())
        lines.append(f"    (its main {k} group {main!r} has {rev} cells in total)")
    return "\n".join(lines)


def _deg_table(ad, cluster_key, cluster, reference, top_n, remove_mask):
    """On-demand wilcoxon DEG for the CURRENT working clustering (any prior
    subcluster splits included) — the precomputed deg_global_*/deg_local_*
    CSVs only cover the original r1.0/r2.0 partition and can't see ids the
    agent creates. reference='rest': one-vs-rest against every other
    current cluster (matches deg_global_* semantics). reference=comma-list
    of other cluster ids: pooled reference group (matches deg_local_*
    semantics, e.g. a subcluster's siblings or PAGA neighbors). remove_mask
    cells (already recommend_removal — see Pre-annotation filtering) are
    excluded first, same as the precomputed CSVs."""
    base = ad[~remove_mask]
    lab = base.obs[cluster_key].astype(str)
    if cluster not in set(lab):
        return f"cluster {cluster!r} has no cells left once recommend_removal cells are excluded"
    if reference == "rest":
        sub = base
    else:
        ref_groups = [g.strip() for g in reference.split(",") if g.strip()]
        sub = base[lab.isin([cluster, *ref_groups])].copy()
    sc.tl.rank_genes_groups(sub, cluster_key, groups=[cluster], reference="rest",
                            method="wilcoxon", use_raw=True, pts=True)
    df = sc.get.rank_genes_groups_df(sub, group=cluster)
    df = df.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"})
    # natural scanpy ranking (by test score), not resorted by raw logFC —
    # sorting by logFC alone surfaces near-zero-expression noise genes with
    # huge fold change but pct1==pct2==0 and padj==1, same trap as anywhere
    # else in msp that reads rank_genes_groups_df
    df = df.head(top_n)
    ref_desc = "rest" if reference == "rest" else reference
    lines = [f"DEG for cluster {cluster!r} vs {ref_desc}, top {len(df)}:"]
    for _, r in df.iterrows():
        lines.append(f"  {r['names']}: logFC={r['logfoldchanges']:.2f} padj={r['pvals_adj']:.2e} "
                     f"pct1={r['pct1']:.2f} pct2={r['pct2']:.2f}")
    return "\n".join(lines)


def _subcluster_once(ad, key, cluster, resolution, new_key, remove_mask):
    """Split one cluster; sizes reported are the FULL split (removed cells
    included, so counts stay honest), but the built-in sibling DE excludes
    remove_mask cells — same DEG-only exclusion as check_deg / the
    precomputed deg_global_*/deg_local_* CSVs."""
    parent_mask = (ad.obs[key].astype(str) == cluster).values
    sc.tl.leiden(ad, restrict_to=(key, [cluster]), resolution=resolution,
                 key_added=new_key, flavor="igraph", n_iterations=2)
    sub_labels = ad.obs[new_key][parent_mask].astype(str)
    subs = _cluster_order(sub_labels)
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

_VERDICTS = ("real", "artifact-doublet", "artifact-lowquality", "artifact-batch",
             "artifact-ambient", "ambiguous")


def _validate_proposal(proposal, clusters, obs):
    problems = []
    entries = proposal.get("clusters")
    if not isinstance(entries, list) or not entries:
        problems.append('missing "clusters" list')
        entries = []
    for e in entries:
        missing = [k for k in ("cluster", "verdict", "action", "confidence", "tests", "rationale") if k not in e]
        if missing:
            problems.append(f"cluster entry missing {missing}: {e}")
            continue
        if e["verdict"] not in _VERDICTS:
            problems.append(f'verdict must be one of {_VERDICTS}: {e}')
        if e["action"] not in ("keep", "flag", "drop"):
            problems.append(f'action must be keep|flag|drop: {e}')
        if not all(k in e["tests"] for k in ("markers", "qc", "composition", "geometry", "stability")):
            problems.append(f'tests must cover markers/qc/composition/geometry/stability: {e}')
    covered = {str(e.get("cluster")) for e in entries}
    missed = [c for c in clusters if c not in covered]
    if missed:
        problems.append(f"clusters without a verdict: {missed}")
    for a in proposal.get("cell_actions", []):
        if str(a.get("cluster")) not in clusters:
            problems.append(f"cell_action cluster {a.get('cluster')!r} is not a current cluster id: {a}")
        metric = a.get("metric")
        if metric not in obs.columns or not pd.api.types.is_numeric_dtype(obs[metric]):
            problems.append(f'cell_action "metric" must be a numeric obs column: {a}')
        if a.get("op") not in _OPS:
            problems.append(f'cell_action "op" must be one of {sorted(_OPS)}: {a}')
        try:
            float(a.get("value"))
        except (TypeError, ValueError):
            problems.append(f'cell_action "value" must be numeric: {a}')
        if a.get("action") not in ("drop", "flag"):
            problems.append(f'cell_action "action" must be drop|flag: {a}')
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
    ad.obs["_msp_verdict"] = lab.map(
        {str(e["cluster"]): e["verdict"] for e in proposal["clusters"]}
    ).astype("category")


def _plot_verdicts(ad, figdir):
    from .plots import UMAP_DPI, umap_axes

    os.makedirs(figdir, exist_ok=True)
    xy = np.asarray(ad.obsm["X_umap"])
    act = ad.obs["_msp_action"].astype(str).values
    base = 120000 / ad.n_obs
    fig, ax = umap_axes(ad)
    for name, color, size in (("keep", "#d3d3d3", base), ("flag", "#b8860b", 1.5 * base),
                              ("drop", "#8b0000", 1.5 * base)):
        m = act == name
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=size, c=color, linewidths=0,
                       label=f"{name} (n={int(m.sum())})")
    ax.set_title("UMAP: inspection action (proposal)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(os.path.join(figdir, "inspect_umap_action.png"), dpi=UMAP_DPI)
    plt.close(fig)


def _file_inventory(outdir):
    patterns = ["*.csv", "figures/*.png"]
    paths = []
    for pat in patterns:
        paths += sorted(os.path.relpath(p, outdir) for p in glob.glob(os.path.join(outdir, pat)))
    return "\n".join(f"- {p}" for p in paths)


def _system_prompt(outdir, cluster_key, clusters, batch_col, species, language):
    context = (f"Context — species: {species}." if species else
               "No species context was provided — infer cautiously and say so.")
    context += f" Sample/batch column: {batch_col!r}."
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
{_file_inventory(outdir)}

What they are:
- deg_global_{{key}}.csv / deg_local_{{key}}.csv, for key in msp_leiden_r1.0 and msp_leiden_r2.0: \
precomputed DEG at both resolutions (global = one-vs-rest; local = vs the cluster's 3 nearest \
PAGA neighbors pooled; pct1/pct2 = expressing fraction in/out). These only cover the ORIGINAL \
clustering, though — once you subcluster, use check_deg for DEG on the refined ids;
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
2. Read deg_global_{cluster_key}.csv / deg_local_{cluster_key}.csv and the QC/composition tables; \
read the standissect diagnosis + drift tables.
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


async def _run_agent(ad, outdir, cluster_key, other_keys, batch_col, species, language,
                     model, effort, max_turns, remove_mask):
    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolUseBlock,
        create_sdk_mcp_server, query, tool,
    )

    state = {"key": cluster_key, "n_sub": 0}
    holder = {}

    def current_clusters():
        return _cluster_order(ad.obs[state["key"]].astype(str))

    @tool("check_genes",
          "Per-cluster mean expression and expressing-cell fraction for the given genes "
          "(case-insensitive). Use to verify markers.", {"genes": list})
    async def check_genes(args):
        genes = args["genes"]
        if isinstance(genes, str):
            genes = [g for g in genes.replace(",", " ").split() if g]
        return {"content": [{"type": "text", "text": _gene_table(ad, genes, state["key"])}]}

    @tool("check_qc_scores",
          "Per-cluster QC (median|p90) + composition (n_samples, dominant-sample share, "
          "inherited flag/drop fractions). No arguments.", {})
    async def check_qc_scores(args):
        return {"content": [{"type": "text", "text": _qc_table(ad, state["key"], batch_col)}]}

    @tool("check_stability",
          "How one cluster decomposes across the other clustering resolutions — test (e). "
          "A one-resolution splinter dissolves elsewhere.", {"cluster": str})
    async def check_stability(args):
        return {"content": [{"type": "text",
                             "text": _stability_table(ad, str(args["cluster"]), state["key"], other_keys)}]}

    @tool("check_deg",
          "On-demand DEG (wilcoxon) for the CURRENT working clustering, including any subcluster "
          "splits already made — use this once a cluster you're investigating has an id the "
          "precomputed deg_global_*/deg_local_* CSVs never saw. reference='rest' (default) is "
          "one-vs-rest against every other current cluster, same semantics as deg_global_*. Pass "
          "a comma-separated list of other cluster ids as reference instead for a pooled-group "
          "comparison (e.g. a subcluster vs its siblings, or vs specific PAGA neighbors), same "
          "semantics as deg_local_*.", {"cluster": str, "reference": str, "top_n": int})
    async def check_deg(args):
        c = str(args["cluster"])
        if c not in current_clusters():
            return {"content": [{"type": "text", "text": f"unknown cluster {c!r}; current: {current_clusters()}"}],
                    "is_error": True}
        reference = str(args.get("reference") or "rest").strip() or "rest"
        if reference != "rest":
            ref_groups = [g.strip() for g in reference.split(",") if g.strip()]
            unknown = [g for g in ref_groups if g not in current_clusters()]
            if unknown:
                return {"content": [{"type": "text",
                                     "text": f"unknown reference cluster(s) {unknown}; current: {current_clusters()}"}],
                        "is_error": True}
        top_n = int(args.get("top_n") or 20)
        text = _deg_table(ad, state["key"], c, reference, top_n, remove_mask)
        return {"content": [{"type": "text", "text": text}]}

    @tool("subcluster",
          "Split one heterogeneous cluster with leiden restrict_to at the given resolution "
          '(0.3-1.0 typical). New ids look like "5,0"; all tools and the final submission '
          "follow the refined clustering.", {"cluster": str, "resolution": float})
    async def subcluster(args):
        c = str(args["cluster"])
        if c not in current_clusters():
            return {"content": [{"type": "text", "text": f"unknown cluster {c!r}; current: {current_clusters()}"}],
                    "is_error": True}
        new_key = f"inspect_sub{state['n_sub'] + 1}"
        n, text = _subcluster_once(ad, state["key"], c, float(args["resolution"]), new_key, remove_mask)
        if n >= 2:
            state["n_sub"] += 1
            state["key"] = new_key
            text += "\n(working clustering refined; all tools and the submission now use the new ids)"
        return {"content": [{"type": "text", "text": text}]}

    @tool("submit_inspection",
          "Submit the final verdicts (mandatory; the run completes only after validation "
          "passes). proposal_json is a JSON string with this schema:\n" + _PROPOSAL_SCHEMA_DOC,
          {"proposal_json": str})
    async def submit_inspection(args):
        try:
            proposal = json.loads(args["proposal_json"])
        except json.JSONDecodeError as e:
            return {"content": [{"type": "text", "text": f"JSON parse error, fix and resubmit: {e}"}],
                    "is_error": True}
        problems = _validate_proposal(proposal, current_clusters(), ad.obs)
        if problems:
            return {"content": [{"type": "text",
                                 "text": "validation failed, fix and resubmit:\n- " + "\n- ".join(problems)}],
                    "is_error": True}
        proposal["cluster_key"] = state["key"]
        holder["proposal"] = proposal
        path = os.path.join(outdir, "inspection_proposal.json")
        with open(path, "w") as fh:
            json.dump(proposal, fh, ensure_ascii=False, indent=2)
        return {"content": [{"type": "text", "text": f"saved to {path}"}]}

    server = create_sdk_mcp_server(name="msp", version="1.0.0",
                                   tools=[check_genes, check_qc_scores, check_stability,
                                          check_deg, subcluster, submit_inspection])
    options = ClaudeAgentOptions(
        mcp_servers={"msp": server},
        allowed_tools=["Read", "Glob", "Grep",
                       "mcp__msp__check_genes", "mcp__msp__check_qc_scores",
                       "mcp__msp__check_stability", "mcp__msp__check_deg",
                       "mcp__msp__subcluster", "mcp__msp__submit_inspection"],
        permission_mode="bypassPermissions",
        max_buffer_size=50_000_000,  # figure Reads exceed the 1MB default pipe buffer
        system_prompt=_system_prompt(outdir, cluster_key,
                                     _cluster_order(ad.obs[cluster_key].astype(str)),
                                     batch_col, species, language),
        cwd=os.path.abspath(outdir),
        max_turns=max_turns,
        **({"model": model} if model else {}),
        **({"effort": effort} if effort else {}),
    )

    result_text = None
    async for message in query(
        prompt="Inspect this msp integration directory following the workflow in the system "
               "prompt exactly, and finish by submitting via submit_inspection.",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    arg_hint = str(next(iter(block.input.values()), ""))[:80]
                    print(f"== agent: {block.name}({arg_hint})", flush=True)
        elif isinstance(message, ResultMessage):
            result_text = message.result
            if message.total_cost_usd:
                print(f"== agent cost: ${message.total_cost_usd:.2f}", flush=True)

    if "proposal" not in holder:
        raise RuntimeError(f"agent finished without a successful submit_inspection call. Final reply:\n{result_text}")
    if result_text:
        with open(os.path.join(outdir, "inspection_notes.md"), "w") as fh:
            fh.write(result_text)
    return holder["proposal"]


def inspect_clusters(outdir, species=None, language="English", cluster_key=None,
                     model=None, effort=None, max_turns=100):
    """Run the per-cluster inspection agent on an msp output directory.

    Writes inspection_proposal.json + inspection_notes.md, maps the accepted
    proposal onto obs["_msp_action"]/obs["_msp_verdict"] in integrated.h5ad,
    renders the verdict UMAP, refreshes report.html. Returns the proposal.
    """
    ad = sc.read_h5ad(os.path.join(outdir, "integrated.h5ad"))
    cluster_key = cluster_key or _detect_primary_key(outdir)
    msp_meta = ad.uns.get("msp", {})
    batch_col = msp_meta.get("batch_col")
    if not batch_col:
        raise ValueError("integrated.h5ad lacks uns['msp']['batch_col'] — not an msp output?")
    other_keys = [k for k in ad.obs.columns
                  if k.startswith("msp_leiden_r") and k != cluster_key]
    species = species or (msp_meta.get("species") or None)

    remove_mask = _load_removal_mask(outdir, ad)
    print(f"== {int(remove_mask.sum())}/{ad.n_obs} cells already recommend_removal "
          "(pre-annotation filtering) — excluded from check_deg / subcluster DE", flush=True)

    proposal = asyncio.run(_run_agent(ad, outdir, cluster_key, other_keys, batch_col,
                                      species, language, model, effort, max_turns, remove_mask))
    _apply_proposal(ad, proposal["cluster_key"], proposal)
    _plot_verdicts(ad, os.path.join(outdir, "figures"))
    tmp = os.path.join(outdir, "integrated.tmp.h5ad")
    ad.write_h5ad(tmp)
    os.replace(tmp, os.path.join(outdir, "integrated.h5ad"))
    print(f"== report refreshed: {generate_report(outdir)}", flush=True)
    return proposal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="msp.inspect", description=__doc__)
    parser.add_argument("outdir", help="msp integration output directory")
    parser.add_argument("--species", default=None, help="defaults to uns['msp']['species']")
    parser.add_argument("--language", default="English")
    parser.add_argument("--cluster-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args()

    proposal = inspect_clusters(args.outdir, species=args.species, language=args.language,
                                cluster_key=args.cluster_key, model=args.model,
                                effort=args.effort, max_turns=args.max_turns)
    for e in proposal["clusters"]:
        print(f"cluster {e['cluster']}: {e['verdict']} -> {e['action']} [{e['confidence']}]")
