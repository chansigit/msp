"""
msp.annotate — cell-type annotation of an msp integration directory with a
Claude agent (claude-agent-sdk), run AFTER msp.inspect.

Unit of annotation: every cluster of the base clustering (msp_leiden_r2.0,
the finer of the two Cluster Annotations resolutions). For each cluster the
agent answers a fixed reasoning chain — (1) is it a distinct entity or a
splinter of its r1.0 parent / siblings? (2) best coarse (lineage) and fine
(subtype) label, or noise/low-quality → remove; (3) merge into a sibling or
neighbour, or keep as is — and submits one JSON per cluster.

100% coverage is enforced twice: the agent tracks progress with Claude
Code's Task tools (TaskCreate/TaskUpdate/TaskList — one task per cluster),
and the host refuses finalize_annotation until every base cluster has a
validated submission. Merge decisions are made in ONE session (the agent
sees its own earlier verdicts), and the host resolves the merge graph
deterministically (union-find) and rejects inconsistent components, so no
separate harmonization agent is needed.

Removal is real here (unlike inspect/integrate, which only propose): the
final removed set = preannotation_removal.csv ∪ inspect's obs["_msp_action"]
== "drop" ∪ clusters the agent marks action=remove. It is archived per cell
with its sources in annotation_removed.csv; integrated.h5ad is left intact
and annotated.h5ad (removed cells dropped, msp_ann_* columns added) is the
downstream deliverable. Live DEG (check_deg) excludes the pre-agent part of
that union (preannotation ∪ inspect drop), same as the precomputed
deg_global_*/deg_local_* CSVs plus inspect's verdicts.

Prior labels (osp's inherited _ann_coarse/_ann_fine and whatever cell-type
columns the original authors shipped) are surfaced per cluster as
composition tables — reference evidence, never ground truth; the column
names are detected, not assumed, because they differ per dataset.

Usage:
    python -m msp.annotate <msp_outdir> [--species human] [--model ...]
"""

import argparse
import asyncio
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc

from .inspect import _cluster_order, _deg_table, _file_inventory, _gene_table, _load_removal_mask
from .report import generate_report

BASE_KEY = "msp_leiden_r2.0"
PARENT_KEY = "msp_leiden_r1.0"
CONFIDENCES = ("high", "medium", "low")
REMOVE_REASONS = ("doublet", "low-quality", "ambient", "stress", "batch", "other")
_STANHUE_DIR = os.path.expanduser("~/.claude/skills/stanhue/scripts")


# ---------------------------------------------------------------- evidence

def _prior_label_columns(ad, batch_col):
    """obs columns that look like categorical cell labels shipped with the
    data (author annotations, osp's _ann_coarse/_ann_fine). Detected, not
    assumed: any string/categorical column with 2..200 levels that is not
    (i) produced by this pipeline (leiden/qc/msp/inspect/standissect
    columns), (ii) boolean-like, or (iii) a sample-identity column (every
    level lives in exactly one sample and there are no more levels than
    samples — 'orig.ident', 'project', 'source_unit' and friends)."""
    deny_prefix = ("msp_", "_msp", "leiden", "qc_", "_qc", "inspect_", "standissect",
                   "decontX_clusters", "predicted_doublet", "low_quality", "original_cluster",
                   "recommended_disposition")
    n_batches = ad.obs[batch_col].nunique()
    out = []
    for c in ad.obs.columns:
        if c == batch_col or c.startswith(deny_prefix):
            continue
        s = ad.obs[c]
        if not (s.dtype == object or str(s.dtype) == "category"):
            continue
        n = s.nunique(dropna=True)
        if n < 2 or n > 200:
            continue
        levels = set(s.dropna().astype(str).str.lower().unique())
        if levels <= {"true", "false", "yes", "no", "0", "1"}:
            continue
        per_level_batches = ad.obs.groupby(c, observed=True)[batch_col].nunique()
        if n <= n_batches and per_level_batches.max() == 1:
            continue  # sample identity in disguise
        out.append(c)
    return out


def _load_paga_neighbors(outdir, key):
    path = os.path.join(outdir, f"paga_neighbors_{key}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str)
    nb = {}
    for c, g in df.groupby("cluster", sort=False):
        nb[str(c)] = list(g.sort_values("rank", key=lambda s: s.astype(int))["neighbor"].astype(str))
    return nb


def _cluster_context(ad, cluster, batch_col, prior_cols, paga, pre_agent_removed):
    """Everything about ONE base cluster that is not gene expression:
    size, how much of it is already slated for removal, its r1.0 parent(s)
    and the r2.0 siblings under that parent, PAGA neighbours, sample
    composition, QC medians, inspect's verdict composition, and the prior
    label compositions — the reasoning-chain context in one call."""
    base = ad.obs[BASE_KEY].astype(str)
    m = (base == cluster).values
    if not m.any():
        return f"unknown cluster {cluster!r}"
    sub = ad.obs.loc[m]
    n = int(m.sum())
    lines = [f"cluster {cluster} ({BASE_KEY}): n={n} cells, "
             f"{int(pre_agent_removed[m].sum())} ({100 * pre_agent_removed[m].mean():.1f}%) already slated for "
             "removal before this step (preannotation filtering ∪ inspect drop)"]
    if PARENT_KEY in ad.obs:
        par = sub[PARENT_KEY].astype(str).value_counts()
        lines.append(f"  {PARENT_KEY} parent composition: " +
                     ", ".join(f"{i}:{v} ({100 * v / n:.0f}%)" for i, v in par.head(5).items()))
        main_parent = par.index[0]
        sib = ad.obs.loc[(ad.obs[PARENT_KEY].astype(str) == main_parent).values, BASE_KEY].astype(str).value_counts()
        sib = sib.drop(cluster, errors="ignore")
        lines.append(f"  siblings under parent {main_parent} ({BASE_KEY}): " +
                     (", ".join(f"{i}:{v}" for i, v in sib.items()) if len(sib) else "none — this cluster IS the parent"))
    if cluster in paga:
        lines.append(f"  PAGA nearest neighbours ({BASE_KEY}): {', '.join(paga[cluster])}")
    vc = sub[batch_col].value_counts(normalize=True)
    lines.append(f"  samples: {sub[batch_col].nunique()}/{ad.obs[batch_col].nunique()} present, "
                 f"dominant sample share {vc.iloc[0]:.2f} ({vc.index[0]})")
    qc = [c for c in ("doublet_score", "decontX_contamination", "pct_counts_mt", "n_genes_by_counts",
                      "total_counts", "dissociation_score") if c in ad.obs]
    lines.append("  QC medians: " + ", ".join(f"{c}={sub[c].median():.3g}" for c in qc))
    for col, name in (("_msp_verdict", "inspect verdict"), ("_msp_action", "inspect action"),
                      ("_qc_action", "osp per-sample qc action")):
        if col in ad.obs:
            cc = sub[col].astype(str).value_counts(normalize=True)
            lines.append(f"  {name} composition: " + ", ".join(f"{i}:{v:.2f}" for i, v in cc.items()))
    if prior_cols:
        lines.append("  prior label compositions (reference only, NOT ground truth; top 5 per column):")
        for c in prior_cols:
            vals = sub[c].dropna().astype(str)
            vals = vals[~vals.str.lower().isin(["nan", "none", ""])]
            if vals.empty:
                lines.append(f"    {c}: (unlabelled in this cluster)")
                continue
            cc = vals.value_counts(normalize=True)
            lines.append(f"    {c}: " + ", ".join(f"{i}:{v:.2f}" for i, v in cc.head(5).items()))
    return "\n".join(lines)


# ---------------------------------------------------------------- proposal

_CLUSTER_SCHEMA_DOC = """{
  "cluster_id": "<base cluster id, e.g. "7">",
  "coarse_label": "<lineage-level label in English, e.g. 'Fibroblast' / 'Macrophage' / 'Endothelial'>",
  "fine_label": "<subtype-level label in English, e.g. 'CTHRC1+ matrix fibroblast'; for removed clusters describe what it is, e.g. 'Fibroblast-immune doublet'>",
  "merge_target": null | "<another base cluster id this one is part of — same population, not a distinct entity>",
  "action": "keep" | "remove",
  "remove_reason": null | "doublet" | "low-quality" | "ambient" | "stress" | "batch" | "other",
  "confidence": "high" | "medium" | "low",
  "evidence": {
    "distinctness": "<step 1: is it distinct from its r1.0 parent / r2.0 siblings? what separates it (local DEG), or nothing?>",
    "markers": "<step 2: the positive markers that fix the identity, verified with check_genes/check_deg>",
    "merge": "<step 3: why merge / why keep separate>"
  },
  "rationale": "<one or two sentences tying evidence to the labels and the merge/remove decision>"
}"""


def _validate_cluster(e, clusters):
    problems = []
    for k in ("cluster_id", "coarse_label", "fine_label", "merge_target", "action", "confidence",
              "evidence", "rationale"):
        if k not in e:
            problems.append(f"missing field {k!r}")
    if problems:
        return problems
    cid = str(e["cluster_id"])
    if cid not in clusters:
        problems.append(f"cluster_id {cid!r} is not a base cluster; base clusters: {clusters}")
    for k in ("coarse_label", "fine_label"):
        if not isinstance(e[k], str) or not e[k].strip():
            problems.append(f"{k} must be a non-empty string")
    if e["action"] not in ("keep", "remove"):
        problems.append("action must be keep|remove")
    if e["action"] == "remove" and e.get("remove_reason") not in REMOVE_REASONS:
        problems.append(f"remove requires remove_reason in {REMOVE_REASONS}")
    if e["confidence"] not in CONFIDENCES:
        problems.append(f"confidence must be one of {CONFIDENCES}")
    mt = e["merge_target"]
    if mt is not None:
        mt = str(mt)
        if mt not in clusters:
            problems.append(f"merge_target {mt!r} is not a base cluster")
        elif mt == cid:
            problems.append("merge_target cannot be the cluster itself")
    ev = e["evidence"]
    if not isinstance(ev, dict) or not all(k in ev for k in ("distinctness", "markers", "merge")):
        problems.append("evidence must be an object with distinctness / markers / merge")
    return problems


def _components(entries):
    """Union-find over merge_target edges → {cluster_id: component members}
    (sorted, numeric-aware)."""
    parent = {c: c for c in entries}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c, e in entries.items():
        mt = e.get("merge_target")
        if mt is not None:
            parent[find(c)] = find(str(mt))
    groups = {}
    for c in entries:
        groups.setdefault(find(c), []).append(c)
    comp = {}
    for members in groups.values():
        members = _cluster_order(members)
        for c in members:
            comp[c] = members
    return comp


def _validate_final(entries, clusters):
    """Cross-cluster consistency, the deterministic replacement for a
    harmonization agent. Every violation names the clusters to fix."""
    problems = []
    missing = [c for c in clusters if c not in entries]
    if missing:
        problems.append(f"no submission yet for clusters {missing} — submit_cluster each of them first")
        return problems
    for c, e in entries.items():
        mt = e.get("merge_target")
        if mt is None:
            continue
        mt = str(mt)
        tgt = entries[mt]
        if tgt["action"] == "remove" and e["action"] != "remove":
            problems.append(f"cluster {c} merges into {mt}, but {mt} is action=remove — either remove {c} too "
                            f"or drop the merge_target")
    comp = _components(entries)
    seen = set()
    for c, members in comp.items():
        key = tuple(members)
        if key in seen or len(members) < 2:
            continue
        seen.add(key)
        kept = [m for m in members if entries[m]["action"] == "keep"]
        for field in ("coarse_label", "fine_label"):
            vals = {entries[m][field].strip() for m in kept}
            if len(vals) > 1:
                problems.append(f"merged group {'+'.join(members)} disagrees on {field}: "
                                + "; ".join(f"{m}={entries[m][field]!r}" for m in kept)
                                + " — resubmit them with one shared label")
    # one fine label ↔ one coarse label, and fine-label equality == merge
    by_fine = {}
    for c, e in entries.items():
        if e["action"] != "keep":
            continue
        by_fine.setdefault(e["fine_label"].strip(), []).append(c)
    for fine, members in by_fine.items():
        coarse = {entries[m]["coarse_label"].strip() for m in members}
        if len(coarse) > 1:
            problems.append(f"fine label {fine!r} sits under several coarse labels {sorted(coarse)} "
                            f"(clusters {members}) — one fine label belongs to exactly one coarse label")
        comps = {tuple(comp[m]) for m in members}
        if len(comps) > 1:
            problems.append(f"clusters {members} share fine label {fine!r} but are not merged — either "
                            "set merge_target between them (same population) or give them distinct fine labels")
    return problems


# ---------------------------------------------------------------- apply

def _apply(ad, proposal, pre_removed, pre_sources):
    """obs columns on the FULL object: msp_ann_cluster (merged id, members
    joined by '+'), msp_ann_coarse / msp_ann_fine, msp_ann_action
    (keep/remove). Returns the removal archive (removed cells only, with
    their sources)."""
    entries = {str(e["cluster_id"]): e for e in proposal["clusters"]}
    comp = _components(entries)
    base = ad.obs[BASE_KEY].astype(str)
    merged_id = {c: "+".join(members) for c, members in comp.items()}
    ad.obs["msp_ann_cluster"] = base.map(merged_id).astype("category")
    ad.obs["msp_ann_coarse"] = base.map({c: e["coarse_label"].strip() for c, e in entries.items()}).astype("category")
    ad.obs["msp_ann_fine"] = base.map({c: e["fine_label"].strip() for c, e in entries.items()}).astype("category")
    agent_remove = base.isin([c for c, e in entries.items() if e["action"] == "remove"]).values
    removed = pre_removed | agent_remove
    ad.obs["msp_ann_action"] = pd.Categorical(np.where(removed, "remove", "keep"), categories=["keep", "remove"])
    archive = pd.DataFrame({"cell": ad.obs_names, BASE_KEY: base.values,
                            **{k: v for k, v in pre_sources.items()},
                            "annotate_remove": agent_remove,
                            "remove_reason": base.map({c: e.get("remove_reason") for c, e in entries.items()
                                                       if e["action"] == "remove"}).values})
    return archive.loc[removed].reset_index(drop=True)


def _palette(ad, col):
    """stanhue hierarchical palette (related labels share a hue family) when
    the skill is importable, else scanpy's default."""
    try:
        if _STANHUE_DIR not in sys.path:
            sys.path.insert(0, _STANHUE_DIR)
        from scatter_colormap import assign_celltype_colors  # type: ignore[import-not-found]
    except Exception:
        return None
    cmap = assign_celltype_colors(np.asarray(ad.obsm["X_umap"]), ad.obs[col].astype(str).to_numpy())
    return [cmap.get(str(c), "#999999") for c in ad.obs[col].cat.categories]


def _plot(ad_full, ad_kept, figdir):
    from .plots import UMAP_DPI, save_single_umap, umap_axes
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    for col, fname in (("msp_ann_coarse", "annotation_umap_coarse.png"),
                       ("msp_ann_fine", "annotation_umap_fine.png")):
        ad_kept.obs[col] = ad_kept.obs[col].cat.remove_unused_categories()
        pal = _palette(ad_kept, col)
        if pal:
            ad_kept.uns[f"{col}_colors"] = pal
        n = ad_kept.obs[col].nunique()
        save_single_umap(ad_kept, col, os.path.join(figdir, fname), repel=True,
                         repel_fontsize=9 if n > 15 else 11,
                         figsize=(9, 9) if n > 15 else None)

    xy = np.asarray(ad_full.obsm["X_umap"])
    act = ad_full.obs["msp_ann_action"].astype(str).values
    base = 120000 / ad_full.n_obs
    fig, ax = umap_axes(ad_full)
    for name, color, size in (("keep", "#d3d3d3", base), ("remove", "#c0392b", 1.5 * base)):
        m = act == name
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=size, c=color, linewidths=0, label=f"{name} (n={int(m.sum())})")
    ax.set_title("UMAP: cells removed at annotation (all sources)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(os.path.join(figdir, "annotation_umap_removed.png"), dpi=UMAP_DPI)
    plt.close(fig)


# ---------------------------------------------------------------- agent

def _system_prompt(outdir, clusters, batch_col, species, prior_cols, language):
    context = (f"Context — species: {species}." if species else
               "No species context was provided — infer cautiously and say so.")
    context += f" Sample/batch column: {batch_col!r}."
    priors = (", ".join(prior_cols) if prior_cols else "none detected")
    return f"""You are a single-cell RNA-seq cell-type annotation expert. The working directory is an \
msp (multi-sample pipeline) integration output that has already been through per-cluster QC \
inspection. Task: annotate EVERY cluster of the base clustering {BASE_KEY} \
({len(clusters)} clusters: {clusters}) with a coarse (lineage) and a fine (subtype) label, decide \
for each whether it is a distinct population or should MERGE into another base cluster, or whether \
it is noise / low-quality cells to REMOVE — and submit one JSON per cluster.
{context}
Prior label columns detected in obs (reference evidence only, never ground truth — datasets name \
these differently, so they were detected, not assumed): {priors}.

Reasoning chain, answered for every cluster in this order:
1. Distinctness — what does this {BASE_KEY} cluster add over its {PARENT_KEY} parent? Compare it \
against its siblings under the same parent (check_deg with reference=<sibling ids>) and against its \
PAGA neighbours (deg_local_{BASE_KEY}.csv). Real subtype signal (specific positive markers) vs a \
resolution splinter (only depth/QC/cell-cycle/stress genes, or nothing).
2. Identity — the best coarse label and fine label in English; OR, if the positive markers are absent \
and the cluster is explained by doublet/ambient/low-quality/stress/batch signals, action=remove with \
remove_reason (labels still describe what it is, e.g. 'Fibroblast-immune doublet').
3. Merge — if it is the same population as another base cluster (a splinter, or two clusters with \
the same identity), set merge_target to that cluster's id; otherwise null. Merge is explicit: two kept \
clusters given the same fine label MUST also be merged (or given distinct fine labels), and a merged \
group must share one coarse and one fine label — finalize_annotation checks this and tells you what to fix.
Coarse labels group fine labels: one fine label belongs to exactly one coarse label across the dataset. \
Keep the vocabulary consistent across clusters (same population → literally the same string).

All relevant files (paths relative to the working directory — Read exactly these, no guessing):
{_file_inventory(outdir)}

What they are:
- deg_global_{{key}}.csv / deg_local_{{key}}.csv for key in {PARENT_KEY} and {BASE_KEY}: precomputed DEG \
(global = one-vs-rest; local = vs the cluster's 3 nearest PAGA neighbours pooled; pct1/pct2 = expressing \
fraction in/out), computed after excluding the cells in figures/umap_preannotation_removal.png;
- paga_neighbors_{{key}}.csv: the PAGA neighbours used for deg_local;
- inspection_proposal.json / inspection_notes.md: the QC inspection that ran before you (five-test \
verdicts on {PARENT_KEY} clusters; its 'drop' clusters are already in the removal set — do not re-litigate \
them, annotate what remains);
- stress_clusters.csv, cluster_qc_*.csv, cell_outlier_summary.csv, per_sample_qc.csv: QC/composition tables;
- minor_sibling_qc.csv, fragments_*.csv: standissect-lite fragment QC;
- figures/umap_*.png, figures/qc_umap_*.png: UMAPs by sample, clusterings at three resolutions, inherited \
annotation, QC metrics; figures/inspect_umap_action.png: the inspection verdict.

Mandatory workflow:
1. Create ONE task per base cluster with TaskCreate (subject "annotate cluster <id>") before any analysis, \
so nothing is skipped; keep TaskList honest — TaskUpdate a task to completed ONLY after its submit_cluster \
call succeeded.
2. Look at the figures first (three resolution UMAPs, inherited annotation, sample mixing, \
inspect_umap_action), then Read deg_global_{BASE_KEY}.csv and deg_local_{BASE_KEY}.csv once.
3. For each cluster: cluster_context (parent/siblings/neighbours/priors/QC/inspect verdict in one call), \
then verify markers with check_genes (batch dozens of genes per call) and, when distinctness is in \
doubt, check_deg against its siblings or a specific neighbour. Then submit_cluster. You may resubmit a \
cluster later to revise it (e.g. after seeing its merge partner) — the last submission wins.
4. When every task is completed, call finalize_annotation. If it reports problems, fix them by \
resubmitting the named clusters and call it again. The run completes only after it succeeds.

Efficiency: parallel Reads in one turn; batch genes; do not re-read files you already read.

Principles: labels in English gene-symbol style vocabulary; rationale/evidence text in {language}. Weak \
evidence → low confidence, never a forced guess. Distinguish genuine expression from ambient \
contamination (decontX evidence exists for that). Respect the inspection verdicts: they came from a \
dedicated QC pass; you annotate identity and decide merges."""


async def _run_agent(ad, outdir, clusters, batch_col, species, prior_cols, paga, pre_agent_removed,
                     language, model, effort, max_turns):
    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolUseBlock,
        create_sdk_mcp_server, query, tool,
    )

    entries = {}
    holder = {}

    @tool("cluster_context",
          "Non-expression context for one base cluster: size, share already slated for removal, "
          "r1.0 parent composition and r2.0 siblings, PAGA neighbours, sample composition, QC "
          "medians, inspection verdict composition, prior label compositions.", {"cluster": str})
    async def cluster_context(args):
        return {"content": [{"type": "text",
                             "text": _cluster_context(ad, str(args["cluster"]), batch_col, prior_cols, paga,
                                                      pre_agent_removed)}]}

    @tool("check_genes",
          f"Per-{BASE_KEY}-cluster mean expression and expressing-cell fraction for the given genes "
          "(case-insensitive). Use to verify markers.", {"genes": list})
    async def check_genes(args):
        genes = args["genes"]
        if isinstance(genes, str):
            genes = [g for g in genes.replace(",", " ").split() if g]
        return {"content": [{"type": "text", "text": _gene_table(ad, genes, BASE_KEY)}]}

    @tool("check_deg",
          f"On-demand DEG (wilcoxon) for one {BASE_KEY} cluster. reference='rest' (default) is "
          "one-vs-rest (deg_global semantics); a comma-separated list of cluster ids is a pooled "
          "reference group (e.g. its siblings under the r1.0 parent, or one specific neighbour). "
          "Cells already slated for removal are excluded, like the precomputed tables.",
          {"cluster": str, "reference": str, "top_n": int})
    async def check_deg(args):
        c = str(args["cluster"])
        if c not in clusters:
            return {"content": [{"type": "text", "text": f"unknown cluster {c!r}; base clusters: {clusters}"}],
                    "is_error": True}
        reference = str(args.get("reference") or "rest").strip() or "rest"
        if reference != "rest":
            unknown = [g.strip() for g in reference.split(",") if g.strip() and g.strip() not in clusters]
            if unknown:
                return {"content": [{"type": "text",
                                     "text": f"unknown reference cluster(s) {unknown}; base clusters: {clusters}"}],
                        "is_error": True}
        top_n = int(args.get("top_n") or 20)
        return {"content": [{"type": "text",
                             "text": _deg_table(ad, BASE_KEY, c, reference, top_n, pre_agent_removed)}]}

    @tool("submit_cluster",
          "Submit (or resubmit — last one wins) the annotation of ONE base cluster. cluster_json is a "
          "JSON string with this schema:\n" + _CLUSTER_SCHEMA_DOC, {"cluster_json": str})
    async def submit_cluster(args):
        try:
            e = json.loads(args["cluster_json"])
        except json.JSONDecodeError as exc:
            return {"content": [{"type": "text", "text": f"JSON parse error, fix and resubmit: {exc}"}],
                    "is_error": True}
        problems = _validate_cluster(e, clusters)
        if problems:
            return {"content": [{"type": "text", "text": "invalid, fix and resubmit:\n- " + "\n- ".join(problems)}],
                    "is_error": True}
        e["cluster_id"] = str(e["cluster_id"])
        if e["merge_target"] is not None:
            e["merge_target"] = str(e["merge_target"])
        entries[e["cluster_id"]] = e
        left = [c for c in clusters if c not in entries]
        print(f"== submitted cluster {e['cluster_id']}: {e['coarse_label']} / {e['fine_label']} "
              f"[{e['action']}{', merge→' + e['merge_target'] if e['merge_target'] else ''}]", flush=True)
        return {"content": [{"type": "text",
                             "text": f"recorded cluster {e['cluster_id']}; {len(entries)}/{len(clusters)} submitted"
                                     + (f", remaining: {left}" if left else " — all covered, call finalize_annotation")}]}

    @tool("finalize_annotation",
          "Validate all submissions together (coverage, merge graph consistency, label hierarchy) "
          "and finish the run. overall is a short overall assessment of the dataset's populations.",
          {"overall": str})
    async def finalize_annotation(args):
        problems = _validate_final(entries, clusters)
        if problems:
            return {"content": [{"type": "text",
                                 "text": "not final yet, fix and call again:\n- " + "\n- ".join(problems)}],
                    "is_error": True}
        comp = _components(entries)
        groups = sorted({tuple(v) for v in comp.values() if len(v) > 1}, key=lambda t: float(t[0]))
        proposal = {"cluster_key": BASE_KEY, "parent_key": PARENT_KEY,
                    "clusters": [entries[c] for c in clusters],
                    "merged_groups": ["+".join(g) for g in groups],
                    "overall": str(args.get("overall") or "")}
        holder["proposal"] = proposal
        path = os.path.join(outdir, "annotation_proposal.json")
        with open(path, "w") as fh:
            json.dump(proposal, fh, ensure_ascii=False, indent=2)
        return {"content": [{"type": "text", "text": f"accepted; saved to {path}"}]}

    server = create_sdk_mcp_server(name="msp", version="1.0.0",
                                   tools=[cluster_context, check_genes, check_deg, submit_cluster,
                                          finalize_annotation])
    options = ClaudeAgentOptions(
        mcp_servers={"msp": server},
        allowed_tools=["Read", "Glob", "Grep", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
                       "mcp__msp__cluster_context", "mcp__msp__check_genes", "mcp__msp__check_deg",
                       "mcp__msp__submit_cluster", "mcp__msp__finalize_annotation"],
        permission_mode="bypassPermissions",
        disallowed_tools=["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch"],
        max_buffer_size=50_000_000,  # figure Reads exceed the 1MB default pipe buffer
        system_prompt=_system_prompt(outdir, clusters, batch_col, species, prior_cols, language),
        cwd=os.path.abspath(outdir),
        max_turns=max_turns,
        **({"model": model} if model else {}),
        **({"effort": effort} if effort else {}),
    )

    result_text = None
    async for message in query(
        prompt="Annotate this msp integration directory following the workflow in the system prompt "
               "exactly: one Task per base cluster, submit_cluster for each, then finalize_annotation.",
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
        raise RuntimeError("agent finished without a successful finalize_annotation call "
                           f"({len(entries)}/{len(clusters)} clusters submitted). Final reply:\n{result_text}")
    if result_text:
        with open(os.path.join(outdir, "annotation_notes.md"), "w") as fh:
            fh.write(result_text)
    return holder["proposal"]


# ---------------------------------------------------------------- entry

def annotate_clusters(outdir, species=None, language="English", model=None, effort=None, max_turns=200):
    """Run the annotation agent on an msp output directory (after msp.inspect).

    Writes annotation_proposal.json, annotation_notes.md, annotation_removed.csv
    (every removed cell with its sources), annotated.h5ad (removed cells
    dropped; msp_ann_cluster / msp_ann_coarse / msp_ann_fine / msp_ann_action
    added), the annotation UMAPs, and refreshes report.html. integrated.h5ad
    is not modified. Returns the proposal.
    """
    ad = sc.read_h5ad(os.path.join(outdir, "integrated.h5ad"))
    for k in (BASE_KEY, PARENT_KEY):
        if k not in ad.obs:
            raise ValueError(f"integrated.h5ad lacks obs[{k!r}] — not an msp output with r1.0/r2.0 clusterings?")
    msp_meta = ad.uns.get("msp", {})
    batch_col = msp_meta.get("batch_col")
    if not batch_col:
        raise ValueError("integrated.h5ad lacks uns['msp']['batch_col'] — not an msp output?")
    species = species or (msp_meta.get("species") or None)
    clusters = _cluster_order(ad.obs[BASE_KEY].astype(str))

    pre_sources = {"preannotation": _load_removal_mask(outdir, ad)}
    if "_msp_action" in ad.obs:
        pre_sources["inspect_drop"] = (ad.obs["_msp_action"].astype(str) == "drop").to_numpy()
    else:
        print("== no obs['_msp_action'] — msp.inspect has not run; only preannotation removals inherited",
              flush=True)
    pre_agent_removed = np.logical_or.reduce(list(pre_sources.values()))
    print(f"== {int(pre_agent_removed.sum())}/{ad.n_obs} cells already slated for removal "
          f"({', '.join(f'{k}={int(v.sum())}' for k, v in pre_sources.items())})", flush=True)

    prior_cols = _prior_label_columns(ad, batch_col)
    print(f"== prior label columns detected: {prior_cols}", flush=True)
    paga = _load_paga_neighbors(outdir, BASE_KEY)

    proposal = asyncio.run(_run_agent(ad, outdir, clusters, batch_col, species, prior_cols, paga,
                                      pre_agent_removed, language, model, effort, max_turns))

    archive = _apply(ad, proposal, pre_agent_removed, pre_sources)
    archive.to_csv(os.path.join(outdir, "annotation_removed.csv"), index=False)
    kept = ad[(ad.obs["msp_ann_action"] == "keep").values].copy()
    print(f"== removed {len(archive)} cells (agent-marked clusters: "
          f"{int(archive['annotate_remove'].sum())}); annotated.h5ad keeps {kept.n_obs}/{ad.n_obs}", flush=True)
    _plot(ad, kept, os.path.join(outdir, "figures"))
    tmp = os.path.join(outdir, "annotated.tmp.h5ad")
    kept.write_h5ad(tmp)
    os.replace(tmp, os.path.join(outdir, "annotated.h5ad"))
    print(f"== report refreshed: {generate_report(outdir)}", flush=True)
    return proposal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="msp.annotate", description=__doc__)
    parser.add_argument("outdir", help="msp integration output directory (after msp.inspect)")
    parser.add_argument("--species", default=None, help="defaults to uns['msp']['species']")
    parser.add_argument("--language", default="English")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-turns", type=int, default=200)
    args = parser.parse_args()

    proposal = annotate_clusters(args.outdir, species=args.species, language=args.language,
                                 model=args.model, effort=args.effort, max_turns=args.max_turns)
    for e in proposal["clusters"]:
        tail = f" merge→{e['merge_target']}" if e["merge_target"] else ""
        print(f"cluster {e['cluster_id']}: {e['coarse_label']} / {e['fine_label']} "
              f"[{e['action']}, {e['confidence']}]{tail}")
    if proposal["merged_groups"]:
        print("merged groups: " + ", ".join(proposal["merged_groups"]))
