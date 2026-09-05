"""
msp.annotate — cell-type annotation of an msp integration directory with an
agent (harness_bridge: HARNESS=claude|deepseek|openai), run AFTER msp.inspect.

Unit of annotation: every cluster of the base clustering (msp_leiden_r2.0,
the finer of the two Cluster Annotations resolutions). For each cluster the
agent answers a fixed reasoning chain — (1) is it a distinct entity or a
splinter of its r1.0 parent / siblings? (2) best coarse (lineage) and fine
(subtype) label, or noise/low-quality → remove; (3) merge into a sibling or
neighbour, or keep as is — and submits one JSON per cluster.

100% coverage is enforced twice: the agent tracks progress with a session
task list (TaskCreate/TaskUpdate/TaskList — one task per cluster; Claude
Code's own under HARNESS=claude, a host-served equivalent under deepseek/openai),
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
import logging
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc
from harness_bridge import AgentIncompleteError, default_model

from .agent_tools import DEG_FILTER_ARGS, deg_filters, shared_tools, text_result
from .evidence import (
    DegCache,
    DegTables,
    cluster_order,
    file_inventory,
    load_paga_neighbors,
    load_removal_mask,
    parse_reference,
)
from .log import configure, ensure
from .report import generate_report
from .steps import begin_step, complete_step, require_upstream_ready

log = logging.getLogger(__name__)

BASE_KEY = "msp_leiden_r2.0"
PARENT_KEY = "msp_leiden_r1.0"
CONFIDENCES = ("high", "medium", "low")
REMOVE_REASONS = ("doublet", "low-quality", "ambient", "stress", "batch", "other")


# ---------------------------------------------------------------- evidence


def _prior_label_columns(ad, batch_col):
    """obs columns that look like categorical cell labels shipped with the
    data (author annotations, osp's _ann_coarse/_ann_fine). Detected, not
    assumed: any string/categorical column with 2..200 levels that is not
    (i) produced by this pipeline (leiden/qc/msp/inspect/standissect
    columns), (ii) boolean-like, or (iii) a sample-identity column (every
    level lives in exactly one sample and there are no more levels than
    samples — 'orig.ident', 'project', 'source_unit' and friends)."""
    deny_prefix = (
        "msp_",
        "_msp",
        "leiden",
        "qc_",
        "_qc",
        "inspect_",
        "standissect",
        "decontX_clusters",
        "predicted_doublet",
        "low_quality",
        "original_cluster",
        "recommended_disposition",
    )
    n_batches = ad.obs[batch_col].nunique()
    out = []
    for c in ad.obs.columns:
        if c == batch_col or c.startswith(deny_prefix):
            continue
        s = ad.obs[c]
        if not (pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype)):
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


def _cluster_context(ad, cluster, batch_col, prior_cols, paga, pre_agent_removed, tables=None):
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
    lines = [
        f"cluster {cluster} ({BASE_KEY}): n={n} cells, "
        f"{int(pre_agent_removed[m].sum())} ({100 * pre_agent_removed[m].mean():.1f}%) already slated for "
        "removal before this step (preannotation filtering ∪ inspect drop)"
    ]
    if PARENT_KEY in ad.obs:
        par = sub[PARENT_KEY].astype(str).value_counts()
        lines.append(
            f"  {PARENT_KEY} parent composition: "
            + ", ".join(f"{i}:{v} ({100 * v / n:.0f}%)" for i, v in par.head(5).items())
        )
        main_parent = par.index[0]
        sib = ad.obs.loc[(ad.obs[PARENT_KEY].astype(str) == main_parent).values, BASE_KEY].astype(str).value_counts()
        sib = sib.drop(cluster, errors="ignore")
        lines.append(
            f"  siblings under parent {main_parent} ({BASE_KEY}): "
            + (", ".join(f"{i}:{v}" for i, v in sib.items()) if len(sib) else "none — this cluster IS the parent")
        )
    if cluster in paga:
        lines.append(f"  PAGA nearest neighbours ({BASE_KEY}): {', '.join(paga[cluster])}")
    if tables is not None:
        mk = tables.markers_text(BASE_KEY, cluster)
        if mk:
            lines.append(mk)
    vc = sub[batch_col].value_counts(normalize=True)
    lines.append(
        f"  samples: {sub[batch_col].nunique()}/{ad.obs[batch_col].nunique()} present, "
        f"dominant sample share {vc.iloc[0]:.2f} ({vc.index[0]})"
    )
    qc = [
        c
        for c in (
            "doublet_score",
            "decontX_contamination",
            "pct_counts_mt",
            "n_genes_by_counts",
            "total_counts",
            "dissociation_score",
        )
        if c in ad.obs
    ]
    lines.append("  QC medians: " + ", ".join(f"{c}={sub[c].median():.3g}" for c in qc))
    for col, name in (
        ("_msp_verdict", "inspect verdict"),
        ("_msp_action", "inspect action"),
        ("_qc_action", "osp per-sample qc action"),
    ):
        if col in ad.obs:
            cc = sub[col].dropna().astype(str).value_counts(normalize=True)
            lines.append(
                f"  {name} composition (observed cells; {sub[col].isna().sum()} missing): "
                + (", ".join(f"{i}:{v:.2f}" for i, v in cc.items()) or "unavailable")
            )
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
    if not isinstance(e, dict):
        return [f"cluster entry must be an object: {e!r}"]
    for k in (
        "cluster_id",
        "coarse_label",
        "fine_label",
        "merge_target",
        "action",
        "confidence",
        "evidence",
        "rationale",
    ):
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
    if not isinstance(ev, dict) or not all(
        isinstance(ev.get(k), str) and ev[k].strip() for k in ("distinctness", "markers", "merge")
    ):
        problems.append(
            "evidence must provide non-empty text for distinctness / markers / merge; "
            "explain unavailable evidence explicitly"
        )
    if not isinstance(e["rationale"], str) or not e["rationale"].strip():
        problems.append("rationale must be non-empty text")
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
        members = cluster_order(members)
        for c in members:
            comp[c] = members
    return comp


def _validate_final(entries, clusters):
    """Cross-cluster consistency, the deterministic replacement for a
    harmonization agent. Every violation names the clusters to fix."""
    problems = []
    if not isinstance(entries, dict):
        return ["entries must be an object keyed by cluster ID"]
    for c, e in entries.items():
        problems.extend(f"cluster {c}: {p}" for p in _validate_cluster(e, clusters))
        if c not in clusters or (isinstance(e, dict) and str(e.get("cluster_id")) != c):
            problems.append(f"entry key {c!r} must match a current cluster_id")
    if problems:
        return problems
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
            problems.append(
                f"cluster {c} merges into {mt}, but {mt} is action=remove — either remove {c} too "
                f"or drop the merge_target"
            )
    comp = _components(entries)
    seen = set()
    for members in comp.values():
        key = tuple(members)
        if key in seen or len(members) < 2:
            continue
        seen.add(key)
        kept = [m for m in members if entries[m]["action"] == "keep"]
        for field in ("coarse_label", "fine_label"):
            vals = {entries[m][field].strip() for m in kept}
            if len(vals) > 1:
                problems.append(
                    f"merged group {'+'.join(members)} disagrees on {field}: "
                    + "; ".join(f"{m}={entries[m][field]!r}" for m in kept)
                    + " — resubmit them with one shared label"
                )
    # one fine label ↔ one coarse label, and fine-label equality == merge
    by_fine = {}
    for c, e in entries.items():
        if e["action"] != "keep":
            continue
        by_fine.setdefault(e["fine_label"].strip(), []).append(c)
    for fine, members in by_fine.items():
        coarse = {entries[m]["coarse_label"].strip() for m in members}
        if len(coarse) > 1:
            problems.append(
                f"fine label {fine!r} sits under several coarse labels {sorted(coarse)} "
                f"(clusters {members}) — one fine label belongs to exactly one coarse label"
            )
        comps = {tuple(comp[m]) for m in members}
        if len(comps) > 1:
            problems.append(
                f"clusters {members} share fine label {fine!r} but are not merged — either "
                "set merge_target between them (same population) or give them distinct fine labels"
            )
    return problems


def _guard_batch_annotation(entry):
    """Retain batch-only suspicions without changing the model's explanation."""
    if entry.get("action") == "remove" and entry.get("remove_reason") == "batch":
        entry["requested_action"] = entry["action"]
        entry["requested_remove_reason"] = entry["remove_reason"]
        entry["action"] = "keep"
        entry["remove_reason"] = None
        entry["host_adjustment"] = {
            "policy": "batch_annotation_non_destructive_v1",
            "reason": "Sample/batch composition alone does not establish invalid cells; retained for review.",
        }
        entry["review_required"] = True
        log.warning("== annotate cluster %s: batch-only removal adjusted to keep for review", entry["cluster_id"])
    return entry


# ---------------------------------------------------------------- apply


def _apply(ad, proposal, pre_removed, pre_sources):
    """obs columns on the FULL object: msp_ann_cluster (merged id, members
    joined by '+'), msp_ann_coarse / msp_ann_fine, msp_ann_action
    (keep/remove). Returns the removal archive (removed cells only, with
    their sources)."""
    entries = {str(e["cluster_id"]): e for e in proposal["clusters"]}
    if any(e.get("action") == "remove" and e.get("remove_reason") == "batch" for e in entries.values()):
        raise ValueError("unguarded batch-only annotation removal; normalize and validate the proposal before applying")
    comp = _components(entries)
    base = ad.obs[BASE_KEY].astype(str)
    merged_id = {c: "+".join(members) for c, members in comp.items()}
    ad.obs["msp_ann_cluster"] = base.map(merged_id).astype("category")
    ad.obs["msp_ann_coarse"] = base.map({c: e["coarse_label"].strip() for c, e in entries.items()}).astype("category")
    ad.obs["msp_ann_fine"] = base.map({c: e["fine_label"].strip() for c, e in entries.items()}).astype("category")
    ad.obs["msp_ann_review"] = base.map({c: bool(e.get("review_required", False)) for c, e in entries.items()}).astype(
        bool
    )
    agent_remove = base.isin([c for c, e in entries.items() if e["action"] == "remove"]).values
    removed = pre_removed | agent_remove
    ad.obs["msp_ann_action"] = pd.Categorical(np.where(removed, "remove", "keep"), categories=["keep", "remove"])
    archive = pd.DataFrame(
        {
            "cell": ad.obs_names,
            BASE_KEY: base.values,
            **pre_sources,
            "annotate_remove": agent_remove,
            "remove_reason": base.map(
                {c: e.get("remove_reason") for c, e in entries.items() if e["action"] == "remove"}
            ).values,
        }
    )
    return archive.loc[removed].reset_index(drop=True)


def _palette(ad, col):
    """stanhue hierarchical palette (related labels share a hue family) in
    category order, or None for scanpy's default when stanhue is missing or
    fails; the fallback is announced so it never passes unnoticed. Failing
    here must not lose the annotation run that precedes it."""
    try:
        from stanhue import assign_celltype_colors

        cmap = assign_celltype_colors(np.asarray(ad.obsm["X_umap"]), ad.obs[col].astype(str).to_numpy())
    except Exception as exc:
        log.warning(f"== stanhue palette unavailable ({exc!r}); using scanpy's default palette for {col}")
        return None
    return [cmap.get(str(c), "#999999") for c in ad.obs[col].cat.categories]


def _plot(ad_full, ad_kept, figdir):
    import matplotlib.pyplot as plt

    from .plots import UMAP_DPI, save_single_umap, umap_axes

    os.makedirs(figdir, exist_ok=True)
    for col, fname in (("msp_ann_coarse", "annotation_umap_coarse.png"), ("msp_ann_fine", "annotation_umap_fine.png")):
        ad_kept.obs[col] = ad_kept.obs[col].cat.remove_unused_categories()
        if ad_kept.n_obs == 0:
            # Preserve the usual figure files and full-run coordinate scale.
            fig, ax = umap_axes(ad_full)
            ax.text(0.5, 0.5, "No cells retained after annotation", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"UMAP: {col} (0 cells)")
            fig.savefig(os.path.join(figdir, fname), dpi=UMAP_DPI)
            plt.close(fig)
            continue
        pal = _palette(ad_kept, col)
        if pal:
            ad_kept.uns[f"{col}_colors"] = pal
        n = ad_kept.obs[col].nunique()
        save_single_umap(
            ad_kept,
            col,
            os.path.join(figdir, fname),
            repel=True,
            repel_fontsize=9 if n > 15 else 11,
            figsize=(9, 9) if n > 15 else None,
        )

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


def _system_prompt(outdir, clusters, batch_col, species, prior_cols, language, n_batches=None):
    context = (
        f"Context — species: {species}."
        if species
        else "No species context was provided — infer cautiously and say so."
    )
    context += f" Sample/batch column: {batch_col!r}."
    if n_batches is not None and n_batches < 2:
        context += (
            " THIS DATASET HAS A SINGLE SAMPLE (harmony was skipped): the per-cluster sample lines are "
            "trivially 1/1 present, share 1.00 — never use sample composition as evidence for or against "
            "an identity, a merge or a removal; rely on markers, QC axes, PAGA neighbourhood and priors."
        )
    priors = ", ".join(prior_cols) if prior_cols else "none detected"
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
and the cluster is explained by independent doublet/ambient/low-quality/stress evidence, action=remove with \
remove_reason (labels still describe what it is, e.g. 'Fibroblast-immune doublet'). Batch/sample composition \
alone never justifies removal: use keep and explain the uncertainty. The host converts batch-only \
remove requests to keep for review and preserves your original request and explanation. Do not relabel \
a batch-only suspicion as another removal reason to bypass this policy.
3. Merge — if it is the same population as another base cluster (a splinter, or two clusters with \
the same identity), set merge_target to that cluster's id; otherwise null. Merge is explicit: two kept \
clusters given the same fine label MUST also be merged (or given distinct fine labels), and a merged \
group must share one coarse and one fine label — finalize_annotation checks this and tells you what to fix.
Coarse labels group fine labels: one fine label belongs to exactly one coarse label across the dataset. \
Keep the vocabulary consistent across clusters (same population → literally the same string). If prior \
label columns named r<NN>_zmip_ann_coarse / r<NN>_msp_ann_coarse exist, they are the PREVIOUS ROUND of this \
same pipeline on these same cells: reuse their coarse vocabulary verbatim for the same populations (a lineage \
called 'Fibroblast' last round stays 'Fibroblast', not 'Stromal fibroblast'), and change a label only where \
the evidence contradicts it — label churn between rounds is noise, not progress.

All relevant files (paths relative to the working directory — Read exactly these, no guessing):
{file_inventory(outdir)}

What they are:
- deg_global_{{key}}.csv / deg_local_{{key}}.csv for key in {PARENT_KEY} and {BASE_KEY}: precomputed DEG \
(global = one-vs-rest; local = vs the cluster's 3 nearest PAGA neighbours pooled; pct1/pct2 = expressing \
fraction in/out), computed after excluding the cells in figures/umap_preannotation_removal.png. TOO LARGE \
to Read whole: cluster_context already carries each cluster's top-12 global and local markers from them; \
deg_lookup (a cluster's full ranked list, or every cluster a gene marks, any key) and deg_sql (one \
SELECT) retrieve the rest — check_deg is for comparisons the tables don't hold (custom references, \
subclustered ids);
- paga_neighbors_{{key}}.csv: the PAGA neighbours used for deg_local;
- inspection_proposal.json / inspection_notes.md: the QC inspection that ran before you (five-test \
verdicts on {PARENT_KEY} clusters; its 'drop' clusters are already in the removal set — do not re-litigate \
them, annotate what remains);
- stress_clusters.csv, cluster_qc_*.csv, cell_outlier_summary.csv, per_sample_qc.csv: QC/composition tables;
- minor_sibling_qc.csv, fragments_*.csv: standissect-lite fragment QC;
- figures/umap_*.png, figures/qc_umap_*.png: UMAPs by sample, clusterings at three resolutions, inherited \
annotation, QC metrics; figures/inspect_umap_action.png: the inspection verdict.

Recovery rule (takes precedence after a fresh-session/context-reset notice): first call \
annotation_status(cluster='', offset=0) and TaskList. Host submissions are authoritative; reconcile task \
statuses with them. Query one saved cluster entry only if needed for label/merge consistency. Continue \
only pending clusters. Do not repeat global figures or completed cluster_context calls merely because \
the conversation reset. If all clusters are submitted, proceed to finalize_annotation and fix only its \
named conflicts.

Mandatory workflow for new work:
1. Create ONE task per base cluster with TaskCreate (subject "annotate cluster <id>") before any analysis, \
so nothing is skipped; keep TaskList honest — TaskUpdate a task to completed ONLY after its submit_cluster \
call succeeded.
2. Look at the figures first (three resolution UMAPs, inherited annotation, sample mixing, \
inspect_umap_action); the DEG tables are reached through cluster_context / deg_lookup / deg_sql, not Read.
3. Work in batches of at most FOUR pending clusters: get cluster_context only for that batch \
(parent/siblings/neighbours/priors/QC/inspect verdict in one call), then verify markers with check_genes (batch dozens of genes per call) and, when distinctness is in \
doubt, check_deg against its siblings or a specific neighbour. Submit those clusters before requesting \
context for the next batch; never collect every cluster context first. You may resubmit a cluster later to revise it (e.g. after seeing its merge partner) — the last submission wins.
4. When every task is completed, call finalize_annotation. If it reports problems, fix them by \
resubmitting the named clusters and call it again. The run completes only after it succeeds.

Efficiency: parallel Reads in one turn; batch genes; do not re-read files you already read.

Principles: labels in English gene-symbol style vocabulary; rationale/evidence text in {language}. Weak \
evidence → low confidence, never a forced guess. Distinguish genuine expression from ambient \
contamination (decontX evidence exists for that). Respect the inspection verdicts: they came from a \
dedicated QC pass; you annotate identity and decide merges."""


_STATUS_MAX_BYTES = 16 * 1024
_STATUS_PAGE_ITEMS = 8
_STATUS_DETAIL_BYTES = 6000


def _annotation_status(entries, clusters, cluster="", offset=0):
    """Bounded recovery view; host submissions, not TaskList, are authoritative."""
    if not isinstance(cluster, str) or type(offset) is not int or offset < 0:
        return text_result("cluster must be a string and offset a nonnegative integer", is_error=True)
    if cluster:
        if cluster not in clusters:
            return text_result("unknown cluster ID", is_error=True)
        if cluster not in entries:
            return text_result(json.dumps({"submitted": False, "message": "This cluster has no saved submission."}))
        # ASCII JSON has unambiguous byte offsets, including Unicode evidence.
        serialized = json.dumps(entries[cluster], ensure_ascii=True, separators=(",", ":"))
        if offset > len(serialized):
            return text_result("offset is past the end of the saved entry", is_error=True)
        end = min(offset + _STATUS_DETAIL_BYTES, len(serialized))
        result = {
            "cluster": cluster,
            "submitted": True,
            "entry_json": serialized[offset:end],
            "offset": offset,
            "next_offset": end if end < len(serialized) else None,
            "total_bytes": len(serialized),
        }
    else:
        pending = [c for c in clusters if c not in entries]
        accepted = [entries[c] for c in clusters if c in entries]
        length = max(len(pending), len(accepted))
        if offset > length:
            return text_result("offset is past the status lists", is_error=True)
        result = None
        for page_size in range(_STATUS_PAGE_ITEMS, 0, -1):
            end = min(offset + page_size, length)
            summaries = []
            for entry in accepted[offset:end]:
                row = {
                    "cluster_id": entry["cluster_id"],
                    "action": entry["action"],
                    "merge_target": entry["merge_target"],
                }
                for field in ("coarse_label", "fine_label"):
                    value = entry[field]
                    row[field] = value if len(value) <= 96 else value[:96] + "…"
                summaries.append(row)
            result = {
                "total_clusters": len(clusters),
                "submitted_count": len(accepted),
                "pending_count": len(pending),
                "pending_ids": pending[offset:end],
                "submitted": summaries,
                "offset": offset,
                "next_offset": end if end < length else None,
                "note": "Host submissions are authoritative. Labels may be shortened; query one cluster for its full saved entry.",
            }
            if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= _STATUS_MAX_BYTES:
                break
    text = json.dumps(result, ensure_ascii=False)
    if len(text.encode("utf-8")) > _STATUS_MAX_BYTES:
        return text_result("status identifier exceeds the response size limit", is_error=True)
    return text_result(text)


async def _run_agent(
    ad, outdir, clusters, batch_col, species, prior_cols, paga, pre_agent_removed, language, model, effort, max_turns
):
    from harness_bridge import ToolSpec, run_agent

    entries = {}
    deg = DegCache(ad, outdir, pre_agent_removed, label="annotate")
    tables = DegTables(outdir, base_key=BASE_KEY)
    log.info(f"== precomputed DEG tables loaded: {tables.n_rows} rows for keys {tables.keys}")

    async def annotation_status(args):
        return _annotation_status(entries, clusters, args.get("cluster", ""), args.get("offset", 0))

    async def cluster_context(args):
        return text_result(
            _cluster_context(ad, str(args["cluster"]), batch_col, prior_cols, paga, pre_agent_removed, tables)
        )

    async def check_deg(args):
        c = str(args["cluster"])
        if c not in clusters:
            return text_result(f"unknown cluster {c!r}; base clusters: {clusters}", is_error=True)
        reference = str(args.get("reference") or "rest").strip() or "rest"
        try:
            ref = parse_reference(reference, clusters)
            if ref != "rest" and c in ref:
                raise ValueError("reference must exclude the target cluster")
        except ValueError as exc:
            return text_result(exc, is_error=True)
        return text_result(deg.table(BASE_KEY, c, reference, int(args.get("top_n") or 20), **deg_filters(args)))

    async def submit_cluster(args):
        try:
            e = json.loads(args["cluster_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            return text_result(f"JSON parse error, fix and resubmit: {exc}", is_error=True)
        problems = _validate_cluster(e, clusters)
        if problems:
            return text_result("invalid, fix and resubmit:\n- " + "\n- ".join(problems), is_error=True)
        e["cluster_id"] = str(e["cluster_id"])
        if e["merge_target"] is not None:
            e["merge_target"] = str(e["merge_target"])
        # Audit fields are host-owned, never accepted from an agent submission.
        for field in ("requested_action", "requested_remove_reason", "host_adjustment", "review_required"):
            e.pop(field, None)
        _guard_batch_annotation(e)
        entries[e["cluster_id"]] = e
        left = [c for c in clusters if c not in entries]
        log.info(
            f"== submitted cluster {e['cluster_id']}: {e['coarse_label']} / {e['fine_label']} "
            f"[{e['action']}{', merge→' + e['merge_target'] if e['merge_target'] else ''}]",
        )
        return text_result(
            f"recorded cluster {e['cluster_id']}; {len(entries)}/{len(clusters)} submitted"
            + (f", remaining: {left}" if left else " — all covered, call finalize_annotation")
            + (
                "; host retained this batch-only removal for review; merge/label consistency still applies"
                if e.get("review_required")
                else ""
            )
        )

    async def finalize_annotation(args):
        problems = _validate_final(entries, clusters)
        if problems:
            return text_result("not final yet, fix and call again:\n- " + "\n- ".join(problems), is_error=True)
        comp = _components(entries)
        # Order merged groups by their first member's position in the base
        # clustering order; cluster IDs are not guaranteed to be numeric.
        position = {c: i for i, c in enumerate(clusters)}
        groups = sorted(
            {tuple(v) for v in comp.values() if len(v) > 1}, key=lambda t: position.get(t[0], len(position))
        )
        proposal = {
            "cluster_key": BASE_KEY,
            "parent_key": PARENT_KEY,
            "clusters": [entries[c] for c in clusters],
            "merged_groups": ["+".join(g) for g in groups],
            "overall": str(args.get("overall") or ""),
        }
        path = os.path.join(outdir, "annotation_proposal.json")
        with open(path, "w") as fh:
            json.dump(proposal, fh, ensure_ascii=False, indent=2)
        return {**text_result(f"accepted; saved to {path}"), "_submitted": proposal}

    tools = [
        ToolSpec(
            "annotation_status",
            "Recover authoritative saved annotation progress. cluster='' gives compact paged pending IDs "
            "and submitted labels/action/merge, never evidence; offset=0 starts, then follow next_offset. "
            "Specify one cluster to recover its complete saved entry as ASCII JSON entry_json fragments "
            "(offset is then a byte cursor; concatenate fragments). Each response is at most 16 KiB. "
            "Call this and TaskList FIRST after a context reset; do not resubmit completed work just to recover it.",
            {"cluster": str, "offset": int},
            annotation_status,
        ),
        ToolSpec(
            "cluster_context",
            "Non-expression context for one base cluster: size, share already slated for removal, "
            "r1.0 parent composition and r2.0 siblings, PAGA neighbours, sample composition, QC "
            "medians, inspection verdict composition, prior label compositions.",
            {"cluster": str},
            cluster_context,
        ),
        *shared_tools(
            tables,
            ad,
            lambda: BASE_KEY,
            f"Per-{BASE_KEY}-cluster mean expression and expressing-cell fraction for the given genes "
            "(case-insensitive). Use to verify markers.",
        ),
        ToolSpec(
            "check_deg",
            f"On-demand DEG (wilcoxon) for one {BASE_KEY} cluster. reference='rest' (default) is "
            "one-vs-rest (deg_global semantics); a comma-separated list of cluster ids is a pooled "
            "reference group (e.g. its siblings under the r1.0 parent, or one specific neighbour). "
            "Cells already slated for removal are excluded, like the precomputed tables. Thresholds "
            "(0/empty = off): min_logfc, max_padj, min_pct1, max_pct2 — ask for exactly the gene list you "
            "need. Cached per (cluster, reference); one-vs-rest and vs-the-3-PAGA-neighbours come from the "
            "precomputed tables.",
            {"cluster": str, "reference": str, **DEG_FILTER_ARGS},
            check_deg,
        ),
        ToolSpec(
            "submit_cluster",
            "Submit (or resubmit — last one wins) the annotation of ONE base cluster. cluster_json is a "
            "JSON string with this schema:\n" + _CLUSTER_SCHEMA_DOC,
            {"cluster_json": str},
            submit_cluster,
        ),
        ToolSpec(
            "finalize_annotation",
            "Validate all submissions together (coverage, merge graph consistency, label hierarchy) "
            "and finish the run. overall is a short overall assessment of the dataset's populations.",
            {"overall": str},
            finalize_annotation,
        ),
    ]
    try:
        result = await run_agent(
            tools=tools,
            submit_tool="finalize_annotation",
            prompt="Annotate this msp integration directory following the workflow in the system prompt "
            "exactly: one Task per base cluster, submit_cluster for each, then finalize_annotation.",
            system_prompt=_system_prompt(
                outdir, clusters, batch_col, species, prior_cols, language, n_batches=int(ad.obs[batch_col].nunique())
            ),
            cwd=os.path.abspath(outdir),
            model=model,
            effort=effort,
            max_turns=max_turns,
            allowed_builtin=("read", "glob", "grep", "tasks"),
            label="annotate",
            max_buffer_size=50_000_000,  # figure Reads exceed the 1MB default pipe buffer
        )
    except AgentIncompleteError as e:
        raise RuntimeError(f"{e} ({len(entries)}/{len(clusters)} clusters submitted)") from None
    finally:
        tables.close()
    if result.transcript_text:
        with open(os.path.join(outdir, "annotation_notes.md"), "w") as fh:
            fh.write(result.transcript_text)
    return result.submitted


# ---------------------------------------------------------------- entry


def annotate_clusters(outdir, species=None, language="English", model=None, effort=None, max_turns=200):
    """Run the annotation agent on an msp output directory (after msp.inspect).

    Writes annotation_proposal.json, annotation_notes.md, annotation_removed.csv
    (every removed cell with its sources), annotated.h5ad (removed cells
    dropped; msp_ann_cluster / msp_ann_coarse / msp_ann_fine / msp_ann_action
    added), the annotation UMAPs, and refreshes report.html. integrated.h5ad
    is not modified. Returns the proposal.
    """
    ensure()
    require_upstream_ready(outdir, "annotate")
    ad = sc.read_h5ad(os.path.join(outdir, "integrated.h5ad"))
    for k in (BASE_KEY, PARENT_KEY):
        if k not in ad.obs:
            raise ValueError(f"integrated.h5ad lacks obs[{k!r}] — not an msp output with r1.0/r2.0 clusterings?")
    msp_meta = ad.uns.get("msp", {})
    batch_col = msp_meta.get("batch_col")
    if not batch_col:
        raise ValueError("integrated.h5ad lacks uns['msp']['batch_col'] — not an msp output?")
    species = species or (msp_meta.get("species") or None)
    clusters = cluster_order(ad.obs[BASE_KEY].astype(str))

    pre_sources = {"preannotation": load_removal_mask(outdir, ad)}
    if "_msp_action" in ad.obs:
        inspect_drop = (ad.obs["_msp_action"].astype(str) == "drop").to_numpy()
        if (
            "_msp_verdict" in ad.obs
            and (inspect_drop & ad.obs["_msp_verdict"].astype(str).eq("artifact-batch").to_numpy()).any()
        ):
            # A saved pre-policy integrated matrix must not bypass the host
            # gate. Explicit cell-level QC may still remove cells within a
            # flagged batch cluster; verify that its guarded proposal agrees.
            from types import SimpleNamespace

            from .inspect import _apply_proposal, _validate_proposal

            path = os.path.join(outdir, "inspection_proposal.json")
            try:
                with open(path, encoding="utf-8") as fh:
                    inspection = json.load(fh)
                key = inspection["cluster_key"]
                problems = _validate_proposal(inspection, cluster_order(ad.obs[key].astype(str)), ad.obs)
                if problems or any(
                    e.get("verdict") == "artifact-batch" and e.get("action") == "drop" for e in inspection["clusters"]
                ):
                    raise ValueError("unguarded inspection proposal")
                probe = SimpleNamespace(obs=ad.obs.copy(), n_obs=ad.n_obs)
                _apply_proposal(probe, key, inspection)
                if not probe.obs["_msp_action"].astype(str).equals(ad.obs["_msp_action"].astype(str)):
                    raise ValueError("inspection proposal and applied actions disagree")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    "legacy artifact-batch drop decisions require reapplying the saved inspection "
                    "with conservative host flags in a new output directory"
                ) from exc
        pre_sources["inspect_drop"] = inspect_drop
    else:
        log.warning("== no obs['_msp_action'] — msp.inspect has not run; only preannotation removals inherited")
    pre_agent_removed = np.logical_or.reduce(list(pre_sources.values()))
    log.info(
        f"== {int(pre_agent_removed.sum())}/{ad.n_obs} cells already slated for removal "
        f"({', '.join(f'{k}={int(v.sum())}' for k, v in pre_sources.items())})",
    )

    prior_cols = _prior_label_columns(ad, batch_col)
    log.info(f"== prior label columns detected: {prior_cols}")
    paga = load_paga_neighbors(outdir, BASE_KEY)

    begin_step(outdir, "annotate")
    proposal = asyncio.run(
        _run_agent(
            ad,
            outdir,
            clusters,
            batch_col,
            species,
            prior_cols,
            paga,
            pre_agent_removed,
            language,
            model or default_model(),
            effort,
            max_turns,
        )
    )

    archive = _apply(ad, proposal, pre_agent_removed, pre_sources)
    archive.to_csv(os.path.join(outdir, "annotation_removed.csv"), index=False)
    kept = ad[(ad.obs["msp_ann_action"] == "keep").values].copy()
    log.info(
        f"== removed {len(archive)} cells (agent-marked clusters: "
        f"{int(archive['annotate_remove'].sum())}); annotated.h5ad keeps {kept.n_obs}/{ad.n_obs}",
    )
    _plot(ad, kept, os.path.join(outdir, "figures"))
    tmp = os.path.join(outdir, "annotated.tmp.h5ad")
    kept.write_h5ad(tmp)
    os.replace(tmp, os.path.join(outdir, "annotated.h5ad"))
    complete_step(outdir, "annotate")
    log.info(f"== report refreshed: {generate_report(outdir)}")
    return proposal


def main(argv=None):
    configure()
    parser = argparse.ArgumentParser(prog="msp.annotate", description=__doc__)
    parser.add_argument("outdir", help="msp integration output directory (after msp.inspect)")
    parser.add_argument("--species", default=None, help="defaults to uns['msp']['species']")
    parser.add_argument("--language", default="English")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-turns", type=int, default=200)
    args = parser.parse_args(argv)

    proposal = annotate_clusters(
        args.outdir,
        species=args.species,
        language=args.language,
        model=args.model,
        effort=args.effort,
        max_turns=args.max_turns,
    )
    for e in proposal["clusters"]:
        tail = f" merge→{e['merge_target']}" if e["merge_target"] else ""
        print(
            f"cluster {e['cluster_id']}: {e['coarse_label']} / {e['fine_label']} "
            f"[{e['action']}, {e['confidence']}]{tail}"
        )
    if proposal["merged_groups"]:
        print("merged groups: " + ", ".join(proposal["merged_groups"]))


if __name__ == "__main__":
    main()
