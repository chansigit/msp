"""Concat per-sample osp outputs and integrate with harmony.

Pipeline (conventions mirror osp where they apply): raw counts →
normalize_total(1e4) → log1p → HVG per batch (flavor="seurat",
batch_key) → scale(max 10) on HVG → PCA (arpack, seed 0) → harmony on the
batch key → neighbors on X_pca_harmony → leiden at several resolutions →
UMAP → fragment/QC evidence → PAGA and global/local DEG. No cells are removed;
msp proposes candidates for later steps.

Inherited per-sample columns (QC metrics, _ann_*, _qc_action, doublet
calls) ride along in obs. Sample-local leiden labels are prefixed with the
sample value ("H12inner:3") so they stay meaningful after the merge; the
integrated clusterings get their own msp_leiden_r* keys.
"""

from __future__ import annotations

import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

from .plots import UMAP_DPI, save_single_umap, slug
from .steps import begin_step, complete_step


def load_and_merge(inputs, batch_col, counts_layer="counts"):
    """Read per-sample clustered.h5ad files and concat them.

    Hard checks: identical var axis (they all come from one organized
    h5ad), exactly one batch value per file, globally unique barcodes —
    any violation is an error, never silently papered over.
    """
    import anndata as ad

    adatas, var_ref = [], None
    for path in inputs:
        a = ad.read_h5ad(path)
        if var_ref is None:
            var_ref = a.var_names
        elif not a.var_names.equals(var_ref):
            raise ValueError(f"{path}: var axis differs from the first input — same-unit samples must share genes")
        vals = a.obs[batch_col].astype(str).unique()
        if len(vals) != 1:
            raise ValueError(f"{path}: expected one {batch_col!r} value, found {list(vals)}")
        sample = vals[0]
        for c in a.obs.columns:
            if c.startswith("leiden_"):
                a.obs[c] = (sample + ":" + a.obs[c].astype(str)).astype("category")
        if counts_layer not in a.layers:
            raise ValueError(f"{path}: missing layers[{counts_layer!r}]")
        counts = a.layers[counts_layer]
        a.X = counts.copy()
        # per-sample embeddings/uns are meaningless after the merge; raw
        # counts is all downstream needs
        a.obsm.clear()
        a.uns.clear()
        for layer in [k for k in a.layers if k != counts_layer]:
            del a.layers[layer]  # deleting keys never touches X, unlike reassigning .layers
        adatas.append(a)

    # Gene axes were checked above, so an outer join only widens obs metadata.
    # Missing QC/labels remain missing, never invented negative calls or zeros.
    merged = ad.concat(adatas, join="outer", merge="same")
    for col in merged.obs:
        values = merged.obs[col]
        if values.dtype == object and values.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
            # Nullable booleans stored as object cannot be serialized by H5AD.
            merged.obs[col] = pd.Categorical(values, categories=[False, True])
    if merged.obs_names.duplicated().any():
        n = int(merged.obs_names.duplicated().sum())
        raise ValueError(f"{n} duplicated barcodes across samples — inputs must be disjoint cells")
    total = sum(a.n_obs for a in adatas)
    if merged.n_obs != total:
        raise ValueError(f"concat lost cells: {merged.n_obs} != {total}")
    return merged


# QC metrics shown on the integrated UMAP. pct_counts_mt gets a FIXED color
# ceiling: an autoscaled colorbar once made a median-2.4%-mt dataset look
# "high mt" (the Liu round-4 illusion) — never let the scale float.
QC_UMAP_METRICS = (
    ("pct_counts_mt", 20),
    ("n_genes_by_counts", None),
    ("total_counts", None),
    ("doublet_score", None),
    ("decontX_contamination", 1.0),
    ("dissociation_score", None),
)

QC_ACTION_PALETTE = {"keep": "#d3d3d3", "flag": "#ff8c00", "drop": "#d62728"}


# thresholds for minor-sibling QC testing (candidate detection, not a final
# verdict — msp.inspect judges what actually matters). "n_hits"/
# "recommend_removal" here are deliberately not "flag" — osp already uses
# keep/flag/drop for a different, per-cell concept and reusing the word
# would be confusing side by side.
MIN_N_FOR_TEST = 5  # below this, Mann-Whitney has no real power — mark insufficient_data
BIG_SIBLING_FRAC = 0.25  # sibling >= this fraction of its own parent's core is skipped, not a "minor" fragment
BIG_SIBLING_N = 800  # sibling >= this many cells (absolute) is skipped too, regardless of frac_of_core
DOUBLET_MEDIAN_THRESH = 0.2  # scrublet score, 0-1 scale
MT_MEDIAN_THRESH = 20.0  # pct_counts_mt is already on a 0-100 scale
DROP_PCT_THRESH = 50.0  # % of a sibling's cells already _qc_action=drop upstream (osp)


def _mwu_greater(sib_vals, core_vals):
    """One-sided Mann-Whitney U: is `sib_vals` stochastically greater than
    `core_vals`? Rank-based — doesn't assume normality, works for the small,
    skewed samples minor siblings usually are."""
    from scipy.stats import mannwhitneyu

    # Partial sample metadata must not turn an otherwise valid comparison into
    # a NaN p-value. Require enough measured values on both sides of the test.
    sib_vals = np.asarray(sib_vals, dtype=float)
    core_vals = np.asarray(core_vals, dtype=float)
    sib_vals = sib_vals[np.isfinite(sib_vals)]
    core_vals = core_vals[np.isfinite(core_vals)]
    if min(len(sib_vals), len(core_vals)) < MIN_N_FOR_TEST:
        return None
    _, p = mannwhitneyu(sib_vals, core_vals, alternative="greater")
    return bool(p < 0.05)


def _minor_sibling_qc(ad, res, outdir):
    """Test minor-sibling fragments (standissect subclusters c{parent}_i,
    i>0) for a QC profile worse than the pooled parent-core cells, plus one
    hard upstream-drop rule — candidates for msp.inspect to verify, but ANY
    one of these criteria hitting is enough to mark the whole fragment
    recommend_removal (this function proposes, it never removes anything
    itself).

    Parent cores (rank 0) are never tested. Siblings holding >=25% of their
    own parent core's cell count, OR >=800 cells outright, are "big"
    fragments, not minor — skipped entirely (no rows below apply to them).

    For every remaining (non-big) sibling: if >50% of its cells already
    carry _qc_action=="drop" from the upstream per-sample annotation, that
    alone is enough — a fragment that's already majority-drop upstream just
    inherits that verdict as a group. This is computed even when the
    sibling is too small for the statistical tests below.

    Siblings with >=5 cells are additionally tested one-sided (sibling >
    pooled cores) via Mann-Whitney U on: decontX_contamination,
    dissociation_score, doublet_score, pct_counts_mt. doublet/mt tests
    additionally require the sibling's own median to clear an absolute
    floor (0.2 and 20% respectively). No multiple-testing correction — this
    half is candidate detection, downstream inspection re-verifies. A
    sibling is recommend_removal if the upstream-drop rule fires OR any one
    of the four stats tests comes back significant."""
    frag = res.fragments
    core_n = frag.loc[frag["rank"] == 0].set_index("parent")["n_cells"]
    core_subclusters = set(frag.loc[frag["rank"] == 0, "subcluster"])
    core_mask = ad.obs["standissect_product"].astype(str).isin(core_subclusters)

    metric_tests = [
        ("decontX_contamination", "decontX", None),
        ("dissociation_score", "dissociation", None),
        ("doublet_score", "doublet", DOUBLET_MEDIAN_THRESH),
        ("pct_counts_mt", "mt", MT_MEDIAN_THRESH),
    ]
    metric_tests = [(col, name, thr) for col, name, thr in metric_tests if col in ad.obs]

    rows = []
    for _, frow in frag[frag["rank"] > 0].iterrows():
        sub, parent, n_cells = frow["subcluster"], frow["parent"], int(frow["n_cells"])
        cn = int(core_n.get(parent, 0))
        row = {
            "subcluster": sub,
            "parent": parent,
            "n_cells": n_cells,
            "core_n_cells": cn,
            "frac_of_core": round(n_cells / cn, 3) if cn else None,
        }
        if (cn and n_cells >= BIG_SIBLING_FRAC * cn) or n_cells >= BIG_SIBLING_N:
            row["status"] = "big_sibling_skip"
            rows.append(row)
            continue

        sib_mask = ad.obs["standissect_product"].astype(str) == sub
        drop_hit = False
        if "_qc_action" in ad.obs:
            pct_drop = float((ad.obs.loc[sib_mask, "_qc_action"] == "drop").mean() * 100)
            row["pct_drop_upstream"] = round(pct_drop, 2)
            drop_hit = pct_drop > DROP_PCT_THRESH

        stat_hit = False
        if n_cells < MIN_N_FOR_TEST:
            row["status"] = "insufficient_data"
        else:
            row["status"] = "tested"
            n_hits = 0
            for col, name, thr in metric_tests:
                sib_vals, core_vals = ad.obs.loc[sib_mask, col], ad.obs.loc[core_mask, col]
                sig = _mwu_greater(sib_vals, core_vals)
                if thr is not None and sig:
                    sig = bool(sib_vals.median() > thr)
                row[f"{name}_median"] = round(float(sib_vals.median()), 4)
                row[f"{name}_significant"] = sig
                n_hits += bool(sig)
            row["n_hits"] = n_hits
            stat_hit = n_hits > 0

        row["recommend_removal"] = bool(drop_hit or stat_hit)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "minor_sibling_qc.csv"), index=False)
    return df


def _qc_outputs(ad, batch_col, primary_key, outdir, figdir, leiden_keys=(), resolutions=()):
    """QC evidence for the integrated space: metric UMAPs, the inherited
    keep/flag/drop overlay, and per-sample / per-cluster QC tables — the
    raw material for the later per-cluster inspection step."""
    # one metric per file — never a multi-panel figure (osp's rule: a human
    # or an agent reads one signal per image)
    metrics = [(m, vmax) for m, vmax in QC_UMAP_METRICS if m in ad.obs]
    for m, vmax in metrics:
        save_single_umap(ad, m, os.path.join(figdir, f"qc_umap_{slug(m)}.png"), vmax=vmax)

    # grouped by the standissect clusters (parent leiden x umap fragment),
    # not the primary leiden — the whole point of that finer split is to
    # see whether a minor sibling's QC profile diverges from its main core
    violin_key = "standissect_product" if "standissect_product" in ad.obs else primary_key
    log_metrics = {"n_genes_by_counts", "total_counts"}  # heavy-tailed counts
    for m, _ in metrics:
        sc.pl.violin(ad, m, groupby=violin_key, stripplot=False, rotation=90, show=False, log=m in log_metrics)
        plt.savefig(os.path.join(figdir, f"qc_violin_{slug(m)}.png"), dpi=UMAP_DPI, bbox_inches="tight")
        plt.close("all")

    if "_qc_action" in ad.obs:
        order = [c for c in ("keep", "flag", "drop") if c in set(ad.obs["_qc_action"].astype(str))]
        ad.obs["_qc_action"] = pd.Categorical(ad.obs["_qc_action"].astype(str), categories=order)
        # inherited from OSP just like _ann_coarse — filename (no qc_
        # prefix) groups it with the OSP-inherited panels, not the
        # integrated-space QC metrics
        save_single_umap(
            ad,
            "_qc_action",
            os.path.join(figdir, "umap__qc_action.png"),
            palette=[QC_ACTION_PALETTE[c] for c in order],
            legend_loc="best",
            na_color="#808080",
        )  # distinct from keep, including in Scanpy's color categories

    def _agg(groupby):
        g = ad.obs.groupby(groupby, observed=True)
        out = pd.DataFrame({"n_cells": g.size()})
        if "_qc_action" in ad.obs:
            out["pct_flag"] = (g["_qc_action"].apply(lambda s: (s.dropna() == "flag").mean()) * 100).round(2)
            out["pct_drop"] = (g["_qc_action"].apply(lambda s: (s.dropna() == "drop").mean()) * 100).round(2)
        for m in ("pct_counts_mt", "doublet_score", "decontX_contamination", "n_genes_by_counts"):
            if m in ad.obs:
                out[f"median_{m}"] = g[m].median().round(4)
        return out

    _agg(batch_col).to_csv(os.path.join(outdir, "per_sample_qc.csv"))
    # standissect clusters, not the primary leiden — same rationale as the
    # violins: a minor sibling's QC/composition can diverge from its core
    cl = _agg(violin_key)
    # how many samples contribute to each cluster — a 1-sample cluster in an
    # integrated space is itself a QC signal
    cl["n_samples"] = ad.obs.groupby(violin_key, observed=True)[batch_col].nunique()
    cl.to_csv(os.path.join(outdir, f"cluster_qc_{violin_key}.csv"))

    # same table at the primary leiden resolutions (r1.0 / r2.0) — coarser
    # than standissect fragments, but this is what Cluster Annotations (DEG,
    # PAGA, stress flags) actually keys off, so its QC belongs alongside it
    res_to_key = dict(zip(resolutions, leiden_keys, strict=True))
    for r in (1.0, 2.0):
        key = res_to_key.get(r)
        if key is None or key == violin_key:
            continue
        lcl = _agg(key)
        lcl["n_samples"] = ad.obs.groupby(key, observed=True)[batch_col].nunique()
        lcl.to_csv(os.path.join(outdir, f"cluster_qc_{key}.csv"))


# Seurat's classic DoHeatmap palette (colorRampPalette(c("purple","black","yellow")))
SEURAT_HEATMAP_CMAP = LinearSegmentedColormap.from_list("seurat_purple_black_yellow", ["purple", "black", "yellow"])


def _fractal_marker_heatmap(ad, res, outdir, figdir, top_n=10):
    """Explain fractal identity from the transcriptome, not just QC stats:
    DE each parent's core cells one-vs-rest against every OTHER parent's
    core cells (core-only, so a minor sibling's cells never leak into
    either side of the comparison) -> per-parent top markers (logFC>0,
    padj<0.05, ribosomal genes excluded) -> dot plot across every
    standissect cluster (cores AND fractals): dot size = fraction of a
    cluster's cells expressing the gene, dot color = row-wise (per-gene)
    z-score of average log1p expression, Seurat-style purple-black-yellow.

    Columns are clustered (optimal leaf ordering) purely to order them — no
    dendrogram is drawn. Rows are NOT clustered: genes stay grouped by the
    parent they're a marker for, with a colored strip per gene showing that
    parent, since that grouping is more legible than a gene-gene dendrogram
    here. Column labels: parent-core clusters (c{parent}_0) bold; clusters
    minor_sibling_qc.csv marked recommend_removal in red."""
    frag = res.fragments
    core_rows = frag.loc[frag["rank"] == 0]
    parent_of_core = dict(zip(core_rows["subcluster"], core_rows["parent"], strict=True))
    core_mask = ad.obs["standissect_product"].astype(str).isin(parent_of_core)
    core_ad = ad[core_mask].copy()
    core_ad.obs["parent"] = (
        core_ad.obs["standissect_product"].astype(str).map(parent_of_core).astype(str).astype("category")
    )

    if core_ad.obs["parent"].nunique() < 2:
        # one-vs-other-parent DE needs >=2 parents; a lineage that standissect
        # never split beyond its root has only one, and rank_genes_groups_df
        # drops the "group" column for a single-group input (KeyError downstream)
        print("== only one parent-core cluster — skipping parent-core DEG/heatmap", flush=True)
        return

    print("== parent-core DEG (one vs other parent cores)", flush=True)
    sc.tl.rank_genes_groups(core_ad, "parent", method="wilcoxon", use_raw=True, pts=True)
    de_df = sc.get.rank_genes_groups_df(core_ad, group=None)
    de_df.to_csv(os.path.join(outdir, "de_parent_core_vs_core.csv"), index=False)

    ribo = set(ad.var_names[ad.var["ribo"]]) if "ribo" in ad.var else set()
    marker_rows, gene_parent = [], {}
    for parent, g in de_df.groupby("group", observed=True):
        g = g[(g["pvals_adj"] < 0.05) & (g["logfoldchanges"] > 0) & (~g["names"].isin(ribo))]
        g = g.sort_values("logfoldchanges", ascending=False).head(top_n)
        for rank, row in enumerate(g.itertuples(), start=1):
            gene_parent.setdefault(row.names, parent)  # first parent wins if a gene repeats
            marker_rows.append(
                {
                    "parent": parent,
                    "gene": row.names,
                    "rank": rank,
                    "logfoldchange": round(row.logfoldchanges, 3),
                    "pvals_adj": row.pvals_adj,
                }
            )
    pd.DataFrame(marker_rows).to_csv(os.path.join(outdir, "fractal_markers.csv"), index=False)
    # row order follows parent order (numeric, not the groupby's alpha-sort), then rank within
    # parent; de-dup a gene shared across parents by keeping its first (lowest-parent) occurrence
    marker_rows_sorted = sorted(marker_rows, key=lambda r: (int(r["parent"]), r["rank"]))
    markers = list(dict.fromkeys(r["gene"] for r in marker_rows_sorted))
    if not markers:
        print("== no marker genes passed the filters — skipping heatmap", flush=True)
        return

    print(
        f"== fractal marker dot plot ({len(markers)} genes x {ad.obs['standissect_product'].nunique()} clusters)",
        flush=True,
    )
    sub = ad[:, markers]
    X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    expr = pd.DataFrame(X, index=ad.obs_names, columns=markers)
    expr["cluster"] = ad.obs["standissect_product"].astype(str).values
    grp = expr.groupby("cluster")
    avg = grp[markers].mean().T  # genes x clusters, average log1p expression
    frac = grp[markers].apply(lambda d: (d > 0).mean()).T  # genes x clusters, fraction expressing
    avg.to_csv(os.path.join(outdir, "fractal_marker_avg_expr.csv"))
    frac.to_csv(os.path.join(outdir, "fractal_marker_frac_expr.csv"))

    z = avg.sub(avg.mean(axis=1), axis=0).div(avg.std(axis=1).replace(0, 1), axis=0).fillna(0)
    z = z.loc[markers]  # parent-then-rank order (see marker_rows_sorted above)
    frac = frac.loc[markers]

    # column order from hierarchical clustering (optimal leaf ordering) — used
    # only to order columns, no dendrogram is drawn
    col_linkage = linkage(z.values.T, method="average", metric="euclidean", optimal_ordering=True)
    leaf_order = dendrogram(col_linkage, no_plot=True)["leaves"]
    cluster_order = [z.columns[i] for i in leaf_order]
    z, frac = z[cluster_order], frac[cluster_order]

    removal_path = os.path.join(outdir, "minor_sibling_qc.csv")
    removal_set = set()
    if os.path.exists(removal_path):
        with open(removal_path) as f:
            removal_set = {r["subcluster"] for r in csv.DictReader(f) if r.get("recommend_removal") == "True"}
    core_pattern = re.compile(r"^c\d+_0$")

    parents = sorted(set(gene_parent.values()), key=int)
    parent_palette = dict(zip(parents, sns.color_palette("tab20", n_colors=max(len(parents), 1)), strict=False))

    n_genes, n_clusters = z.shape

    # Every margin below is a fixed inch amount, not a figure-fraction — so
    # whitespace stays constant (tight) regardless of n_genes/n_clusters,
    # instead of growing with figure size the way fraction-based offsets did.
    left_margin, strip_w, col_gap, right_margin = 1.3, 0.18, 0.06, 0.15
    top_margin, plot_col_w = 0.1, n_clusters * 0.32
    xtick_h, legend_gap, legend_h, bottom_margin = 0.9, 0.12, 0.8, 0.05
    plot_row_h = max(2.0, n_genes * 0.16)

    fig_w = left_margin + strip_w + col_gap + plot_col_w + right_margin
    fig_h = top_margin + plot_row_h + xtick_h + legend_gap + legend_h + bottom_margin
    fig = plt.figure(figsize=(fig_w, fig_h))

    plot_bottom = (xtick_h + legend_gap + legend_h + bottom_margin) / fig_h
    plot_h = plot_row_h / fig_h
    ax_strip = fig.add_axes((left_margin / fig_w, plot_bottom, strip_w / fig_w, plot_h))
    ax = fig.add_axes(
        ((left_margin + strip_w + col_gap) / fig_w, plot_bottom, plot_col_w / fig_w, plot_h), sharey=ax_strip
    )

    strip = np.array([parent_palette[gene_parent[g]] for g in markers])[:, None, :]
    ax_strip.imshow(strip, aspect="auto")
    ax_strip.set_xticks([])
    ax_strip.set_yticks(range(n_genes))
    ax_strip.set_yticklabels(markers, fontsize=8)
    ax_strip.set_ylabel("marker gene")

    xs, ys, sizes, colors = [], [], [], []
    for xi, cl in enumerate(cluster_order):
        for yi, gene in enumerate(markers):
            xs.append(xi)
            ys.append(yi)
            sizes.append(frac.loc[gene, cl])
            colors.append(z.loc[gene, cl])
    max_dot_area = 200.0
    sca = ax.scatter(
        xs,
        ys,
        s=np.asarray(sizes) * max_dot_area + 2,
        c=colors,
        cmap=SEURAT_HEATMAP_CMAP,
        vmin=-2.5,
        vmax=2.5,
        edgecolor="none",
    )
    for x in range(1, n_clusters):  # thin separators between columns only
        ax.axvline(x - 0.5, color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_xlim(-0.5, n_clusters - 0.5)
    ax.set_ylim(n_genes - 0.5, -0.5)
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels(cluster_order, rotation=90)
    # NOT ax.set_yticks([]) — sharey means the y Locator is shared with
    # ax_strip; clearing ticks here would silently wipe its gene labels too
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("standissect cluster (core + fractals)")
    for label in ax.get_xticklabels():
        name = label.get_text()
        if core_pattern.match(name):
            label.set_fontweight("bold")
        if name in removal_set:
            label.set_color("#c0392b")

    # colorbar (dot color = z-score) and size legend, packed into the fixed
    # legend_h/bottom_margin band reserved above — no negative coordinates,
    # so there's no bbox_inches="tight" guesswork; every element's text stays
    # within its own axes' [0,1] range so nothing depends on overflow room
    cax = fig.add_axes(
        ((left_margin + strip_w + col_gap) / fig_w, (bottom_margin + 0.4) / fig_h, 1.6 / fig_w, 0.15 / fig_h)
    )
    fig.colorbar(sca, cax=cax, orientation="horizontal", label="z-score")

    lax = fig.add_axes(
        ((left_margin + strip_w + col_gap + 2.1) / fig_w, bottom_margin / fig_h, 2.2 / fig_w, legend_h / fig_h)
    )
    lax.set_xlim(0, 4)
    lax.set_ylim(0, 1)
    lax.axis("off")
    for i, frac_ref in enumerate((0.25, 0.5, 0.75, 1.0)):
        lax.scatter([i], [0.55], s=frac_ref * max_dot_area + 2, c="grey")
        lax.text(i, 0.15, f"{frac_ref:g}", ha="center", va="top", fontsize=7)
    lax.text(1.5, 0.95, "fraction expressing", ha="center", va="top", fontsize=8)

    fig.savefig(os.path.join(figdir, "fractal_marker_heatmap.png"), dpi=UMAP_DPI)
    plt.close("all")


# Conservative "dissociation stress" core panel (not osp's full ~130-gene
# DISSOCIATION_GENES_HS — that one includes ECM/lineage genes like DCN,
# LMNA, SERPINE1 that are real cell-identity markers in plenty of tissues).
# Two independent, mechanistically distinct, well-established acute-
# dissociation-stress axes: heat-shock/chaperone response, and AP-1/
# immediate-early transcription — both firing together is far more specific
# than one broad gene list. Human symbols; matched case-insensitively
# (dataset gene names uppercased before lookup) so mouse data works too.
STRESS_GENES_CORE = [
    "HSPA1A",
    "HSPA1B",
    "HSPA8",
    "HSPB1",
    "HSP90AA1",
    "HSP90AB1",
    "HSPH1",
    "HSPE1",
    "DNAJA1",
    "DNAJB1",
    "DNAJB4",
    "FOS",
    "FOSB",
    "JUN",
    "JUNB",
    "JUND",
    "EGR1",
    "EGR2",
    "ATF3",
    "NR4A1",
    "PPP1R15A",
    "ZFP36",
    "IER2",
    "IER3",
    "DUSP1",
]
STRESS_GENE_SET = set(STRESS_GENES_CORE)
STRESS_HIT_THRESHOLD = 3  # a cluster is "stress" if MORE than this many top genes hit
STRESS_CHECK_TOP_N = 10  # matches what the report displays, independent of top_n_de
MIN_DE_GROUP_SIZE = 10  # clusters smaller than this are excluded from DE comparisons
# (both global one-vs-rest and local vs-PAGA-neighbors views)


def _is_stress_gene(symbol) -> bool:
    su = str(symbol).upper()
    return su in STRESS_GENE_SET or su.startswith("MT-")


def _stress_hits(names) -> list[str]:
    return [n for n in names if _is_stress_gene(n)]


def _build_removal_mask(ad, msq_df, cell_outliers_df, outdir):
    """Union of every recommend_removal source proposed so far: whole
    standissect fragments (minor_sibling_qc), individual cells (cell-level
    doublet/ambient outliers), AND cells osp itself already proposed
    dropping per-sample (obs["_qc_action"]=="drop", inherited from
    persample annotation — cross-sample clustering is exactly the evidence
    this step should weigh, not evidence to discard). This is the set
    Cluster Annotations excludes and the pre-annotation UMAP visualizes.
    Persisted to preannotation_removal.csv (cell, recommend_removal) so any
    later live tool (msp.inspect's check_deg etc.) can apply the identical
    exclusion without recomputing/duplicating this logic. Boolean numpy
    array, aligned to ad.obs_names order."""
    remove_set = set()
    if msq_df is not None and "recommend_removal" in msq_df.columns:
        remove_set = set(msq_df.loc[msq_df["recommend_removal"] == True, "subcluster"])  # noqa: E712
    from_fragments = (
        ad.obs["standissect_product"].astype(str).isin(remove_set).values
        if "standissect_product" in ad.obs
        else np.zeros(ad.n_obs, dtype=bool)
    )
    if cell_outliers_df is not None and "recommend_removal" in cell_outliers_df.columns:
        from_cells = cell_outliers_df["recommend_removal"].reindex(ad.obs_names).fillna(False).values
    else:
        from_cells = np.zeros(ad.n_obs, dtype=bool)
    from_osp_drop = (
        ad.obs["_qc_action"].astype(str).values == "drop" if "_qc_action" in ad.obs else np.zeros(ad.n_obs, dtype=bool)
    )
    mask = from_fragments | from_cells | from_osp_drop
    pd.DataFrame({"cell": ad.obs_names, "recommend_removal": mask}).to_csv(
        os.path.join(outdir, "preannotation_removal.csv"), index=False
    )
    return mask


def _preannotation_removal_umap(ad, remove_mask, figdir):
    """UMAP of every cell proposed for removal so far, before Cluster
    Annotations excludes them — one before/after picture of what 'propose,
    never remove' actually selected. Computation-only: the column lives on
    ad.obs just long enough to plot, never persisted to the written h5ad."""
    ad.obs["_preannotation_removal"] = pd.Categorical(
        np.where(remove_mask, "remove", "keep"), categories=["keep", "remove"]
    )
    save_single_umap(
        ad,
        "_preannotation_removal",
        os.path.join(figdir, "umap_preannotation_removal.png"),
        palette=["#cccccc", "#c0392b"],
        legend_loc="best",
    )
    del ad.obs["_preannotation_removal"]


def _cluster_annotations(ad, remove_mask, leiden_keys, resolutions, outdir, top_n_de=50):
    """Cluster Annotations: for the r1.0 and r2.0 leiden clusterings, two DE
    views per cluster — global (one-vs-rest, as before) and local (one-vs-
    its-top-3-PAGA-neighbors, pooled). remove_mask (see
    _preannotation_removal, the union of minor_sibling_qc fragments and
    cell-level doublet/ambient outliers) is excluded from BOTH views (and
    from the PAGA graph itself) — this is a computation-only exclusion, msp
    never drops cells from ad or the written h5ad.

    Each (key, cluster) is also checked for a dissociation-stress signature
    (STRESS_GENES_CORE / mitochondrial genes) among its top
    STRESS_CHECK_TOP_N genes, in each view separately — but if EITHER view
    hits the threshold, the whole (key, cluster) is recommend_removal,
    written to stress_clusters.csv. Same rule as everywhere else in msp:
    propose, never remove cells directly.

    The wilcoxon runs (2 global + one local per cluster per key, all
    independent of each other) go through a thread pool sized to the CPUs
    this process may actually use (msp.resources): on a 59k-cell object the
    stage dropped from ~5 min to ~2.3 min. Outputs are assembled in the
    same key/cluster order as the old sequential loop, so the CSVs are
    byte-identical; each global run gets its own uns key so the two keys
    don't clobber each other's rank_genes_groups slot."""
    from concurrent.futures import ThreadPoolExecutor

    from .resources import available_cpus

    keep_mask = ~remove_mask
    n_excluded = int((~keep_mask).sum())
    ad_excl = ad[keep_mask].copy()
    print(
        f"== cluster annotations: excluding {n_excluded} recommend_removal cells ({ad_excl.n_obs}/{ad.n_obs} remain)",
        flush=True,
    )

    res_to_key = dict(zip(resolutions, leiden_keys, strict=True))
    target = [(r, res_to_key[r]) for r in (1.0, 2.0) if r in res_to_key]
    if not target:
        print("== cluster annotations: neither r1.0 nor r2.0 in resolutions — skipping", flush=True)
        return

    print("== cluster annotations: neighbors on the excluded subset", flush=True)
    sc.pp.neighbors(ad_excl, use_rep="X_pca_harmony")

    # phase 1 (sequential, cheap): PAGA + neighbour tables + the task list
    plan = []  # per key: dict(key, cats, valid_groups, top3)
    for r, key in target:
        print(f"== cluster annotations on {key} (r={r})", flush=True)
        sc.tl.paga(ad_excl, groups=key)
        conn = ad_excl.uns["paga"]["connectivities"].toarray()
        cats = list(ad_excl.obs[key].cat.categories)

        top3, neighbor_rows = {}, []
        for i, c in enumerate(cats):
            order = np.argsort(conn[i])[::-1]
            picked = [cats[j] for j in order if j != i and conn[i, j] > 0][:3]
            top3[c] = picked
            for rank, nb in enumerate(picked, start=1):
                neighbor_rows.append(
                    {
                        "cluster": c,
                        "neighbor": nb,
                        "rank": rank,
                        "connectivity": round(float(conn[i, cats.index(nb)]), 4),
                    }
                )
        # Keep the existing columns even when this graph has no positive edges.
        pd.DataFrame(neighbor_rows, columns=["cluster", "neighbor", "rank", "connectivity"]).to_csv(
            os.path.join(outdir, f"paga_neighbors_{key}.csv"), index=False
        )

        # wilcoxon needs >=2 cells per group to run at all, but a cluster that
        # tiny (can survive this far when it's PAGA-connected enough not to get
        # merged) doesn't give trustworthy DE either; require MIN_DE_GROUP_SIZE
        sizes = ad_excl.obs[key].value_counts()
        valid_groups = [c for c in cats if sizes.get(c, 0) >= MIN_DE_GROUP_SIZE]
        if len(valid_groups) < len(cats):
            skipped = [c for c in cats if c not in valid_groups]
            print(
                f"== cluster annotations: skipping global DE for undersized "
                f"(<{MIN_DE_GROUP_SIZE} cells) cluster(s) {skipped} in {key}",
                flush=True,
            )
        if not valid_groups:
            continue
        plan.append({"key": key, "cats": cats, "valid": valid_groups, "top3": top3, "sizes": sizes})

    # phase 2 (parallel): the DE runs. Global writes into ad_excl.uns under a
    # per-key slot; local runs work on their own copies.
    def run_global(item):
        key = item["key"]
        slot = f"_rgg_{key}"
        sc.tl.rank_genes_groups(
            ad_excl, key, groups=item["valid"], method="wilcoxon", use_raw=True, pts=True, key_added=slot
        )
        gdf = sc.get.rank_genes_groups_df(ad_excl, group=None, key=slot)
        del ad_excl.uns[slot]
        # Scanpy omits group when only one group qualifies for testing.
        if "group" not in gdf and len(item["valid"]) == 1:
            gdf.insert(0, "group", item["valid"][0])
        return gdf.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"})

    def run_local(item, c):
        key = item["key"]
        neighbors = item["top3"].get(c, [])
        if not neighbors:
            return None
        sub = ad_excl[ad_excl.obs[key].isin([c, *neighbors])].copy()
        if int((sub.obs[key] == c).sum()) < MIN_DE_GROUP_SIZE:
            return None
        sc.tl.rank_genes_groups(sub, key, groups=[c], reference="rest", method="wilcoxon", use_raw=True, pts=True)
        ldf = sc.get.rank_genes_groups_df(sub, group=c)
        ldf = ldf.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"})
        # rank_genes_groups_df drops the "group" column when `group` is a scalar
        # (only keeps it for group=None/list) — put it back for schema parity
        # with the global-view CSV and so the report can key off it
        ldf.insert(0, "group", c)
        ldf["neighbors"] = "|".join(neighbors)
        return ldf

    n_workers = max(1, min(available_cpus(), 8))
    print(f"== cluster annotations: DE on {n_workers} thread(s)", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for item in plan:
            futures.append((item, "global", None, pool.submit(run_global, item)))
            for c in item["cats"]:
                futures.append((item, "local", c, pool.submit(run_local, item, c)))
        results = [(item, view, c, f.result()) for item, view, c, f in futures]

    # phase 3 (sequential): write tables + stress rows in the old loop's order
    stress_rows = []
    for item in plan:
        key = item["key"]
        gdf = next(res for it, view, c, res in results if it is item and view == "global")
        gdf.groupby("group", observed=True).head(top_n_de).reset_index(drop=True).to_csv(
            os.path.join(outdir, f"deg_global_{key}.csv"), index=False
        )
        for c, sub in gdf.groupby("group", observed=True).head(STRESS_CHECK_TOP_N).groupby("group", observed=True):
            hits = _stress_hits(sub["names"].tolist())
            stress_rows.append(
                {
                    "key": key,
                    "cluster": c,
                    "view": "global",
                    "n_hits": len(hits),
                    "hit_genes": "|".join(hits),
                    "stress": len(hits) > STRESS_HIT_THRESHOLD,
                }
            )
        local_rows = []
        for it, view, c, ldf in results:
            if it is not item or view != "local" or ldf is None:
                continue
            hits = _stress_hits(ldf.head(STRESS_CHECK_TOP_N)["names"].tolist())
            stress_rows.append(
                {
                    "key": key,
                    "cluster": c,
                    "view": "local",
                    "n_hits": len(hits),
                    "hit_genes": "|".join(hits),
                    "stress": len(hits) > STRESS_HIT_THRESHOLD,
                }
            )
            local_rows.append(ldf.head(top_n_de))
        if local_rows:
            pd.concat(local_rows, ignore_index=True).to_csv(os.path.join(outdir, f"deg_local_{key}.csv"), index=False)

    if stress_rows:
        stress_df = pd.DataFrame(stress_rows)
        # a cluster is recommend_removal overall if EITHER its global or its
        # local view hit the stress threshold — not judged per-view
        overall = (
            stress_df.groupby(["key", "cluster"])["stress"]
            .any()
            .reset_index()
            .rename(columns={"stress": "recommend_removal"})
        )
        stress_df = stress_df.merge(overall, on=["key", "cluster"])
        stress_df.to_csv(os.path.join(outdir, "stress_clusters.csv"), index=False)
        n_removal = int(overall["recommend_removal"].sum())
        print(
            f"== cluster annotations: {n_removal}/{len(overall)} (key, cluster) pairs "
            f"recommend_removal (stress signature)",
            flush=True,
        )


CELL_OUTLIER_METRICS = ("doublet_score", "decontX_contamination")
CELL_OUTLIER_MAD_K = 3.0
CELL_OUTLIER_HARD_FLOOR = 0.5


def _cell_level_outliers(ad, leiden_keys, resolutions, outdir):
    """Per-cell doublet/ambient-RNA outlier flagging, at both the r1.0 and
    r2.0 leiden resolutions. A cell is a metric outlier in a cluster if it
    clears BOTH gates together: cluster median + 3*MAD, AND an absolute
    floor of 0.5 (the MAD rule alone is too permissive in near-clean
    clusters where MAD itself is tiny). recommend_removal is the OR across
    both metrics and both resolutions — propose-only, this CSV is the only
    place the flag lives, never written back to ad.obs/h5ad."""
    metrics = [m for m in CELL_OUTLIER_METRICS if m in ad.obs]
    if not metrics:
        return None
    res_to_key = dict(zip(resolutions, leiden_keys, strict=True))
    targets = [(r, res_to_key[r]) for r in (1.0, 2.0) if r in res_to_key]
    if not targets:
        return None

    df = pd.DataFrame(index=ad.obs_names)
    for m in metrics:
        df[m] = ad.obs[m]

    flag_cols = []
    for _r, key in targets:
        df[key] = ad.obs[key].values
        g = ad.obs.groupby(key, observed=True)
        for m in metrics:
            med = g[m].transform("median")
            mad = g[m].transform(lambda s: (s - s.median()).abs().median())
            outlier = ad.obs[m] > (med + CELL_OUTLIER_MAD_K * mad)
            hard = ad.obs[m] > CELL_OUTLIER_HARD_FLOOR
            col = f"{key}_{m}_outlier"
            df[col] = (outlier & hard).values
            flag_cols.append(col)

    df["recommend_removal"] = df[flag_cols].any(axis=1)
    df.index.name = "cell"
    df.reset_index().to_csv(os.path.join(outdir, "cell_outliers.csv"), index=False)

    summary_rows = []
    for _r, key in targets:
        g = df.groupby(key, observed=True)
        for cl, sub in g:
            row = {"key": key, "cluster": cl, "n_cells": len(sub)}
            for m in metrics:
                col = f"{key}_{m}_outlier"
                row[f"n_{m}_outlier"] = int(sub[col].sum())
                row[f"pct_{m}_outlier"] = round(100 * sub[col].mean(), 2)
            row["n_recommend_removal"] = int(sub["recommend_removal"].sum())
            row["pct_recommend_removal"] = round(100 * sub["recommend_removal"].mean(), 2)
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(outdir, "cell_outlier_summary.csv"), index=False)

    n_removal = int(df["recommend_removal"].sum())
    print(
        f"== cell-level outliers: {n_removal}/{len(df)} cells recommend_removal "
        f"(doublet/ambient, MAD+floor, r1.0 or r2.0)",
        flush=True,
    )
    return df


def _leiden_cluster_qc_violins(ad, leiden_keys, resolutions, figdir):
    """Per-cluster violins for the cell-level outlier metrics (doublet_score,
    decontX_contamination), one file per (metric, resolution): the visual
    counterpart to _cell_level_outliers. Each cluster's own outlier cutoff —
    max of (cluster median + 3*MAD) and the absolute floor 0.5, exactly the
    threshold a cell must clear to be flagged — is drawn as a red bar."""
    metrics = [m for m in CELL_OUTLIER_METRICS if m in ad.obs]
    if not metrics:
        return
    res_to_key = dict(zip(resolutions, leiden_keys, strict=True))
    targets = [(r, res_to_key[r]) for r in (1.0, 2.0) if r in res_to_key]
    for _r, key in targets:
        order = [str(c) for c in ad.obs[key].cat.categories]
        g = ad.obs.groupby(key, observed=True)
        for m in metrics:
            med = g[m].median()
            mad = g[m].apply(lambda s: (s - s.median()).abs().median())
            cutoff = pd.concat(
                [med + CELL_OUTLIER_MAD_K * mad, pd.Series(CELL_OUTLIER_HARD_FLOOR, index=med.index)], axis=1
            ).max(axis=1)

            fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.5), 4))
            sns.violinplot(
                data=ad.obs, x=key, y=m, order=order, cut=0, inner="quartile", color="#cfe3f7", linewidth=0.8, ax=ax
            )
            for xi, cl in enumerate(order):
                if cl in cutoff.index:
                    ax.hlines(cutoff.loc[cl], xi - 0.4, xi + 0.4, colors="#c0392b", linewidth=2, zorder=5)
            ax.set_title(f"{m} by {key} (red = per-cluster outlier cutoff)")
            ax.set_xlabel(key)
            ax.tick_params(axis="x", rotation=90)
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, f"leiden_qc_violin_{slug(m)}_{key}.png"), dpi=UMAP_DPI)
            plt.close(fig)


def run_multi_sample_pipeline(
    inputs,
    batch_col,
    outdir,
    species=None,
    resolutions=(0.3, 1.0, 2.0),
    n_top_genes=2000,
    n_pcs=50,
    n_neighbors=15,
    counts_layer="counts",
    top_n_de=50,
    harmony_kwargs=None,
):
    """Load osp per-sample outputs, merge, and run integrate_adata on the
    result — see there for the parameters. Returns (ad, summary)."""
    os.makedirs(outdir, exist_ok=True)
    ad = load_and_merge(inputs, batch_col, counts_layer=counts_layer)
    return integrate_adata(
        ad,
        batch_col,
        outdir,
        species=species,
        resolutions=resolutions,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        counts_layer=counts_layer,
        top_n_de=top_n_de,
        harmony_kwargs=harmony_kwargs,
        inputs=inputs,
    )


def integrate_adata(
    ad,
    batch_col,
    outdir,
    species=None,
    resolutions=(0.3, 1.0, 2.0),
    n_top_genes=2000,
    n_pcs=50,
    n_neighbors=15,
    counts_layer="counts",
    top_n_de=50,
    harmony_kwargs=None,
    inputs=(),
    meta_extra=None,
):
    """The integration core on an in-memory AnnData: X is reset from
    layers[counts_layer] (so any prior normalization/embedding on the object
    is discarded), then normalize → HVG per batch → PCA → harmony → neighbors
    → leiden at each resolution → UMAP → standissect-lite → QC/DEG artifacts
    → integrated.h5ad + figures in outdir. Used by run_multi_sample_pipeline
    on merged osp outputs and by zmip on one lineage subset of annotated.h5ad
    (same artifacts, same report).

    harmony_kwargs: passed straight to harmonypy.run_harmony on top of msp's
    fixed random_state=0/device — theta (diversity penalty, default 2 per
    covariate), lamb (ridge, default 1; -1 = auto-estimate), sigma (soft
    k-means width, 0.1), nclust (default min(round(N/30), 100)), tau,
    block_size, max_iter_harmony (10), max_iter_kmeans (20),
    epsilon_cluster, epsilon_harmony, alpha. Recorded in uns["msp"]["harmony"].
    meta_extra: extra keys merged into uns["msp"] (zmip records its lineage)."""
    if batch_col not in ad.obs:
        raise ValueError(f"missing obs[{batch_col!r}] — batch_col must name an obs column")
    if not resolutions:
        raise ValueError("resolutions must contain at least one positive value")
    if any(not math.isfinite(float(r)) or float(r) <= 0 for r in resolutions):
        raise ValueError(f"resolutions must be finite and positive, got {resolutions!r}")
    if len({float(r) for r in resolutions}) != len(resolutions):
        raise ValueError(f"resolutions must be unique, got {resolutions!r}")
    if n_top_genes <= 0 or n_pcs <= 0 or n_neighbors <= 0:
        raise ValueError("n_top_genes, n_pcs, and n_neighbors must be positive")
    if ad.n_obs < 3:
        raise ValueError(f"at least 3 cells are required for integration, found {ad.n_obs}")
    if ad.n_vars < 2:
        raise ValueError(f"at least 2 genes are required for PCA, found {ad.n_vars}")
    if n_neighbors >= ad.n_obs:
        raise ValueError(f"n_neighbors must be smaller than n_obs ({ad.n_obs}), got {n_neighbors}")
    harmony_kwargs = dict(harmony_kwargs or {})
    os.makedirs(outdir, exist_ok=True)
    if counts_layer not in ad.layers:
        raise ValueError(f"missing layers[{counts_layer!r}] — raw counts are required")
    begin_step(outdir, "integrate")
    # Prior inspection decisions and old resolutions do not describe this run.
    for key in list(ad.obs.columns):
        if key in ("_msp_action", "_msp_verdict") or key.startswith(("inspect_sub", "msp_leiden_r")):
            del ad.obs[key]
    ad.X = ad.layers[counts_layer].copy()
    for k in list(ad.obsm.keys()):
        del ad.obsm[k]
    for k in list(ad.obsp.keys()):
        del ad.obsp[k]
    ad.uns.clear()
    ad.raw = None
    n_samples = ad.obs[batch_col].astype(str).nunique()
    print(f"== integrating: {ad.shape}, {n_samples} samples (batch={batch_col!r})", flush=True)

    print("== normalize/log1p", flush=True)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.raw = ad

    print(f"== HVG per batch (n_top_genes={n_top_genes})", flush=True)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor="seurat", batch_key=batch_col)

    hvg = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(hvg, max_value=10)
    n_comps = min(n_pcs, hvg.n_vars - 1, ad.n_obs - 1)
    if n_comps < 1:
        raise ValueError(f"not enough variable genes/cells for PCA: {hvg.shape}, n_comps={n_comps}")
    print(f"== PCA ({n_comps} comps on {hvg.n_vars} HVGs)", flush=True)
    ad.obsm["X_pca"] = PCA(n_components=n_comps, svd_solver="arpack", random_state=0).fit_transform(hvg.X)
    del hvg

    # call harmonypy directly: the installed fork returns Z_corr already
    # cells-by-PCs, which scanpy's wrapper transposes into garbage — accept
    # either orientation and assert the final shape
    import harmonypy
    import numpy as np
    import torch

    # GPU auto-detect (same order as harmonypy's own get_device); MSP_DEVICE
    # env overrides (cpu|cuda|mps)
    device = os.environ.get("MSP_DEVICE") or None
    auto = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    if n_samples < 2:
        # nothing to correct across: one sample / one batch level. The rest of
        # the chain (neighbors, leiden, UMAP, QC tables, agents) runs unchanged
        # on plain PCA so single-sample datasets go through the same steps
        print("== harmony skipped: single batch — X_pca_harmony = X_pca", flush=True)
        Z = np.array(ad.obsm["X_pca"], copy=True)
        harmony_record = "skipped: single batch"
    else:
        print(
            f"== harmony on {device or auto}{'' if device else ' (auto-detected)'}"
            + (f", overrides {harmony_kwargs}" if harmony_kwargs else ", harmonypy defaults"),
            flush=True,
        )
        ho = harmonypy.run_harmony(
            ad.obsm["X_pca"], ad.obs[[batch_col]], batch_col, random_state=0, device=device, **harmony_kwargs
        )
        Z = np.asarray(ho.Z_corr)
        if Z.shape[0] != ad.n_obs:
            Z = Z.T
        harmony_record = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in harmony_kwargs.items()}
    if Z.shape != (ad.n_obs, n_comps):
        raise ValueError(f"harmony returned shape {Z.shape}, expected ({ad.n_obs}, {n_comps})")
    ad.obsm["X_pca_harmony"] = Z

    print("== neighbors (use_rep=X_pca_harmony)", flush=True)
    sc.pp.neighbors(ad, use_rep="X_pca_harmony", n_neighbors=n_neighbors)
    leiden_keys = []
    for r in resolutions:
        key = f"msp_leiden_r{r}"
        print(f"== leiden {key}", flush=True)
        sc.tl.leiden(ad, resolution=r, key_added=key, flavor="igraph", n_iterations=2)
        leiden_keys.append(key)

    print("== umap", flush=True)
    sc.tl.umap(ad)

    primary_key = leiden_keys[min(1, len(leiden_keys) - 1)]  # middle resolution

    # standissect-lite (>=0.2.0): cross the RNA-side leiden with a UMAP-side
    # clustering; "subcluster" (c{parent}_{rank}) is its headline per-cell
    # identifier — rank 0 is always the largest fragment WITHIN that parent,
    # strictly descending. That's exactly the clustering scheme we visualize
    # and label, same as the leiden panels: detection only, no grey/colored
    # main-vs-minor framing here — msp.inspect judges which cells matter.
    # Parent comes from the LOWEST resolution: the product hunts strays
    # inside broad clusters — a high-res leiden has already split them.
    standissect_key = leiden_keys[int(np.argmin(resolutions))]
    print(f"== standissect-lite on {standissect_key} (leiden x umap product)", flush=True)
    from standissect_lite import dissect_partition

    res = dissect_partition(ad, cluster_col=standissect_key, umap_key="X_umap")
    ad.obs["standissect_product"] = res.labels["subcluster"].astype("category")
    res.fragments.to_csv(os.path.join(outdir, f"fragments_{standissect_key}.csv"), index=False)
    res.overlap.to_csv(os.path.join(outdir, f"overlap_{standissect_key}.csv"))
    print("== minor-sibling QC", flush=True)
    msq_df = _minor_sibling_qc(ad, res, outdir)

    print("== cell-level doublet/ambient-RNA outliers (per-cluster MAD + hard floor)", flush=True)
    cell_outliers_df = _cell_level_outliers(ad, leiden_keys, resolutions, outdir)

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    print("== leiden cluster QC violins (per-cluster cutoffs)", flush=True)
    _leiden_cluster_qc_violins(ad, leiden_keys, resolutions, figdir)

    remove_mask = _build_removal_mask(ad, msq_df, cell_outliers_df, outdir)
    print(
        f"== pre-annotation filtering: {int(remove_mask.sum())}/{ad.n_obs} cells "
        "recommend_removal (minor-sibling fragments ∪ cell-level outliers ∪ osp _qc_action=drop)",
        flush=True,
    )
    _preannotation_removal_umap(ad, remove_mask, figdir)

    print("== cluster annotations (PAGA + global/local DEG)", flush=True)
    _cluster_annotations(ad, remove_mask, leiden_keys, resolutions, outdir, top_n_de=top_n_de)

    ad.uns["msp"] = {
        "batch_col": batch_col,
        "species": species or "",
        "inputs": [str(p) for p in inputs],
        "resolutions": list(resolutions),
        "n_top_genes": n_top_genes,
        "n_pcs_requested": n_pcs,
        "n_pcs": n_comps,
        "n_neighbors": n_neighbors,
        # what actually reached harmonypy.run_harmony beyond its defaults
        # (empty dict = harmonypy defaults: theta=2/covariate, lamb=1,
        # sigma=0.1, nclust=min(round(N/30),100), max_iter_harmony=10)
        "harmony": harmony_record,  # kwargs beyond harmonypy defaults, or "skipped: single batch"
        "n_batches": int(n_samples),
        **(meta_extra or {}),
    }

    print("== figures", flush=True)
    save_single_umap(ad, batch_col, os.path.join(figdir, f"umap_{slug(batch_col)}.png"), legend_fontsize=6)
    for color in leiden_keys:  # cluster ids on the clusters, repelled apart
        save_single_umap(ad, color, os.path.join(figdir, f"umap_{slug(color)}.png"), repel=True)
    # inherited per-sample annotation — coarse only (the fine labels are far
    # too many across samples to render legibly)
    if "_ann_coarse" in ad.obs:
        # many near-duplicate labels across samples — needs a much bigger
        # canvas for repel to actually separate them
        save_single_umap(
            ad,
            "_ann_coarse",
            os.path.join(figdir, "umap__ann_coarse.png"),
            repel=True,
            repel_fontsize=7,
            figsize=(14, 14),
        )

    # the product clustering itself: every (parent, umap_cluster) cell as its
    # own category, labeled and repelled just like the leiden panels — no
    # main/minor distinction, standissect-lite only detects, msp.inspect judges
    save_single_umap(
        ad, "standissect_product", os.path.join(figdir, "standissect_product.png"), repel=True, repel_fontsize=7
    )

    print("== QC figures/tables", flush=True)
    _qc_outputs(ad, batch_col, primary_key, outdir, figdir, leiden_keys, resolutions)

    print("== fractal marker heatmap", flush=True)
    _fractal_marker_heatmap(ad, res, outdir, figdir)

    summary = {
        "n_cells": int(ad.n_obs),
        "n_genes": int(ad.n_vars),
        "n_samples": n_samples,
        "batch_col": batch_col,
        **{k: int(ad.obs[k].nunique()) for k in leiden_keys},
    }
    pd.Series(summary).to_csv(os.path.join(outdir, "integration_summary.csv"))

    tmp = os.path.join(outdir, "integrated.tmp.h5ad")
    ad.write_h5ad(tmp)  # never in place: tmp + rename
    os.replace(tmp, os.path.join(outdir, "integrated.h5ad"))
    complete_step(outdir, "integrate")
    print(f"== wrote {os.path.join(outdir, 'integrated.h5ad')}", flush=True)
    return ad, summary
