"""Concat per-sample osp outputs and integrate with harmony.

Pipeline (conventions mirror osp where they apply): raw counts →
normalize_total(1e4) → log1p → HVG per batch (flavor="seurat",
batch_key) → scale(max 10) on HVG → PCA (arpack, seed 0) → harmony on the
batch key → neighbors on X_pca_harmony → leiden at several resolutions →
UMAP. No PAGA, no DEG, no cell removal — msp proposes, later steps decide.

Inherited per-sample columns (QC metrics, _ann_*, _qc_action, doublet
calls) ride along in obs. Sample-local leiden labels are prefixed with the
sample value ("H12inner:3") so they stay meaningful after the merge; the
integrated clusterings get their own msp_leiden_r* keys.
"""

from __future__ import annotations

import csv
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
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA

from .plots import UMAP_DPI, save_single_umap, slug


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
        a.layers = {counts_layer: counts}
        adatas.append(a)

    merged = ad.concat(adatas, join="inner", merge="same")
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
MIN_N_FOR_TEST = 5          # below this, Mann-Whitney has no real power — mark insufficient_data
BIG_SIBLING_FRAC = 0.25     # sibling >= this fraction of its own parent's core is skipped, not a "minor" fragment
BIG_SIBLING_N = 800         # sibling >= this many cells (absolute) is skipped too, regardless of frac_of_core
DOUBLET_MEDIAN_THRESH = 0.2   # scrublet score, 0-1 scale
MT_MEDIAN_THRESH = 20.0        # pct_counts_mt is already on a 0-100 scale
DROP_PCT_THRESH = 50.0         # % of a sibling's cells already _qc_action=drop upstream (osp)


def _mwu_greater(sib_vals, core_vals):
    """One-sided Mann-Whitney U: is `sib_vals` stochastically greater than
    `core_vals`? Rank-based — doesn't assume normality, works for the small,
    skewed samples minor siblings usually are."""
    from scipy.stats import mannwhitneyu

    if len(sib_vals) < MIN_N_FOR_TEST:
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
        row = {"subcluster": sub, "parent": parent, "n_cells": n_cells,
               "core_n_cells": cn, "frac_of_core": round(n_cells / cn, 3) if cn else None}
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


def _qc_outputs(ad, batch_col, primary_key, outdir, figdir):
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
        sc.pl.violin(ad, m, groupby=violin_key, stripplot=False, rotation=90, show=False,
                    log=m in log_metrics)
        plt.savefig(os.path.join(figdir, f"qc_violin_{slug(m)}.png"), dpi=UMAP_DPI,
                    bbox_inches="tight")
        plt.close("all")

    if "_qc_action" in ad.obs:
        order = [c for c in ("keep", "flag", "drop") if c in set(ad.obs["_qc_action"].astype(str))]
        ad.obs["_qc_action"] = pd.Categorical(ad.obs["_qc_action"].astype(str), categories=order)
        # inherited from OSP just like _ann_coarse — filename (no qc_
        # prefix) groups it with the OSP-inherited panels, not the
        # integrated-space QC metrics
        save_single_umap(ad, "_qc_action", os.path.join(figdir, "umap__qc_action.png"),
                         palette=[QC_ACTION_PALETTE[c] for c in order], legend_loc="best")

    def _agg(groupby):
        g = ad.obs.groupby(groupby, observed=True)
        out = pd.DataFrame({"n_cells": g.size()})
        if "_qc_action" in ad.obs:
            out["pct_flag"] = (g["_qc_action"].apply(lambda s: (s == "flag").mean()) * 100).round(2)
            out["pct_drop"] = (g["_qc_action"].apply(lambda s: (s == "drop").mean()) * 100).round(2)
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


# Seurat's classic DoHeatmap palette (colorRampPalette(c("purple","black","yellow")))
SEURAT_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "seurat_purple_black_yellow", ["purple", "black", "yellow"])


def _fractal_marker_heatmap(ad, res, outdir, figdir, top_n=10):
    """Explain fractal identity from the transcriptome, not just QC stats:
    DE each parent's core cells one-vs-rest against every OTHER parent's
    core cells (core-only, so a minor sibling's cells never leak into
    either side of the comparison) -> per-parent top markers (logFC>0,
    padj<0.05, ribosomal genes excluded) -> average log1p expression of
    those markers across every standissect cluster (cores AND fractals) ->
    row-wise (per-gene) z-score, Seurat-style purple-black-yellow heatmap.

    Columns are clustered (optimal leaf ordering) but the dendrogram is
    hidden — it's ordering, not something to read. Rows are NOT clustered:
    genes stay grouped by the parent they're a marker for, with a colored
    strip per gene showing that parent, since that grouping is more legible
    than a gene-gene dendrogram here. Column labels: parent-core clusters
    (c{parent}_0) bold; clusters minor_sibling_qc.csv marked
    recommend_removal in red."""
    frag = res.fragments
    core_rows = frag.loc[frag["rank"] == 0]
    parent_of_core = dict(zip(core_rows["subcluster"], core_rows["parent"]))
    core_mask = ad.obs["standissect_product"].astype(str).isin(parent_of_core)
    core_ad = ad[core_mask].copy()
    core_ad.obs["parent"] = (
        core_ad.obs["standissect_product"].astype(str).map(parent_of_core).astype(str).astype("category")
    )

    print("== parent-core DEG (one vs other parent cores)", flush=True)
    sc.tl.rank_genes_groups(core_ad, "parent", method="wilcoxon", use_raw=True, pts=True)
    de_df = sc.get.rank_genes_groups_df(core_ad, group=None)
    de_df.to_csv(os.path.join(outdir, "de_parent_core_vs_core.csv"), index=False)

    ribo = set(ad.var_names[ad.var["ribo"]]) if "ribo" in ad.var else set()
    markers, marker_rows, gene_parent = [], [], {}
    for parent, g in de_df.groupby("group", observed=True):
        g = g[(g["pvals_adj"] < 0.05) & (g["logfoldchanges"] > 0) & (~g["names"].isin(ribo))]
        g = g.sort_values("logfoldchanges", ascending=False).head(top_n)
        for rank, row in enumerate(g.itertuples(), start=1):
            markers.append(row.names)
            gene_parent.setdefault(row.names, parent)  # first parent wins if a gene repeats
            marker_rows.append({"parent": parent, "gene": row.names, "rank": rank,
                                 "logfoldchange": round(row.logfoldchanges, 3),
                                 "pvals_adj": row.pvals_adj})
    pd.DataFrame(marker_rows).to_csv(os.path.join(outdir, "fractal_markers.csv"), index=False)
    # row order follows parent order (numeric, not the groupby's alpha-sort), then rank within
    # parent; de-dup a gene shared across parents by keeping its first (lowest-parent) occurrence
    marker_rows_sorted = sorted(marker_rows, key=lambda r: (int(r["parent"]), r["rank"]))
    markers = list(dict.fromkeys(r["gene"] for r in marker_rows_sorted))
    if not markers:
        print("== no marker genes passed the filters — skipping heatmap", flush=True)
        return

    print(f"== fractal marker heatmap ({len(markers)} genes x "
          f"{ad.obs['standissect_product'].nunique()} clusters)", flush=True)
    sub = ad[:, markers]
    X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    expr = pd.DataFrame(X, index=ad.obs_names, columns=markers)
    expr["cluster"] = ad.obs["standissect_product"].astype(str).values
    avg = expr.groupby("cluster").mean().T  # genes x clusters, average log1p expression
    avg.to_csv(os.path.join(outdir, "fractal_marker_avg_expr.csv"))

    z = avg.sub(avg.mean(axis=1), axis=0).div(avg.std(axis=1).replace(0, 1), axis=0).fillna(0)
    z = z.loc[markers]  # parent-then-rank order (see marker_rows_sorted above)

    col_linkage = linkage(z.values.T, method="average", metric="euclidean", optimal_ordering=True)

    removal_path = os.path.join(outdir, "minor_sibling_qc.csv")
    removal_set = set()
    if os.path.exists(removal_path):
        with open(removal_path) as f:
            removal_set = {r["subcluster"] for r in csv.DictReader(f) if r.get("recommend_removal") == "True"}
    core_pattern = re.compile(r"^c\d+_0$")

    parents = sorted({p for p in gene_parent.values()}, key=int)
    parent_palette = dict(zip(parents, sns.color_palette("tab20", n_colors=max(len(parents), 1))))
    row_colors = pd.Series({g: parent_palette[gene_parent[g]] for g in markers}, name="parent")

    n_genes, n_clusters = z.shape
    cg = sns.clustermap(
        z, row_cluster=False, col_linkage=col_linkage, row_colors=row_colors,
        cmap=SEURAT_HEATMAP_CMAP, center=0, vmin=-2.5, vmax=2.5,
        figsize=(max(6.0, n_clusters * 0.35 + 3), max(6.0, n_genes * 0.18 + 2)),
        dendrogram_ratio=(0.01, 0.08),
        xticklabels=True, yticklabels=True,
        cbar_kws={"label": "z-score", "orientation": "horizontal"},
        cbar_pos=(0.02, -0.05, 0.12, 0.015),  # below the x-axis cluster-name labels, not beside them
    )
    cg.ax_col_dendrogram.set_visible(False)  # ordering used, tree not shown — it's not the point
    for label in cg.ax_heatmap.get_xticklabels():
        name = label.get_text()
        if core_pattern.match(name):
            label.set_fontweight("bold")
        if name in removal_set:
            label.set_color("#c0392b")
    for x in range(1, n_clusters):  # thin white separators between columns only
        cg.ax_heatmap.axvline(x, color="white", linewidth=0.6)
    cg.ax_heatmap.set_xlabel("standissect cluster (core + fractals)")
    cg.ax_heatmap.set_ylabel("marker gene")
    cg.savefig(os.path.join(figdir, "fractal_marker_heatmap.png"), dpi=UMAP_DPI, bbox_inches="tight")
    plt.close("all")


def run_multi_sample_pipeline(inputs, batch_col, outdir, species=None,
                              resolutions=(0.3, 1.0, 2.0), n_top_genes=2000,
                              n_pcs=50, n_neighbors=15, counts_layer="counts",
                              top_n_de=50):
    os.makedirs(outdir, exist_ok=True)
    ad = load_and_merge(inputs, batch_col, counts_layer=counts_layer)
    n_samples = ad.obs[batch_col].astype(str).nunique()
    print(f"== merged: {ad.shape}, {n_samples} samples (batch={batch_col!r})", flush=True)

    print("== normalize/log1p", flush=True)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.raw = ad

    print(f"== HVG per batch (n_top_genes={n_top_genes})", flush=True)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor="seurat", batch_key=batch_col)

    hvg = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(hvg, max_value=10)
    n_comps = min(n_pcs, hvg.n_vars - 1, ad.n_obs - 1)
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
    auto = ("cuda" if torch.cuda.is_available() else
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu")
    print(f"== harmony on {device or auto}{'' if device else ' (auto-detected)'}", flush=True)
    ho = harmonypy.run_harmony(ad.obsm["X_pca"], ad.obs[[batch_col]], batch_col,
                               random_state=0, device=device)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != ad.n_obs:
        Z = Z.T
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

    print(f"== rank_genes_groups on {primary_key}", flush=True)
    sc.tl.rank_genes_groups(ad, primary_key, method="wilcoxon", use_raw=True, pts=True)
    de_df = sc.get.rank_genes_groups_df(ad, group=None)
    # pct1/pct2 naming mirrors osp's de_top_genes export
    de_df = de_df.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"})
    de_top = de_df.groupby("group", observed=True).head(top_n_de).reset_index(drop=True)
    de_top.to_csv(os.path.join(outdir, f"de_top_genes_{primary_key}.csv"), index=False)

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
    _minor_sibling_qc(ad, res, outdir)

    ad.uns["msp"] = {
        "batch_col": batch_col,
        "species": species or "",
        "inputs": [str(p) for p in inputs],
        "resolutions": list(resolutions),
        "n_top_genes": n_top_genes,
    }

    print("== figures", flush=True)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    save_single_umap(ad, batch_col, os.path.join(figdir, f"umap_{slug(batch_col)}.png"),
                     legend_fontsize=6)
    for color in leiden_keys:  # cluster ids on the clusters, repelled apart
        save_single_umap(ad, color, os.path.join(figdir, f"umap_{slug(color)}.png"), repel=True)
    # inherited per-sample annotation — coarse only (the fine labels are far
    # too many across samples to render legibly)
    if "_ann_coarse" in ad.obs:
        # many near-duplicate labels across samples — needs a much bigger
        # canvas for repel to actually separate them
        save_single_umap(ad, "_ann_coarse", os.path.join(figdir, "umap__ann_coarse.png"),
                         repel=True, repel_fontsize=7, figsize=(14, 14))

    # the product clustering itself: every (parent, umap_cluster) cell as its
    # own category, labeled and repelled just like the leiden panels — no
    # main/minor distinction, standissect-lite only detects, msp.inspect judges
    save_single_umap(ad, "standissect_product", os.path.join(figdir, "standissect_product.png"),
                     repel=True, repel_fontsize=7)

    print("== QC figures/tables", flush=True)
    _qc_outputs(ad, batch_col, primary_key, outdir, figdir)

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
    print(f"== wrote {os.path.join(outdir, 'integrated.h5ad')}", flush=True)
    return ad, summary
