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

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA


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


def _qc_outputs(ad, batch_col, primary_key, outdir, figdir):
    """QC evidence for the integrated space: metric UMAPs, the inherited
    keep/flag/drop overlay, and per-sample / per-cluster QC tables — the
    raw material for the later per-cluster inspection step."""
    metrics = [(m, vmax) for m, vmax in QC_UMAP_METRICS if m in ad.obs]
    if metrics:
        sc.pl.umap(ad, color=[m for m, _ in metrics], vmax=[v for _, v in metrics],
                   ncols=3, show=False)
        plt.savefig(os.path.join(figdir, "qc_umap_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

    if "_qc_action" in ad.obs:
        order = [c for c in ("keep", "flag", "drop") if c in set(ad.obs["_qc_action"].astype(str))]
        ad.obs["_qc_action"] = pd.Categorical(ad.obs["_qc_action"].astype(str), categories=order)
        sc.pl.umap(ad, color="_qc_action", palette=[QC_ACTION_PALETTE[c] for c in order],
                   show=False)
        plt.savefig(os.path.join(figdir, "qc_umap_qc_action.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

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
    cl = _agg(primary_key)
    # how many samples contribute to each cluster — a 1-sample cluster in an
    # integrated space is itself a QC signal
    cl["n_samples"] = ad.obs.groupby(primary_key, observed=True)[batch_col].nunique()
    cl.to_csv(os.path.join(outdir, f"cluster_qc_{primary_key}.csv"))


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

    print("== harmony", flush=True)
    # call harmonypy directly: the installed fork returns Z_corr already
    # cells-by-PCs, which scanpy's wrapper transposes into garbage — accept
    # either orientation and assert the final shape
    import harmonypy
    import numpy as np

    ho = harmonypy.run_harmony(ad.obsm["X_pca"], ad.obs[[batch_col]], batch_col, random_state=0)
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

    print(f"== standissect-lite on {primary_key} (rule mode)", flush=True)
    from standissect import run_dissect_pipeline

    run_dissect_pipeline(
        ad,
        cluster_col=primary_key,
        output_dir=os.path.join(outdir, "standissect"),
        umap_key="X_umap",
        sample_col=batch_col,
        batch_col=batch_col,
        annotation_col="_ann_coarse" if "_ann_coarse" in ad.obs else None,
        doublet_score_col="doublet_score" if "doublet_score" in ad.obs else None,
        mito_col="pct_counts_mt" if "pct_counts_mt" in ad.obs else None,
        feature_count_col="n_genes_by_counts" if "n_genes_by_counts" in ad.obs else None,
        umi_count_col="total_counts" if "total_counts" in ad.obs else None,
        diagnosis_mode="rule",  # candidates only — verdicts belong to msp.inspect
        random_state=0,
    )

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
    colorings = [batch_col] + leiden_keys + (["_ann_coarse"] if "_ann_coarse" in ad.obs else [])
    for color in colorings:
        sc.pl.umap(ad, color=color, legend_fontsize=6, show=False)
        plt.savefig(os.path.join(figdir, f"umap_{color.replace('.', '_')}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close("all")

    print("== QC figures/tables", flush=True)
    _qc_outputs(ad, batch_col, primary_key, outdir, figdir)

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
