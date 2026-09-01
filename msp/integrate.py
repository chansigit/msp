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


def run_multi_sample_pipeline(inputs, batch_col, outdir, species=None,
                              resolutions=(0.3, 1.0, 2.0), n_top_genes=2000,
                              n_pcs=50, n_neighbors=15, counts_layer="counts"):
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
