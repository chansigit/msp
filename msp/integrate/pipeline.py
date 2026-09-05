"""The integration pipeline: merge per-sample files, then normalize → HVG per
batch → PCA → harmony → neighbors → leiden → UMAP → fragment / QC / DEG
evidence → integrated.h5ad and figures.

``integrate_adata`` runs the stage functions below in order; each one prints
its progress the same way the original single function did, so Slurm logs
read the same.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA

from ..log import ensure
from ..plots import save_single_umap, slug
from ..steps import begin_step, complete_step
from .deg import _cluster_annotations
from .fragments import _fractal_marker_heatmap, _minor_sibling_qc
from .outliers import _build_removal_mask, _cell_level_outliers, _leiden_cluster_qc_violins, _preannotation_removal_umap
from .qc import _qc_outputs

log = logging.getLogger(__name__)


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
    ensure()
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


# ---------------------------------------------------------------- stages


def _validate_inputs(ad, batch_col, resolutions, n_top_genes, n_pcs, n_neighbors, counts_layer):
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
    if counts_layer not in ad.layers:
        raise ValueError(f"missing layers[{counts_layer!r}] — raw counts are required")


def _reset_state(ad, counts_layer):
    """Start from raw counts: prior inspection decisions, old resolutions,
    embeddings, graphs and uns do not describe this run."""
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


def _preprocess(ad, batch_col, n_top_genes):
    """normalize_total(1e4) → log1p (kept as .raw) → HVG per batch."""
    log.info("== normalize/log1p")
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.raw = ad

    log.info(f"== HVG per batch (n_top_genes={n_top_genes})")
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor="seurat", batch_key=batch_col)


def _embed(ad, batch_col, n_pcs, n_samples, harmony_kwargs):
    """Scaled-HVG PCA, then harmony on the batch key (skipped for a single
    batch). Returns ``(n_comps, harmony_record)``."""
    hvg = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(hvg, max_value=10)
    n_comps = min(n_pcs, hvg.n_vars - 1, ad.n_obs - 1)
    if n_comps < 1:
        raise ValueError(f"not enough variable genes/cells for PCA: {hvg.shape}, n_comps={n_comps}")
    log.info(f"== PCA ({n_comps} comps on {hvg.n_vars} HVGs)")
    ad.obsm["X_pca"] = PCA(n_components=n_comps, svd_solver="arpack", random_state=0).fit_transform(hvg.X)
    del hvg

    # harmonypy >= 2.0 (C++ backend, numpy-only): Z_corr is cells-by-PCs.
    # Accept either orientation anyway and assert the final shape below.
    import harmonypy

    if n_samples < 2:
        # nothing to correct across: one sample / one batch level. The rest of
        # the chain (neighbors, leiden, UMAP, QC tables, agents) runs unchanged
        # on plain PCA so single-sample datasets go through the same steps
        log.info("== harmony skipped: single batch — X_pca_harmony = X_pca")
        Z = np.array(ad.obsm["X_pca"], copy=True)
        harmony_record = "skipped: single batch"
    else:
        from ..resources import available_cpus

        # BLAS threads for the C++ solver: the CPUs this process may really
        # use (affinity mask / cgroup), not the node's core count.
        kwargs = {"random_state": 0, "ncores": available_cpus(), **harmony_kwargs}
        log.info(
            f"== harmony (harmonypy {getattr(harmonypy, '__version__', '?')}, {kwargs['ncores']} thread(s))"
            + (f", overrides {harmony_kwargs}" if harmony_kwargs else ", harmonypy defaults"),
        )
        ho = harmonypy.run_harmony(ad.obsm["X_pca"], ad.obs[[batch_col]], batch_col, **kwargs)
        Z = np.asarray(ho.Z_corr)
        if Z.shape[0] != ad.n_obs:
            Z = Z.T
        harmony_record = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in harmony_kwargs.items()}
    if Z.shape != (ad.n_obs, n_comps):
        raise ValueError(f"harmony returned shape {Z.shape}, expected ({ad.n_obs}, {n_comps})")
    ad.obsm["X_pca_harmony"] = Z
    return n_comps, harmony_record


def _cluster(ad, resolutions, n_neighbors):
    """Neighbors on X_pca_harmony, leiden at every resolution, UMAP. Returns
    the leiden keys in resolution order."""
    log.info("== neighbors (use_rep=X_pca_harmony)")
    sc.pp.neighbors(ad, use_rep="X_pca_harmony", n_neighbors=n_neighbors)
    leiden_keys = []
    for r in resolutions:
        key = f"msp_leiden_r{r}"
        log.info(f"== leiden {key}")
        sc.tl.leiden(ad, resolution=r, key_added=key, flavor="igraph", n_iterations=2)
        leiden_keys.append(key)

    log.info("== umap")
    sc.tl.umap(ad)
    return leiden_keys


def _dissect(ad, leiden_keys, resolutions, outdir):
    """standissect-lite (>=0.2.0): cross the RNA-side leiden with a UMAP-side
    clustering; "subcluster" (c{parent}_{rank}) is its headline per-cell
    identifier — rank 0 is always the largest fragment WITHIN that parent,
    strictly descending. That's exactly the clustering scheme we visualize
    and label, same as the leiden panels: detection only, no grey/colored
    main-vs-minor framing here — msp.inspect judges which cells matter.
    Parent comes from the LOWEST resolution: the product hunts strays
    inside broad clusters — a high-res leiden has already split them."""
    standissect_key = leiden_keys[int(np.argmin(resolutions))]
    log.info(f"== standissect-lite on {standissect_key} (leiden x umap product)")
    from standissect_lite import dissect_partition

    res = dissect_partition(ad, cluster_col=standissect_key, umap_key="X_umap")
    ad.obs["standissect_product"] = res.labels["subcluster"].astype("category")
    res.fragments.to_csv(os.path.join(outdir, f"fragments_{standissect_key}.csv"), index=False)
    res.overlap.to_csv(os.path.join(outdir, f"overlap_{standissect_key}.csv"))
    return res


def _evidence(ad, res, leiden_keys, resolutions, outdir, figdir, top_n_de):
    """Fragment QC, cell-level outliers, the pre-annotation removal union,
    then PAGA + global/local DEG on the survivors."""
    log.info("== minor-sibling QC")
    msq_df = _minor_sibling_qc(ad, res, outdir)

    log.info("== cell-level doublet/ambient-RNA outliers (per-cluster MAD + hard floor)")
    cell_outliers_df = _cell_level_outliers(ad, leiden_keys, resolutions, outdir)

    log.info("== leiden cluster QC violins (per-cluster cutoffs)")
    _leiden_cluster_qc_violins(ad, leiden_keys, resolutions, figdir)

    remove_mask = _build_removal_mask(ad, msq_df, cell_outliers_df, outdir)
    log.info(
        f"== pre-annotation filtering: {int(remove_mask.sum())}/{ad.n_obs} cells "
        "recommend_removal (minor-sibling fragments ∪ cell-level outliers ∪ osp _qc_action=drop)",
    )
    _preannotation_removal_umap(ad, remove_mask, figdir)

    log.info("== cluster annotations (PAGA + global/local DEG)")
    _cluster_annotations(ad, remove_mask, leiden_keys, resolutions, outdir, top_n_de=top_n_de)


def _figures(ad, batch_col, leiden_keys, figdir):
    log.info("== figures")
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


def _write(ad, batch_col, leiden_keys, n_samples, outdir):
    """integration_summary.csv and integrated.h5ad (tmp + rename), then the
    step is complete. Returns the summary dict."""
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
    log.info(f"== wrote {os.path.join(outdir, 'integrated.h5ad')}")
    return summary


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

    harmony_kwargs: passed straight to harmonypy.run_harmony (>= 2.0) on top
    of msp's fixed random_state=0 and ncores=available_cpus() — theta
    (diversity penalty, default 2 per covariate), lamb (ridge; default None =
    auto-estimate), sigma (soft k-means width, 0.1), nclust (default
    min(round(N/30), 100)), tau, block_size, max_iter_harmony (10),
    max_iter_kmeans (4), epsilon_cluster (1e-3), epsilon_harmony (1e-2),
    alpha, batch_prop_cutoff. Recorded in uns["msp"]["harmony"].
    meta_extra: extra keys merged into uns["msp"] (zmip records its lineage)."""
    ensure()
    _validate_inputs(ad, batch_col, resolutions, n_top_genes, n_pcs, n_neighbors, counts_layer)
    harmony_kwargs = dict(harmony_kwargs or {})
    os.makedirs(outdir, exist_ok=True)
    begin_step(outdir, "integrate")
    _reset_state(ad, counts_layer)
    n_samples = ad.obs[batch_col].astype(str).nunique()
    log.info(f"== integrating: {ad.shape}, {n_samples} samples (batch={batch_col!r})")

    _preprocess(ad, batch_col, n_top_genes)
    n_comps, harmony_record = _embed(ad, batch_col, n_pcs, n_samples, harmony_kwargs)
    leiden_keys = _cluster(ad, resolutions, n_neighbors)
    primary_key = leiden_keys[min(1, len(leiden_keys) - 1)]  # middle resolution
    res = _dissect(ad, leiden_keys, resolutions, outdir)

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    _evidence(ad, res, leiden_keys, resolutions, outdir, figdir, top_n_de)

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

    _figures(ad, batch_col, leiden_keys, figdir)

    log.info("== QC figures/tables")
    _qc_outputs(ad, batch_col, primary_key, outdir, figdir, leiden_keys, resolutions)

    log.info("== fractal marker heatmap")
    _fractal_marker_heatmap(ad, res, outdir, figdir)

    summary = _write(ad, batch_col, leiden_keys, n_samples, outdir)
    return ad, summary
