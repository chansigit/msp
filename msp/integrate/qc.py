"""QC evidence for the integrated space: metric UMAPs, the inherited
keep/flag/drop overlay, and per-sample / per-cluster QC tables."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from ..plots import UMAP_DPI, save_single_umap, slug

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
