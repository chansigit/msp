"""Cell-level doublet / ambient-RNA outliers, their per-cluster cutoff
violins, and the union of every removal source proposed before Cluster
Annotations (the pre-annotation filtering mask and UMAP)."""

from __future__ import annotations

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..plots import UMAP_DPI, save_single_umap, slug

log = logging.getLogger(__name__)

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
    log.info(
        f"== cell-level outliers: {n_removal}/{len(df)} cells recommend_removal "
        f"(doublet/ambient, MAD+floor, r1.0 or r2.0)",
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
        # Nullable boolean: cells absent from the table are not flagged, without object-dtype downcasting.
        flags = cell_outliers_df["recommend_removal"].astype("boolean").reindex(ad.obs_names).fillna(False)
        from_cells = flags.to_numpy(dtype=bool)
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
