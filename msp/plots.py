"""Shared UMAP rendering conventions — same rules as osp.cluster:

- ONE plot per file, never a scanpy multi-panel figure (a human or an agent
  should read one signal per image);
- fixed figsize / axes rect / dpi so every UMAP panel the pipeline produces
  is pixel-for-pixel comparable;
- square limits + equal aspect; point size 120000/n_obs; ticks restored;
- no axis labels (redundant at one file per panel), title carries the name.
"""

from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap

import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text

UMAP_FIGSIZE = (5.5, 5.5)
UMAP_AXES_RECT = (0.14, 0.12, 0.78, 0.78)  # left, bottom, width, height
UMAP_DPI = 150

# axes box in inches, derived from the figsize/rect above — held CONSTANT
# across every panel. A side legend never shrinks this box: when a legend
# needs more room than the default canvas has, the canvas grows to the
# right instead (so long sample names are never clipped, and every UMAP's
# axes are pixel-for-pixel the same size regardless of legend length).
_AXES_W_IN = UMAP_FIGSIZE[0] * UMAP_AXES_RECT[2]
_AXES_H_IN = UMAP_FIGSIZE[1] * UMAP_AXES_RECT[3]
_LEFT_IN = UMAP_FIGSIZE[0] * UMAP_AXES_RECT[0]
_BOTTOM_IN = UMAP_FIGSIZE[1] * UMAP_AXES_RECT[1]
_RIGHT_PAD_IN = UMAP_FIGSIZE[0] * (1 - UMAP_AXES_RECT[0] - UMAP_AXES_RECT[2])


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def square_limits(xy, pad=1.05):
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 * pad
    return (cx - half, cx + half), (cy - half, cy + half)


def umap_axes(ad):
    """Fresh (fig, ax) with the shared rect, square limits and equal aspect
    already applied — for custom scatter panels that must line up with the
    save_single_umap ones."""
    xy = np.asarray(ad.obsm["X_umap"])
    fig = plt.figure(figsize=UMAP_FIGSIZE)
    ax = fig.add_axes(UMAP_AXES_RECT)
    xlim, ylim = square_limits(xy)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def _repel_on_data_labels(ad, color_col, ax, fontsize=12, fontweight="bold"):
    """Place one text label per category at its centroid, then repel them
    apart with adjustText (the ggrepel equivalent) so dense/adjacent
    clusters don't collide — scanpy's own legend_loc='on data' has no
    collision avoidance and clumps labels together."""
    xy = np.asarray(ad.obsm["X_umap"])
    lab = ad.obs[color_col].astype(str)
    texts = []
    for cat in pd.unique(lab):
        cat = str(cat)
        m = (lab == cat).values
        cx, cy = float(xy[m, 0].mean()), float(xy[m, 1].mean())
        wrapped = ("\n".join(textwrap.wrap(cat, width=14, break_long_words=False,
                                          break_on_hyphens=False))
                  if len(cat) > 14 else cat)
        texts.append(ax.text(cx, cy, wrapped, fontsize=fontsize, fontweight=fontweight,
                             ha="center", va="center",
                             path_effects=[pe.withStroke(linewidth=3, foreground="white")]))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7),
               expand=(1.2, 1.4), force_text=(0.5, 0.7), max_move=200)


def save_single_umap(ad, color_col, out_path, repel=False, repel_fontsize=12,
                     figsize=None, **kwargs):
    """One UMAP colored by one column, saved to its own file (osp's
    _save_single_umap conventions verbatim). repel=True: draw with no
    scanpy legend and place collision-free on-data labels via adjustText
    instead (for categorical columns with many/adjacent groups). figsize
    overrides UMAP_FIGSIZE — a bigger canvas gives repel more room to spread
    a crowded category set apart (the panel still displays at whatever size
    the report scales it to; a bigger source image just fits more labels)."""
    xlim, ylim = square_limits(np.asarray(ad.obsm["X_umap"]))
    kwargs.setdefault("size", 120000 / ad.n_obs)

    fig = plt.figure(figsize=figsize or UMAP_FIGSIZE)
    ax = fig.add_axes(UMAP_AXES_RECT)
    if repel:
        kwargs["legend_loc"] = None
        sc.pl.umap(ad, color=color_col, ax=ax, show=False, **kwargs)
        _repel_on_data_labels(ad, color_col, ax, fontsize=repel_fontsize)
    else:
        sc.pl.umap(ad, color=color_col, ax=ax, show=False, **kwargs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    # scanpy clears tick positions; reset the locator before re-enabling
    ax.xaxis.set_major_locator(mticker.AutoLocator())
    ax.yaxis.set_major_locator(mticker.AutoLocator())
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"UMAP: {ax.get_title()}")

    legend = ax.get_legend()
    if legend is not None:
        # measure the legend, then widen the CANVAS (never the axes box) so
        # nothing is clipped — bbox_to_anchor is axes-relative, so it tracks
        # the axes automatically once the figure is resized
        fig.canvas.draw()
        bbox_in = legend.get_window_extent(fig.canvas.get_renderer()) \
            .transformed(fig.dpi_scale_trans.inverted())
        needed_w = _LEFT_IN + _AXES_W_IN + bbox_in.width + 0.25
        if needed_w > fig.get_figwidth():
            fig.set_size_inches(needed_w, fig.get_figheight())
            ax.set_position([_LEFT_IN / needed_w, UMAP_AXES_RECT[1],
                             _AXES_W_IN / needed_w, UMAP_AXES_RECT[3]])

    fig.savefig(out_path, dpi=UMAP_DPI)
    plt.close(fig)
