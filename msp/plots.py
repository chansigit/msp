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
import matplotlib.ticker as mticker
import numpy as np
import scanpy as sc

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


def save_single_umap(ad, color_col, out_path, **kwargs):
    """One UMAP colored by one column, saved to its own file (osp's
    _save_single_umap conventions verbatim)."""
    xlim, ylim = square_limits(np.asarray(ad.obsm["X_umap"]))
    kwargs.setdefault("size", 120000 / ad.n_obs)

    fig = plt.figure(figsize=UMAP_FIGSIZE)
    ax = fig.add_axes(UMAP_AXES_RECT)
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
