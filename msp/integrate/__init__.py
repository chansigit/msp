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

Submodules: ``pipeline`` (merge and the staged ``integrate_adata``),
``fragments`` (minor-sibling QC, fractal marker dot plot), ``outliers``
(cell-level outliers, removal union), ``deg`` (PAGA, global/local DEG,
stress signature), ``qc`` (QC UMAPs and tables). The names below are
re-exported so ``from msp.integrate import ...`` keeps working; tests and
ZMIP use the underscore ones.
"""

from .deg import (
    MIN_DE_GROUP_SIZE,
    STRESS_CHECK_TOP_N,
    STRESS_GENE_SET,
    STRESS_GENES_CORE,
    STRESS_HIT_THRESHOLD,
    _cluster_annotations,
    _is_stress_gene,
    _stress_hits,
)
from .fragments import (
    BIG_SIBLING_FRAC,
    BIG_SIBLING_N,
    DOUBLET_MEDIAN_THRESH,
    DROP_PCT_THRESH,
    MIN_N_FOR_TEST,
    MT_MEDIAN_THRESH,
    SEURAT_HEATMAP_CMAP,
    _fractal_marker_heatmap,
    _minor_sibling_qc,
    _mwu_greater,
    _select_fractal_markers,
)
from .outliers import (
    CELL_OUTLIER_HARD_FLOOR,
    CELL_OUTLIER_MAD_K,
    CELL_OUTLIER_METRICS,
    _build_removal_mask,
    _cell_level_outliers,
    _leiden_cluster_qc_violins,
    _preannotation_removal_umap,
)
from .pipeline import integrate_adata, load_and_merge, run_multi_sample_pipeline
from .qc import QC_ACTION_PALETTE, QC_UMAP_METRICS, _qc_outputs

__all__ = [
    "BIG_SIBLING_FRAC",
    "BIG_SIBLING_N",
    "CELL_OUTLIER_HARD_FLOOR",
    "CELL_OUTLIER_MAD_K",
    "CELL_OUTLIER_METRICS",
    "DOUBLET_MEDIAN_THRESH",
    "DROP_PCT_THRESH",
    "MIN_DE_GROUP_SIZE",
    "MIN_N_FOR_TEST",
    "MT_MEDIAN_THRESH",
    "QC_ACTION_PALETTE",
    "QC_UMAP_METRICS",
    "SEURAT_HEATMAP_CMAP",
    "STRESS_CHECK_TOP_N",
    "STRESS_GENES_CORE",
    "STRESS_GENE_SET",
    "STRESS_HIT_THRESHOLD",
    "integrate_adata",
    "load_and_merge",
    "run_multi_sample_pipeline",
    "_build_removal_mask",
    "_cell_level_outliers",
    "_cluster_annotations",
    "_fractal_marker_heatmap",
    "_is_stress_gene",
    "_leiden_cluster_qc_violins",
    "_minor_sibling_qc",
    "_mwu_greater",
    "_preannotation_removal_umap",
    "_qc_outputs",
    "_select_fractal_markers",
    "_stress_hits",
]
