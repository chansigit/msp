"""Summarize only Scanpy's undefined log-fold-change warnings per DEG call."""

import logging
import warnings

import scanpy as sc

log = logging.getLogger(__name__)


def rank_genes_groups(*args, **kwargs):
    """Run the unchanged Scanpy test; preserve other warnings and all errors.

    Zero expression can produce infinite/NaN log-fold changes. These values
    remain in the results; the repeated warning is summarized, not repaired.
    """
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings(
                "always", message=r"(divide by zero|invalid value) encountered in log2$",
                category=RuntimeWarning, module=r"scanpy\.tools\._rank_genes_groups$",
            )
            return sc.tl.rank_genes_groups(*args, **kwargs)
    finally:
        count = 0
        for warning in caught:
            is_logfc = (
                issubclass(warning.category, RuntimeWarning)
                and warning.filename.replace("\\", "/").endswith("/scanpy/tools/_rank_genes_groups.py")
                and str(warning.message) in {
                    "divide by zero encountered in log2", "invalid value encountered in log2",
                }
            )
            if is_logfc:
                count += 1
            else:
                warnings.warn_explicit(
                    warning.message, warning.category, warning.filename, warning.lineno,
                )
        if count:
            log.warning(
                "== DEG: %d Scanpy log-fold-change warnings (zero/invalid expression); "
                "non-finite values remain in the DEG results", count,
            )
