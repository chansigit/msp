"""eca-rsi's contract with msp: every msp name eca-rsi uses, under a public name.

eca-rsi imports msp only from here (its test_layers.py enforces that). Renaming, removing or changing
the behaviour of a name below breaks eca-rsi; everything else in msp is free to change. Names resolve
on first use, so importing this module loads nothing beyond the msp package itself,
in particular not msp.inspect / msp.annotate and their optional agent dependencies.
"""
import importlib

# public name: (module, attribute)
_NAMES = {
    "integrate_adata": ("msp.integrate", "integrate_adata"),
    "load_and_merge": ("msp.integrate", "load_and_merge"),
    "compute_deg_task": ("msp.integrate.deg", "compute_deg_task"),
    "load_deg_input": ("msp.integrate.deg", "load_deg_input"),
    "prepare_deg": ("msp.integrate.deg", "prepare_deg"),
    "save_deg_input": ("msp.integrate.deg", "save_deg_input"),
    "write_deg_results": ("msp.integrate.deg", "write_deg_results"),
    "qc_outputs": ("msp.integrate.qc", "_qc_outputs"),
    "DEG_SQL_DOC": ("msp.evidence", "DEG_SQL_DOC"),
    "DEG_TOOL_DOC": ("msp.evidence", "DEG_TOOL_DOC"),
    "DegCache": ("msp.evidence", "DegCache"),
    "DegTables": ("msp.evidence", "DegTables"),
    "gene_table": ("msp.evidence", "gene_table"),
    "load_paga_neighbors": ("msp.evidence", "load_paga_neighbors"),
    "load_removal_mask": ("msp.evidence", "load_removal_mask"),
    "plot_annotation": ("msp.evidence", "plot_annotation"),
    "qc_table": ("msp.evidence", "qc_table"),
    "INSPECTION_OPS": ("msp.inspect", "_OPS"),
    "INSPECTION_SCHEMA_DOC": ("msp.inspect", "_PROPOSAL_SCHEMA_DOC"),
    "apply_inspection": ("msp.inspect", "_apply_proposal"),
    "guard_batch_actions": ("msp.inspect", "_guard_batch_actions"),
    "subcluster_once": ("msp.inspect", "_subcluster_once"),
    "validate_inspection": ("msp.inspect", "_validate_proposal"),
    "CLUSTER_SCHEMA_DOC": ("msp.annotate", "_CLUSTER_SCHEMA_DOC"),
    "apply_annotation": ("msp.annotate", "_apply"),
    "check_coarse_boundaries": ("msp.annotate", "_check_coarse_boundaries"),
    "components": ("msp.annotate", "_components"),
    "guard_batch_annotation": ("msp.annotate", "_guard_batch_annotation"),
    "plot_annotated": ("msp.annotate", "_plot"),
    "validate_annotation": ("msp.annotate", "_validate_final"),
    "validate_cluster": ("msp.annotate", "_validate_cluster"),
    "save_single_umap": ("msp.plots", "save_single_umap"),
    "slug": ("msp.plots", "slug"),
    "compose_title": ("msp.report", "compose_title"),
    "generate_report": ("msp.report", "generate_report"),
}
__all__ = sorted(_NAMES)


def __getattr__(name):
    if name not in _NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _NAMES[name]
    return getattr(importlib.import_module(module), attribute)
