"""Step invalidation and recovery without changing the public output formats.

Only the step currently being rerun needs a persistent pending marker. Its
old outputs and downstream outputs leave the active directory before work
starts. A crash leaves the marker in place, so neither resume nor report
rendering can mistake partial files for completed results. Use one writer
per output directory.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

STEPS = ("integrate", "inspect", "annotate")
_OUTPUTS = {
    "integrate": (
        "integrated.h5ad",
        "integrated.tmp.h5ad",
        "integration_summary.csv",
        "per_sample_qc.csv",
        "cluster_qc_*.csv",
        "fragments_*.csv",
        "overlap_*.csv",
        "minor_sibling_qc.csv",
        "cell_outliers.csv",
        "cell_outlier_summary.csv",
        "preannotation_removal.csv",
        "deg_global_*.csv",
        "deg_local_*.csv",
        "paga_neighbors_*.csv",
        "stress_clusters.csv",
        "de_parent_core_vs_core.csv",
        "fractal_*.csv",
        "figures/umap_*.png",
        "figures/qc_*.png",
        "figures/standissect_*.png",
        "figures/leiden_qc_violin_*.png",
        "figures/fractal_*.png",
    ),
    "inspect": (
        "inspection_proposal.json",
        "inspection_notes.md",
        "figures/inspect_*.png",
        "integrated.tmp.h5ad",
    ),
    "annotate": (
        "annotated.h5ad",
        "annotated.tmp.h5ad",
        "annotation_proposal.json",
        "annotation_notes.md",
        "annotation_*.csv",
        "figures/annotation_*.png",
    ),
}


def step_pending(outdir, step):
    """An interrupted ancestor also prevents this step from being current."""
    root = Path(outdir) / ".msp-state"
    return any((root / f"{name}.pending").exists() for name in STEPS[: STEPS.index(step) + 1])


def require_upstream_ready(outdir, step):
    """Refuse downstream work after an interrupted upstream rerun."""
    for upstream in STEPS[: STEPS.index(step)]:
        if step_pending(outdir, upstream):
            raise RuntimeError(f"{upstream} is incomplete in {outdir}; rerun it before {step}")


def begin_step(outdir, step):
    """Invalidate first, then archive only the outputs owned by affected steps.

    Caller inputs such as sample_decisions.csv, report_context.txt, and
    unrelated files stay in place. Inspection shares integrated.h5ad with
    integration: retain a snapshot in the archive while keeping it available
    as input. All H5AD writers replace files atomically, so a hard link is a
    safe snapshot and avoids copying a potentially large input file.
    """
    require_upstream_ready(outdir, step)
    root = Path(outdir)
    state = root / ".msp-state"
    state.mkdir(parents=True, exist_ok=True)
    (state / f"{step}.pending").touch()

    affected = STEPS[STEPS.index(step) :]
    paths = {root / "report.html", root / "report.html.tmp"}
    for name in affected:
        for pattern in _OUTPUTS[name]:
            paths.update(root.glob(pattern))
        if name != step:
            paths.add(state / f"{name}.pending")
    paths = {p for p in paths if p.is_file()}
    snapshot = root / "integrated.h5ad" if step == "inspect" else None
    if not paths and not (snapshot and snapshot.is_file()):
        return

    history = root / ".msp-history"
    history.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix=f"{step}-", dir=history))
    # Remove the old report first; it must not advertise invalidated results.
    for src in sorted(paths, key=lambda p: (p.name != "report.html", str(p))):
        dst = archive / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
    if snapshot and snapshot.is_file():
        dst = archive / "integrated.h5ad"
        try:
            os.link(snapshot, dst)
        except OSError:
            shutil.copy2(snapshot, dst)
    log.info(f"== [{step}] previous outputs archived in {archive}")


def complete_step(outdir, step):
    """Called only after the step's data, tables, and figures are written.

    Report rendering is independent: if it fails, the completed computation
    remains reusable and the report can be rebuilt without another agent run.
    """
    (Path(outdir) / ".msp-state" / f"{step}.pending").unlink()
