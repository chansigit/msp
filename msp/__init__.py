"""msp — multi-sample pipeline: integrate osp per-sample outputs (harmony).

Strictly propose-only: no cell is deleted here. Per-cluster inspection
(an agent step) will live in msp.inspect as an optional module, mirroring
osp.annotate.
"""

from .integrate import load_and_merge, run_multi_sample_pipeline
from .report import generate_report

__all__ = ["load_and_merge", "run_multi_sample_pipeline", "generate_report"]
