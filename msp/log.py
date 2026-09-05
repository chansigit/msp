"""Progress output for msp goes through ``logging`` (the ``msp`` logger family).

CLI entry points call :func:`configure` once; library callers get the same
stdout handler from :func:`ensure` when nothing else is configured. Both
also cover the ``harness_bridge`` family, so agent traces and pipeline lines
share one stream and one format.
"""

from __future__ import annotations

import logging

import harness_bridge


def configure(level: int | str = logging.INFO, stream=None) -> logging.Handler:
    """Route ``msp`` and ``harness_bridge`` records to ``stream`` (default:
    stdout), one flushed ``%(message)s`` line per record."""
    return harness_bridge.configure_logging("msp", level=level, stream=stream)


def ensure() -> None:
    """Attach the default stdout handler unless one is already reachable."""
    harness_bridge.ensure_logging("msp")
