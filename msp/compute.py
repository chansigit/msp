"""Pluggable compute endpoint: decouples *what* a heavy step computes from
*where* it runs, symmetric to the inference-side split (``HARNESS`` /
``AGENT_MODEL_POOL`` in ``agent-harness-bridge``) that already exists.

Design record: docs/compute-endpoint-design.md. Only the ``local`` backend
is implemented; it is a same-process, same-timing wrapper around a direct
call, so ``MSP_COMPUTE_ENDPOINT`` unset (the default) changes nothing about
existing behavior. ``dask-local`` / ``dask-slurm`` are reserved names for
later work and raise ``NotImplementedError`` until built -- they are not a
silent fallback to ``local``, so a misconfigured environment variable never
passes unnoticed.

Callers never import a concrete backend; they call :func:`resolve_endpoint`
and use the result as a context manager scoped to one heavy step::

    with resolve_endpoint() as ep:
        fut = ep.submit(pure_fn, array_in, ...)
        result = fut.result()

``pure_fn`` must take and return plain data (numpy arrays, small pandas
objects) -- never an AnnData or other msp-internal object -- so a future
networked backend only ever serializes small, well-typed payloads.
"""
from __future__ import annotations

import os
from concurrent.futures import Future
from typing import Any, Callable, Protocol


class ComputeEndpoint(Protocol):
    """Shaped like ``concurrent.futures.Executor``: dask's ``Client``
    already implements ``submit() -> Future`` natively, so a Dask backend
    is close to a passthrough rather than an adapter layer."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future: ...
    def __enter__(self) -> "ComputeEndpoint": ...
    def __exit__(self, *exc: Any) -> None: ...


class LocalEndpoint:
    """Runs ``fn`` synchronously in-process. Same call, same thread, same
    timing as calling ``fn`` directly -- the default, and the only backend
    that needs no extra dependency."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # mirror Executor.submit: exceptions surface via .result()
            fut.set_exception(exc)
        return fut

    def __enter__(self) -> "LocalEndpoint":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


_RESERVED_NOT_YET_IMPLEMENTED = ("dask-local", "dask-slurm")


def resolve_endpoint() -> ComputeEndpoint:
    """``MSP_COMPUTE_ENDPOINT`` (default ``local``) picks the backend.
    Reserved names raise loudly instead of silently falling back to
    ``local``, so a typo or an unbuilt backend never passes unnoticed."""
    kind = os.environ.get("MSP_COMPUTE_ENDPOINT", "local")
    if kind == "local":
        return LocalEndpoint()
    if kind in _RESERVED_NOT_YET_IMPLEMENTED:
        raise NotImplementedError(f"MSP_COMPUTE_ENDPOINT={kind!r} is not implemented yet")
    raise ValueError(f"unknown MSP_COMPUTE_ENDPOINT={kind!r}")
