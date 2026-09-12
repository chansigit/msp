"""Pluggable compute endpoint: decouples *what* a heavy step computes from
*where* it runs, symmetric to the inference-side split (``HARNESS`` /
``AGENT_MODEL_POOL`` in ``agent-harness-bridge``) that already exists.

Design record: docs/compute-endpoint-design.md. Backends, picked by
``MSP_COMPUTE_ENDPOINT``:

``local`` (default)
    same-process, same-timing wrapper around a direct call -- unset changes
    nothing about existing behavior.
``dask-local``
    a throwaway single-node ``distributed.LocalCluster``, one per heavy step.
``dask``
    attach to an *already running* Dask scheduler named by
    ``MSP_DASK_SCHEDULER`` (``tcp://host:port`` or a scheduler-file path) --
    the shared warm pool of eca-rsi#8. The pool outlives any one step or run;
    this backend only ever opens/closes a client to it.

An unknown name raises rather than silently falling back to ``local``, so a
misconfigured environment variable never passes unnoticed.

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


class DaskLocalEndpoint:
    """A single-node, multi-process ``distributed.LocalCluster`` -- no Slurm
    needed. Started fresh in ``__enter__`` and torn down in ``__exit__``,
    scoped to one heavy step rather than held open across a whole pipeline
    run, same as :class:`LocalEndpoint`. ``dask[distributed]`` is imported
    lazily here, not at module import time, so it is only ever required
    when this backend is actually selected (``msp-sc[dask]``).

    Real multi-process workers (not threads) on purpose: a numpy array
    handed to ``submit()`` really does cross a process boundary and get
    pickled/unpickled, which is the same constraint the ``dask`` backend
    (workers on other nodes) imposes. Threads would hide that."""

    def __init__(self, n_workers: int | None = None):
        self._n_workers = n_workers
        self._cluster = None
        self._client = None

    def __enter__(self) -> "DaskLocalEndpoint":
        try:
            from distributed import Client, LocalCluster
        except ImportError as exc:
            raise ImportError(
                "MSP_COMPUTE_ENDPOINT=dask-local needs dask[distributed]: "
                "pip install 'msp-sc[dask]'"
            ) from exc
        from .resources import available_cpus

        n_workers = self._n_workers or max(1, min(4, available_cpus()))
        self._cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True)
        self._client = Client(self._cluster)
        return self

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        if self._client is None:
            raise RuntimeError("DaskLocalEndpoint.submit() called outside its `with` block")
        return self._client.submit(fn, *args, **kwargs)

    def __exit__(self, *exc: Any) -> None:
        if self._client is not None:
            self._client.close()
        if self._cluster is not None:
            self._cluster.close()
        self._client = self._cluster = None


class DaskEndpoint:
    """Attach to a Dask scheduler somebody else started (the shared warm
    pool: one scheduler, workers in whatever Slurm allocations exist, any
    number of runs connecting concurrently). ``MSP_DASK_SCHEDULER`` is
    either ``tcp://host:port`` or the path of the scheduler file the
    scheduler wrote (``dask scheduler --scheduler-file F``; the natural
    form on a shared filesystem, since nobody has to know the node's IP).

    Only the client is opened/closed here -- the pool is not ours to tear
    down. Workers must import the same ``msp`` the client runs, because
    ``submit(fn)`` pickles module-level functions by reference: launch them
    with the same interpreter and ``PYTHONPATH`` (``container/dask-pool.sh``
    in eca-rsi does exactly that)."""

    def __init__(self, scheduler: str | None = None):
        self._scheduler = scheduler
        self._client = None

    def __enter__(self) -> "DaskEndpoint":
        try:
            from distributed import Client
        except ImportError as exc:
            raise ImportError(
                "MSP_COMPUTE_ENDPOINT=dask needs dask[distributed]: pip install 'msp-sc[dask]'"
            ) from exc
        target = self._scheduler or os.environ.get("MSP_DASK_SCHEDULER")
        if not target:
            raise ValueError(
                "MSP_COMPUTE_ENDPOINT=dask needs MSP_DASK_SCHEDULER "
                "(tcp://host:port or a scheduler-file path)"
            )
        if "://" in target:
            self._client = Client(target)
        else:
            self._client = Client(scheduler_file=target)
        return self

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        if self._client is None:
            raise RuntimeError("DaskEndpoint.submit() called outside its `with` block")
        return self._client.submit(fn, *args, **kwargs)

    def __exit__(self, *exc: Any) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None


def resolve_endpoint() -> ComputeEndpoint:
    """``MSP_COMPUTE_ENDPOINT`` (default ``local``) picks the backend. An
    unknown name raises loudly instead of falling back to ``local``."""
    kind = os.environ.get("MSP_COMPUTE_ENDPOINT", "local")
    if kind == "local":
        return LocalEndpoint()
    if kind == "dask-local":
        return DaskLocalEndpoint()
    if kind == "dask":
        return DaskEndpoint()
    raise ValueError(f"unknown MSP_COMPUTE_ENDPOINT={kind!r}")
