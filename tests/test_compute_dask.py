"""DaskLocalEndpoint: a real single-node multi-process cluster, started and
torn down per `with` block. Skipped when dask[distributed] (msp-sc[dask])
is not installed -- it is an optional dependency, not a required one."""

import os

import numpy as np
import pytest

pytest.importorskip("distributed")

from msp.compute import DaskLocalEndpoint, resolve_endpoint  # noqa: E402


def test_dask_local_endpoint_runs_in_a_different_process():
    with DaskLocalEndpoint(n_workers=1) as ep:
        fut = ep.submit(lambda: os.getpid())
        worker_pid = fut.result()
    assert worker_pid != os.getpid()  # ran in the worker process, not this one


def test_dask_local_endpoint_round_trips_a_real_array():
    arr = np.arange(1000, dtype="float64")

    def stats(x):
        return float(x.sum()), float(x.mean())

    with DaskLocalEndpoint(n_workers=1) as ep:
        total, mean = ep.submit(stats, arr).result()
    assert total == arr.sum()
    assert mean == arr.mean()


def test_dask_local_endpoint_surfaces_exceptions_through_the_future():
    def boom():
        raise ValueError("bad input")

    with DaskLocalEndpoint(n_workers=1) as ep:
        fut = ep.submit(boom)
        with pytest.raises(ValueError, match="bad input"):
            fut.result()


def test_dask_local_endpoint_closes_the_cluster_on_exit():
    ep = DaskLocalEndpoint(n_workers=1)
    with ep:
        pass
    assert ep._client is None and ep._cluster is None
    with pytest.raises(RuntimeError):
        ep.submit(lambda: 1)


def test_resolve_endpoint_returns_dask_local(monkeypatch):
    monkeypatch.setenv("MSP_COMPUTE_ENDPOINT", "dask-local")
    ep = resolve_endpoint()
    assert isinstance(ep, DaskLocalEndpoint)
