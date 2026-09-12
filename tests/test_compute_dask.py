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


# --- DaskEndpoint: attach to a scheduler we do not own -------------------

@pytest.fixture(scope="module")
def pool():
    from distributed import LocalCluster

    with LocalCluster(n_workers=1, threads_per_worker=1, processes=True) as cluster:
        yield cluster


def test_dask_endpoint_attaches_by_address_and_leaves_the_pool_running(pool):
    from msp.compute import DaskEndpoint

    with DaskEndpoint(pool.scheduler_address) as ep:
        assert ep.submit(lambda: os.getpid()).result() != os.getpid()
    assert pool.scheduler.status.name == "running"  # exit closed the client only
    with DaskEndpoint(pool.scheduler_address) as ep:  # pool still usable
        assert ep.submit(lambda x: x * 2, 21).result() == 42


def test_dask_endpoint_never_shares_results_between_identical_submits(pool):
    import time

    from msp.compute import DaskEndpoint

    with DaskEndpoint(pool.scheduler_address) as ep:
        a = ep.submit(time.perf_counter_ns)
        b = ep.submit(time.perf_counter_ns)
        assert a.key != b.key  # pure=False: identical calls are still separate tasks
        assert a.result() != b.result()


def test_dask_endpoint_attaches_by_scheduler_file(pool, tmp_path, monkeypatch):
    import json

    from msp.compute import DaskEndpoint

    f = tmp_path / "scheduler.json"
    f.write_text(json.dumps({"address": pool.scheduler_address}))
    monkeypatch.setenv("MSP_COMPUTE_ENDPOINT", "dask")
    monkeypatch.setenv("MSP_DASK_SCHEDULER", str(f))
    ep = resolve_endpoint()
    assert isinstance(ep, DaskEndpoint)
    with ep:
        assert ep.submit(sum, [1, 2, 3]).result() == 6


def test_dask_endpoint_requires_a_scheduler(monkeypatch):
    from msp.compute import DaskEndpoint

    monkeypatch.delenv("MSP_DASK_SCHEDULER", raising=False)
    with pytest.raises(ValueError, match="MSP_DASK_SCHEDULER"):
        with DaskEndpoint():
            pass
