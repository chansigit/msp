"""LocalEndpoint is a same-process, same-timing passthrough; resolve_endpoint()
never silently falls back to local for an unknown backend name. Needs no
optional dependency -- the Dask backends have their own test file, skipped
when dask[distributed] (msp-sc[dask]) is not installed."""

import pytest

from msp.compute import LocalEndpoint, resolve_endpoint


def test_local_endpoint_runs_synchronously_and_returns_the_result():
    ep = LocalEndpoint()
    with ep as scoped:
        fut = scoped.submit(lambda a, b, scale=1: (a + b) * scale, 2, 3, scale=10)
        assert fut.done()  # ran synchronously before submit() returned
        assert fut.result() == 50


def test_local_endpoint_surfaces_exceptions_through_the_future():
    ep = LocalEndpoint()

    def boom():
        raise ValueError("bad input")

    fut = ep.submit(boom)
    with pytest.raises(ValueError, match="bad input"):
        fut.result()


def test_resolve_endpoint_defaults_to_local(monkeypatch):
    monkeypatch.delenv("MSP_COMPUTE_ENDPOINT", raising=False)
    assert isinstance(resolve_endpoint(), LocalEndpoint)


def test_resolve_endpoint_honors_explicit_local(monkeypatch):
    monkeypatch.setenv("MSP_COMPUTE_ENDPOINT", "local")
    assert isinstance(resolve_endpoint(), LocalEndpoint)


def test_resolve_endpoint_rejects_an_unknown_name(monkeypatch):
    monkeypatch.setenv("MSP_COMPUTE_ENDPOINT", "quantum")
    with pytest.raises(ValueError, match="quantum"):
        resolve_endpoint()
