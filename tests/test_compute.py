"""LocalEndpoint is a same-process, same-timing passthrough; resolve_endpoint()
never silently falls back for a reserved-but-unimplemented backend name."""

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


@pytest.mark.parametrize("kind", ["dask-local", "dask-slurm"])
def test_resolve_endpoint_raises_not_implemented_for_reserved_names(monkeypatch, kind):
    monkeypatch.setenv("MSP_COMPUTE_ENDPOINT", kind)
    with pytest.raises(NotImplementedError, match=kind):
        resolve_endpoint()


def test_resolve_endpoint_rejects_an_unknown_name(monkeypatch):
    monkeypatch.setenv("MSP_COMPUTE_ENDPOINT", "quantum")
    with pytest.raises(ValueError, match="quantum"):
        resolve_endpoint()
