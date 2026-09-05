"""Concurrent DEG metadata writes cannot invalidate local AnnData copies."""

from threading import Event, Lock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import msp.integrate.deg as D
import msp.resources as resources


@pytest.mark.parametrize("sparse_input", [False, True])
def test_parallel_deg_matches_serial_with_raw_axis_and_forced_overlap(tmp_path, monkeypatch, sparse_input):
    rng = np.random.default_rng(21)
    values = np.log1p(rng.poisson(3, (36, 7))).astype(float)
    if sparse_input:
        values = sparse.csr_matrix(values)
    raw = ad.AnnData(
        values,
        obs=pd.DataFrame({"k": pd.Categorical(np.repeat(["0", "1", "2"], 12))}, index=[f"c{i}" for i in range(36)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(7)]),
    )
    raw.uns["log1p"] = {"base": 2}
    data = raw[:, :3].copy()
    data.raw = raw
    before = data.raw.X.copy()
    workspace = D._global_deg_workspace(data)
    assert workspace.raw.var_names.tolist() == raw.var_names.tolist()
    assert workspace.uns["log1p"] == {"base": 2}
    assert workspace.X is data.X and workspace.raw.X is data.raw.X
    workspace.uns["log1p"]["base"] = 10
    assert data.uns["log1p"]["base"] == 2

    active = False
    armed = False
    global_started, copy_started, written = Event(), Event(), Event()
    selected = Lock()
    monkeypatch.setattr(D.sc.pp, "neighbors", lambda *args, **kwargs: None)

    def paga(subset, groups):
        nonlocal armed
        armed = active
        subset.uns["paga"] = {"connectivities": sparse.csr_matrix(np.ones((3, 3)) - np.eye(3))}

    monkeypatch.setattr(D.sc.tl, "paga", paga)
    monkeypatch.setattr(resources, "available_cpus", lambda: 1 if not active else 8)
    serial = tmp_path / "serial"
    serial.mkdir()
    D._cluster_annotations(data, np.zeros(36, bool), ["k"], [1.0], serial)
    original_rank, original_copy = D.rank_genes_groups, ad.AnnData.copy

    def rank(work, *args, **kwargs):
        if "key_added" in kwargs:
            global_started.set()
            assert copy_started.wait(10), "local copy never overlapped global computation"
            work.uns["_forced_global_result"] = True
            written.set()
        return original_rank(work, *args, **kwargs)

    def copy_with_overlap(view, *args, **kwargs):
        if armed and view.is_view and selected.acquire(blocking=False):
            assert global_started.wait(10), "global task did not start"
            # AnnData.copy deep-copies uns using dictionary iteration. Force a
            # global write between iterator creation and consumption: shared
            # metadata raises RuntimeError here, deterministically.
            iterator = iter(view._adata_ref.uns)
            next(iterator)
            copy_started.set()
            assert written.wait(10), "global metadata mutation did not occur"
            list(iterator)
        return original_copy(view, *args, **kwargs)

    monkeypatch.setattr(D, "rank_genes_groups", rank)
    monkeypatch.setattr(ad.AnnData, "copy", copy_with_overlap)
    active = True
    parallel = tmp_path / "parallel"
    parallel.mkdir()
    D._cluster_annotations(data, np.zeros(36, bool), ["k"], [1.0], parallel)
    assert copy_started.is_set() and written.is_set()
    assert {p.name for p in serial.iterdir()} == {p.name for p in parallel.iterdir()}
    for path in serial.iterdir():
        assert path.read_bytes() == (parallel / path.name).read_bytes(), path.name
    np.testing.assert_array_equal(
        data.raw.X.toarray() if sparse_input else data.raw.X, before.toarray() if sparse_input else before
    )
    assert set(data.uns) == {"log1p"}
