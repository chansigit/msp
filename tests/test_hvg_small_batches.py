"""Per-batch HVG must survive batches with a handful of cells (lineage subsets)."""
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from msp.integrate import pipeline


def _adata(sizes, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    X = rng.poisson(rng.gamma(0.5, 2.0, size=(1, n_genes)), size=(n, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"sample": np.repeat([f"s{i}" for i in range(len(sizes))], sizes)})
    obs.index = [f"c{i}" for i in range(n)]
    a = ad.AnnData(X, obs=obs)
    a.layers["counts"] = a.X.copy()
    return a


def test_tiny_batches_are_excluded_from_the_vote():
    a = _adata([1, 2, 4, 150, 160])
    pipeline._preprocess(a, "sample", n_top_genes=50)
    assert a.var["highly_variable"].sum() == 50
    assert a.var["highly_variable_nbatches"].max() <= 2  # only the two eligible batches voted


def test_single_eligible_batch_falls_back_to_global():
    a = _adata([3, 5, 200])
    pipeline._preprocess(a, "sample", n_top_genes=40)
    assert a.var["highly_variable"].sum() == 40
    assert "highly_variable_nbatches" not in a.var


def test_all_batches_large_unchanged():
    a = _adata([100, 120])
    pipeline._preprocess(a, "sample", n_top_genes=30)
    assert a.var["highly_variable"].sum() == 30
    assert a.var["highly_variable_nbatches"].max() == 2


def test_old_behaviour_reproduces_the_crash(monkeypatch):
    monkeypatch.setattr(pipeline, "MIN_HVG_BATCH_CELLS", 0)
    with pytest.raises(IndexError):
        pipeline._preprocess(_adata([1, 2, 150, 160]), "sample", n_top_genes=50)
