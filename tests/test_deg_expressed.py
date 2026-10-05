"""DEG leaves out genes no cell of the comparison expresses; the tables of the other genes do not change."""

import anndata as an
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from msp.integrate.deg import compute_deg_task

COLUMNS = ["scores", "logfoldchanges", "pvals", "pvals_adj", "pct1", "pct2"]


def data():
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.3, (300, 60)).astype(np.float32)
    counts[:, :10] = 0  # expressed by no cell
    counts[:, 10] = 0
    counts[:100, 10] = rng.poisson(3, 100) + 1  # on in a, off in b and c: the strongest marker
    counts[100:, 11:14] = 0  # expressed in a only, weakly
    counts[:200, 14] = 0  # expressed in c only: zero across a + b, the local comparison drops it
    obs = pd.DataFrame({"k": pd.Categorical(np.repeat(["a", "b", "c"], 100))}, index=[str(i) for i in range(300)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(60)])
    return an.AnnData(X=sparse.csr_matrix(np.log1p(counts)), obs=obs, var=var, uns={"log1p": {}})


def full_test(ad, **kwargs):
    """The test as it ran before: every gene."""
    work = ad.copy()
    sc.tl.rank_genes_groups(work, "k", method="wilcoxon", use_raw=False, pts=True, **kwargs)
    group = kwargs.get("groups", [None])[0] if "reference" in kwargs else None
    df = sc.get.rank_genes_groups_df(work, group=group)
    return df.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"})


def same(new, old):
    old = old[old["names"].isin(new["names"])]
    keys = ["group", "names"] if "group" in old else ["names"]
    new, old = new.sort_values(keys).reset_index(drop=True), old.sort_values(keys).reset_index(drop=True)
    assert list(new["names"]) == list(old["names"])
    for column in COLUMNS:
        assert np.allclose(new[column], old[column], equal_nan=True), column


@pytest.mark.parametrize("view", ["global", "local"])
def test_unexpressed_genes_are_left_out_and_the_rest_is_unchanged(view):
    ad = data()
    item = {"key": "k", "valid": ["a", "b", "c"], "top3": {"a": ["b"]}}
    if view == "global":
        new = compute_deg_task(ad, item)
        old = full_test(ad, groups=["a", "b", "c"])
        dropped = {f"g{i}" for i in range(10)}
    else:
        new = compute_deg_task(ad, item, "a")
        old = full_test(ad[ad.obs["k"].isin(["a", "b"])], groups=["a"], reference="rest")
        dropped = {f"g{i}" for i in range(10)} | {"g14"}
    assert set(old["names"]) - set(new["names"]) == dropped
    same(new, old)
    top = new[new["group"] == "a"] if "group" in new else new
    assert top.iloc[0]["names"] == "g10"  # one group expresses it, the other does not: kept, ranked first
    assert {"g11", "g12", "g13"} <= set(top["names"])
