from types import SimpleNamespace

import pandas as pd
import pytest

from msp.evidence import prior_label_columns


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
@pytest.mark.parametrize("nullable", [False, True])
def test_prior_labels_accept_string_storage_and_missing_values(dtype, nullable):
    values = ["T", "B", None if nullable else "T", "T", "B", None if nullable else "B"]
    obs = pd.DataFrame({"sample": ["a"] * 3 + ["b"] * 3})
    obs["prior"] = pd.Series(values, dtype=dtype)
    obs["source_identity"] = pd.Series(["a"] * 3 + ["b"] * 3, dtype=dtype)
    obs["boolean_label"] = pd.Series(["yes", "no", "yes"] * 2, dtype=dtype)
    assert prior_label_columns(SimpleNamespace(obs=obs), "sample") == ["prior"]


@pytest.mark.parametrize("dtype", ["int64", "float64", "Int64", "Float64", "boolean"])
def test_numeric_and_boolean_dtypes_are_not_prior_labels(dtype):
    values = [0, 1, 0, 1] if dtype == "boolean" else [2, 3, 2, 3]
    obs = pd.DataFrame({"sample": ["a", "a", "b", "b"], "value": pd.Series(values, dtype=dtype)})
    assert prior_label_columns(SimpleNamespace(obs=obs), "sample") == []


def test_inferred_string_dtype_remains_a_prior_label():
    obs = pd.DataFrame({"sample": ["a", "a", "b", "b"], "prior": ["T", "B", "T", "B"]})
    assert prior_label_columns(SimpleNamespace(obs=obs), "sample") == ["prior"]
