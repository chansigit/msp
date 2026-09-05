import asyncio

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from msp.agent_tools import shared_tools
from msp.evidence import gene_table


def fixture():
    values = np.arange(77 * 40, dtype=float).reshape(77, 40) % 13
    return ad.AnnData(
        sparse.csr_matrix(values),
        obs=pd.DataFrame({"cluster": [str(i) for i in range(77)]}, index=[f"c{i}" for i in range(77)]),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(40)]),
    )


def query(obj, **args):
    tool = next(t for t in shared_tools(None, obj, lambda: "cluster", "Expression evidence") if t.name == "check_genes")
    return asyncio.run(tool.handler(args))


def test_large_all_cluster_query_is_explicit_error_not_partial_evidence():
    obj = fixture()
    result = query(obj, genes=list(obj.var_names), clusters=[])
    assert result["is_error"]
    text = result["content"][0]["text"]
    assert "no expression rows returned" in text and "77" in text
    assert len(text.encode()) < 16 * 1024


def test_targeted_expression_values_match_full_matrix_selected_columns():
    obj = fixture()
    genes = ["gene0", "Gene2"]
    result = query(obj, genes=genes, clusters=["76", "0", "76"])
    assert not result.get("is_error")
    expected = gene_table(obj, genes, "cluster", cluster_ids=["76", "0"])
    assert result["content"][0]["text"] == expected
    assert "11.00|100%" in expected  # row 76, gene 0 = 3040 % 13
    assert "0.00|0%" in expected


def test_legacy_path_only_genes_still_returns_small_complete_table():
    obj = fixture()[:2].copy()
    result = query(obj, genes=["Gene0"])
    assert result["content"][0]["text"] == gene_table(obj, ["Gene0"], "cluster")


def test_unknown_clusters_return_error_instead_of_empty_numeric_evidence():
    result = query(fixture(), genes=["Gene0"], clusters=["absent"])
    assert result["is_error"] and "unknown cluster IDs" in result["content"][0]["text"]


def test_invalid_cluster_selector_has_actionable_error():
    result = query(fixture(), genes=["Gene0"], clusters="0")
    assert result["is_error"] and "list" in result["content"][0]["text"]
