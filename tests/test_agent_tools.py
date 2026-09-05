"""The tool factory both agents share."""

import asyncio

import anndata as ad
import numpy as np
import pandas as pd

from msp.agent_tools import DEG_FILTER_ARGS, deg_filters, parse_gene_list, shared_tools, text_result
from msp.evidence import DegTables


def test_text_result_envelope():
    assert text_result("hello") == {"content": [{"type": "text", "text": "hello"}]}
    assert text_result(ValueError("bad"), is_error=True) == {
        "content": [{"type": "text", "text": "bad"}],
        "is_error": True,
    }


def test_argument_helpers():
    assert parse_gene_list("CD3D, MS4A1  LYZ") == ["CD3D", "MS4A1", "LYZ"]
    assert parse_gene_list(["a", "b"]) == ["a", "b"]
    assert deg_filters({"min_logfc": 1, "top_n": 5}) == {
        "min_logfc": 1,
        "max_padj": None,
        "min_pct1": None,
        "max_pct2": None,
    }
    assert set(DEG_FILTER_ARGS) == {"top_n", "min_logfc", "max_padj", "min_pct1", "max_pct2"}


def test_shared_tools_answer_from_tables_and_expression(tmp_path):
    pd.DataFrame(
        {
            "group": ["0", "0"],
            "names": ["G0", "G1"],
            "logfoldchanges": [0.5, 2.0],
            "pvals_adj": [0.01, 0.01],
            "pct1": [0.9, 0.9],
            "pct2": [0.1, 0.1],
        }
    ).to_csv(tmp_path / "deg_global_k.csv", index=False)
    data = ad.AnnData(
        np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame({"k": pd.Categorical(["0", "1"])}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["G0", "G1"]),
    )
    with DegTables(tmp_path, base_key="k") as tables:
        tools = shared_tools(tables, data, lambda: "k", "DOC")
        assert [t.name for t in tools] == ["deg_lookup", "deg_sql", "check_genes"]
        assert tools[2].description.startswith("DOC ")
        assert "16 KiB" in tools[2].description and "clusters" in tools[2].description
        handlers = {t.name: t.handler for t in tools}
        lookup = asyncio.run(handlers["deg_lookup"]({"cluster": "0", "min_logfc": 1}))["content"][0]["text"]
        assert "G1 #2" in lookup and "G0 #1" not in lookup
        assert "deg (2 rows)" in asyncio.run(handlers["deg_sql"]({"query": "schema"}))["content"][0]["text"]
        genes = asyncio.run(handlers["check_genes"]({"genes": "g1, missing"}))["content"][0]["text"]
        assert "G1" in genes and "not found in var_names: ['missing']" in genes
