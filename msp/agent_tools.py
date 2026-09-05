"""Tool plumbing the inspect and annotate agents share.

Both agents expose the same three evidence tools over the precomputed DEG
tables and the loaded AnnData — ``deg_lookup``, ``deg_sql`` and
``check_genes`` — with identical argument schemas and identical handlers;
this module builds them once. ``check_deg`` stays per agent on purpose:
inspect validates cluster IDs against its live, possibly subclustered key,
annotate against the fixed base key.
"""

from __future__ import annotations

from harness_bridge import ToolSpec

from .evidence import DEG_SQL_DOC, DEG_TOOL_DOC, gene_table

# Argument schema shared by every DEG-returning tool (0 / empty = off).
DEG_FILTER_ARGS = {"top_n": int, "min_logfc": float, "max_padj": float, "min_pct1": float, "max_pct2": float}


def text_result(text, is_error=False):
    """The tool-result envelope the harness expects."""
    result: dict = {"content": [{"type": "text", "text": str(text)}]}
    if is_error:
        result["is_error"] = True
    return result


def deg_filters(args):
    """The four threshold arguments of a DEG tool call, as keyword arguments."""
    return {k: args.get(k) for k in ("min_logfc", "max_padj", "min_pct1", "max_pct2")}


def parse_gene_list(genes):
    """Accept a JSON list or a whitespace/comma separated string."""
    if isinstance(genes, str):
        return [g for g in genes.replace(",", " ").split() if g]
    return list(genes)


def shared_tools(tables, ad, current_key, check_genes_doc):
    """ToolSpecs for deg_lookup / deg_sql / check_genes.

    ``tables`` is a DegTables over the working directory; ``current_key`` is
    a zero-argument callable returning the clustering key check_genes should
    group by (inspect's changes after subclustering, annotate's is fixed)."""

    async def deg_lookup(args):
        return text_result(
            tables.lookup(
                cluster=args.get("cluster", ""),
                gene=args.get("gene", ""),
                view=args.get("view", "both"),
                key=args.get("key", ""),
                top_n=args.get("top_n") or 20,
                **deg_filters(args),
            )
        )

    async def deg_sql(args):
        return text_result(tables.sql(args.get("query", "")))

    async def check_genes(args):
        return text_result(gene_table(ad, parse_gene_list(args["genes"]), current_key()))

    return [
        ToolSpec(
            "deg_lookup",
            DEG_TOOL_DOC,
            {"cluster": str, "gene": str, "view": str, "key": str, **DEG_FILTER_ARGS},
            deg_lookup,
        ),
        ToolSpec("deg_sql", DEG_SQL_DOC, {"query": str}, deg_sql),
        ToolSpec("check_genes", check_genes_doc, {"genes": list}, check_genes),
    ]
