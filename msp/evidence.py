"""Evidence the agents query: precomputed DEG tables, live DEG, per-cluster
expression / QC / stability views, and the removal mask that keeps every
live test consistent with the precomputed CSVs.

Everything here is read-only over an msp output directory plus the loaded
``integrated.h5ad``; nothing writes back. ``msp.inspect`` and ``msp.annotate``
build their agent tools on top of these functions, and ``zmip`` reuses them
for its per-lineage sessions.
"""

from __future__ import annotations

import csv
import glob
import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from .deg_logging import rank_genes_groups

log = logging.getLogger(__name__)

QC_COLS = (
    "pct_counts_mt",
    "n_genes_by_counts",
    "total_counts",
    "doublet_score",
    "decontX_contamination",
    "dissociation_score",
    "pct_counts_malat1",
)


def load_removal_mask(outdir, ad):
    """Cells already proposed for removal before Cluster Annotations ran
    (msp.integrate._build_removal_mask / the Pre-annotation filtering
    UMAP) — excluded from every live DEG test here too (check_deg,
    subcluster's built-in sibling DE), matching the precomputed
    deg_global_*/deg_local_* CSVs exactly. Missing file (older msp output,
    predates this artifact) → nothing excluded. Boolean numpy array,
    aligned to ad.obs_names order."""
    path = os.path.join(outdir, "preannotation_removal.csv")
    if not os.path.exists(path):
        return np.zeros(ad.n_obs, dtype=bool)
    # Cell IDs are opaque strings: preserve leading zeros and literal "NA" IDs.
    df = pd.read_csv(path, dtype={"cell": str}, keep_default_na=False).set_index("cell")
    # Nullable boolean: cells absent from the file are not flagged, without object-dtype downcasting.
    flags = df["recommend_removal"].astype("boolean").reindex(ad.obs_names).fillna(False)
    return flags.to_numpy(dtype=bool)


def cluster_order(labels):
    """Distinct labels in a stable, numeric-aware order.

    Subcluster IDs are comma-joined ("5,10"); every comma-separated part is
    compared as a number, so "5,2" precedes "5,10". Labels that are not
    numeric throughout fall back to plain string order."""
    seen = list(dict.fromkeys(labels))  # first-seen order, any iterable
    try:
        return sorted(seen, key=lambda x: tuple(float(p) for p in str(x).split(",")))
    except ValueError:
        return sorted(seen)


def gene_table(ad, genes, cluster_key, cluster_ids=None):
    """Mean log-normalized expression and expressing fraction per cluster
    for the given genes (case-insensitive symbol match).

    cluster_ids optionally selects comparison clusters, preserving their order;
    omitted or empty retains the complete-table behavior for Python callers.
    """
    upper = {g.upper(): g for g in ad.var_names}
    found = {q: upper[q.upper()] for q in genes if q.upper() in upper}
    missing = [q for q in genes if q.upper() not in upper]
    if not found:
        return f"none of these genes are in var_names: {genes}"
    idx = [ad.var_names.get_loc(g) for g in found.values()]
    X = ad.X[:, idx]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    cl = ad.obs[cluster_key].astype(str)
    cols = {}
    selected = cluster_order(cl) if not cluster_ids else list(dict.fromkeys(map(str, cluster_ids)))
    unknown = sorted(set(selected) - set(cl))
    if unknown:
        return f"unknown cluster IDs: {unknown}; available: {cluster_order(cl)}"
    for c in selected:
        m = (cl == c).values
        mean = X[m].mean(axis=0)
        pct = 100 * (X[m] > 0).mean(axis=0)
        cols[c] = [f"{mn:.2f}|{p:.0f}%" for mn, p in zip(mean, pct, strict=True)]
    df = pd.DataFrame(cols, index=list(found.values()))
    out = "mean lognorm expr | pct expressing, per cluster:\n" + df.to_string()
    if missing:
        out += f"\nnot found in var_names: {missing}"
    return out


def qc_table(ad, cluster_key, batch_col):
    """Per-cluster QC (median|p90) + composition: n_samples, dominant-sample
    share, inherited flag/drop fractions — tests (b) and (c) in one view."""
    cols = [c for c in QC_COLS if c in ad.obs]
    cl = ad.obs[cluster_key].astype(str)
    rows = {}
    for c in cluster_order(cl):
        m = (cl == c).values
        sub = ad.obs.loc[m]
        row = [
            int(m.sum()),
            int(sub[batch_col].nunique()),
            f"{sub[batch_col].value_counts(normalize=True).iloc[0]:.2f}",
        ]
        if "_qc_action" in ad.obs:
            act = sub["_qc_action"].dropna().astype(str)
            row += [f"{(act == 'flag').mean():.2f}", f"{(act == 'drop').mean():.2f}"]
        row += [f"{sub[col].median():.3g}|{sub[col].quantile(0.9):.3g}" for col in cols]
        rows[c] = row
    index = ["n_cells", "n_samples", "max_sample_share"]
    if "_qc_action" in ad.obs:
        index += ["frac_flag_inherited", "frac_drop_inherited"]
    index += cols
    df = pd.DataFrame(rows, index=index).T
    df.index.name = cluster_key
    return (
        "per-cluster QC (median|p90) and composition "
        "(frac_* among cells with inherited QC; nan means unavailable):\n" + df.to_string()
    )


def stability_table(ad, cluster, cluster_key, other_keys):
    """How one cluster decomposes across the other clustering resolutions —
    test (e). A cluster that dissolves at neighboring resolutions is a
    one-resolution splinter."""
    m = (ad.obs[cluster_key].astype(str) == cluster).values
    if not m.any():
        return f"unknown cluster {cluster!r}"
    lines = [f"cluster {cluster} (n={int(m.sum())}) across resolutions:"]
    for k in other_keys:
        vc = ad.obs.loc[m, k].astype(str).value_counts()
        top = ", ".join(f"{i}:{v}" for i, v in vc.head(6).items())
        lines.append(f"  {k}: {top}")
        main = vc.index[0]
        rev = int((ad.obs[k].astype(str) == main).sum())
        lines.append(f"    (its main {k} group {main!r} has {rev} cells in total)")
    return "\n".join(lines)


def load_paga_neighbors(outdir, key):
    """{cluster: [top-3 PAGA neighbours]} as integrate wrote them (paga_neighbors_<key>.csv);
    {} when absent or empty (including headerless files from older outputs)."""
    path = os.path.join(outdir, f"paga_neighbors_{key}.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return {}
    nb = {}
    for c, g in df.groupby("cluster", sort=False):
        nb[str(c)] = list(g.sort_values("rank", key=lambda s: s.astype(int))["neighbor"].astype(str))
    return nb


def file_inventory(outdir):
    """Relative paths of every CSV and figure, for the system prompts."""
    patterns = ["*.csv", "figures/*.png"]
    paths = []
    for pat in patterns:
        paths += sorted(os.path.relpath(p, outdir) for p in glob.glob(os.path.join(outdir, pat)))
    return "\n".join(f"- {p}" for p in paths)


# ---------------------------------------------------------------- live DEG


def deg_frame(ad, cluster_key, cluster, ref_groups, remove_mask):
    """Live wilcoxon for one cluster of the CURRENT working clustering — the
    full ranked table (all genes, scanpy's natural score order), or None when
    the cluster has no cells left once remove_mask is excluded. ref_groups is
    "rest" (one-vs-rest, deg_global_* semantics) or a tuple of other cluster
    ids pooled (deg_local_* semantics). remove_mask cells (recommend_removal,
    see Pre-annotation filtering) are excluded first, same as the precomputed
    CSVs."""
    base = ad[~remove_mask]
    lab = base.obs[cluster_key].astype(str)
    if cluster not in set(lab):
        return None
    sub = base if ref_groups == "rest" else base[lab.isin([cluster, *ref_groups])].copy()
    rank_genes_groups(sub, cluster_key, groups=[cluster], reference="rest", method="wilcoxon", use_raw=True, pts=True)
    df = sc.get.rank_genes_groups_df(sub, group=cluster)
    # natural scanpy ranking (by test score), not resorted by raw logFC —
    # sorting by logFC alone surfaces near-zero-expression noise genes with
    # huge fold change but pct1==pct2==0 and padj==1, same trap as anywhere
    # else in msp that reads rank_genes_groups_df
    return df.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"}).reset_index(drop=True)


def filter_deg(df, min_logfc=None, max_padj=None, min_pct1=None, max_pct2=None):
    """Row filter shared by every DEG surface; None/0/1 defaults mean no filter."""
    m = np.ones(len(df), dtype=bool)
    if min_logfc:
        m &= df["logfoldchanges"].to_numpy() >= float(min_logfc)
    if max_padj is not None and 0 < float(max_padj) < 1:
        m &= df["pvals_adj"].to_numpy() <= float(max_padj)
    if min_pct1:
        m &= df["pct1"].to_numpy() >= float(min_pct1)
    if max_pct2 is not None and 0 < float(max_pct2) < 1:
        m &= df["pct2"].to_numpy() <= float(max_pct2)
    return df[m]


def filter_desc(min_logfc=None, max_padj=None, min_pct1=None, max_pct2=None):
    parts = []
    if min_logfc:
        parts.append(f"logFC>={float(min_logfc):g}")
    if max_padj is not None and 0 < float(max_padj) < 1:
        parts.append(f"padj<={float(max_padj):g}")
    if min_pct1:
        parts.append(f"pct1>={float(min_pct1):g}")
    if max_pct2 is not None and 0 < float(max_pct2) < 1:
        parts.append(f"pct2<={float(max_pct2):g}")
    return ", ".join(parts)


def format_deg(cluster, ref_desc, df, n_total=None, filters=""):
    """One line per gene, terse — gene logFC padj pct1/pct2 — so a 20-gene
    answer costs ~200 tokens, not 600; the header carries the comparison,
    the filters applied and how many genes passed."""
    head = f"DEG cluster {cluster} vs {ref_desc}"
    if filters:
        head += f" [{filters}]"
    head += f": {len(df)} gene(s)" + (f" of {n_total} passing" if n_total is not None and n_total != len(df) else "")
    head += " (gene logFC padj pct1/pct2, by wilcoxon score):"
    body = ", ".join(
        f"{r.names} {r.logfoldchanges:.1f} {r.pvals_adj:.0e} {r.pct1:.2f}/{r.pct2:.2f}"
        for r in df.itertuples(index=False)
    )
    return head + "\n  " + (body if len(df) else "(none)")


def parse_reference(reference, clusters=None):
    """Parse the existing string API without splitting an exact subcluster ID.

    CSV quoting disambiguates pooled IDs containing commas: '"5,0","5,1"'.
    Without a cluster vocabulary, retain the legacy comma-list interpretation.
    """
    reference = str(reference or "rest").strip() or "rest"
    if reference == "rest":
        return "rest"
    known = None if clusters is None else {str(c) for c in clusters}
    if known is not None and reference in known:
        return (reference,)
    if known is not None and '"' not in reference:
        parts = [p.strip() for p in reference.split(",")]
        if any(",".join(parts[i:j]) in known for i in range(len(parts)) for j in range(i + 2, len(parts) + 1)):
            raise ValueError('ambiguous reference; CSV-quote each ID, e.g. \'"5,0","5,1"\'')
    try:
        groups = tuple(
            sorted({g.strip() for g in next(csv.reader([reference], skipinitialspace=True, strict=True)) if g.strip()})
        )
    except csv.Error as exc:
        raise ValueError(f"invalid reference CSV: {exc}") from exc
    if not groups or (known is not None and any(g not in known for g in groups)):
        raise ValueError(
            'unknown reference cluster(s); use current IDs and CSV-quote IDs containing commas, e.g. \'"5,0","5,1"\''
        )
    return groups


def deg_table(ad, cluster_key, cluster, reference, top_n, remove_mask):
    """One-shot check_deg text (no cache) — kept for callers outside the
    agent loops; the agents go through DegCache."""
    ref = parse_reference(reference, ad.obs[cluster_key].astype(str).unique())
    df = deg_frame(ad, cluster_key, cluster, ref, remove_mask)
    if df is None:
        return f"cluster {cluster!r} has no cells left once recommend_removal cells are excluded"
    return format_deg(cluster, "rest" if ref == "rest" else ",".join(ref), df.head(top_n), len(df))


# ---------------------------------------------------------------- precomputed tables

DEG_TOOL_DOC = """Targeted retrieval over integrate's precomputed DEG tables (deg_global_*/deg_local_*, top-50 \
per cluster per view, every leiden key) — the CSVs themselves are larger than Read allows, use this instead. \
Selectors: cluster (its ranked markers), gene (which clusters have it among their top markers), view \
('global' = one-vs-rest, 'local' = vs the cluster's 3 pooled PAGA neighbours, 'both'), key (leiden key; \
default = the base key). Thresholds (0 / empty = off): min_logfc, max_padj, min_pct1, max_pct2 — e.g. \
min_logfc=1, max_padj=1e-10, max_pct2=0.3 returns just the specific positive markers. top_n per view \
(default 20). At least one of cluster/gene is required. For an arbitrary a-vs-b comparison use check_deg \
(same thresholds)."""

DEG_SQL_DOC = """Read-only SQL (one SELECT, ≤200 rows returned) over the working directory's tables: deg(key \
TEXT, view TEXT 'global'|'local', cluster TEXT, rank INTEGER 1=best, gene TEXT, logfc REAL, padj REAL, \
pct1 REAL, pct2 REAL, neighbors TEXT 'a|b|c' for local rows) = every precomputed DEG table, plus one table \
per other CSV in the directory named by its file stem (cluster_qc_msp_leiden_r1_0, paga_neighbors_..., \
stress_clusters, cell_outlier_summary, ...; non-alphanumeric characters in names/columns become '_'). \
Send the query 'schema' to list tables and columns. Example: SELECT cluster, gene, logfc FROM deg WHERE \
key='msp_leiden_r2.0' AND view='global' AND gene IN ('CD3D','MS4A1') ORDER BY cluster, rank"""


def _sql_name(text):
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text))


class DegTables:
    """integrate's precomputed DEG tables (deg_global_<key>.csv / deg_local_<key>.csv,
    every key present in outdir) loaded into an in-memory, read-only sqlite so the
    agent can retrieve exactly what it needs: Claude Code's Read refuses files
    over 25k tokens and a 40-cluster deg_global CSV is ~50k, so before this the
    agent could not read the tables it was told to read and fell back to 35-s
    check_deg calls. Three surfaces: lookup() (structured filters), sql() (one
    SELECT), markers_text() (a compact per-cluster summary cluster_context
    appends). The tables are evidence the agent already had on disk — no new
    computation, same numbers."""

    _COLS = ("key", "view", "cluster", "rank", "gene", "logfc", "padj", "pct1", "pct2", "neighbors")
    # Per-cell tables (one row per cell) are not evidence the agents query by
    # SQL, and loading them costs memory on every session — skip by file stem.
    PER_CELL_TABLES = frozenset({"cell_outliers", "preannotation_removal", "annotation_removed"})
    # Any other CSV above this size is listed in the schema as skipped instead of
    # being loaded: the summaries the agents need are small, and a directory that
    # accumulates large exports must not slow down every session.
    MAX_EXTRA_TABLE_BYTES = 64 << 20

    def __init__(self, outdir, base_key=None):
        import sqlite3

        self.base_key = base_key
        self.keys: list[str] = []
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE deg (key TEXT, view TEXT, cluster TEXT, rank INTEGER, gene TEXT, "
            "logfc REAL, padj REAL, pct1 REAL, pct2 REAL, neighbors TEXT)"
        )
        rows = []
        for path in sorted(glob.glob(os.path.join(outdir, "deg_*_*.csv"))):
            name = os.path.basename(path)[len("deg_") : -len(".csv")]
            view, key = name.split("_", 1)
            if view not in ("global", "local"):
                continue
            df = pd.read_csv(path, dtype={"group": str})
            if "group" not in df or df.empty:
                continue
            if key not in self.keys:
                self.keys.append(key)
            df = df.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"})
            for c, sub in df.groupby("group", sort=False):
                nbs = str(sub["neighbors"].iloc[0]) if "neighbors" in sub and pd.notna(sub["neighbors"].iloc[0]) else ""
                for rank, r in enumerate(sub.itertuples(index=False), 1):
                    rows.append(
                        (
                            key,
                            view,
                            str(c),
                            rank,
                            str(r.names),
                            float(r.logfoldchanges),
                            float(r.pvals_adj),
                            float(r.pct1),
                            float(r.pct2),
                            nbs,
                        )
                    )
        self.conn.executemany("INSERT INTO deg VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.execute("CREATE INDEX ix_ckv ON deg(key, view, cluster, rank)")
        self.conn.execute("CREATE INDEX ix_gene ON deg(gene)")
        self.n_rows = len(rows)
        # every other CSV in the directory (QC tables, paga_neighbors_*, stress_clusters,
        # fragments, ...) as its own table named by file stem — the agents reach for
        # them in SQL as soon as they see one table exists, and several are large
        self.extra_tables: dict[str, tuple[int, list[str]]] = {}  # name -> (rows, columns)
        self.skipped_tables: list[str] = []
        for path in sorted(glob.glob(os.path.join(outdir, "*.csv"))):
            stem = os.path.basename(path)[: -len(".csv")]
            if stem.startswith("deg_") or stem in self.PER_CELL_TABLES:
                continue
            name = _sql_name(stem)
            if os.path.getsize(path) > self.MAX_EXTRA_TABLE_BYTES:
                self.skipped_tables.append(name)
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or df.shape[1] == 0:
                continue
            df.columns = [_sql_name(c) or f"c{i}" for i, c in enumerate(df.columns)]
            df.to_sql(name, self.conn, index=False, if_exists="replace")
            self.extra_tables[name] = (len(df), list(df.columns))
        self.conn.commit()

        def _authorizer(action, *_):
            return (
                sqlite3.SQLITE_OK
                if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION)
                else sqlite3.SQLITE_DENY
            )

        self.conn.set_authorizer(_authorizer)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _fmt_rows(self, rows, header):
        """Grouped per (view, cluster): one terse line per group — gene #rank logFC padj pct1/pct2."""
        groups = {}
        for _key, view, cluster, rank, gene, logfc, padj, pct1, pct2, nbs in rows:
            ref = "rest" if view == "global" else "PAGA nbrs " + nbs.replace("|", ",")
            groups.setdefault((view, cluster, ref), []).append(
                f"{gene} #{rank} {logfc:.1f} {padj:.0e} {pct1:.2f}/{pct2:.2f}"
            )
        lines = [header]
        for (view, cluster, ref), items in groups.items():
            lines.append(f"  {view} cluster {cluster} vs {ref}: " + ", ".join(items))
        return "\n".join(lines)

    def lookup(
        self,
        cluster="",
        gene="",
        view="both",
        key="",
        top_n=20,
        min_logfc=None,
        max_padj=None,
        min_pct1=None,
        max_pct2=None,
    ):
        cluster, gene, key = str(cluster or "").strip(), str(gene or "").strip(), str(key or "").strip()
        view = str(view or "both").strip().lower() or "both"
        if view not in ("global", "local", "both"):
            return "view must be global | local | both"
        if not cluster and not gene:
            return "give cluster and/or gene"
        key = key or self.base_key or (self.keys[0] if self.keys else "")
        if key not in self.keys:
            return f"no precomputed tables for key {key!r}; available: {self.keys} (a subclustered key has none — use check_deg)"
        where, params = ["key = ?"], [key]
        if view != "both":
            where.append("view = ?")
            params.append(view)
        if cluster:
            where.append("cluster = ?")
            params.append(cluster)
        if gene:
            where.append("upper(gene) = ?")
            params.append(gene.upper())
        if min_logfc:
            where.append("logfc >= ?")
            params.append(float(min_logfc))
        if max_padj is not None and 0 < float(max_padj) < 1:
            where.append("padj <= ?")
            params.append(float(max_padj))
        if min_pct1:
            where.append("pct1 >= ?")
            params.append(float(min_pct1))
        if max_pct2 is not None and 0 < float(max_pct2) < 1:
            where.append("pct2 <= ?")
            params.append(float(max_pct2))
        filters = filter_desc(min_logfc, max_padj, min_pct1, max_pct2)
        top_n = max(1, min(int(top_n or 20), 200))
        sql = f"SELECT * FROM deg WHERE {' AND '.join(where)} ORDER BY view, rank"
        rows = self.conn.execute(sql, params).fetchall()
        n_total = len(rows)
        if cluster:  # per view, the best top_n of the rows that pass
            per_view, selected = {}, []
            for row in rows:
                count = per_view.get(row[1], 0)
                if count < top_n:
                    selected.append(row)
                    per_view[row[1]] = count + 1
            rows = selected
        else:
            rows = rows[:top_n]
        if not rows:
            what = f"cluster {cluster!r}" if cluster else f"gene {gene!r}"
            if gene and not cluster:
                return f"{what} is not among any cluster's top-50 markers in {key} ({view}); use check_genes for its expression per cluster"
            return f"nothing for {what} in {key} ({view}); clusters present: {self.clusters(key)}"
        head = (
            f"precomputed DEG, {key}, "
            + (f"cluster {cluster}" if cluster else f"gene {gene}")
            + (f" ∩ gene {gene}" if cluster and gene else "")
            + f", view={view}"
            + (f" [{filters}]" if filters else "")
            + f": {len(rows)} row(s)"
            + (f" of {n_total} passing" if n_total != len(rows) else "")
            + " (tables hold each cluster's top-50 per view; for other references or deeper lists use check_deg):"
        )
        return self._fmt_rows(rows, head)

    def schema_text(self):
        """One line per table for the tool description / a schema query."""
        lines = [f"deg ({self.n_rows} rows): key, view, cluster, rank, gene, logfc, padj, pct1, pct2, neighbors"]
        for name, (n, cols) in self.extra_tables.items():  # (PRAGMA is blocked by the read-only authorizer)
            lines.append(f"{name} ({n} rows): " + ", ".join(cols))
        if self.skipped_tables:
            lines.append(
                f"not loaded (over {self.MAX_EXTRA_TABLE_BYTES >> 20} MB; read the CSV directly): "
                + ", ".join(self.skipped_tables)
            )
        return "\n".join(lines)

    def clusters(self, key):
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT cluster FROM deg WHERE key = ? ORDER BY CAST(cluster AS REAL), cluster", [key]
            )
        ]

    def sql(self, query, max_rows=200):
        q = str(query or "").strip().rstrip(";").strip()
        if q.lower() in ("schema", "tables", ".tables", "show tables"):
            return self.schema_text()
        if not q.lower().startswith(("select", "with")):
            return "only a single SELECT is allowed (or 'schema' to list tables and columns)"
        if ";" in q:
            return "one statement only"
        try:
            cur = self.conn.execute(q)
            rows = cur.fetchmany(max_rows + 1)
        except Exception as exc:  # sqlite3 errors — feed the message back verbatim
            return f"SQL error: {exc}"
        cols = [d[0] for d in cur.description]
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        if not rows:
            return "no rows"
        df = pd.DataFrame(rows, columns=cols)
        out = df.to_string(index=False, float_format=lambda v: f"{v:.3g}")
        if truncated:
            out += f"\n... truncated at {max_rows} rows — narrow the query"
        return out

    def markers_text(self, key, cluster, n=12):
        """Two compact lines for cluster_context: top global and top local
        markers of one cluster from the precomputed tables ('' if absent)."""
        if key not in self.keys:
            return ""
        lines = []
        for view in ("global", "local"):
            rows = self.conn.execute(
                "SELECT gene, logfc, pct1, pct2, neighbors FROM deg WHERE key=? AND view=? AND cluster=? "
                "AND rank<=? ORDER BY rank",
                [key, view, str(cluster), n],
            ).fetchall()
            if not rows:
                continue
            ref = "rest" if view == "global" else "PAGA nbrs " + rows[0][4].replace("|", ",")
            lines.append(
                f"  top {view} markers (vs {ref}, precomputed; gene logFC pct1/pct2): "
                + ", ".join(f"{g} {lf:.1f} {p1:.2f}/{p2:.2f}" for g, lf, p1, p2, _ in rows)
            )
        return "\n".join(lines)


class DegCache:
    """check_deg for one agent session: every (clustering key, cluster,
    reference set) is computed at most once, and when the request is exactly
    what integrate already tabulated it is answered from disk instead of
    recomputed — one-vs-rest on an original key == deg_global_<key>.csv, and
    "vs its top-3 PAGA neighbours" == deg_local_<key>.csv, provided this
    session excludes exactly the cells those tables excluded (inspect always
    does; annotate/zmip only when inspect proposed no drops). A live
    one-vs-rest wilcoxon on 59k cells costs ~35 s, so a 40-cluster annotate
    session used to spend 5-10 min recomputing tables it could have read.
    Numbers are identical either way (same test, same exclusion, same
    use_raw, same ranking). If too few cached rows pass the requested filters,
    compute the full table before claiming that fewer genes are available."""

    def __init__(self, ad, outdir, remove_mask, label="check_deg"):
        self.ad, self.outdir, self.mask, self.label = ad, outdir, np.asarray(remove_mask, dtype=bool), label
        self._memo = {}  # (key, cluster, ref) -> (df, complete)
        self._csv = {}  # (key, view) -> DataFrame | None
        self._paga = {}  # key -> {cluster: [neighbours]}
        self.tables_usable = bool(np.array_equal(load_removal_mask(outdir, ad), self.mask))
        self.n_computed = self.n_precomputed = self.n_memo = 0

    def _csv_rows(self, key, view, cluster):
        k = (key, view)
        if k not in self._csv:
            path = os.path.join(self.outdir, f"deg_{view}_{key}.csv")
            self._csv[k] = pd.read_csv(path, dtype={"group": str}) if os.path.exists(path) else None
        df = self._csv[k]
        if df is None or "group" not in df:
            return None
        sub = df[df["group"].astype(str) == cluster]
        return sub.reset_index(drop=True) if len(sub) else None

    def _precomputed(self, key, cluster, ref):
        if not self.tables_usable:
            return None
        if ref == "rest":
            return self._csv_rows(key, "global", cluster)
        if key not in self._paga:
            self._paga[key] = load_paga_neighbors(self.outdir, key)
        nbs = self._paga[key].get(cluster)
        if not nbs or set(nbs) != set(ref):
            return None
        return self._csv_rows(key, "local", cluster)

    def table(self, key, cluster, reference, top_n, min_logfc=None, max_padj=None, min_pct1=None, max_pct2=None):
        ref = parse_reference(reference, self.ad.obs[key].astype(str).unique())
        mk = (key, cluster, ref)
        hit = self._memo.get(mk)
        if hit is not None and (hit[1] or top_n <= len(hit[0])):
            df, source = hit[0], "memo"
            self.n_memo += 1
        else:
            df = self._precomputed(key, cluster, ref) if hit is None else None
            if df is not None and top_n <= len(df):
                self._memo[mk] = (df, False)
                source = "precomputed"
                self.n_precomputed += 1
            else:
                df = deg_frame(self.ad, key, cluster, ref, self.mask)
                if df is None:
                    return f"cluster {cluster!r} has no cells left once recommend_removal cells are excluded"
                self._memo[mk] = (df, True)
                source = "computed"
                self.n_computed += 1
        ref_desc = "rest" if ref == "rest" else ",".join(ref)
        log.info(f"== [{self.label}] check_deg {cluster} vs {ref_desc}: {source}")
        kept = filter_deg(df, min_logfc, max_padj, min_pct1, max_pct2)
        if not self._memo[mk][1] and len(kept) < top_n:
            df = deg_frame(self.ad, key, cluster, ref, self.mask)
            if df is None:
                return f"cluster {cluster!r} has no cells left once recommend_removal cells are excluded"
            self._memo[mk] = (df, True)
            self.n_computed += 1
            kept = filter_deg(df, min_logfc, max_padj, min_pct1, max_pct2)
            log.info(f"== [{self.label}] check_deg {cluster} vs {ref_desc}: computed after filtering")
        complete = self._memo[mk][1]
        text = format_deg(
            cluster,
            ref_desc,
            kept.head(top_n),
            len(kept) if complete else None,
            filter_desc(min_logfc, max_padj, min_pct1, max_pct2),
        )
        return text if complete else text + "\n(cached ranked prefix; more genes may pass)"


# Lazy imports keep the evidence module usable during annotate/inspect import.
# These stable public entry points retain the existing 0.3 call signatures.
def prior_label_columns(ad, batch_col):
    """Return candidate prior label columns, excluding sample identities."""
    from .annotate import _prior_label_columns

    return _prior_label_columns(ad, batch_col)


def components(entries):
    """Return connected annotation merge components from cluster entries."""
    from .annotate import _components

    return _components(entries)


def palette(ad, col):
    """Assign the annotation palette for an observation column."""
    from .annotate import _palette

    return _palette(ad, col)


def plot_annotation(ad_full, ad_kept, figdir):
    """Render full and retained annotation views into a figure directory."""
    from .annotate import _plot

    return _plot(ad_full, ad_kept, figdir)


def subcluster_once(ad, key, cluster, resolution, new_key, remove_mask):
    """Split one cluster and report sibling markers excluding removed cells."""
    from .inspect import _subcluster_once

    return _subcluster_once(ad, key, cluster, resolution, new_key, remove_mask)
