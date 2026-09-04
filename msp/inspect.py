"""
msp.inspect — per-cluster inspection of an msp integration directory with an
agent (msp.harness: HARNESS=claude|deepseek), mirroring osp.annotate's
architecture.

Every integrated cluster goes through five tests (the R11 battery):

  (a) markers     — own specific positive markers; ribo/mito/stress modules
                    with flat logFC → noise partition;
  (b) QC axis     — separation from neighbors along QC metrics → technical;
  (c) composition — multi-sample presence; single-sample dominance → batch
                    artifact (unless metadata explains it);
  (d) geometry    — sits between two populations with intermediate
                    signatures → doublet candidate;
  (e) stability   — persists across resolutions; one-resolution splinters
                    are not actionable.

Evidence on disk (deg_global_*/deg_local_* at both leiden resolutions,
cluster/per-sample QC tables, the standissect-lite bundle, figures) plus
live MCP tools (check_genes, check_qc_scores, check_stability, check_deg,
subcluster). The agent submits a
structured proposal; the host validates it, writes
inspection_proposal.json / inspection_notes.md, maps it onto
obs["_msp_action"] (keep/flag/drop) in integrated.h5ad, renders the
verdict UMAP and refreshes report.html.

Verdicts are PROPOSALS only — msp deletes nothing; execution is a later,
separate step. Cells flagged/dropped per-sample (obs["_qc_action"]) are in
here on purpose: their cross-sample clustering is exactly the evidence
this step weighs — check_genes/check_qc_scores/check_stability/subcluster's
own splitting still see them. Only DEG (check_deg, and subcluster's
built-in sibling DE) excludes preannotation_removal.csv's cells, matching
the precomputed deg_global_*/deg_local_* CSVs exactly.

Usage:
    python -m msp.inspect <msp_outdir> [--species human] [--model ...]
"""

import argparse
import asyncio
import glob
import json
import operator
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from .harness import default_model
from .report import generate_report

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}

QC_COLS = ("pct_counts_mt", "n_genes_by_counts", "total_counts", "doublet_score",
           "decontX_contamination", "dissociation_score", "pct_counts_malat1")


def _detect_primary_key(outdir):
    """Cluster Annotations computes DEG for both r1.0 and r2.0 (deg_global_*/
    deg_local_*.csv); default to r1.0 when both exist (matches the old
    single-resolution 'primary_key' convention), else whichever is present."""
    paths = sorted(glob.glob(os.path.join(outdir, "deg_global_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no deg_global_*.csv in {outdir} — run the msp pipeline first")
    keys = [os.path.basename(p)[len("deg_global_"):-len(".csv")] for p in paths]
    return "msp_leiden_r1.0" if "msp_leiden_r1.0" in keys else keys[0]


def _load_removal_mask(outdir, ad):
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
    df = pd.read_csv(path).set_index("cell")
    return df["recommend_removal"].reindex(ad.obs_names).fillna(False).to_numpy()


def _cluster_order(labels):
    seen = pd.unique(labels)
    try:
        return sorted(seen, key=lambda x: float(str(x).split(",")[0]))
    except ValueError:
        return sorted(seen)


def _gene_table(ad, genes, cluster_key):
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
    for c in _cluster_order(cl):
        m = (cl == c).values
        mean = X[m].mean(axis=0)
        pct = 100 * (X[m] > 0).mean(axis=0)
        cols[c] = [f"{mn:.2f}|{p:.0f}%" for mn, p in zip(mean, pct)]
    df = pd.DataFrame(cols, index=list(found.values()))
    out = "mean lognorm expr | pct expressing, per cluster:\n" + df.to_string()
    if missing:
        out += f"\nnot found in var_names: {missing}"
    return out


def _qc_table(ad, cluster_key, batch_col):
    """Per-cluster QC (median|p90) + composition: n_samples, dominant-sample
    share, inherited flag/drop fractions — tests (b) and (c) in one view."""
    cols = [c for c in QC_COLS if c in ad.obs]
    cl = ad.obs[cluster_key].astype(str)
    rows = {}
    for c in _cluster_order(cl):
        m = (cl == c).values
        sub = ad.obs.loc[m]
        row = [int(m.sum()), int(sub[batch_col].nunique()),
               f"{sub[batch_col].value_counts(normalize=True).iloc[0]:.2f}"]
        if "_qc_action" in ad.obs:
            act = sub["_qc_action"].astype(str)
            row += [f"{(act == 'flag').mean():.2f}", f"{(act == 'drop').mean():.2f}"]
        row += [f"{sub[col].median():.3g}|{sub[col].quantile(0.9):.3g}" for col in cols]
        rows[c] = row
    index = ["n_cells", "n_samples", "max_sample_share"]
    if "_qc_action" in ad.obs:
        index += ["frac_flag_inherited", "frac_drop_inherited"]
    index += cols
    df = pd.DataFrame(rows, index=index).T
    df.index.name = cluster_key
    return ("per-cluster QC (median|p90) and composition "
            "(frac_* inherited from per-sample annotation):\n" + df.to_string())


def _stability_table(ad, cluster, cluster_key, other_keys):
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


def _load_paga_neighbors(outdir, key):
    """{cluster: [top-3 PAGA neighbours]} as integrate wrote them (paga_neighbors_<key>.csv);
    {} when the file is absent (older output, or a key integrate never saw)."""
    path = os.path.join(outdir, f"paga_neighbors_{key}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str)
    nb = {}
    for c, g in df.groupby("cluster", sort=False):
        nb[str(c)] = list(g.sort_values("rank", key=lambda s: s.astype(int))["neighbor"].astype(str))
    return nb


def _deg_frame(ad, cluster_key, cluster, ref_groups, remove_mask):
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
    sc.tl.rank_genes_groups(sub, cluster_key, groups=[cluster], reference="rest",
                            method="wilcoxon", use_raw=True, pts=True)
    df = sc.get.rank_genes_groups_df(sub, group=cluster)
    # natural scanpy ranking (by test score), not resorted by raw logFC —
    # sorting by logFC alone surfaces near-zero-expression noise genes with
    # huge fold change but pct1==pct2==0 and padj==1, same trap as anywhere
    # else in msp that reads rank_genes_groups_df
    return df.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"}).reset_index(drop=True)


def _filter_deg(df, min_logfc=None, max_padj=None, min_pct1=None, max_pct2=None):
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


def _filter_desc(min_logfc=None, max_padj=None, min_pct1=None, max_pct2=None):
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


def _format_deg(cluster, ref_desc, df, n_total=None, filters=""):
    """One line per gene, terse — gene logFC padj pct1/pct2 — so a 20-gene
    answer costs ~200 tokens, not 600; the header carries the comparison,
    the filters applied and how many genes passed."""
    head = f"DEG cluster {cluster} vs {ref_desc}"
    if filters:
        head += f" [{filters}]"
    head += f": {len(df)} gene(s)" + (f" of {n_total} passing" if n_total is not None and n_total != len(df) else "")
    head += " (gene logFC padj pct1/pct2, by wilcoxon score):"
    body = ", ".join(f"{r.names} {r.logfoldchanges:.1f} {r.pvals_adj:.0e} {r.pct1:.2f}/{r.pct2:.2f}"
                     for r in df.itertuples(index=False))
    return head + "\n  " + (body if len(df) else "(none)")


def _parse_reference(reference):
    reference = str(reference or "rest").strip() or "rest"
    if reference == "rest":
        return "rest"
    return tuple(sorted({g.strip() for g in reference.split(",") if g.strip()}))


def _deg_table(ad, cluster_key, cluster, reference, top_n, remove_mask):
    """One-shot check_deg text (no cache) — kept for callers outside the
    agent loops; the agents go through DegCache."""
    ref = _parse_reference(reference)
    df = _deg_frame(ad, cluster_key, cluster, ref, remove_mask)
    if df is None:
        return f"cluster {cluster!r} has no cells left once recommend_removal cells are excluded"
    return _format_deg(cluster, "rest" if ref == "rest" else ",".join(ref), df.head(top_n), len(df))


_DEG_TOOL_DOC = """Targeted retrieval over integrate's precomputed DEG tables (deg_global_*/deg_local_*, top-50 \
per cluster per view, every leiden key) — the CSVs themselves are larger than Read allows, use this instead. \
Selectors: cluster (its ranked markers), gene (which clusters have it among their top markers), view \
('global' = one-vs-rest, 'local' = vs the cluster's 3 pooled PAGA neighbours, 'both'), key (leiden key; \
default = the base key). Thresholds (0 / empty = off): min_logfc, max_padj, min_pct1, max_pct2 — e.g. \
min_logfc=1, max_padj=1e-10, max_pct2=0.3 returns just the specific positive markers. top_n per view \
(default 20). At least one of cluster/gene is required. For an arbitrary a-vs-b comparison use check_deg \
(same thresholds)."""

_DEG_SQL_DOC = """Read-only SQL (one SELECT, ≤200 rows returned) over the working directory's tables: deg(key \
TEXT, view TEXT 'global'|'local', cluster TEXT, rank INTEGER 1=best, gene TEXT, logfc REAL, padj REAL, \
pct1 REAL, pct2 REAL, neighbors TEXT 'a|b|c' for local rows) = every precomputed DEG table, plus one table \
per other CSV in the directory named by its file stem (cluster_qc_msp_leiden_r1_0, paga_neighbors_..., \
stress_clusters, cell_outlier_summary, ...; non-alphanumeric characters in names/columns become '_'). \
Send the query 'schema' to list tables and columns. Example: SELECT cluster, gene, logfc FROM deg WHERE \
key='msp_leiden_r2.0' AND view='global' AND gene IN ('CD3D','MS4A1') ORDER BY cluster, rank"""


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

    def __init__(self, outdir, base_key=None):
        import glob as _glob
        import sqlite3

        self.base_key = base_key
        self.keys: list[str] = []
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("CREATE TABLE deg (key TEXT, view TEXT, cluster TEXT, rank INTEGER, gene TEXT, "
                          "logfc REAL, padj REAL, pct1 REAL, pct2 REAL, neighbors TEXT)")
        rows = []
        for path in sorted(_glob.glob(os.path.join(outdir, "deg_*_*.csv"))):
            name = os.path.basename(path)[len("deg_"):-len(".csv")]
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
                    rows.append((key, view, str(c), rank, str(r.names), float(r.logfoldchanges),
                                 float(r.pvals_adj), float(r.pct1), float(r.pct2), nbs))
        self.conn.executemany("INSERT INTO deg VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.execute("CREATE INDEX ix_ckv ON deg(key, view, cluster, rank)")
        self.conn.execute("CREATE INDEX ix_gene ON deg(gene)")
        self.n_rows = len(rows)
        # every other CSV in the directory (QC tables, paga_neighbors_*, stress_clusters,
        # fragments, ...) as its own table named by file stem — the agents reach for
        # them in SQL as soon as they see one table exists, and several are large
        self.extra_tables: dict[str, tuple[int, list[str]]] = {}  # name -> (rows, columns)
        for path in sorted(_glob.glob(os.path.join(outdir, "*.csv"))):
            stem = os.path.basename(path)[:-len(".csv")]
            if stem.startswith("deg_"):
                continue
            name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or df.shape[1] == 0:
                continue
            df.columns = ["".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(c)) or f"c{i}"
                          for i, c in enumerate(df.columns)]
            df.to_sql(name, self.conn, index=False, if_exists="replace")
            self.extra_tables[name] = (len(df), list(df.columns))
        self.conn.commit()

        def _authorizer(action, *_):
            return sqlite3.SQLITE_OK if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                                                   sqlite3.SQLITE_FUNCTION) else sqlite3.SQLITE_DENY
        self.conn.set_authorizer(_authorizer)

    def _fmt_rows(self, rows, header):
        """Grouped per (view, cluster): one terse line per group — gene #rank logFC padj pct1/pct2."""
        groups = {}
        for key, view, cluster, rank, gene, logfc, padj, pct1, pct2, nbs in rows:
            ref = "rest" if view == "global" else "PAGA nbrs " + nbs.replace("|", ",")
            groups.setdefault((view, cluster, ref), []).append(f"{gene} #{rank} {logfc:.1f} {padj:.0e} {pct1:.2f}/{pct2:.2f}")
        lines = [header]
        for (view, cluster, ref), items in groups.items():
            lines.append(f"  {view} cluster {cluster} vs {ref}: " + ", ".join(items))
        return "\n".join(lines)

    def lookup(self, cluster="", gene="", view="both", key="", top_n=20, min_logfc=None, max_padj=None,
               min_pct1=None, max_pct2=None):
        cluster, gene, key = str(cluster or "").strip(), str(gene or "").strip(), str(key or "").strip()
        view = (str(view or "both").strip().lower() or "both")
        if view not in ("global", "local", "both"):
            return "view must be global | local | both"
        if not cluster and not gene:
            return "give cluster and/or gene"
        key = key or self.base_key or (self.keys[0] if self.keys else "")
        if key not in self.keys:
            return f"no precomputed tables for key {key!r}; available: {self.keys} (a subclustered key has none — use check_deg)"
        where, params = ["key = ?"], [key]
        if view != "both":
            where.append("view = ?"); params.append(view)
        if cluster:
            where.append("cluster = ?"); params.append(cluster)
        if gene:
            where.append("upper(gene) = ?"); params.append(gene.upper())
        if min_logfc:
            where.append("logfc >= ?"); params.append(float(min_logfc))
        if max_padj is not None and 0 < float(max_padj) < 1:
            where.append("padj <= ?"); params.append(float(max_padj))
        if min_pct1:
            where.append("pct1 >= ?"); params.append(float(min_pct1))
        if max_pct2 is not None and 0 < float(max_pct2) < 1:
            where.append("pct2 <= ?"); params.append(float(max_pct2))
        filters = _filter_desc(min_logfc, max_padj, min_pct1, max_pct2)
        top_n = max(1, min(int(top_n or 20), 200))
        sql = f"SELECT * FROM deg WHERE {' AND '.join(where)} ORDER BY view, rank"
        rows = self.conn.execute(sql, params).fetchall()
        n_total = len(rows)
        if cluster:  # per view, the best top_n of the rows that pass
            per_view = {}
            rows = [r for r in rows if per_view.setdefault(r[1], []).append(r) or len(per_view[r[1]]) <= top_n]
        else:
            rows = rows[:top_n]
        if not rows:
            what = f"cluster {cluster!r}" if cluster else f"gene {gene!r}"
            if gene and not cluster:
                return f"{what} is not among any cluster's top-50 markers in {key} ({view}); use check_genes for its expression per cluster"
            return f"nothing for {what} in {key} ({view}); clusters present: {self.clusters(key)}"
        head = (f"precomputed DEG, {key}, " + (f"cluster {cluster}" if cluster else f"gene {gene}")
                + (f" ∩ gene {gene}" if cluster and gene else "") + f", view={view}"
                + (f" [{filters}]" if filters else "") + f": {len(rows)} row(s)"
                + (f" of {n_total} passing" if n_total != len(rows) else "")
                + " (tables hold each cluster's top-50 per view; for other references or deeper lists use check_deg):")
        return self._fmt_rows(rows, head)

    def schema_text(self):
        """One line per table for the tool description / a schema query."""
        lines = [f"deg ({self.n_rows} rows): key, view, cluster, rank, gene, logfc, padj, pct1, pct2, neighbors"]
        for name, (n, cols) in self.extra_tables.items():  # (PRAGMA is blocked by the read-only authorizer)
            lines.append(f"{name} ({n} rows): " + ", ".join(cols))
        return "\n".join(lines)

    def clusters(self, key):
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT cluster FROM deg WHERE key = ? ORDER BY CAST(cluster AS REAL), cluster", [key])]

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
                "AND rank<=? ORDER BY rank", [key, view, str(cluster), n]).fetchall()
            if not rows:
                continue
            ref = "rest" if view == "global" else "PAGA nbrs " + rows[0][4].replace("|", ",")
            lines.append(f"  top {view} markers (vs {ref}, precomputed; gene logFC pct1/pct2): "
                         + ", ".join(f"{g} {lf:.1f} {p1:.2f}/{p2:.2f}" for g, lf, p1, p2, _ in rows))
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
    use_raw, same ranking); only top_n beyond the tabulated 50 rows falls
    back to computing."""

    def __init__(self, ad, outdir, remove_mask, label="check_deg"):
        self.ad, self.outdir, self.mask, self.label = ad, outdir, np.asarray(remove_mask, dtype=bool), label
        self._memo = {}  # (key, cluster, ref) -> (df, complete)
        self._csv = {}   # (key, view) -> DataFrame | None
        self._paga = {}  # key -> {cluster: [neighbours]}
        self.tables_usable = bool(np.array_equal(_load_removal_mask(outdir, ad), self.mask))
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
            self._paga[key] = _load_paga_neighbors(self.outdir, key)
        nbs = self._paga[key].get(cluster)
        if not nbs or set(nbs) != set(ref):
            return None
        return self._csv_rows(key, "local", cluster)

    def table(self, key, cluster, reference, top_n, min_logfc=None, max_padj=None, min_pct1=None, max_pct2=None):
        ref = _parse_reference(reference)
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
                df = _deg_frame(self.ad, key, cluster, ref, self.mask)
                if df is None:
                    return f"cluster {cluster!r} has no cells left once recommend_removal cells are excluded"
                self._memo[mk] = (df, True)
                source = "computed"
                self.n_computed += 1
        ref_desc = "rest" if ref == "rest" else ",".join(ref)
        print(f"== [{self.label}] check_deg {cluster} vs {ref_desc}: {source}", flush=True)
        kept = _filter_deg(df, min_logfc, max_padj, min_pct1, max_pct2)
        return _format_deg(cluster, ref_desc, kept.head(top_n), len(kept),
                           _filter_desc(min_logfc, max_padj, min_pct1, max_pct2))


def _subcluster_once(ad, key, cluster, resolution, new_key, remove_mask):
    """Split one cluster; sizes reported are the FULL split (removed cells
    included, so counts stay honest), but the built-in sibling DE excludes
    remove_mask cells — same DEG-only exclusion as check_deg / the
    precomputed deg_global_*/deg_local_* CSVs."""
    parent_mask = (ad.obs[key].astype(str) == cluster).values
    sc.tl.leiden(ad, restrict_to=(key, [cluster]), resolution=resolution,
                 key_added=new_key, flavor="igraph", n_iterations=2)
    sub_labels = ad.obs[new_key][parent_mask].astype(str)
    subs = _cluster_order(sub_labels)
    if len(subs) < 2:
        del ad.obs[new_key]
        return 0, f"cluster {cluster} did not split at resolution {resolution}; try higher"
    sub = ad[parent_mask].copy()
    sub.obs["_sub"] = pd.Categorical(sub_labels.values)
    sub_clean = sub[~remove_mask[parent_mask]]
    top = None
    if sub_clean.obs["_sub"].nunique(dropna=True) >= 2:
        sc.tl.rank_genes_groups(sub_clean, "_sub", method="wilcoxon", use_raw=False)
        top = sc.get.rank_genes_groups_df(sub_clean, group=None).groupby("group", observed=True).head(10)
    sizes = sub_labels.value_counts()
    lines = [f"cluster {cluster} split into {len(subs)} subclusters at resolution {resolution}:"]
    for s in subs:
        genes = ", ".join(top.loc[top["group"] == s, "names"]) if top is not None else ""
        note = "" if genes else " (no DE — too few non-removed cells)"
        lines.append(f"  {s} (n={int(sizes[s])}) top genes vs siblings: {genes}{note}")
    return len(subs), "\n".join(lines)


_PROPOSAL_SCHEMA_DOC = """{
  "clusters": [
    {"cluster": "<id>", "verdict": "real|artifact-doublet|artifact-lowquality|artifact-batch|artifact-ambient|ambiguous",
     "action": "keep|flag|drop", "confidence": "high|medium|low",
     "tests": {"markers": "<test a finding>", "qc": "<test b>", "composition": "<test c>",
               "geometry": "<test d>", "stability": "<test e>"},
     "rationale": "<one or two sentences tying the tests to the verdict>"}
    // must cover EVERY cluster of the current clustering (incl. subcluster ids like "5,0")
  ],
  "cell_actions": [
    // optional finer-than-cluster records: only cells of that cluster matching metric op value
    {"cluster": "<id>", "metric": "<numeric obs column>", "op": ">|>=|<|<=", "value": 0.3,
     "action": "drop|flag", "reason": "doublet|ambient|debris|low-quality|other", "note": "<free text>"}
  ],
  "overall": "<overall assessment of the integration>"
}"""

_VERDICTS = ("real", "artifact-doublet", "artifact-lowquality", "artifact-batch",
             "artifact-ambient", "ambiguous")


def _validate_proposal(proposal, clusters, obs):
    problems = []
    entries = proposal.get("clusters")
    if not isinstance(entries, list) or not entries:
        problems.append('missing "clusters" list')
        entries = []
    for e in entries:
        missing = [k for k in ("cluster", "verdict", "action", "confidence", "tests", "rationale") if k not in e]
        if missing:
            problems.append(f"cluster entry missing {missing}: {e}")
            continue
        if e["verdict"] not in _VERDICTS:
            problems.append(f'verdict must be one of {_VERDICTS}: {e}')
        if e["action"] not in ("keep", "flag", "drop"):
            problems.append(f'action must be keep|flag|drop: {e}')
        if not all(k in e["tests"] for k in ("markers", "qc", "composition", "geometry", "stability")):
            problems.append(f'tests must cover markers/qc/composition/geometry/stability: {e}')
    covered = {str(e.get("cluster")) for e in entries}
    missed = [c for c in clusters if c not in covered]
    if missed:
        problems.append(f"clusters without a verdict: {missed}")
    for a in proposal.get("cell_actions", []):
        if str(a.get("cluster")) not in clusters:
            problems.append(f"cell_action cluster {a.get('cluster')!r} is not a current cluster id: {a}")
        metric = a.get("metric")
        if metric not in obs.columns or not pd.api.types.is_numeric_dtype(obs[metric]):
            problems.append(f'cell_action "metric" must be a numeric obs column: {a}')
        if a.get("op") not in _OPS:
            problems.append(f'cell_action "op" must be one of {sorted(_OPS)}: {a}')
        try:
            float(a.get("value"))
        except (TypeError, ValueError):
            problems.append(f'cell_action "value" must be numeric: {a}')
        if a.get("action") not in ("drop", "flag"):
            problems.append(f'cell_action "action" must be drop|flag: {a}')
    return problems


def _apply_proposal(ad, key, proposal):
    """obs["_msp_action"] in keep/flag/drop: cluster actions first, then
    cell_actions refine; within each pass flags before drops so drop wins."""
    lab = ad.obs[key].astype(str)
    action = np.array(["keep"] * ad.n_obs, dtype=object)
    for verb in ("flag", "drop"):
        for e in proposal["clusters"]:
            if e["action"] == verb:
                action[(lab == str(e["cluster"])).values] = verb
        for a in proposal.get("cell_actions", []):
            if a["action"] == verb:
                mask = (lab == str(a["cluster"])).values
                mask &= _OPS[a["op"]](ad.obs[a["metric"]].to_numpy(dtype=float), float(a["value"]))
                action[mask] = verb
    ad.obs["_msp_action"] = pd.Categorical(action, categories=["keep", "flag", "drop"])
    ad.obs["_msp_verdict"] = lab.map(
        {str(e["cluster"]): e["verdict"] for e in proposal["clusters"]}
    ).astype("category")


def _plot_verdicts(ad, figdir):
    from .plots import UMAP_DPI, umap_axes

    os.makedirs(figdir, exist_ok=True)
    xy = np.asarray(ad.obsm["X_umap"])
    act = ad.obs["_msp_action"].astype(str).values
    base = 120000 / ad.n_obs
    fig, ax = umap_axes(ad)
    for name, color, size in (("keep", "#d3d3d3", base), ("flag", "#b8860b", 1.5 * base),
                              ("drop", "#8b0000", 1.5 * base)):
        m = act == name
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=size, c=color, linewidths=0,
                       label=f"{name} (n={int(m.sum())})")
    ax.set_title("UMAP: inspection action (proposal)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(os.path.join(figdir, "inspect_umap_action.png"), dpi=UMAP_DPI)
    plt.close(fig)


def _file_inventory(outdir):
    patterns = ["*.csv", "figures/*.png"]
    paths = []
    for pat in patterns:
        paths += sorted(os.path.relpath(p, outdir) for p in glob.glob(os.path.join(outdir, pat)))
    return "\n".join(f"- {p}" for p in paths)


def _system_prompt(outdir, cluster_key, clusters, batch_col, species, language, n_batches=None):
    context = (f"Context — species: {species}." if species else
               "No species context was provided — infer cautiously and say so.")
    context += f" Sample/batch column: {batch_col!r}."
    if n_batches is not None and n_batches < 2:
        context += (" THIS DATASET HAS A SINGLE SAMPLE (harmony was skipped): test (c) composition carries no "
                    "information — n_samples=1 and share=1.00 are expected for every cluster, never evidence of "
                    "a batch artifact, and the verdict artifact-batch is unavailable; decide on (a), (b), (d), (e).")
    return f"""You are a single-cell RNA-seq integration QC expert. The working directory is an \
msp (multi-sample pipeline) integration output. Task: put EVERY integrated cluster \
({cluster_key}, {len(clusters)} clusters: {clusters}) through the five-test battery and submit \
a verdict + action per cluster.
{context}

The five tests (all five must be answered for every cluster):
(a) markers — own specific positive markers? ribo/mito/stress modules with flat logFC → noise;
(b) QC axis — separated from neighbors mainly along QC metrics (mt/doublet/contamination/depth) → technical;
(c) composition — cells from several samples, or dominated by one sample → batch artifact unless \
biology explains it; clusters rich in per-sample flag/drop cells that CO-CLUSTER across samples \
are the point of this integration — judge whether the cross-sample agreement confirms the artifact \
(e.g. shared doublet signature) or rescues the cells (a real rare population every sample flagged);
(d) geometry — between two populations with intermediate signatures → doublet candidate;
(e) stability — persists across resolutions (check_stability); one-resolution splinters are not actionable.

Standard verdict mapping: passes a+c+e → real/keep; fails a and trips b → artifact-lowquality; \
trips d → artifact-doublet; single-sample + no biological explanation → artifact-batch; \
unclear → ambiguous/flag (defer, never force). "drop" proposes removal, "flag" requests review — \
NOTHING is executed here; deletion is a later separate step, so be precise, not timid.

All relevant files (paths relative to the working directory — Read exactly these, no guessing):
{_file_inventory(outdir)}

What they are:
- deg_global_{{key}}.csv / deg_local_{{key}}.csv, for key in msp_leiden_r1.0 and msp_leiden_r2.0: \
precomputed DEG at both resolutions (global = one-vs-rest; local = vs the cluster's 3 nearest \
PAGA neighbors pooled; pct1/pct2 = expressing fraction in/out). They are TOO LARGE to Read whole — \
query them with deg_lookup (a cluster's ranked markers, or every cluster a gene marks) and deg_sql \
(one SELECT for anything else). These only cover the ORIGINAL clustering, though — once you \
subcluster, use check_deg for DEG on the refined ids;
- stress_clusters.csv: (key, cluster) pairs whose deg_global/deg_local top genes already trip the \
dissociation-stress/mitochondrial signature check — a head start, not a substitute for your own judgment;
- cluster_qc_*.csv, per_sample_qc.csv, cell_outlier_summary.csv: QC + composition tables (the \
latter is the same cluster-median-MAD-plus-floor doublet/ambient outlier test as check_qc_scores \
gives you live, precomputed at both resolutions);
- fragments_*.csv, minor_sibling_qc.csv: standissect-lite's cartesian product (leiden × UMAP-side \
clustering) and its per-fragment QC verdict; rows/fragments recommend_removal are candidate stray \
fragments of a parent cluster — detection only, judge each one yourself with the five tests \
(obs["standissect_product"] holds the per-cell fragment labels, so subcluster on a parent \
reproduces them);
- figures/standissect_product.png: the fragments on the UMAP; figures/umap_preannotation_removal.png: \
every cell already proposed for removal (standissect fragments ∪ cell-level outliers ∪ inherited \
osp _qc_action=="drop") BEFORE the precomputed deg_global/deg_local tables were computed — those \
tables already exclude these cells, so don't re-flag them for the same reasons found here;
- figures/umap_*.png: sample mixing, clusterings at three resolutions, inherited annotation; \
figures/qc_umap_*.png, figures/qc_violin_*.png: QC metrics on the UMAP / per-cluster violins, \
plus inherited keep/flag/drop.

Mandatory workflow:
1. Figures BEFORE conclusions: sample-mixing UMAP, the three resolution UMAPs, qc_umap_metrics, \
qc_umap_qc_action, umap_preannotation_removal.
2. deg_lookup(cluster=<id>) for every cluster of {cluster_key} (batch several calls in one turn; the \
CSVs themselves exceed the Read limit), then Read the QC/composition tables and the standissect \
diagnosis + drift tables.
3. Verify markers with check_genes (batch dozens of genes per call); QC/composition with \
check_qc_scores (one call, no arguments); stability with check_stability per suspicious cluster.
4. If a cluster is heterogeneous, split it with subcluster (ids like "5,0") — all tools and the \
final submission follow the refined clustering automatically. subcluster's own reply only DEs \
the new subclusters against each other (their immediate siblings); for a real global or local \
view on a subcluster (or on any custom comparison you want — a merged group of cluster ids, a \
specific pair), call check_deg — the precomputed deg_global_*/deg_local_* CSVs only cover the \
ORIGINAL clustering and cannot see ids you created.
5. Finish by calling submit_inspection — conclusions only in the submitted JSON.

Efficiency: parallel Reads in one turn; batch genes; check_qc_scores once; get the JSON right \
first try (schema in the tool description).

Principles: output language {language} (gene symbols excepted). Weak evidence → low confidence \
+ ambiguous/flag, never a forced guess. Distinguish genuine expression from ambient \
contamination (decontX evidence exists for that)."""


async def _run_agent(ad, outdir, cluster_key, other_keys, batch_col, species, language,
                     model, effort, max_turns, remove_mask):
    from .harness import ToolSpec, run_agent

    state = {"key": cluster_key, "n_sub": 0}
    deg = DegCache(ad, outdir, remove_mask, label="inspect")
    tables = DegTables(outdir, base_key=cluster_key)
    print(f"== precomputed DEG tables loaded: {tables.n_rows} rows for keys {tables.keys}", flush=True)

    async def deg_lookup(args):
        return {"content": [{"type": "text", "text": tables.lookup(
            args.get("cluster", ""), args.get("gene", ""), args.get("view", "both"), args.get("key", ""),
            args.get("top_n") or 20, args.get("min_logfc"), args.get("max_padj"), args.get("min_pct1"),
            args.get("max_pct2"))}]}

    async def deg_sql(args):
        return {"content": [{"type": "text", "text": tables.sql(args.get("query", ""))}]}

    def current_clusters():
        return _cluster_order(ad.obs[state["key"]].astype(str))

    async def check_genes(args):
        genes = args["genes"]
        if isinstance(genes, str):
            genes = [g for g in genes.replace(",", " ").split() if g]
        return {"content": [{"type": "text", "text": _gene_table(ad, genes, state["key"])}]}

    async def check_qc_scores(args):
        return {"content": [{"type": "text", "text": _qc_table(ad, state["key"], batch_col)}]}

    async def check_stability(args):
        return {"content": [{"type": "text",
                             "text": _stability_table(ad, str(args["cluster"]), state["key"], other_keys)}]}

    async def check_deg(args):
        c = str(args["cluster"])
        if c not in current_clusters():
            return {"content": [{"type": "text", "text": f"unknown cluster {c!r}; current: {current_clusters()}"}],
                    "is_error": True}
        reference = str(args.get("reference") or "rest").strip() or "rest"
        if reference != "rest":
            ref_groups = [g.strip() for g in reference.split(",") if g.strip()]
            unknown = [g for g in ref_groups if g not in current_clusters()]
            if unknown:
                return {"content": [{"type": "text",
                                     "text": f"unknown reference cluster(s) {unknown}; current: {current_clusters()}"}],
                        "is_error": True}
        return {"content": [{"type": "text", "text": deg.table(
            state["key"], c, reference, int(args.get("top_n") or 20), args.get("min_logfc"), args.get("max_padj"),
            args.get("min_pct1"), args.get("max_pct2"))}]}

    async def subcluster(args):
        c = str(args["cluster"])
        if c not in current_clusters():
            return {"content": [{"type": "text", "text": f"unknown cluster {c!r}; current: {current_clusters()}"}],
                    "is_error": True}
        new_key = f"inspect_sub{state['n_sub'] + 1}"
        n, text = _subcluster_once(ad, state["key"], c, float(args["resolution"]), new_key, remove_mask)
        if n >= 2:
            state["n_sub"] += 1
            state["key"] = new_key
            text += "\n(working clustering refined; all tools and the submission now use the new ids)"
        return {"content": [{"type": "text", "text": text}]}

    async def submit_inspection(args):
        try:
            proposal = json.loads(args["proposal_json"])
        except json.JSONDecodeError as e:
            return {"content": [{"type": "text", "text": f"JSON parse error, fix and resubmit: {e}"}],
                    "is_error": True}
        problems = _validate_proposal(proposal, current_clusters(), ad.obs)
        if problems:
            return {"content": [{"type": "text",
                                 "text": "validation failed, fix and resubmit:\n- " + "\n- ".join(problems)}],
                    "is_error": True}
        proposal["cluster_key"] = state["key"]
        path = os.path.join(outdir, "inspection_proposal.json")
        with open(path, "w") as fh:
            json.dump(proposal, fh, ensure_ascii=False, indent=2)
        return {"content": [{"type": "text", "text": f"saved to {path}"}], "_submitted": proposal}

    tools = [
        ToolSpec("deg_lookup", _DEG_TOOL_DOC,
                 {"cluster": str, "gene": str, "view": str, "key": str, "top_n": int, "min_logfc": float, "max_padj": float, "min_pct1": float, "max_pct2": float}, deg_lookup),
        ToolSpec("deg_sql", _DEG_SQL_DOC, {"query": str}, deg_sql),
        ToolSpec("check_genes",
                 "Per-cluster mean expression and expressing-cell fraction for the given genes "
                 "(case-insensitive). Use to verify markers.", {"genes": list}, check_genes),
        ToolSpec("check_qc_scores",
                 "Per-cluster QC (median|p90) + composition (n_samples, dominant-sample share, "
                 "inherited flag/drop fractions). No arguments.", {}, check_qc_scores),
        ToolSpec("check_stability",
                 "How one cluster decomposes across the other clustering resolutions — test (e). "
                 "A one-resolution splinter dissolves elsewhere.", {"cluster": str}, check_stability),
        ToolSpec("check_deg",
                 "On-demand DEG (wilcoxon) for the CURRENT working clustering, including any subcluster "
                 "splits already made — use this once a cluster you're investigating has an id the "
                 "precomputed deg_global_*/deg_local_* CSVs never saw. reference='rest' (default) is "
                 "one-vs-rest against every other current cluster, same semantics as deg_global_*. Pass "
                 "a comma-separated list of other cluster ids as reference instead for a pooled-group "
                 "comparison (e.g. a subcluster vs its siblings, or vs specific PAGA neighbors), same "
                 "semantics as deg_local_*. Thresholds (0/empty = off): min_logfc, max_padj, min_pct1, "
                 "max_pct2 — ask for exactly the gene list you need (e.g. min_logfc=1, max_padj=1e-10). "
                 "Results are cached per (cluster, reference); one-vs-rest on the base clustering is answered "
                 "from the precomputed table.",
                 {"cluster": str, "reference": str, "top_n": int, "min_logfc": float, "max_padj": float,
                  "min_pct1": float, "max_pct2": float}, check_deg),
        ToolSpec("subcluster",
                 "Split one heterogeneous cluster with leiden restrict_to at the given resolution "
                 '(0.3-1.0 typical). New ids look like "5,0"; all tools and the final submission '
                 "follow the refined clustering.", {"cluster": str, "resolution": float}, subcluster),
        ToolSpec("submit_inspection",
                 "Submit the final verdicts (mandatory; the run completes only after validation "
                 "passes). proposal_json is a JSON string with this schema:\n" + _PROPOSAL_SCHEMA_DOC,
                 {"proposal_json": str}, submit_inspection),
    ]
    result = await run_agent(
        tools=tools, submit_tool="submit_inspection",
        prompt="Inspect this msp integration directory following the workflow in the system "
               "prompt exactly, and finish by submitting via submit_inspection.",
        system_prompt=_system_prompt(outdir, cluster_key,
                                     _cluster_order(ad.obs[cluster_key].astype(str)),
                                     batch_col, species, language,
                                     n_batches=int(ad.obs[batch_col].nunique())),
        cwd=os.path.abspath(outdir), model=model, effort=effort, max_turns=max_turns,
        allowed_builtin=("read", "glob", "grep"), label="inspect",
        max_buffer_size=50_000_000,  # figure Reads exceed the 1MB default pipe buffer
    )
    if result.transcript_text:
        with open(os.path.join(outdir, "inspection_notes.md"), "w") as fh:
            fh.write(result.transcript_text)
    return result.submitted


def inspect_clusters(outdir, species=None, language="English", cluster_key=None,
                     model=None, effort=None, max_turns=100):
    """Run the per-cluster inspection agent on an msp output directory.

    Writes inspection_proposal.json + inspection_notes.md, maps the accepted
    proposal onto obs["_msp_action"]/obs["_msp_verdict"] in integrated.h5ad,
    renders the verdict UMAP, refreshes report.html. Returns the proposal.
    """
    ad = sc.read_h5ad(os.path.join(outdir, "integrated.h5ad"))
    cluster_key = cluster_key or _detect_primary_key(outdir)
    msp_meta = ad.uns.get("msp", {})
    batch_col = msp_meta.get("batch_col")
    if not batch_col:
        raise ValueError("integrated.h5ad lacks uns['msp']['batch_col'] — not an msp output?")
    other_keys = [k for k in ad.obs.columns
                  if k.startswith("msp_leiden_r") and k != cluster_key]
    species = species or (msp_meta.get("species") or None)

    remove_mask = _load_removal_mask(outdir, ad)
    print(f"== {int(remove_mask.sum())}/{ad.n_obs} cells already recommend_removal "
          "(pre-annotation filtering) — excluded from check_deg / subcluster DE", flush=True)

    proposal = asyncio.run(_run_agent(ad, outdir, cluster_key, other_keys, batch_col, species, language,
                                      model or default_model(), effort, max_turns, remove_mask))
    _apply_proposal(ad, proposal["cluster_key"], proposal)
    _plot_verdicts(ad, os.path.join(outdir, "figures"))
    tmp = os.path.join(outdir, "integrated.tmp.h5ad")
    ad.write_h5ad(tmp)
    os.replace(tmp, os.path.join(outdir, "integrated.h5ad"))
    print(f"== report refreshed: {generate_report(outdir)}", flush=True)
    return proposal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="msp.inspect", description=__doc__)
    parser.add_argument("outdir", help="msp integration output directory")
    parser.add_argument("--species", default=None, help="defaults to uns['msp']['species']")
    parser.add_argument("--language", default="English")
    parser.add_argument("--cluster-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args()

    proposal = inspect_clusters(args.outdir, species=args.species, language=args.language,
                                cluster_key=args.cluster_key, model=args.model,
                                effort=args.effort, max_turns=args.max_turns)
    for e in proposal["clusters"]:
        print(f"cluster {e['cluster']}: {e['verdict']} -> {e['action']} [{e['confidence']}]")
