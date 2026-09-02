"""
msp HTML report generator — same design as osp's report:

  - Human: base64-inline images (self-contained file), osp's CSS, a sticky
    TOC that jumps between numbered sections.
  - Agent: every number in a figure also exists in a plain <table>; headers
    are plain <h2 id=...> text so grepping the raw HTML is enough.

Section order (a section is omitted, and the numbering closes up, when its
artifacts are absent — so the same generator renders every stage):
  1. Sample Summary        — merged size, samples, per-sample QC, sample decisions
  2. UMAPs                 — inherited annotation, samples, leiden resolutions,
                             standissect clusters, QC metrics
  3. Per-cluster QC        — standissect clusters: table, minor-sibling QC,
                             violins, fractal marker heatmap
  4. Leiden Cluster QC     — r1.0/r2.0 tabs: cluster table, cell-level
                             doublet/ambient outliers, per-cluster cutoff violins
  5. Cluster Annotations   — pre-annotation filtering UMAP, global/local DEG tabs
  6. Cell Type Annotation  — msp.annotate: coarse/fine UMAPs, removed cells,
                             per-cluster decisions, merged groups

Usage:
    python -m msp.report /path/to/msp_out [--out report.html]
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import html
import json
import os
import re

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 1400px; color: #1a1a1a; padding: 0 1rem; }
h1 { border-bottom: 2px solid #333; padding-bottom: .3rem; margin-bottom: .3rem; }
h2 { margin-top: 3rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; scroll-margin-top: 1rem; }
h3 { margin-top: 1.5rem; }
table { border-collapse: collapse; margin: .5rem 0 1rem 0; font-size: .9rem; }
th, td { border: 1px solid #ddd; padding: .3rem .6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th, tr th { background: #f2f2f2; }
tr:nth-child(even) { background: #fafafa; }
img.fig { max-width: 100%; display: block; border: 1px solid #ddd; }
.row { display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: flex-start; }
.row > div { flex: 1 1 45%; min-width: 320px; }
.row img.fig { width: 100%; }
.grid { display: flex; flex-wrap: wrap; gap: 1.2rem; margin: .5rem 0 1rem 0; }
.grid-item { flex: 0 0 330px; margin: 0; }
.grid-item img.fig { width: 330px; }
.grid-item figcaption { text-align: center; }
/* fixed height regardless of how many panels share the row, so a 2-panel
   and a 3-panel row render at the same visual scale */
.trio { display: flex; gap: 1.2rem; margin: .5rem 0 1rem 0; align-items: flex-start; flex-wrap: wrap; }
.trio figure { flex: 0 0 auto; margin: 0; }
.trio img.fig { height: 330px; width: auto; max-width: 100%; }
.trio figcaption { text-align: center; }
figure.natural img.fig { height: 330px; width: auto; max-width: 100%; }
figure { margin: 1rem 0; }
figcaption { font-size: .78rem; color: #555; margin-top: .25rem; }
#annotation-tables td { text-align: left; vertical-align: top; }
pre.notes { white-space: pre-wrap; font-size: .85rem; background: #f7f7f7; padding: .8rem; border-radius: 6px; }
.meta { color: #666; font-size: .9rem; margin: .2rem 0 1rem 0; }
.hint { color: #555; font-size: .88rem; margin: .3rem 0 1rem 0; max-width: 75ch; }
.tabset { margin: .5rem 0 1rem 0; }
.tab-input { display: none; }
.tab-label { display: inline-block; padding: .4rem .9rem; margin: 0 .3rem -1px 0; cursor: pointer;
             border: 1px solid #ccc; border-bottom: none; border-radius: 6px 6px 0 0;
             background: #f2f2f2; font-size: .9rem; color: #444; }
.tab-input:checked + .tab-label { background: #fff; border-color: #999; font-weight: 600; color: #1a1a1a; }
.tab-panels { border-top: 1px solid #999; padding-top: .8rem; }
.tab-panel { display: none; }
.layout { display: flex; gap: 4rem; align-items: flex-start; }
.content { flex: 1; min-width: 0; }
nav.toc { position: -webkit-sticky; position: sticky; top: 50%; transform: translateY(-50%);
          align-self: flex-start;
          z-index: 10; flex: 0 0 200px; max-height: calc(100vh - 2rem); overflow-y: auto;
          display: flex; flex-direction: column; gap: .5rem;
          background: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 6px;
          padding: .8rem 1rem; font-size: .9rem;
          box-shadow: 0 2px 4px rgba(0,0,0,.06); }
nav.toc a { color: #24578a; text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
nav.toc .toc-sub { display: flex; flex-direction: column; gap: .3rem; margin: -.15rem 0 .15rem 1rem; }
nav.toc .toc-sub a { color: #5a6b7a; font-size: .82rem; }
@media (max-width: 800px) {
  .layout { flex-direction: column; }
  nav.toc { position: static; transform: none; flex-direction: row; flex-wrap: wrap; width: auto; }
}
"""

# `position: sticky` in CSS is the fallback (and what mobile keeps), but it
# silently no-ops in some flex/ancestor configurations that are hard to spot
# from CSS alone; this pins the TOC with measured `position: fixed`
# coordinates instead, which floats unconditionally regardless of any
# ancestor's overflow/flex quirks. A same-width placeholder keeps .content
# from jumping left into the space the TOC vacated.
TOC_PIN_SCRIPT = """<script>
(function () {
  var toc = document.querySelector("nav.toc");
  if (!toc) return;
  var placeholder = document.createElement("div");
  placeholder.style.display = "none";
  toc.parentNode.insertBefore(placeholder, toc);

  function pin() {
    if (!window.matchMedia("(min-width: 801px)").matches) {
      toc.style.position = "";
      toc.style.left = "";
      toc.style.width = "";
      placeholder.style.display = "none";
      return;
    }
    placeholder.style.display = "none";
    toc.style.position = "";
    toc.style.left = "";
    toc.style.width = "";
    var rect = toc.getBoundingClientRect();
    placeholder.style.display = "block";
    placeholder.style.flex = "0 0 " + rect.width + "px";
    placeholder.style.width = rect.width + "px";
    toc.style.position = "fixed";
    toc.style.left = rect.left + "px";
    toc.style.width = rect.width + "px";
  }
  window.addEventListener("resize", pin);
  window.addEventListener("load", pin);
  pin();
})();
</script>"""

_SECTION_LABELS = {
    "sample-summary": "Sample Summary",
    "umaps": "UMAPs (integrated space)",
    "per-cluster-qc": "Per-cluster QC (standissect clusters)",
    "leiden-qc": "Leiden Cluster QC",
    "deg": "Cluster Annotations",
    "annotation": "Cell Type Annotation",
}


def _h2(anchor: str) -> str:
    return f'<h2 id="{anchor}">{_SECTION_LABELS[anchor]}</h2>'


def _img(path: str, cls: str = "") -> str:
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    ext = os.path.splitext(path)[1].lstrip(".") or "png"
    name = html.escape(os.path.basename(path))
    cls_attr = f' class="{cls}"' if cls else ""
    return (f'<figure{cls_attr}><img class="fig" src="data:image/{ext};base64,{b64}" '
            f'alt="{name}"><figcaption>{name}</figcaption></figure>')


def _tabs(group_id: str, items: list[tuple[str, str]]) -> str:
    """CSS-only tabset (radio input + label; no JS) — checked radio's panel
    is shown via an ID-selector rule that outranks the default .tab-panel
    { display:none }, no !important needed."""
    if not items:
        return ""
    inputs, labels, panels, rules = [], [], [], []
    for i, (label, content) in enumerate(items):
        tid = f"{group_id}-tab-{i}"
        checked = " checked" if i == 0 else ""
        inputs.append(f'<input type="radio" name="{group_id}" id="{tid}"{checked} class="tab-input">')
        labels.append(f'<label for="{tid}" class="tab-label">{html.escape(label)}</label>')
        panels.append(f'<div class="tab-panel" id="{tid}-panel">{content}</div>')
        rules.append(f"#{tid}:checked ~ .tab-panels #{tid}-panel")
    style = f"<style>{', '.join(rules)} {{ display: block; }}</style>"
    return (f'<div class="tabset">'
            + "".join(x for pair in zip(inputs, labels) for x in pair)
            + f'<div class="tab-panels">{"".join(panels)}</div></div>{style}')


def _grid(paths: list[str]) -> str:
    """Tile a family of single-metric panels into a grid (osp's report
    pattern: one file per plot, tiled here so the family reads as one view)."""
    if not paths:
        return ""
    return '<div class="grid">' + "".join(_img(p, cls="grid-item") for p in paths) + "</div>"


def _fmt_cell(header: str, value: str) -> str:
    """Display-only rounding — never touches the underlying CSV. Counts read
    as integers; other medians only need 2 decimals to be legible."""
    try:
        v = float(value)
    except ValueError:
        return html.escape(value)
    if "n_genes_by_counts" in header:
        return f"{v:.0f}"
    if header.startswith("median_"):
        return f"{v:.2f}"
    return html.escape(value)


def _csv_table(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt_cell(h, c)}</td>" for h, c in zip(rows[0], r)) + "</tr>"
        for r in rows[1:]
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _kv_table(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        rows = [r for r in csv.reader(f) if len(r) == 2 and r[0]]
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


# ---------------------------------------------------------------- sections


def _section_sample_decisions(outdir: str) -> str:
    path = os.path.join(outdir, "sample_decisions.csv")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n_incl = sum(1 for r in rows if r["decision"] == "include")
    head = "".join(f"<th>{c}</th>" for c in ("sample", "decision", "n_cells", "reason"))
    body = "".join(
        f'<tr style="color:{"#1a7f37" if r["decision"] == "include" else "#c0392b"}">'
        + "".join(f"<td>{html.escape(r.get(c, ''))}</td>" for c in ("sample", "decision", "n_cells", "reason"))
        + "</tr>"
        for r in rows
    )
    return (f"<h3>Sample inclusion ({n_incl}/{len(rows)} entered integration)</h3>"
            f"<table><tr>{head}</tr>{body}</table>")


def _section_sample_summary(outdir: str) -> str:
    summary_t = _kv_table(os.path.join(outdir, "integration_summary.csv"))
    qc_t = _csv_table(os.path.join(outdir, "per_sample_qc.csv"))
    if not summary_t and not qc_t:
        return ""
    parts = [_h2("sample-summary")]
    if summary_t:
        parts.append(summary_t)
    parts.append(_section_sample_decisions(outdir))
    if qc_t:
        parts += ["<h3>Per-sample QC</h3>", qc_t]
    return "".join(parts)


def _section_minor_sibling(outdir: str) -> str:
    path = os.path.join(outdir, "minor_sibling_qc.csv")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ""
    metric_names = sorted({c[: -len("_median")] for c in rows[0] if c.endswith("_median")})
    head_cols = ("subcluster", "parent", "n_cells", "core_n_cells", "frac_of_core", "status",
                 "pct_drop_upstream", "n_hits", "recommend_removal")
    head = "".join(f"<th>{c}</th>" for c in head_cols)
    body_rows = []
    for r in rows:
        if r.get("recommend_removal") == "True":
            style = ' style="color:#8b0000;font-weight:bold"'
        elif r["status"] == "big_sibling_skip":
            style = ' style="color:#888"'
        else:
            style = ""

        def _cell(c, v=r):
            val = v.get(c) or ""
            if c == "n_hits" and val:
                val = str(int(float(val)))
            return f"<td>{html.escape(val)}</td>"

        cells = "".join(_cell(c) for c in head_cols)
        body_rows.append(f"<tr{style}>{cells}</tr>")
    table = f"<table><tr>{head}</tr>{''.join(body_rows)}</table>"

    detail_head = "".join(f"<th>{m}_median</th><th>{m}_significant</th>" for m in metric_names)
    detail_rows = "".join(
        "<tr><td>" + html.escape(r["subcluster"]) + "</td>"
        + "".join(f"<td>{html.escape(r.get(f'{m}_median') or '')}</td>"
                  f"<td>{html.escape(r.get(f'{m}_significant') or '')}</td>" for m in metric_names)
        + "</tr>"
        for r in rows if r["status"] == "tested"
    )
    details = ("<details><summary>per-metric values (tested siblings only)</summary>"
               f"<table><tr><th>subcluster</th>{detail_head}</tr>{detail_rows}</table></details>")

    n_removal = sum(1 for r in rows if r.get("recommend_removal") == "True")
    n_tested = sum(1 for r in rows if r["status"] == "tested")
    hint = (f"<p class=\"hint\">Each minor sibling (standissect fragment, rank&gt;0), other than "
            "\"big\" ones, is checked against two kinds of criteria — hitting ANY ONE marks the "
            "whole fragment recommend_removal (dark red bold below; a candidate for msp.inspect "
            "to verify, not an automatic removal). (1) upstream: &gt;50% of the sibling's cells "
            "already carry _qc_action=\"drop\" from the per-sample annotation. (2) stats "
            "(only run when the sibling has &ge;5 cells): one-sided Mann-Whitney U, sibling vs. "
            "the pooled parent-core cells, p&lt;0.05, no multiple-testing correction, on "
            "decontX_contamination / dissociation_score / doublet_score / pct_counts_mt — the "
            "latter two also require the sibling's own median above an absolute floor (0.2 and "
            "20% respectively). Siblings holding ≥25% of their own parent core's cell count, or "
            "≥800 cells outright, are skipped as \"big\", not minor. "
            f"{n_removal}/{len(rows)} siblings recommend_removal, {n_tested} stats-tested.</p>")
    return f"<h3>StanDissect Minor sibling fractals QC</h3>{hint}{table}{details}"


def _section_per_cluster(outdir: str, violins: list[str], fractal_figs: list[str]) -> str:
    parts = []
    for p in sorted(glob.glob(os.path.join(outdir, "cluster_qc_standissect_product.csv"))):
        key = os.path.basename(p)[len("cluster_qc_"):-len(".csv")]
        parts += [f"<h3>{html.escape(key)}</h3>",
                  '<p class="hint">flag/drop fractions carried over from per-sample '
                  "annotation; a cluster fed by a single sample is itself a signal.</p>",
                  _csv_table(p)]
    parts.append(_section_minor_sibling(outdir))
    if violins:
        parts += ["<h3>StanDissect Per-cluster QC violins</h3>",
                  '<p class="hint">Grouped by the standissect clusters (leiden × UMAP-fragment '
                  "product), not the primary leiden — so a minor sibling's QC profile can be "
                  "compared against its main core.</p>",
                  _grid(violins)]
    parts.append(_section_fractal_heatmap(outdir, fractal_figs))

    return _h2("per-cluster-qc") + "".join(parts) if parts else ""


def _section_leiden_qc(outdir: str, leiden_violin_figs: list[str]) -> str:
    """QC at the primary leiden resolutions (r1.0/r2.0) — distinct from the
    standissect-fragment QC above (finer, detection-oriented); this is what
    Cluster Annotations (DEG, PAGA, stress flags) below actually keys off.
    One tab per resolution: the cluster_qc table, the cell-level
    doublet/ambient outlier summary, and per-cluster cutoff violins."""
    qc_paths = sorted(glob.glob(os.path.join(outdir, "cluster_qc_msp_leiden_r*.csv")))
    if not qc_paths:
        return ""
    outlier_path = os.path.join(outdir, "cell_outlier_summary.csv")
    outlier_rows: list[dict] = []
    if os.path.exists(outlier_path):
        with open(outlier_path) as f:
            outlier_rows = list(csv.DictReader(f))

    parts = ['<p class="hint">Same cells as Per-cluster QC above, aggregated by the primary '
             "leiden clustering instead of standissect fragments. Includes the cell-level "
             "doublet/ambient-RNA outlier test: a cell is an outlier for a metric "
             "(doublet_score or decontX_contamination) only if it clears BOTH gates together — "
             "cluster median + 3×MAD, AND an absolute floor of 0.5 (the MAD rule alone is too "
             "permissive in near-clean clusters where MAD itself is tiny); recommend_removal is "
             "the OR across both metrics. Red bar on each violin = that cluster's cutoff "
             "(max of median+3×MAD and the floor); propose-only, see cell_outliers.csv for the "
             "full per-cell table.</p>"]
    tabs = []
    for p in qc_paths:
        key = os.path.basename(p)[len("cluster_qc_"):-len(".csv")]
        tab_parts = [_csv_table(p)]
        krows = [r for r in outlier_rows if r.get("key") == key]
        if krows:
            head_cols = list(krows[0].keys())
            head = "".join(f"<th>{html.escape(c)}</th>" for c in head_cols)
            body = "".join(
                "<tr>" + "".join(f"<td>{html.escape(r.get(c, ''))}</td>" for c in head_cols) + "</tr>"
                for r in krows
            )
            tab_parts += ["<h4>Cell-level doublet / ambient-RNA outliers</h4>",
                          f"<table><tr>{head}</tr>{body}</table>"]
        key_figs = [fp for fp in leiden_violin_figs if os.path.basename(fp).endswith(f"_{key}.png")]
        if key_figs:
            tab_parts += ["<h4>Per-cluster cutoff violins</h4>", _grid(key_figs)]
        tabs.append((key, "".join(tab_parts)))
    parts.append(_tabs("leidenqc", tabs))
    return _h2("leiden-qc") + "".join(parts)


def _section_fractal_heatmap(outdir: str, fractal_figs: list[str]) -> str:
    path = os.path.join(outdir, "fractal_markers.csv")
    if not os.path.exists(path) and not fractal_figs:
        return ""
    parts = ["<h3>Fractal marker dot plot</h3>",
             '<p class="hint">Transcriptomic evidence for what a fractal actually is — e.g. a '
             "doublet fractal co-expressing two parents' marker sets shows up as a column lit up "
             "in both parents' colored rows, not just one. Each parent's CORE cells (rank 0) DE'd "
             "one-vs-rest against every other parent's core cells (core-only, so minor siblings "
             "never leak into either side); "
             "per parent, top 10 markers with logFC&gt;0, padj&lt;0.05, ribosomal genes excluded. "
             "Dot plot across every standissect cluster (cores AND fractals): dot size = fraction "
             "of a cluster's cells expressing the gene, dot color = row-wise z-score of average "
             "log1p expression. Columns are clustered (optimal leaf ordering) purely to order them "
             "— no dendrogram shown. Rows are grouped by the parent they're a marker for (colored "
             "strip), not clustered. Bold column labels = parent-core clusters; red = "
             "recommend_removal (see StanDissect Minor sibling fractals QC above).</p>"]
    parts += [_img(p) for p in fractal_figs]
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        body = "".join(
            "<tr><td>" + "</td><td>".join(html.escape(r[c]) for c in
                                           ("parent", "gene", "rank", "logfoldchange", "pvals_adj")) + "</td></tr>"
            for r in rows
        )
        parts.append("<details><summary>marker gene list</summary>"
                     "<table><tr><th>parent</th><th>gene</th><th>rank</th><th>logfoldchange</th>"
                     f"<th>pvals_adj</th></tr>{body}</table></details>")
    return "".join(parts)


def _load_stress_lookup(outdir: str) -> dict[tuple[str, str, str], dict]:
    """stress_clusters.csv, written by msp.integrate._cluster_annotations —
    the report only reads/displays it, computation lives with the rest of
    the pipeline. Keyed (key, cluster, view); recommend_removal is already
    merged across global+local for that (key, cluster) there."""
    path = os.path.join(outdir, "stress_clusters.csv")
    lookup: dict[tuple[str, str, str], dict] = {}
    if not os.path.exists(path):
        return lookup
    with open(path) as f:
        for r in csv.DictReader(f):
            lookup[(r["key"], r["cluster"], r["view"])] = {
                "hit_genes": set(r["hit_genes"].split("|")) if r["hit_genes"] else set(),
                "recommend_removal": r["recommend_removal"] == "True",
            }
    return lookup


def _deg_row(cluster: str, genes: list[tuple[str, float]], stress_info: dict | None,
             middle_td: str = "") -> str:
    info = stress_info or {}
    hit_genes = info.get("hit_genes", set())
    recommend_removal = info.get("recommend_removal", False)
    cluster_cell = html.escape(cluster)
    if recommend_removal:
        cluster_cell += (' <span style="color:#c0392b;font-weight:bold" title="stress signature '
                         'in this view or its global/local pair — see Cluster Annotations hint">'
                         "[recommend_removal]</span>")
    gene_html = ", ".join(
        f'<b style="color:#c0392b">{html.escape(name)}</b> ({lfc:.1f})' if name in hit_genes
        else f"{html.escape(name)} ({lfc:.1f})"
        for name, lfc in genes
    )
    row_style = ' style="background:#fdecea"' if recommend_removal else ""
    return f"<tr{row_style}><td>{cluster_cell}</td>{middle_td}<td style='text-align:left'>{gene_html}</td></tr>"


def _deg_global_table(path: str, top_n: int, key: str, stress_lookup: dict) -> str:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_group: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        g = by_group.setdefault(r["group"], [])
        if len(g) < top_n:
            g.append((r["names"], float(r["logfoldchanges"])))
    body = "".join(
        _deg_row(g, genes, stress_lookup.get((key, g, "global"))) for g, genes in by_group.items()
    )
    return f"<table><tr><th>cluster</th><th>top genes (logFC)</th></tr>{body}</table>"


def _deg_local_table(path: str, top_n: int, key: str, stress_lookup: dict) -> str:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_group: dict[str, list[tuple[str, float]]] = {}
    neighbors_by_group: dict[str, str] = {}
    for r in rows:
        g = by_group.setdefault(r["group"], [])
        neighbors_by_group.setdefault(r["group"], r.get("neighbors", ""))
        if len(g) < top_n:
            g.append((r["names"], float(r["logfoldchanges"])))
    body = "".join(
        _deg_row(g, genes, stress_lookup.get((key, g, "local")),
                middle_td=f"<td>{html.escape(neighbors_by_group.get(g, ''))}</td>")
        for g, genes in by_group.items()
    )
    return f"<table><tr><th>cluster</th><th>neighbors</th><th>top genes (logFC)</th></tr>{body}</table>"


def _section_deg(outdir: str, top_n: int = 10, preannotation_figs: list[str] | None = None) -> str:
    global_paths = sorted(glob.glob(os.path.join(outdir, "deg_global_*.csv")))
    if not global_paths:
        return ""
    parts = []
    if preannotation_figs:
        parts += ["<h3>Pre-annotation filtering</h3>",
                  '<p class="hint">Every cell proposed for removal before Cluster Annotations '
                  "runs — the union of three sources: whole standissect fragments flagged by "
                  "StanDissect Minor sibling fractals QC, individual cells flagged by the "
                  "cell-level doublet / ambient-RNA outlier test above, and cells osp itself "
                  "already proposed dropping per-sample (_qc_action==\"drop\", inherited from "
                  "persample annotation — their cross-sample clustering here is evidence to "
                  "weigh, not evidence to discard). Red = recommend_removal, grey = kept. "
                  "Computation-only: excluded from the DEG/PAGA analyses below, never dropped "
                  "from the data.</p>"]
        parts.append('<div class="trio">' + "".join(_img(p) for p in preannotation_figs) + "</div>")
    parts += [
        '<p class="hint">Cells marked recommend_removal above (see Pre-annotation filtering) '
        "are excluded from every comparison below — "
        "computation-only, no cells are dropped from the data. Global view: cluster vs every "
        "other cluster (one-vs-rest). Local view: cluster vs its 3 nearest neighbors by PAGA "
        "connectivity, pooled into one reference group — a sharper comparison when neighbors "
        "are transcriptionally close and get washed out by the global one-vs-rest. A (key, "
        "cluster) is marked <b style=\"color:#c0392b\">[recommend_removal]</b> when EITHER its "
        "global or local view has more than 3 of its displayed top genes in the conservative "
        "heat-shock/AP-1 dissociation-stress core panel (STRESS_GENES_CORE) or mitochondrial "
        "(MT-*) — the verdict is merged, so both the global and local rows for that cluster show "
        "it even if only one of the two actually crossed the threshold. Flagged only, nothing is "
        "removed from the data (see stress_clusters.csv for per-view hit genes).</p>"]

    stress_lookup = _load_stress_lookup(outdir)
    tabs = []
    for p in global_paths:
        key = os.path.basename(p)[len("deg_global_"):-len(".csv")]
        tabs.append((f"{key} — global", _deg_global_table(p, top_n, key, stress_lookup)))
        local_p = os.path.join(outdir, f"deg_local_{key}.csv")
        if os.path.exists(local_p):
            tabs.append((f"{key} — local", _deg_local_table(local_p, top_n, key, stress_lookup)))
    parts.append(_tabs("deg", tabs))
    return _h2("deg") + "".join(parts)


def _section_annotation(outdir: str, annotation_figs: list[str]) -> str:
    """msp.annotate's deliverable: coarse/fine UMAPs of the cells that
    survived removal, the removal UMAP, one row per base cluster, merged
    groups, and the removal archive summary."""
    path = os.path.join(outdir, "annotation_proposal.json")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        prop = json.load(f)
    key = prop.get("cluster_key", "msp_leiden_r2.0")
    by_name = {os.path.basename(p): p for p in annotation_figs}
    parts = [f'<p class="hint">Per-cluster identity on {html.escape(key)}: coarse (lineage) and fine '
             "(subtype) labels, explicit merge decisions (merged clusters share one label), and "
             "clusters judged noise/low-quality. Removal here is real: removed cells = pre-annotation "
             "filtering ∪ inspection drop ∪ agent-removed clusters, archived per cell with sources in "
             "annotation_removed.csv; annotated.h5ad holds the survivors, integrated.h5ad is untouched.</p>"]
    labelled = [by_name[n] for n in ("annotation_umap_coarse.png", "annotation_umap_fine.png") if n in by_name]
    if labelled:
        parts += ["<h3>Annotated UMAPs (removed cells excluded)</h3>",
                  '<div class="trio">' + "".join(_img(p) for p in labelled) + "</div>"]
    if "annotation_umap_removed.png" in by_name:
        parts += ["<h3>Removed cells (all sources)</h3>",
                  '<div class="trio">' + _img(by_name["annotation_umap_removed.png"]) + "</div>"]
        rm = os.path.join(outdir, "annotation_removed.csv")
        if os.path.exists(rm):
            with open(rm) as f:
                rows = list(csv.DictReader(f))
            srcs = [c for c in (rows[0].keys() if rows else []) if c not in ("cell", key, "remove_reason")]
            counts = {c: sum(r.get(c) == "True" for r in rows) for c in srcs}
            parts.append("<p>" + html.escape(f"{len(rows)} cells removed — by source (a cell may have several): "
                                             + ", ".join(f"{c}={n}" for c, n in counts.items())) + "</p>")
    cols = ("cluster_id", "coarse_label", "fine_label", "merge_target", "action", "remove_reason",
            "confidence", "rationale")
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape('' if e.get(c) is None else str(e.get(c)))}</td>" for c in cols) + "</tr>"
        for e in prop.get("clusters", [])
    )
    parts += ['<div id="annotation-tables">', "<h3>Per-cluster decisions</h3>",
              f"<table><tr>{head}</tr>{body}</table>"]
    if prop.get("merged_groups"):
        parts.append("<p><b>Merged groups:</b> " + html.escape(", ".join(prop["merged_groups"])) + "</p>")
    ev_rows = "".join(
        f"<tr><td>{html.escape(str(e.get('cluster_id')))}</td>"
        + "".join(f"<td>{html.escape(str(e.get('evidence', {}).get(t, '')))}</td>"
                  for t in ("distinctness", "markers", "merge"))
        + "</tr>"
        for e in prop.get("clusters", [])
    )
    parts.append("<details><summary>reasoning chain per cluster</summary>"
                 "<table><tr><th>cluster</th><th>1. distinctness</th><th>2. markers</th>"
                 f"<th>3. merge</th></tr>{ev_rows}</table></details>")
    if prop.get("overall"):
        parts.append(f"<p><b>Overall:</b> {html.escape(prop['overall'])}</p>")
    notes = os.path.join(outdir, "annotation_notes.md")
    if os.path.exists(notes):
        with open(notes) as f:
            parts.append("<details><summary>agent notes</summary>"
                         f'<pre class="notes">{html.escape(f.read())}</pre></details>')
    parts.append("</div>")
    return _h2("annotation") + "".join(parts)


def _section_umaps(umap_figs: list[str], standissect_figs: list[str], qc_figs: list[str]) -> str:
    if not umap_figs and not standissect_figs and not qc_figs:
        return ""
    ann_names = ("_ann_coarse", "_qc_action")  # both inherited from OSP per-sample runs
    ann_by_name = {n: p for p in umap_figs for n in ann_names if n in os.path.basename(p)}
    ann = [ann_by_name[n] for n in ann_names if n in ann_by_name]
    rest = [p for p in umap_figs if p not in ann]
    leiden = sorted(p for p in rest if "leiden" in os.path.basename(p))
    samples = [p for p in rest if p not in leiden]
    parts = [_h2("umaps")]
    if ann:
        parts += ["<h3>Inherited annotations from One-sample Pipeline (OSP)</h3>",
                  '<div class="trio">' + "".join(_img(p) for p in ann) + "</div>"]
    if samples:
        # natural image size: the sample legend carries long names — never
        # squeeze this panel into a fixed grid cell
        parts += ["<h3>Samples</h3>"] + [_img(p, cls="natural") for p in samples]
    if leiden:
        parts += ["<h3>Leiden clusterings</h3>",
                  '<div class="trio">' + "".join(_img(p) for p in leiden) + "</div>"]
    if standissect_figs:
        parts += ["<h3>standissect clusters</h3>",
                  '<div class="trio">' + "".join(_img(p) for p in standissect_figs) + "</div>"]
    if qc_figs:
        qc_umaps = [p for p in qc_figs if "umap" in os.path.basename(p)]
        other = [p for p in qc_figs if p not in qc_umaps and "violin" not in os.path.basename(p)]
        parts += ["<h3>QC metrics</h3>",
                  '<p class="hint">One metric per panel; pct_counts_mt uses a fixed color '
                  "ceiling (vmax=20) — the scale never autoscales.</p>",
                  _grid(qc_umaps)]
        parts += [_grid(other)]
    return "".join(parts)


def _add_subsection_anchors(section_html, sec_anchor):
    """Give every bare <h3>Text</h3> in this section an id (sec_anchor-N) so
    the TOC can link straight to it. Returns (new_html, [(id, text), ...])."""
    entries = []
    counter = [0]

    def repl(m):
        counter[0] += 1
        sub_id = f"{sec_anchor}-{counter[0]}"
        entries.append((sub_id, m.group(1)))
        return f'<h3 id="{sub_id}">{m.group(1)}</h3>'

    return re.sub(r"<h3>(.*?)</h3>", repl, section_html), entries


def _number_sections(section_htmls):
    """Number the sections that actually rendered so a missing one doesn't
    leave a gap (same mechanism as osp.report); also anchor every h3
    subsection and nest it under its parent in the TOC."""
    present = [
        (anchor, label) for anchor, label in _SECTION_LABELS.items()
        if any(f'<h2 id="{anchor}">{label}</h2>' in s for s in section_htmls)
    ]
    numbered = {anchor: f"{i}. {label}" for i, (anchor, label) in enumerate(present, start=1)}
    numbered_htmls = []
    toc_groups = []
    for s in section_htmls:
        sec_anchor = next((a for a, l in present if f'<h2 id="{a}">{l}</h2>' in s), None)
        for anchor, label in present:
            s = s.replace(f'<h2 id="{anchor}">{label}</h2>',
                          f'<h2 id="{anchor}">{numbered[anchor]}</h2>', 1)
        subs = []
        if sec_anchor:
            s, subs = _add_subsection_anchors(s, sec_anchor)
            toc_groups.append((sec_anchor, subs))
        numbered_htmls.append(s)

    toc_by_anchor = dict(toc_groups)
    parts = []
    for anchor, _ in present:
        parts.append(f'<a href="#{anchor}">{html.escape(numbered[anchor])}</a>')
        subs = toc_by_anchor.get(anchor, [])
        if subs:
            parts.append('<div class="toc-sub">' + "".join(
                f'<a href="#{sid}">{html.escape(text)}</a>' for sid, text in subs
            ) + "</div>")
    toc = "".join(parts)
    return numbered_htmls, (f'<nav class="toc">{toc}</nav>' if toc else "")


def generate_report(outdir: str, out_html: str | None = None, title: str | None = None) -> str:
    out_html = out_html or os.path.join(outdir, "report.html")
    figdir = os.path.join(outdir, "figures")
    figs = sorted(glob.glob(os.path.join(figdir, "*.png")))
    qc_figs = [p for p in figs if os.path.basename(p).startswith("qc_")]
    standissect_figs = [p for p in figs if os.path.basename(p).startswith("standissect_")]
    fractal_figs = [p for p in figs if os.path.basename(p).startswith("fractal_")]
    preannotation_figs = [p for p in figs if os.path.basename(p) == "umap_preannotation_removal.png"]
    leiden_violin_figs = [p for p in figs if os.path.basename(p).startswith("leiden_qc_violin_")]
    annotation_figs = [p for p in figs if os.path.basename(p).startswith("annotation_")]
    umap_figs = [p for p in figs if p not in qc_figs and not os.path.basename(p).startswith("inspect_")
                 and p not in standissect_figs and p not in fractal_figs and p not in preannotation_figs
                 and p not in leiden_violin_figs and p not in annotation_figs]

    violins = [p for p in qc_figs if "violin" in os.path.basename(p)]
    qc_figs = [p for p in qc_figs if p not in violins]

    title = title or f"msp Integration Report — {os.path.basename(os.path.abspath(outdir))}"
    sections = [
        _section_sample_summary(outdir),
        _section_umaps(umap_figs, standissect_figs, qc_figs),
        _section_per_cluster(outdir, violins, fractal_figs),
        _section_leiden_qc(outdir, leiden_violin_figs),
        _section_deg(outdir, preannotation_figs=preannotation_figs),
        _section_annotation(outdir, annotation_figs),
    ]
    sections, toc = _number_sections(sections)

    header = (f"<h1>{html.escape(title)}</h1>"
              f'<p class="meta">source dir: {html.escape(os.path.abspath(outdir))}</p>')
    body = f'{header}<div class="layout">{toc}<div class="content">{"".join(sections)}</div></div>'
    html_doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
                f"<body>{body}{TOC_PIN_SCRIPT}</body></html>")
    with open(out_html, "w") as fh:
        fh.write(html_doc)
    return out_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="msp.report", description=__doc__)
    parser.add_argument("outdir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(f"wrote {generate_report(args.outdir, out_html=args.out)}")
