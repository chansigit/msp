"""
msp HTML report generator — same design as osp's report:

  - Human: base64-inline images (self-contained file), osp's CSS, a sticky
    TOC that jumps between numbered sections.
  - Agent: every number in a figure also exists in a plain <table>; headers
    are plain <h2 id=...> text so grepping the raw HTML is enough.

Section order:
  1. Summary            — merged size, samples, cluster counts
  2. Per-sample QC      — inherited per-sample metrics after integration
  3. Per-cluster QC     — QC + composition per integrated cluster
  4. Cluster DEG        — top genes per cluster (wilcoxon)
  5. standissect-lite   — rule-mode tiny/fragmented-subcluster candidates
  6. Inspection Verdicts — the msp.inspect agent's five-test proposals
  7. QC UMAP            — QC metrics + inherited keep/flag/drop on the UMAP
  8. Samples & Clusters — sample mixing, clusterings, inherited annotation

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
#inspection td, #agent-tables td { text-align: left; vertical-align: top; }
pre.notes { white-space: pre-wrap; font-size: .85rem; background: #f7f7f7; padding: .8rem; border-radius: 6px; }
.meta { color: #666; font-size: .9rem; margin: .2rem 0 1rem 0; }
.hint { color: #555; font-size: .88rem; margin: .3rem 0 1rem 0; max-width: 75ch; }
.layout { display: flex; gap: 2rem; align-items: flex-start; }
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
    "per-cluster-qc": "Per-cluster QC",
    "deg": "Cluster DEG",
    "inspection": "Inspection Verdicts",
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
    return f"<h3>Minor sibling fractals QC</h3>{hint}{table}{details}"


def _section_per_cluster(outdir: str, violins: list[str]) -> str:
    parts = []
    for p in sorted(glob.glob(os.path.join(outdir, "cluster_qc_*.csv"))):
        key = os.path.basename(p)[len("cluster_qc_"):-len(".csv")]
        parts += [f"<h3>{html.escape(key)}</h3>",
                  '<p class="hint">flag/drop fractions carried over from per-sample '
                  "annotation; a cluster fed by a single sample is itself a signal.</p>",
                  _csv_table(p)]
    parts.append(_section_minor_sibling(outdir))
    if violins:
        parts += ["<h3>Per-cluster QC violins</h3>",
                  '<p class="hint">Grouped by the standissect clusters (leiden × UMAP-fragment '
                  "product), not the primary leiden — so a minor sibling's QC profile can be "
                  "compared against its main core.</p>",
                  _grid(violins)]
    return _h2("per-cluster-qc") + "".join(parts) if parts else ""


def _section_fractal_heatmap(outdir: str, fractal_figs: list[str]) -> str:
    path = os.path.join(outdir, "fractal_markers.csv")
    if not os.path.exists(path) and not fractal_figs:
        return ""
    parts = ["<h3>Fractal marker dot plot</h3>",
             '<p class="hint">Each parent\'s CORE cells (rank 0) DE\'d one-vs-rest against every '
             "other parent's core cells (core-only, so minor siblings never leak into either side); "
             "per parent, top 10 markers with logFC&gt;0, padj&lt;0.05, ribosomal genes excluded. "
             "Dot plot across every standissect cluster (cores AND fractals): dot size = fraction "
             "of a cluster's cells expressing the gene, dot color = row-wise z-score of average "
             "log1p expression. Columns are clustered (optimal leaf ordering) purely to order them "
             "— no dendrogram shown. Rows are grouped by the parent they're a marker for (colored "
             "strip), not clustered. Bold column labels = parent-core clusters; red = "
             "recommend_removal (see Minor sibling fractals QC above).</p>"]
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


def _section_deg(outdir: str, fractal_figs: list[str], top_n: int = 10) -> str:
    parts = []
    for p in sorted(glob.glob(os.path.join(outdir, "de_top_genes_*.csv"))):
        key = os.path.basename(p)[len("de_top_genes_"):-len(".csv")]
        with open(p) as f:
            rows = list(csv.DictReader(f))
        by_group: dict[str, list[str]] = {}
        for r in rows:
            g = by_group.setdefault(r["group"], [])
            if len(g) < top_n:
                g.append(f"{r['names']} ({float(r['logfoldchanges']):.1f})")
        body = "".join(
            f"<tr><td>{html.escape(g)}</td><td style='text-align:left'>{html.escape(', '.join(genes))}</td></tr>"
            for g, genes in by_group.items()
        )
        parts += [f"<h3>{html.escape(key)}</h3>",
                  f"<table><tr><th>cluster</th><th>top genes (logFC)</th></tr>{body}</table>"]
    parts.append(_section_fractal_heatmap(outdir, fractal_figs))
    return _h2("deg") + "".join(parts) if parts else ""


def _section_inspection(outdir: str) -> str:
    path = os.path.join(outdir, "inspection_proposal.json")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        prop = json.load(f)
    parts = ['<div id="inspection">',
             '<p class="hint">Five-test battery (markers / QC axis / composition / '
             "geometry / stability). Proposals only — nothing is removed by msp.</p>"]
    cols = ("cluster", "verdict", "action", "confidence", "rationale")
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(e.get(c, '')))}</td>" for c in cols) + "</tr>"
        for e in prop.get("clusters", [])
    )
    parts.append(f"<table><tr>{head}</tr>{body}</table>")
    tests_rows = "".join(
        f"<tr><td>{html.escape(str(e.get('cluster')))}</td>"
        + "".join(f"<td>{html.escape(str(e.get('tests', {}).get(t, '')))}</td>"
                  for t in ("markers", "qc", "composition", "geometry", "stability"))
        + "</tr>"
        for e in prop.get("clusters", [])
    )
    parts.append("<details><summary>five-test details per cluster</summary>"
                 "<table><tr><th>cluster</th><th>markers</th><th>qc</th>"
                 f"<th>composition</th><th>geometry</th><th>stability</th></tr>{tests_rows}"
                 "</table></details>")
    cells = prop.get("cell_actions", [])
    if cells:
        parts.append("<p>cell-level actions: " + html.escape(
            "; ".join(f"{a['cluster']}: {a['metric']} {a['op']} {a['value']} → {a['action']} ({a['reason']})"
                      for a in cells)) + "</p>")
    if prop.get("overall"):
        parts.append(f"<p><b>Overall:</b> {html.escape(prop['overall'])}</p>")
    # the verdict UMAP (keep/flag/drop) is intentionally not repeated here —
    # it looks like the OSP-inherited _qc_action panel shown earlier
    notes = os.path.join(outdir, "inspection_notes.md")
    if os.path.exists(notes):
        with open(notes) as f:
            parts.append("<details><summary>agent notes</summary>"
                         f'<pre class="notes">{html.escape(f.read())}</pre></details>')
    parts.append("</div>")
    return _h2("inspection") + "".join(parts)


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
    umap_figs = [p for p in figs if p not in qc_figs and not os.path.basename(p).startswith("inspect_")
                 and p not in standissect_figs and p not in fractal_figs]

    violins = [p for p in qc_figs if "violin" in os.path.basename(p)]
    qc_figs = [p for p in qc_figs if p not in violins]

    title = title or f"msp Integration Report — {os.path.basename(os.path.abspath(outdir))}"
    sections = [
        _section_sample_summary(outdir),
        _section_umaps(umap_figs, standissect_figs, qc_figs),
        _section_per_cluster(outdir, violins),
        _section_deg(outdir, fractal_figs),
        _section_inspection(outdir),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(f"wrote {generate_report(args.outdir, out_html=args.out)}")
