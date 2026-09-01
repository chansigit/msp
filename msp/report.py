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
.trio { display: flex; gap: 1.2rem; margin: .5rem 0 1rem 0; align-items: flex-start; flex-wrap: wrap; }
.trio figure { margin: 0; }
.trio img.fig { width: auto; max-width: 100%; }
.trio figcaption { text-align: center; }
figure.natural img.fig { max-width: 100%; width: auto; }
figure { margin: 1rem 0; }
figcaption { font-size: .78rem; color: #555; margin-top: .25rem; }
#inspection td, #agent-tables td { text-align: left; vertical-align: top; }
pre.notes { white-space: pre-wrap; font-size: .85rem; background: #f7f7f7; padding: .8rem; border-radius: 6px; }
.meta { color: #666; font-size: .9rem; margin: .2rem 0 1rem 0; }
.hint { color: #555; font-size: .88rem; margin: .3rem 0 1rem 0; max-width: 75ch; }
.layout { display: flex; gap: 2rem; align-items: flex-start; }
.content { flex: 1; min-width: 0; }
nav.toc { position: sticky; top: 1rem; z-index: 10; flex: 0 0 200px;
          display: flex; flex-direction: column; gap: .5rem;
          background: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 6px;
          padding: .8rem 1rem; font-size: .9rem;
          box-shadow: 0 2px 4px rgba(0,0,0,.06); }
nav.toc a { color: #24578a; text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
@media (max-width: 800px) {
  .layout { flex-direction: column; }
  nav.toc { position: static; flex-direction: row; flex-wrap: wrap; width: auto; }
}
"""

_SECTION_LABELS = {
    "summary": "Summary",
    "per-sample-qc": "Per-sample QC",
    "umaps": "UMAPs",
    "per-cluster-qc": "Per-cluster QC",
    "deg": "Cluster DEG",
    "standissect": "standissect-lite (minor siblings)",
    "inspection": "Inspection Verdicts",
    "qc-umap": "QC (integrated space)",
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


def _csv_table(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows[1:]
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


def _section_summary(outdir: str) -> str:
    t = _kv_table(os.path.join(outdir, "integration_summary.csv"))
    return _h2("summary") + t if t else ""


def _section_per_sample(outdir: str) -> str:
    t = _csv_table(os.path.join(outdir, "per_sample_qc.csv"))
    return _h2("per-sample-qc") + t if t else ""


def _section_per_cluster(outdir: str) -> str:
    parts = []
    for p in sorted(glob.glob(os.path.join(outdir, "cluster_qc_*.csv"))):
        key = os.path.basename(p)[len("cluster_qc_"):-len(".csv")]
        parts += [f"<h3>{html.escape(key)}</h3>",
                  '<p class="hint">flag/drop fractions carried over from per-sample '
                  "annotation; a cluster fed by a single sample is itself a signal.</p>",
                  _csv_table(p)]
    return _h2("per-cluster-qc") + "".join(parts) if parts else ""


def _section_deg(outdir: str, top_n: int = 10) -> str:
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
    return _h2("deg") + "".join(parts) if parts else ""


_FRAGMENT_COLS = ("subcluster", "parent", "umap_label", "n_cells", "frac_of_parent", "rank")


def _section_standissect(outdir: str) -> str:
    hits = sorted(glob.glob(os.path.join(outdir, "fragments_*.csv")))
    if not hits:
        return ""
    parts = []
    for path in hits:
        key = os.path.basename(path)[len("fragments_"):-len(".csv")]
        with open(path) as f:
            rows = list(csv.DictReader(f))
        minors = [r for r in rows if str(r.get("is_minor_sibling", "")).lower() in ("true", "1")]
        parts.append(f"<h3>{html.escape(key)} × umap_cluster</h3>"
                     '<p class="hint">Cartesian product of the RNA-side leiden with a '
                     "UMAP-side clustering; within each parent, fragments are ranked by "
                     "size — rank 0 is the main core, the rest are minor siblings. "
                     "Detection only, candidates not verdicts. "
                     f"{len(minors)} minor sibling(s) among {len(rows)} fragments.</p>")
        head = "".join(f"<th>{html.escape(c)}</th>" for c in _FRAGMENT_COLS)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(r.get(c, ''))}</td>" for c in _FRAGMENT_COLS) + "</tr>"
            for r in minors
        )
        parts.append(f"<table><tr>{head}</tr>{body}</table>")
    return _h2("standissect") + "".join(parts)


def _section_inspection(outdir: str, inspect_figs: list[str]) -> str:
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
    parts += [_img(p) for p in inspect_figs]
    notes = os.path.join(outdir, "inspection_notes.md")
    if os.path.exists(notes):
        with open(notes) as f:
            parts.append("<details><summary>agent notes</summary>"
                         f'<pre class="notes">{html.escape(f.read())}</pre></details>')
    parts.append("</div>")
    return _h2("inspection") + "".join(parts)


def _section_qc_umap(qc_figs: list[str]) -> str:
    if not qc_figs:
        return ""
    umaps = [p for p in qc_figs if "umap" in os.path.basename(p)]
    violins = [p for p in qc_figs if "violin" in os.path.basename(p)]
    other = [p for p in qc_figs if p not in umaps and p not in violins]
    return (_h2("qc-umap")
            + '<p class="hint">One metric per panel; pct_counts_mt uses a fixed color '
            "ceiling (vmax=20) — the scale never autoscales.</p>"
            + _grid(umaps)
            + ("<h3>per-cluster violins</h3>" + _grid(violins) if violins else "")
            + _grid(other))


def _section_umaps(umap_figs: list[str], standissect_figs: list[str]) -> str:
    if not umap_figs and not standissect_figs:
        return ""
    ann = [p for p in umap_figs if "_ann_coarse" in os.path.basename(p)]
    rest = [p for p in umap_figs if "_ann_" not in os.path.basename(p)]
    leiden = sorted(p for p in rest if "leiden" in os.path.basename(p))
    samples = [p for p in rest if p not in leiden]
    parts = [_h2("umaps")]
    if ann:
        parts += ["<h3>Inherited annotations from One-sample Pipeline (OSP)</h3>", _grid(ann)]
    if samples:
        # natural image size: the sample legend carries long names — never
        # squeeze this panel into a fixed grid cell
        parts += ["<h3>Samples</h3>"] + [_img(p, cls="natural") for p in samples]
    if leiden:
        parts += ["<h3>Leiden clusterings</h3>",
                  '<div class="trio">' + "".join(_img(p) for p in leiden) + "</div>"]
    if standissect_figs:
        parts += ["<h3>standissect-lite derived</h3>"]
        parts += [_img(p) for p in standissect_figs]
    return "".join(parts)


def _number_sections(section_htmls):
    """Number the sections that actually rendered so a missing one doesn't
    leave a gap (same mechanism as osp.report)."""
    present = [
        (anchor, label) for anchor, label in _SECTION_LABELS.items()
        if any(f'<h2 id="{anchor}">{label}</h2>' in s for s in section_htmls)
    ]
    numbered = {anchor: f"{i}. {label}" for i, (anchor, label) in enumerate(present, start=1)}
    numbered_htmls = []
    for s in section_htmls:
        for anchor, label in present:
            s = s.replace(f'<h2 id="{anchor}">{label}</h2>',
                          f'<h2 id="{anchor}">{numbered[anchor]}</h2>', 1)
        numbered_htmls.append(s)
    toc = "".join(f'<a href="#{anchor}">{html.escape(numbered[anchor])}</a>' for anchor, _ in present)
    return numbered_htmls, (f'<nav class="toc">{toc}</nav>' if toc else "")


def generate_report(outdir: str, out_html: str | None = None, title: str | None = None) -> str:
    out_html = out_html or os.path.join(outdir, "report.html")
    figdir = os.path.join(outdir, "figures")
    figs = sorted(glob.glob(os.path.join(figdir, "*.png")))
    qc_figs = [p for p in figs if os.path.basename(p).startswith("qc_")]
    inspect_figs = [p for p in figs if os.path.basename(p).startswith("inspect_")]
    standissect_figs = [p for p in figs if os.path.basename(p).startswith("standissect_")]
    umap_figs = [p for p in figs if p not in qc_figs and p not in inspect_figs
                 and p not in standissect_figs]

    title = title or f"msp Integration Report — {os.path.basename(os.path.abspath(outdir))}"
    sections = [
        _section_summary(outdir),
        _section_per_sample(outdir),
        _section_umaps(umap_figs, standissect_figs),
        _section_per_cluster(outdir),
        _section_deg(outdir),
        _section_standissect(outdir),
        _section_inspection(outdir, inspect_figs),
        _section_qc_umap(qc_figs),
    ]
    sections, toc = _number_sections(sections)

    header = (f"<h1>{html.escape(title)}</h1>"
              f'<p class="meta">source dir: {html.escape(os.path.abspath(outdir))}</p>')
    body = f'{header}<div class="layout">{toc}<div class="content">{"".join(sections)}</div></div>'
    html_doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
                f"<body>{body}</body></html>")
    with open(out_html, "w") as fh:
        fh.write(html_doc)
    return out_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(f"wrote {generate_report(args.outdir, out_html=args.out)}")
