"""Self-contained HTML report for an msp integration run.

Images are embedded inline (base64) like osp's report — one file, no
broken links. Sections: summary, per-sample QC, per-cluster QC (the raw
material for per-cluster inspection), QC UMAPs, cluster/annotation UMAPs.
"""

from __future__ import annotations

import base64
import csv
import glob
import html
import os


def _img(path: str) -> str:
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    ext = os.path.splitext(path)[1].lstrip(".")
    name = html.escape(os.path.basename(path))
    return (f'<figure><img style="max-width:100%" '
            f'src="data:image/{ext};base64,{b64}" alt="{name}">'
            f"<figcaption>{name}</figcaption></figure>")


def _csv_table(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
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


def generate_report(outdir: str, out_html: str | None = None) -> str:
    out_html = out_html or os.path.join(outdir, "report.html")
    figdir = os.path.join(outdir, "figures")
    figs = sorted(glob.glob(os.path.join(figdir, "*.png")))
    qc_figs = [p for p in figs if os.path.basename(p).startswith("qc_")]
    umap_figs = [p for p in figs if p not in qc_figs]

    parts = [
        "<!doctype html><meta charset='utf-8'><title>msp integration report</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:4px 10px}"
        "figure{margin:1.5em 0}</style>",
        "<h1>msp integration report</h1>",
        "<h2>Summary</h2>",
        _kv_table(os.path.join(outdir, "integration_summary.csv")),
        "<h2>Per-sample QC</h2>",
        _csv_table(os.path.join(outdir, "per_sample_qc.csv")),
    ]
    for p in sorted(glob.glob(os.path.join(outdir, "cluster_qc_*.csv"))):
        key = os.path.basename(p)[len("cluster_qc_"):-len(".csv")]
        parts += [f"<h2>Per-cluster QC ({html.escape(key)})</h2>",
                  "<p>flag/drop carried over from per-sample annotation; a cluster "
                  "fed by a single sample is itself a signal.</p>",
                  _csv_table(p)]
    if qc_figs:
        parts.append("<h2>QC on the integrated UMAP</h2>"
                     "<p>pct_counts_mt uses a fixed color ceiling (vmax=20) — "
                     "the scale never autoscales.</p>")
        parts += [_img(p) for p in qc_figs]
    parts.append("<h2>Samples, clusters, inherited annotation</h2>")
    parts += [_img(p) for p in umap_figs]

    with open(out_html, "w") as f:
        f.write("\n".join(parts))
    return out_html
