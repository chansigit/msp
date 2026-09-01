"""Minimal self-contained HTML report for an msp integration run.

Images are embedded inline (base64) like osp's report — one file, no
broken links. Current scope: the integration UMAPs and the summary table;
grows step by step with the pipeline.
"""

from __future__ import annotations

import base64
import csv
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


def generate_report(outdir: str, out_html: str | None = None) -> str:
    out_html = out_html or os.path.join(outdir, "report.html")

    rows = []
    spath = os.path.join(outdir, "integration_summary.csv")
    if os.path.exists(spath):
        with open(spath) as f:
            rows = [r for r in csv.reader(f) if len(r) == 2 and r[0]]

    figdir = os.path.join(outdir, "figures")
    figs = sorted(
        os.path.join(figdir, f) for f in os.listdir(figdir) if f.endswith(".png")
    ) if os.path.isdir(figdir) else []

    parts = [
        "<!doctype html><meta charset='utf-8'><title>msp integration report</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 10px}"
        "figure{margin:1.5em 0}</style>",
        "<h1>msp integration report</h1>",
        "<h2>Summary</h2><table>",
    ]
    parts += [f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows]
    parts.append("</table><h2>UMAPs</h2>")
    parts += [_img(p) for p in figs]

    with open(out_html, "w") as f:
        f.write("\n".join(parts))
    return out_html
