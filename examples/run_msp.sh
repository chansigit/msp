#!/usr/bin/env bash
# Full msp chain over a folder of osp per-sample outputs:
#   <osp_root>/<sample>/clustered.h5ad  →  <outdir>/{integrated.h5ad, annotated.h5ad, report.html, ...}
# Re-running resumes (finished steps are skipped). Usage:
#   examples/run_msp.sh <osp_root> <outdir> <batch_col> [species] [model]
set -euo pipefail

osp_root=${1:?osp_root}; outdir=${2:?outdir}; batch_col=${3:?batch_col}
species=${4:-}; model=${5:-claude-sonnet-5}

mapfile -t inputs < <(ls "$osp_root"/*/clustered.h5ad)
[[ ${#inputs[@]} -ge 2 ]] || { echo "need at least two clustered.h5ad under $osp_root" >&2; exit 1; }

python -m msp "${inputs[@]}" --batch-col "$batch_col" --outdir "$outdir" \
    ${species:+--species "$species"} --annotate --model "$model" "${@:6}"
echo "report: $outdir/report.html"
