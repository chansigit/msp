# msp — multi-sample-pipeline

Integrates the per-sample outputs of [osp](https://github.com/chansigit/osp)
(one `clustered.h5ad` per 10x run) into one harmony-corrected space, then
runs two optional Claude-agent steps on the result: a per-cluster QC
**inspection** (proposals only) and a cell-type **annotation** (coarse/fine
labels, explicit merges, real removal). Every step writes a self-contained
`report.html` (all figures base64-embedded) that the next step reads.

```
osp per-sample ──▶ integrate ──▶ inspect ──▶ annotate
                   propose-only   propose-only   removes cells
                   integrated.h5ad              annotated.h5ad
```

## Install

```bash
pip install git+https://github.com/chansigit/msp.git
# with the agent steps (needs claude-agent-sdk + Claude Code CLI credentials):
pip install "msp[agent] @ git+https://github.com/chansigit/msp.git"
```

Dependencies of note: `harmonypy>=0.2.0` (the torch-based fork — set
`MSP_DEVICE=cpu|cuda|mps` to override device auto-detection) and
`standissect-lite` (minor-sibling fragment detection inside clusters).

## Quick usage

```bash
# integration + report
python -m msp A/clustered.h5ad B/clustered.h5ad --batch-col project --outdir msp_out --species human

# the whole chain (integration → inspection agent → annotation agent)
python -m msp A/clustered.h5ad B/clustered.h5ad --batch-col project --outdir msp_out \
    --species human --annotate --model claude-sonnet-5

# individual steps
python -m msp.inspect  msp_out --model claude-sonnet-5   # QC verdicts
python -m msp.annotate msp_out --model claude-sonnet-5   # identity + merges + removal (after inspect)
python -m msp.report   msp_out                           # rebuild report.html only
```

Re-running the same `python -m msp` command resumes: a step is skipped when
its contract files already exist (`integrated.h5ad`+`report.html`,
`inspection_proposal.json`, `annotation_proposal.json`+`annotated.h5ad`);
`--force` redoes everything.

```python
from msp import run_multi_sample_pipeline, generate_report
run_multi_sample_pipeline(["A/clustered.h5ad", "B/clustered.h5ad"], batch_col="project", outdir="msp_out")
generate_report("msp_out")

from msp.inspect import inspect_clusters     # optional agent steps
from msp.annotate import annotate_clusters
```

## The three steps

### 1. integrate (`msp.integrate`, propose-only)

concat (hard checks: identical gene axis, one batch value per file,
globally unique barcodes, no cell lost) → normalize from raw
`layers["counts"]` → per-batch HVG → scale/PCA on the merged cells → harmony
→ neighbors on `X_pca_harmony` → leiden at each resolution (`msp_leiden_r*`,
default 0.3/1.0/2.0) → UMAP → standissect-lite fragments on the coarsest
resolution → QC/DEG artifacts → `integrated.h5ad` + `report.html`.

Removal *candidates* are computed but never applied:

- `minor_sibling_qc.csv` — standissect fragments failing QC against their parent;
- `cell_outliers.csv` — per-cluster doublet/ambient outliers (cell flagged only
  when it clears BOTH gates: cluster median + 3×MAD **and** an absolute floor
  of 0.5; OR across metrics and across r1.0/r2.0);
- osp's own per-sample `_qc_action == "drop"` cells;
- their union is `preannotation_removal.csv` and the "Pre-annotation
  filtering" UMAP. The precomputed DEG tables (`deg_global_*` one-vs-rest,
  `deg_local_*` vs the 3 nearest PAGA neighbours, at r1.0 and r2.0) exclude
  these cells; `stress_clusters.csv` flags clusters whose top genes are a
  dissociation-stress/mitochondrial signature.

### 2. inspect (`msp.inspect`, propose-only)

One agent session puts every r1.0 cluster through the five-test battery
(markers / QC axis / composition / geometry / stability) with live tools
(`check_genes`, `check_qc_scores`, `check_stability`, `check_deg`,
`subcluster`). Output: `inspection_proposal.json`, `inspection_notes.md`,
`obs["_msp_action"]` (keep/flag/drop) and `obs["_msp_verdict"]` written
into `integrated.h5ad`, the verdict UMAP. Live DEG excludes the
pre-annotation removal set, matching the precomputed tables.

### 3. annotate (`msp.annotate`, removes cells)

One agent session annotates every **r2.0** cluster. Coverage is enforced
twice: the agent keeps one Claude Code Task per cluster
(TaskCreate/TaskUpdate/TaskList), and the host refuses `finalize_annotation`
until every cluster has a validated `submit_cluster`. Per cluster the agent
answers a fixed reasoning chain — (1) distinct entity or splinter of its
r1.0 parent/siblings, (2) coarse + fine label, or noise/low-quality →
remove, (3) merge target or keep separate — using `cluster_context`
(parent/siblings/PAGA neighbours/sample composition/QC/inspect verdict/prior
label compositions), `check_genes` and `check_deg`. Prior label columns
(osp's `_ann_coarse`/`_ann_fine`, the authors' own cell-type columns) are
detected, not assumed, and shown as reference evidence only.

Merge decisions are made in one session and validated deterministically on
the host (union-find over `merge_target`; a merged group shares one
coarse/fine label; one fine label belongs to one coarse label; equal fine
labels must be merged explicitly; nothing merges into a removed cluster) —
no separate harmonization pass.

Removal is real at this step: removed = `preannotation_removal.csv` ∪
inspect drop ∪ agent-removed clusters, archived per cell with sources in
`annotation_removed.csv`. `annotated.h5ad` keeps the survivors with
`msp_ann_cluster` (merged id, e.g. `1+2+4`), `msp_ann_coarse`,
`msp_ann_fine`, `msp_ann_action`; `integrated.h5ad` is left untouched.

## Output directory

| file | written by | what |
|---|---|---|
| `integrated.h5ad` | integrate (+inspect adds `_msp_*`) | all cells, harmony space, `msp_leiden_r*`, `standissect_product` |
| `report.html` | every step | self-contained report: Sample Summary · UMAPs · Per-cluster QC (standissect) · Leiden Cluster QC · Cluster Annotations (DEG) · Cell Type Annotation |
| `integration_summary.csv`, `per_sample_qc.csv`, `sample_decisions.csv`* | integrate | sample-level tables (*optional, written by the caller) |
| `cluster_qc_*.csv`, `cell_outliers.csv`, `cell_outlier_summary.csv` | integrate | per-cluster / per-cell QC |
| `fragments_*.csv`, `overlap_*.csv`, `minor_sibling_qc.csv`, `fractal_markers.csv` | integrate | standissect-lite bundle |
| `deg_global_*.csv`, `deg_local_*.csv`, `paga_neighbors_*.csv`, `stress_clusters.csv` | integrate | DEG at r1.0/r2.0 |
| `preannotation_removal.csv` | integrate | union of removal candidates (cell, recommend_removal) |
| `inspection_proposal.json`, `inspection_notes.md` | inspect | five-test verdicts per cluster |
| `annotation_proposal.json`, `annotation_notes.md` | annotate | per-cluster labels, merges, evidence, merged groups |
| `annotation_removed.csv` | annotate | every removed cell with its sources |
| `annotated.h5ad` | annotate | survivors with `msp_ann_*` columns |
| `figures/*.png` | all | one signal per file, fixed UMAP geometry |

## Conventions

- **Propose, never remove** until `annotate`: `integrate` and `inspect` add
  columns and CSVs, never drop cells; computation-only exclusions (DEG) are
  documented where they happen.
- Inherited per-sample obs columns (QC metrics, `_ann_*`, `_qc_action`,
  doublet calls) ride along; sample-local leiden labels are prefixed with
  the sample value (`H12inner:3`). Per-sample embeddings/uns/layers are
  dropped — only raw counts travel; everything integrated is recomputed.
- Doublet detection is NOT rerun: it belongs to the per-sample stage.
- `checkpoint`-style writes: h5ad files are written to `*.tmp.h5ad` and
  renamed, never in place.
- All heavy computation lives in `msp.integrate`; `msp.report` only renders
  artifacts already on disk, so `python -m msp.report` is always safe.

Driven in production by `ecarsi.crosssample`, which decides which samples
enter integration (agent decision, archived) before calling this package.

## Tuning integration

`python -m msp` exposes the integration knobs; the Python entry point takes
the same names as keyword arguments (`run_multi_sample_pipeline(...,
n_top_genes=, n_pcs=, n_neighbors=, resolutions=, harmony_kwargs={...})`).

| knob | default | CLI |
|---|---|---|
| HVGs per batch | 2000 | `--n-top-genes` |
| PCs | 50 | `--n-pcs` |
| kNN neighbours (on `X_pca_harmony`) | 15 | `--n-neighbors` |
| leiden resolutions | 0.3 1.0 2.0 | `--resolutions` (1.0 and 2.0 required by inspect/annotate) |
| harmony | harmonypy defaults | `--harmony KEY=VALUE` (repeatable) |

Harmony is called as `harmonypy.run_harmony(X_pca, obs[[batch_col]],
batch_col, random_state=0, device=<auto or $MSP_DEVICE>, **harmony_kwargs)`;
anything not overridden is harmonypy's default:

| harmony parameter | default | meaning |
|---|---|---|
| `theta` | 2 (per covariate) | diversity penalty — higher = stronger mixing across batches |
| `lamb` | 1 | ridge penalty on the correction; `-1` = auto-estimate (R behaviour, uses `alpha`=0.2) |
| `sigma` | 0.1 | soft k-means width — larger = softer cluster assignment |
| `nclust` | min(round(N/30), 100) | number of harmony clusters |
| `tau` | 0 | discounting for small batches (expected cells per cluster) |
| `block_size` | 0.05 | fraction of cells updated per block |
| `max_iter_harmony` | 10 | outer iterations |
| `max_iter_kmeans` | 20 | inner clustering iterations |
| `epsilon_cluster` / `epsilon_harmony` | 1e-5 / 1e-4 | convergence tolerances |

Example: gentler correction that keeps more within-batch structure and runs longer:

```bash
python -m msp ... --harmony theta=1 --harmony max_iter_harmony=20
```

The effective overrides are recorded in `uns["msp"]["harmony"]` of
`integrated.h5ad` (empty = all defaults).
