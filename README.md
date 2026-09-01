# msp — multi-sample pipeline

Integrates the per-sample outputs of [osp](../osp) (one `clustered.h5ad`
per 10x run) into one harmony-corrected space. Strictly propose-only: no
cell is deleted, no annotation is decided here.

```bash
python -m msp sampleA/clustered.h5ad sampleB/clustered.h5ad \
    --batch-col project --outdir out [--species human] [--resolutions 0.3 1.0 2.0]
```

Pipeline: concat (hard checks: identical gene axis, one batch value per
file, globally unique barcodes, no cell lost) → normalize from raw
`layers["counts"]` → per-batch HVG (`flavor="seurat"`, `batch_key`) →
scale/PCA recomputed on the merged cells → harmony → neighbors on
`X_pca_harmony` → leiden at each resolution (`msp_leiden_r*`) → UMAP →
`integrated.h5ad` + self-contained `report.html`.

Conventions:

- Inherited per-sample obs columns (QC metrics, `_ann_*`, `_qc_action`,
  doublet calls) ride along; sample-local leiden labels are prefixed with
  the sample value (`H12inner:3`) so they survive the merge unambiguously.
- Per-sample embeddings/uns/extra layers are dropped — only raw counts
  travel; everything integrated is recomputed on the merged cells.
- Doublet detection is NOT rerun: it belongs to the per-sample stage.

Driven in production by `ecarsi.crosssample`, which decides which samples
enter integration (agent decision, archived) before calling this package.
