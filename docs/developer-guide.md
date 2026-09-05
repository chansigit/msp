# MSP developer guide

[README](../README.md) · [User guide](user-guide.md)

## Code map

MSP owns integration, biological tools, proposal validation, output updates,
and reports. [Agent Harness Bridge](https://github.com/chansigit/agent-harness-bridge)
owns backend execution and shared agent controls; `msp.harness` re-exports
its public objects for compatibility.

| Module | Responsibility |
| --- | --- |
| [`integrate.py`](../msp/integrate.py) | Load samples, integrate, cluster, compute QC and marker evidence. |
| [`inspect.py`](../msp/inspect.py) | Inspect cluster quality, query evidence, validate and apply proposals. |
| [`annotate.py`](../msp/annotate.py) | Assign labels, resolve merges, apply removals. |
| [`steps.py`](../msp/steps.py) | Invalidate and archive outputs; track pending stages. |
| [`report.py`](../msp/report.py) | Build a self-contained HTML report from completed stages. |
| [`__main__.py`](../msp/__main__.py) | Parse CLI options, check resume conditions, run stages in order. |

## Python entry points

`load_and_merge` returns an AnnData; `run_multi_sample_pipeline` and
`integrate_adata` return `(ad, summary)`. The latter operates on the supplied
AnnData, replacing its analysis state, so pass a copy if you need to preserve
it. `inspect_clusters` and `annotate_clusters` return accepted proposals;
`generate_report` returns the report path. Python integration calls require
an explicit report call, unlike the main CLI.

```python
from msp import run_multi_sample_pipeline, generate_report

ad, summary = run_multi_sample_pipeline(
    ["A/clustered.h5ad", "B/clustered.h5ad"],
    batch_col="sample_id",
    outdir="msp_out",
    species="human",
)
report_path = generate_report("msp_out")
```

## Input contract

Supply raw counts in `layers[counts_layer]` (default `counts`) and a batch
column. File merging requires equal `var_names` in order, one batch value per
file, and unique cell IDs across files. It retains the union of `obs` columns
with missing values, prefixes inherited `leiden_*` labels with sample IDs,
and keeps only the counts layer. Integration needs at least three cells, two
genes, usable HVGs, and `n_neighbors < n_obs`.

## Integration and matrices

Integration resets `X` from counts, normalizes to 10,000 counts per cell,
applies log1p, selects HVGs by batch, and runs scaled-HVG PCA, Harmony,
neighbors, Leiden, and UMAP. All genes remain in the output. A single batch
skips Harmony and uses PCA directly. The project pins `harmonypy==0.2.0`;
repeatable CLI `--harmony KEY=VALUE` options pass overrides to Harmony, while
`MSP_DEVICE` can select `cpu`, `cuda`, or `mps`.

| Environment variable | Effect |
| --- | --- |
| `MSP_DEVICE` | Harmony device: `cpu`, `cuda`, or `mps` (default: auto-detect). |
| `MSP_MAX_THREADS` | Cap on the CPUs the DEG thread pool may use (default: the affinity mask). |
| `HARNESS` | Agent backend when `--harness` is not given; see Agent Harness Bridge. |

| Location | Meaning |
| --- | --- |
| `layers["counts"]` | Preserved raw counts under the default layer name. |
| `X`, `raw.X` | Full normalized, log1p expression; `raw.X` is not raw counts. |
| `var["highly_variable"]` | Genes selected for PCA. |
| `obsm["X_pca"]`, `obsm["X_pca_harmony"]` | Original and corrected PCA coordinates. |
| `obsm["X_umap"]` | UMAP coordinates from the integrated neighbor graph. |
| `obs["msp_leiden_r..."]` | Clusters at each requested resolution; CLI defaults are 0.3, 1.0, and 2.0. |
| `obs["standissect_product"]` | Fragment labels used in geometry and QC evidence. |
| `uns["msp"]` | Input paths, species, batch column, dimensions, and integration settings. |

## Fractal structure and fragment detection

The working principle is that recurring sample-level outliers can pool into
dense satellites around major populations; the repeated core-and-satellite
geometry provides a way to locate noise. MSP uses `standissect-lite` to
intersect the lowest-resolution RNA-side Leiden partition with UMAP-side
clustering. Within each parent, the largest fragment is rank 0 (the core),
and smaller fragments receive increasing ranks. These labels describe
structure; QC and marker evidence supply its interpretation.

## Fragment QC rules

`_minor_sibling_qc` tests eligible non-core fragments against the pooled cells
from all rank-0 cores. A fragment becomes a removal candidate if any available
QC test passes or a majority of its cells carry inherited OSP drop proposals.
The rules below govern this fragment test; separate cell-level outlier and
inherited-drop rules also contribute to the final preannotation candidate set.

| Rule | Current implementation |
| --- | --- |
| Large fragments | Skip this test at ≥25% of their own parent core's size, or ≥800 cells. |
| Inherited OSP decisions | Propose the fragment if >50% of its cells have `_qc_action == "drop"`, including fragments too small for statistical testing. |
| Statistical comparison | One-sided Mann–Whitney U, p < 0.05, with at least five finite measurements in each comparison group; no multiple-testing correction. |
| Contamination and stress | Test available `decontX_contamination` and `dissociation_score` values for an upward shift. |
| Doublets and mitochondrial counts | Require the upward shift plus a fragment median above 0.2 for `doublet_score`, or above 20 for `pct_counts_mt`. |
| Filtering | Record candidates in `preannotation_removal.csv`; annotation applies the removal union described below. |

## Inspection contract

Inspection reviews markers, QC, sample composition, geometry, and stability.
Tools query gene expression, QC summaries, resolution overlap, DEG, and local
subclusters. The host validates coverage, actions, and cell-level rules before
mapping the accepted proposal onto `obs["_msp_action"]` and
`obs["_msp_verdict"]` in `integrated.h5ad`. Actions are `keep`, `flag`, or
`drop`; overlapping actions use the strongest, and no cells are removed here.

## Annotation contract

Annotation requires `msp_leiden_r1.0` and `msp_leiden_r2.0`, and submits one
decision per r2.0 cluster covering distinctness, markers, labels, and merges.
The host checks complete coverage, merge consistency, and label hierarchy;
shared fine labels must correspond to explicit merges. Accepted merges resolve
to connected components. Prior labels are reference evidence. The main CLI
runs inspection first; direct annotation can run without inspection actions
and then inherits only preannotation removals.

| Output field in `annotated.h5ad.obs` | Meaning |
| --- | --- |
| `msp_ann_cluster` | Resolved cluster ID, joining merged member IDs with `+`. |
| `msp_ann_coarse` | Broad cell-type label. |
| `msp_ann_fine` | Fine cell-type label. |
| `msp_ann_action` | `keep` for retained cells; removed rows are recorded separately. |

## Removal and marker evidence

Preannotation candidates combine minor-fragment QC, cell-level outliers, and
inherited OSP `_qc_action == "drop"`. Annotation removes their union with
inspection `_msp_action == "drop"` and annotation `action == "remove"`;
keep decisions cannot subtract from that union. Global/local marker tables
and inspection's live DEG exclude preannotation candidates; annotation's live
DEG also excludes inspection drops. Inspection's other evidence tools still
see all cells. The original integrated dataset remains available.

| File | Contract |
| --- | --- |
| `preannotation_removal.csv` | Per-cell `recommend_removal` flags, aligned by cell ID. |
| `annotation_removed.csv` | Removed cells with `preannotation`, optional `inspect_drop`, and `annotate_remove` source flags; `remove_reason` describes annotation removals. |
| `deg_global_*.csv`, `deg_local_*.csv` | Precomputed global and PAGA-neighbor marker comparisons on preannotation survivors. |
| `inspection_proposal.json`, `annotation_proposal.json` | Validated submissions; their presence alone does not establish completed application. |

## Stage recovery

`begin_step` marks the stage pending and archives its outputs and downstream
outputs under `.msp-history/`; inspection also snapshots `integrated.h5ad`.
H5AD writes use a temporary file followed by replacement. `complete_step`
clears the pending marker after data, tables, and plots are written, before
report rendering. Pending ancestors block downstream work, and reports omit
pending stages. Use one writer per directory; this is not a directory-wide
transaction or concurrent-writer lock.

## Resume checks

The CLI checks pending markers, required outputs, and integration metadata:
input path strings, batch column, species, resolutions, HVGs, requested PCs,
neighbors, and Harmony overrides. Inspection also requires mapped action
columns and its action plot; annotation requires its proposal and H5AD.
Input contents and agent configuration are not fingerprinted. Use `--force`
or a new directory after changing them, and record successful process
completion when driving MSP externally.

## Development checks

Install an editable checkout and run the regression suite below. Tests cover
evidence contracts, proposal validation, removal handling, and stage recovery;
they do not establish biological annotation accuracy. Keep changes to shared
backend execution in Agent Harness Bridge and changes to MSP's scientific
behavior in this repository.

```bash
python -m pip install -e ".[agent]"
python -m pip install pytest ruff
ruff check msp tests && ruff format --check msp tests
python -m pytest
```

Lint and test settings live in `pyproject.toml`; the GitHub Actions workflow
runs the same commands on pushes and pull requests. The test suite treats
`FutureWarning`s raised from `msp` as errors, so a pandas deprecation shows up
as a failure rather than as noise. Open refactoring items are tracked in
[TODO.md](../TODO.md).
