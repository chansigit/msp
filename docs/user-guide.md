# MSP user guide

[README](../README.md) · [Developer guide](developer-guide.md)

## Prepare inputs

MSP accepts one H5AD per sample, including OSP's `clustered.h5ad`. Each needs
raw counts in `layers["counts"]`, one sample identity in the chosen `obs`
column, matching gene names and order, and globally unique cell IDs. Preserve
available QC measurements and previous labels for review; metadata missing
from some samples stays missing. The combined data and analysis must fit in
memory.

| Check | If it fails |
| --- | --- |
| Raw counts are present | Recover the original counts; do not relabel normalized expression as counts. |
| Genes match, including order | Standardize and align genes before running MSP; it does not harmonize gene identifiers. |
| Cell IDs are unique | Add a sample prefix to repeated barcodes and retain the mapping to the original IDs. |
| The batch column identifies samples | Use that column with `--batch-col`; keep condition and cell-type labels in separate columns. |

## Integrate sample files

Install with the [README command](../README.md#1-install), then replace the
paths, sample column, and species below. Integration needs no model credentials.
Keep the default clustering resolutions if you plan to add AI inspection or
annotation; those stages need resolutions 1.0 and 2.0.

```bash
python -m msp A/clustered.h5ad B/clustered.h5ad \
    --batch-col sample_id --species human --outdir msp_out
```

## Start from a combined file

Use `--from-h5ad` instead of separate sample paths for a combined dataset with
raw counts and sample metadata. MSP recomputes normalization, integration,
and clustering from the counts layer; previous embeddings are replaced, while
prior labels can provide evidence for the agents.

```bash
python -m msp --from-h5ad combined.h5ad \
    --batch-col sample_id --species human --outdir msp_out
```

## Configure AI

Install with `[agent]`. The example below explicitly selects the OpenAI Agents
runtime and a Doubao model accessed through Volcengine Ark; an Ark API key and
model access are required, and calls may incur charges. MSP also supports
`--harness claude` and `--harness deepseek`; their setup and authentication are
documented in the [Agent Harness Bridge guide](https://github.com/chansigit/agent-harness-bridge#configuration).

```bash
export ARK_API_KEY="YOUR_ARK_API_KEY"
python -m msp A/clustered.h5ad B/clustered.h5ad \
    --batch-col sample_id --species human --outdir msp_out \
    --annotate --harness openai --model doubao-seed-2-1-turbo-260628
```

## Choose how far to run

Use `--inspect` for quality review alone or `--annotate` for inspection followed
by cell-type annotation. To rerun just one AI stage on existing results, use
its module below; these commands select the runtime through `HARNESS`.
Rerunning inspection invalidates annotation, so run annotation again afterward
if you need an updated annotated dataset.

```bash
HARNESS=openai python -m msp.inspect msp_out --species human
HARNESS=openai python -m msp.annotate msp_out --species human
```

## Read the report

Open `msp_out/report.html`, downloading it first if needed. Follow the questions
below to assess the result; a group dominated by one sample may reflect real
biology, and UMAP proximity alone does not establish identity. Missing QC
measurements limit the available evidence. Accepted AI submissions have passed
format and consistency checks, so their biological interpretation still needs
review.

| Report section | What to check |
| --- | --- |
| Sample Summary | Did the expected samples enter, and how do their cell counts and QC compare? |
| UMAPs (integrated space) | Where do samples and inherited labels overlap or separate? |
| Per-cluster QC / Leiden Cluster QC | Do suspicious groups differ in depth, mitochondrial signal, doublet score, or other available metrics? |
| Cluster Annotations | Do global and local marker comparisons support separate populations? This section contains computed evidence before AI annotation. |
| Integration QC Inspection | Which groups were kept, flagged, or proposed for removal, and why? |
| Cell Type Annotation | Do broad and fine labels, merges, and removals agree with the evidence? |

## Use the outputs

Use `integrated.h5ad` to inspect all input cells, and `annotated.h5ad` for the
retained cells after annotation. The latter carries `msp_ann_coarse` and
`msp_ann_fine` in `obs`. Raw counts remain in `layers["counts"]`; `X` holds
normalized, log-transformed expression. Individual plots and decision files
are available beside the report.

| Path | Contents |
| --- | --- |
| `figures/` | Individual plots used in the report. |
| `inspection_proposal.json`, `annotation_proposal.json` | Structured cluster decisions and evidence. |
| `inspection_notes.md`, `annotation_notes.md` | Agent notes when returned by the runtime. |
| `preannotation_removal.csv` | Integration's removal candidates, including inherited OSP drop proposals. |
| `annotation_removed.csv` | Every removed cell, with flags identifying its removal sources. |

## Understand removals

Integration and inspection retain all input cells. Annotation removes the
union of preannotation candidates, inspection drops, and annotation removals;
a later keep decision does not cancel an earlier removal source. It leaves
`integrated.h5ad` intact and writes survivors separately. If no cells survive,
`annotated.h5ad` is empty and `annotation_removed.csv` still records all removed
cells. See the [developer guide](developer-guide.md#removal-and-marker-evidence)
for the exact rules.

## Continue or rerun

Repeat the main command to reuse completed stages with matching integration
settings. Changing input contents at the same path or changing the model is
not detected automatically: use `--force` or a new output directory. A rerun
archives affected outputs under `.msp-history/` and invalidates later stages.
Use one active process per output directory; successful process completion
matters more than finding an old report after an error.

## Rebuild only the report

If report rendering failed after computation completed, or you need to refresh
the HTML, run the command below. It does not rerun integration or call an AI
model; sections from pending stages are omitted until those stages complete.

```bash
python -m msp.report msp_out
```

## Troubleshoot

Use `python -m msp --help` for command options. For a failed run, start with
the reported error and the remedies below; when reporting a problem, include
the command, package versions, traceback, and input dimensions without API keys.

| Symptom | Next step |
| --- | --- |
| Gene axis differs or barcodes are duplicated | Align the inputs as described under [Prepare inputs](#prepare-inputs). |
| Inspection or annotation rejects the resolutions | Include both `1.0` and `2.0` in `--resolutions`, or use the defaults. |
| Authentication, model access, or runtime startup fails | Check the selected backend against the bridge configuration guide. |
| A stage is marked incomplete | Rerun that stage before proceeding downstream. |
| Process runs out of memory | Allocate more memory for the combined dataset; MSP integration loads it into memory. |
