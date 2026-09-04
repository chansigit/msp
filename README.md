# MSP — Multi-sample Pipeline

**Bring single-cell samples together, review their quality, and annotate cell types.**

MSP combines processed single-cell RNA-seq samples into a shared analysis. It
corrects for differences between samples with Harmony, groups similar cells,
and produces a report you can open in your browser. Optional AI agents then
review cluster quality and assign cell-type labels, recording the evidence
behind their decisions.

Use MSP when you have analyzed samples individually and want to understand
which cell populations they share, which groups need closer inspection, and
how to label them consistently across samples.

## From samples to annotated cells

```mermaid
flowchart LR
    A[Per-sample data] --> B[Integrate samples]
    B --> C[Inspect cluster quality]
    C --> D[Annotate cell types]
```

| Step | What it does | What happens to the cells |
| --- | --- | --- |
| **Integrate** | Builds a shared representation, clusters cells, and summarizes quality and marker genes. | All input cells are retained. |
| **Inspect** · optional AI step | Reviews markers, quality measurements, sample composition, cluster geometry, and stability. | Records keep, flag, or drop proposals; retains all cells. |
| **Annotate** · optional AI step | Assigns broad cell types and finer subtypes, records cluster merges, and applies removal decisions. | Writes retained cells to a separate annotated dataset. |

The report brings together sample summaries, cell maps (UMAPs), quality
measurements, marker genes, inspection evidence, and cell-type annotations as
those steps complete. Figures are embedded in the HTML, so you can share the
report as a single file.

## Prepare your data

The usual input is one `.h5ad` file per sample, such as the `clustered.h5ad`
files produced by [OSP — One-sample Pipeline](https://github.com/chansigit/osp).
H5AD is the AnnData format commonly used with Scanpy.

Each file needs:

- **Raw counts** in `layers["counts"]`.
- **A sample column** in `obs`, with one sample identifier per file. You choose
  its name with `--batch-col`.
- **The same genes in the same order** across files.
- **Unique cell IDs** across all input files.

Existing quality measurements and cell-type labels are retained as evidence.
Columns available in only some samples remain missing in the others. MSP can
also start from one combined file using `--from-h5ad`.

## Install

Use Python 3.10 or newer in a separate environment. Install the current
GitHub version with the optional AI dependencies:

```bash
python -m pip install "msp-sc[agent] @ git+https://github.com/chansigit/msp.git"
```

For integration only, omit `[agent]`. The package is named `msp-sc`; its Python
import and command module are `msp`. Dependencies include `harmonypy==0.2.0`,
the version currently pinned by the project. Integration requires no model
credentials.

## Run your first analysis

Replace the example paths with your sample files and `sample_id` with the name
of their sample column.

**Integrate samples and generate a report:**

```bash
python -m msp A/clustered.h5ad B/clustered.h5ad \
    --batch-col sample_id --species human --outdir msp_out
```

Open `msp_out/report.html` in a browser to review the result.

**Include AI inspection and cell-type annotation:**

The default configuration uses the OpenAI Agents SDK to access a Doubao model
through Volcengine Ark. It requires an **Ark API key** and model access in that
account. Model calls may incur provider charges.

```bash
export ARK_API_KEY="your-ark-api-key"
python -m msp A/clustered.h5ad B/clustered.h5ad \
    --batch-col sample_id --species human --outdir msp_out \
    --annotate --harness openai --model doubao-seed-2-1-turbo-260628
```

`--annotate` includes inspection. To request inspection alone, use `--inspect`.
The `--harness` option selects the agent runtime; `--model` selects its model.
Claude and DeepSeek Harness runtimes are also supported. For runtime-specific
installation, authentication, and configuration, see the
[Agent Harness Bridge guide](https://github.com/chansigit/agent-harness-bridge#configuration).

## Find and understand your results

| File | Use it to… |
| --- | --- |
| `report.html` | Review the analysis, including inspection evidence and annotation decisions when available. |
| `integrated.h5ad` | Continue working with all input cells in the integrated space; inspection adds its proposed actions here. |
| `annotated.h5ad` | Analyze retained cells with broad and fine cell-type labels after annotation. |
| `annotation_removed.csv` | See every removed cell and the sources of its removal decision. |
| `inspection_proposal.json`, `annotation_proposal.json` | Read the structured cluster decisions. |
| `inspection_notes.md`, `annotation_notes.md` | Read agent session notes when provided by the runtime. |
| `figures/` | Reuse individual plots. |

**Annotation applies removals from all three sources:** integration's removal
candidates, inspection's drop proposals, and clusters the annotation agent
marks for removal. It preserves `integrated.h5ad` and writes survivors to
`annotated.h5ad`. If every cell is removed, the annotated file has zero cells
and the removal record still accounts for them all.

Review the evidence before using labels or removal decisions downstream.
MSP checks submission completeness and consistency; these checks do not
establish that a biological interpretation is correct.

## Continue an analysis

Repeat the same command to resume completed steps. When an upstream step is
rerun, its downstream results must be regenerated; previous outputs are kept
under `.msp-history/`. Use one running process per output directory.

Use `--force` to recompute the requested steps. Use it, or choose a new output
directory, if you replace an input file at the same path or want to rerun with
a different model: resume does not detect those changes automatically.

To rebuild the report without rerunning computation or agents:

```bash
python -m msp.report msp_out
```

## More information

- Run `python -m msp --help` for command options.
- See [the folder-based example](examples/run_msp.sh) for processing a directory
  of OSP sample outputs.
- A basic user guide and an advanced developer guide are planned as separate
  documents; this README is the starting point.
- Report problems or discuss usage in [GitHub Issues](https://github.com/chansigit/msp/issues).

MSP is available under the [MIT License](LICENSE).
