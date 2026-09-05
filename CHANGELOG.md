# Changelog

All notable changes to msp-sc. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## 0.3.0 - 2026-09-04

### Added
- `msp.evidence`: the evidence layer (precomputed DEG tables, live DEG,
  per-cluster expression / QC / stability views, removal mask, PAGA
  neighbours) with public names. The underscore names `msp.inspect` and
  `msp.annotate` used to export (`_parse_reference`, `_cluster_order`,
  `_gene_table`, ...) are gone; ZMIP 0.3 imports `msp.evidence`.
- `msp.agent_tools`: the `deg_lookup` / `deg_sql` / `check_genes` tools both
  agents share, built once.
- `msp.__version__`, read from the installed package metadata.
- `DegTables` lists CSVs above 64 MB in its schema instead of loading them.
- Tests for the core-and-satellite statistics (`_minor_sibling_qc`,
  `_cell_level_outliers`, `_build_removal_mask`, stress detection, fractal
  marker selection), one real end-to-end integration on a synthetic
  two-sample dataset, keyword snapshots of every report section, and cgroup
  fixture tests for `msp.resources`.

### Changed
- Progress output goes through `logging` (the `msp` logger family) instead
  of `print`; message text is unchanged. CLI entry points call
  `msp.log.configure()` (one flushed stdout handler shared with
  `harness_bridge`), library entry points call `msp.log.ensure()` so
  notebook callers still see the lines. Degraded paths log at `WARNING`.
- `msp.integrate` is a package: `pipeline` (merge and the staged
  `integrate_adata`), `fragments`, `outliers`, `deg`, `qc`. Every previous
  `from msp.integrate import ...` name still resolves; outputs of the
  synthetic dataset are byte-identical before and after the split.
- Annotation UMAP palettes come from the `stanhue` package (`stanhue>=1.1.0`
  on PyPI). The `MSP_PALETTE_DIR` variable and the Claude-skill lookup are
  gone.
- `harmonypy>=2.0.0,<3` replaces the `harmonypy==0.2.0` pin: the C++ rewrite
  is numpy-only and about 11x faster on CPU (17 s vs 199 s for 81k cells);
  msp passes `ncores` from `msp.resources.available_cpus()`. `MSP_DEVICE` is
  gone (no torch, no GPU path). Corrected coordinates correlate at >= 0.99
  per PC with 0.2.0 on the reference dataset; see the developer guide for the
  full comparison. Harmony defaults change with it (`lamb` auto-estimated,
  `max_iter_kmeans` 4, looser convergence thresholds).
- Requires `agent-harness-bridge>=0.2.0,<0.3` (progress lines via
  `logging`, `configure_logging`).
- `cluster_order` compares every comma-separated part of a subcluster ID
  numerically, so `5,2` now sorts before `5,10`.
- Report sections are described by a `Section` dataclass and numbered from
  `(anchor, body)` pairs; an empty Per-cluster QC section no longer renders a
  bare heading.
- Fractal marker strip colours follow the lowest parent of a shared gene,
  matching its row position.
- `torch` is no longer needed at all. Lower bounds added for scanpy, anndata, pandas, numpy, scipy,
  scikit-learn, matplotlib, seaborn, igraph and adjustText.
- In-tree code imports `harness_bridge` directly.

### Deprecated
- `msp.harness` emits a `DeprecationWarning`; it will be removed in 0.4.

### Fixed
- `_build_removal_mask` aligns a partial cell-outlier table by cell ID
  without pandas' object-dtype downcasting warning.

## 0.2.0 - 2026-09-02

- PyPI distribution name `msp-sc` (import name stays `msp`);
  `standissect-lite>=0.2.0` from PyPI.
- Step invalidation and recovery (`.msp-state`, `.msp-history`), report-only
  rebuilds, resume checks on integration metadata.
