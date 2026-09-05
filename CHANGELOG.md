# Changelog

All notable changes to msp-sc. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added
- `msp.evidence`: the evidence layer (precomputed DEG tables, live DEG,
  per-cluster expression / QC / stability views, removal mask, PAGA
  neighbours) with public names; `msp.inspect` and `msp.annotate` keep their
  old underscore names as aliases for one release line.
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
- `msp.integrate` is a package: `pipeline` (merge and the staged
  `integrate_adata`), `fragments`, `outliers`, `deg`, `qc`. Every previous
  `from msp.integrate import ...` name still resolves; outputs of the
  synthetic dataset are byte-identical before and after the split.
- Annotation UMAP palettes come from the `stanhue` package (a git dependency
  pinned to v1.1.0 until it is on PyPI). The `MSP_PALETTE_DIR` variable and
  the Claude-skill lookup are gone.
- `cluster_order` compares every comma-separated part of a subcluster ID
  numerically, so `5,2` now sorts before `5,10`.
- Report sections are described by a `Section` dataclass and numbered from
  `(anchor, body)` pairs; an empty Per-cluster QC section no longer renders a
  bare heading.
- Fractal marker strip colours follow the lowest parent of a shared gene,
  matching its row position.
- `torch` is no longer a direct dependency (harmonypy pulls it in); msp never
  imports it. Lower bounds added for scanpy, anndata, pandas, numpy, scipy,
  scikit-learn, matplotlib, seaborn, igraph and adjustText.
- In-tree code imports `harness_bridge` directly.

### Deprecated
- `msp.harness` emits a `DeprecationWarning`; it will be removed in 0.3.

### Fixed
- `_build_removal_mask` aligns a partial cell-outlier table by cell ID
  without pandas' object-dtype downcasting warning.

## 0.2.0 - 2026-09-02

- PyPI distribution name `msp-sc` (import name stays `msp`);
  `standissect-lite>=0.2.0` from PyPI.
- Step invalidation and recovery (`.msp-state`, `.msp-history`), report-only
  rebuilds, resume checks on integration metadata.
