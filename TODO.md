# TODO

Open items from the 2026-09-04 code review. Phases 0 and 1 (tooling, dead
code, fragile spots, `main()` entry points) are done; the rest is listed here
in the recommended order. See the developer guide for the current contracts.

## Phase 2: module boundaries

Do the first two items first; they are low risk. Leave the `integrate.py`
split until Phase 3 has locked its statistics in tests.

- [ ] **`msp/evidence.py`**: move `DegTables`, `DegCache`, `_gene_table`,
      `_qc_table`, `_cluster_order`, `_parse_reference`, `_load_paga_neighbors`,
      `_load_removal_mask`, `_filter_deg`, `_format_deg` out of `inspect.py`
      and give them public names. `annotate.py` then stops importing
      underscore names from `inspect.py`. Keep re-exports in `inspect.py`
      for one release because ZMIP imports `_load_paga_neighbors` from
      `msp.annotate`.
- [ ] **`msp/agent_tools.py`**: one factory returning the `ToolSpec`s that
      inspect and annotate share verbatim (`deg_lookup`, `deg_sql`,
      `check_genes`). Keep `check_deg` per agent: inspect validates against
      the live subclustered key, annotate against the fixed base key, and
      forcing one signature would need callbacks that obscure that.
- [ ] **Split `integrate.py`** (about 1000 lines) into `integrate/pipeline.py`
      (`integrate_adata`, `run_multi_sample_pipeline`, `load_and_merge`),
      `integrate/fragments.py` (`_minor_sibling_qc`, `_fractal_marker_heatmap`),
      `integrate/outliers.py` (`_cell_level_outliers`,
      `_leiden_cluster_qc_violins`, `_build_removal_mask`), and
      `integrate/deg.py` (`_cluster_annotations`, stress panel). Keep
      `from msp.integrate import integrate_adata` working. Break
      `integrate_adata` into preprocess / embed / cluster / evidence / write
      steps. Verify with a byte-identical CSV comparison on a real run.
- [ ] **`print` to `logging`**: `logging.getLogger("msp")`, handler configured
      in the CLI `main()`s with immediate flush (Slurm logs must stay live).
      Coordinate with Agent Harness Bridge so agent output uses the same
      logger; otherwise two styles interleave.
- [ ] Drop the `msp.harness` shim after in-tree callers and ZMIP / ECA-RSI
      import `harness_bridge` directly.

## Phase 3: tests for the core statistics

Coverage is 63 percent overall and 53 percent in `integrate.py`. The
functions behind the core-and-satellite method are untested.

- [ ] `_minor_sibling_qc`: synthetic fragments around the `BIG_SIBLING_FRAC`,
      `BIG_SIBLING_N`, `MIN_N_FOR_TEST`, `DROP_PCT_THRESH` boundaries; assert
      `status`, `n_hits`, `recommend_removal` and the CSV columns.
- [ ] `_cell_level_outliers`: MAD plus hard floor gates at both resolutions;
      assert `cell_outliers.csv` and the summary agree.
- [ ] `_build_removal_mask`: union of fragment, cell, and inherited drop
      sources; alignment to `obs_names`.
- [ ] Stress detection in `_cluster_annotations`: `STRESS_HIT_THRESHOLD`
      across global and local views, `MT-` prefix, case-insensitive symbols.
- [ ] `_fractal_marker_heatmap`: separate marker selection from drawing, then
      test selection (ribosomal exclusion, per-parent top-N, first-parent-wins
      de-duplication).
- [ ] `report.py`: one keyword-snapshot test per section, and a test that
      `_number_sections` closes gaps when a section is absent.
- [ ] `resources.py`: cgroup v1 and v2 parsing against fixture files.
- [ ] Replace scanpy monkeypatching in `test_step_recovery.py` with a tiny
      real dataset where feasible; the patches break on any internal rename.
- [ ] Coverage gates in CI once the above land: `integrate` at 80 percent,
      total at 75 percent.

## Phase 4: dependencies and release

- [ ] `torch` is only needed by harmonypy 0.2.0; move it out of the direct
      dependencies (let harmonypy pull it) or into a `[gpu]` extra.
- [ ] Add lower bounds for `scanpy`, `anndata`, `pandas` that match the APIs
      used (`observed=True`, nullable `boolean` dtype, `flavor="igraph"`).
- [ ] Evaluate `harmonypy` 2.0.0 against the pinned 0.2.0 on a reference run
      (Z_corr orientation, defaults, numerics); either upgrade or document why
      the pin stays.
- [ ] `agent-harness-bridge` is already a core dependency; either make the
      base install truly agent-free or drop the "optional agent dependencies"
      wording from `msp/__init__.py` and `__main__.py`.
- [ ] Add `msp.__version__` sourced from package metadata, a CHANGELOG, and a
      `v0.2.0` tag.

## Smaller items noticed during review

- [ ] `_cluster_order` sorts by the float value before the first comma, so
      `"5,10"` sorts before `"5,2"`. Acceptable today; fix when subcluster IDs
      get deeper.
- [ ] `DegTables` still loads every remaining CSV into SQLite on each agent
      session; consider lazy `ATTACH` or a size cap if directories grow.
- [ ] `integrate_adata` re-imports `numpy` locally; clean up with the split.
- [ ] `report._number_sections` matches `<h2>` tags by string; move section
      metadata into a small dataclass when touching the report next.
