# TODO

Open items from the 2026-09-04 code review, after the 2026-09-05 pass that
closed the evidence/tool split, the statistics tests, the report and
resources tests, and the dependency cleanup (see `CHANGELOG.md`). Items
below are what remains, in the recommended order.

## Needs a decision or an external step

- [ ] **stanhue on PyPI.** `pyproject.toml` pins
      `stanhue @ git+https://github.com/chansigit/stanhue.git@v1.1.0`. PyPI
      rejects uploads whose dependencies are direct URL references, so the
      next msp-sc release cannot be published until stanhue is on PyPI and
      the pin becomes `stanhue>=1.1.0`.
- [ ] **harmonypy 2.0.0.** Evaluate against the pinned 0.2.0 on a reference
      run (Z_corr orientation, defaults, numerics); either upgrade or record
      why the pin stays. Needs a real dataset and a GPU or a long CPU job.
- [ ] **Remove `msp.harness`** in 0.3 (it already warns). ZMIP and ECA-RSI
      import `harness_bridge` directly.
- [ ] **Release 0.3.0**: bump `version`, move the Unreleased section of
      `CHANGELOG.md`, tag. ZMIP pins `msp-sc<0.3` and imports underscore
      names from `msp.inspect` / `msp.annotate`; update it to the public
      `msp.evidence` names before removing the aliases.

## Phase 2 leftovers

- [ ] **`print` to `logging`**: `logging.getLogger("msp")`, a handler with
      immediate flush configured in the CLI `main()`s (Slurm logs must stay
      live). Coordinate with Agent Harness Bridge so agent output uses the
      same logger; otherwise two styles interleave. Tests that read stdout
      (`capsys`) move to `caplog`.

## Phase 3 leftovers

- [ ] `test_step_recovery.py::test_completed_integration_allows_external_annotation_report`
      still monkeypatches scanpy internals; the end-to-end test covers the
      real path, so either fold its recovery assertions into that test or
      leave this one as the fast variant.
- [ ] Raise the coverage floor in `pyproject.toml` as `integrate.py` and
      `annotate.py` gain tests (target: `integrate` 80 percent, total 80
      percent). `_cluster_context`, `_prior_label_columns`, and
      `_subcluster_once` are the largest untested pieces.

## Smaller items

- [ ] Figures are not byte-reproducible: two runs of the same code on the
      synthetic dataset differ in `figures/umap_msp_leiden_r2.0.png`
      (adjustText's label repulsion is randomized). Seed it in
      `plots._repel_on_data_labels` (adjustText >= 1.0 accepts a seed) if
      figure diffs are ever needed; every CSV and the H5AD are already
      reproducible.
- [ ] `DegTables` still loads every remaining CSV under 64 MB into SQLite on
      each agent session; consider lazy `ATTACH` if session start-up becomes
      noticeable.
- [ ] `report.py` section functions read their CSVs with the `csv` module and
      hand-build tables; a shared `_table(rows, columns, style=...)` helper
      would remove most of the repetition.
