# TODO

Open items from the 2026-09-04 code review, after the passes that closed the
evidence/tool split, the statistics tests, the report and resources tests,
the dependency cleanup, the logging switch and the harmonypy 2.0 upgrade
(see `CHANGELOG.md`). Items below are what remains, in the recommended order.

## Needs a decision or an external step

- [ ] **Publish 0.3.0**: `agent-harness-bridge` 0.2.0 is on PyPI; publish
      `msp-sc` 0.3.0, then `zmip` 0.3.0 (it depends on msp-sc 0.3), tagging
      each after upload.
- [ ] **Remove `msp.harness`** in 0.4 (it warns since 0.3). ZMIP and ECA-RSI
      import `harness_bridge` directly.
- [ ] **ECA-RSI** still pins `agent-harness-bridge[all]==0.1.0`; it keeps
      working (the bridge attaches its default handler when nobody configured
      logging), but move it to `>=0.2.0,<0.3` and `configure_logging` when it
      is next touched.

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
