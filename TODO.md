# TODO

Open items from the 2026-09-04 code review, after the passes that closed the
evidence/tool split, the statistics tests, the report and resources tests,
the dependency cleanup, the logging switch and the harmonypy 2.0 upgrade
(see `CHANGELOG.md`). Items below are what remains, in the recommended order.

## Needs a decision or an external step

- [ ] **Remove `msp.harness`** in 0.4 (it warns since 0.3). ZMIP and ECA-RSI
      already import `harness_bridge` directly (both on bridge 0.2).

## Phase 3 leftovers

- [x] Keep `test_step_recovery.py::test_completed_integration_allows_external_annotation_report`
      as a fast recovery regression; the separate end-to-end integration test
      exercises the real numerical path. These cover different failure modes.
- [x] The configured total coverage floor is already 80 percent. Add direct
      `_cluster_context` / `_prior_label_columns` tests and real-graph
      `_subcluster_once` tests, including removed and singleton siblings.
      Remaining agent-session branch coverage can grow with concrete bugs.

## Smaller items

- [ ] Figures are not byte-reproducible: two runs of the same code on the
      synthetic dataset differ in `figures/umap_msp_leiden_r2.0.png`
      (adjustText's label repulsion is randomized and time-limited). The
      installed adjustText API has no dedicated seed argument: reproducible
      rendering needs controlled RNG state plus a fixed iteration budget,
      with concurrency and label-quality checks. Defer until byte-level
      figure reproducibility is required; this is not a scientific-output
      correctness fix.
- [ ] `DegTables` still loads every remaining CSV under 64 MB into SQLite on
      each agent session; consider lazy `ATTACH` if session start-up becomes
      noticeable.
- [ ] `report.py` section functions read their CSVs with the `csv` module and
      hand-build tables; a shared `_table(rows, columns, style=...)` helper
      would remove most of the repetition.

## Integration maintenance (2026-09-05)

- [x] Summarize Scanpy's repeated log2 numerical warnings in every MSP DEG
      path without replacing non-finite results or suppressing other errors.
- [x] Quiet Harmony iteration logs by default, with explicit verbose override.
- [x] Expose all seven helpers ZMIP used privately; keep old names for 0.3.
- [ ] Record the full 19Liu inspect/annotate/report content audit once the
      coordinated validation run completes (file existence is insufficient).
