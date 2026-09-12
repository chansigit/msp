# ComputeEndpoint design outline

Branch `compute-endpoint` (worktree `$SCRATCH/worktrees/msp-compute-endpoint`), tracking
[eca-rsi#8](https://github.com/chansigit/eca-rsi/issues/8). This is a design record, not yet
implemented code — see "Status" at the bottom for what actually exists.

## Goal

Decouple *what to compute* from *where it runs*, symmetric to the inference-side split that
already exists (`HARNESS` / `AGENT_MODEL_POOL` choose which LLM answers a decision; the kernel
code never talks to a provider directly). Today every heavy local computation (Harmony
integration, DEG, clustering) runs in-process, in whatever Slurm allocation the kernel itself is
running in. `ComputeEndpoint` lets a specific heavy call site hand its work to a different
process, possibly on a different node, without the caller changing.

Motivating waste: an agent call (e.g. annotate) can hold a 240 GB bigmem node for two hours while
barely touching CPU/RAM. Decoupling means the *coordinator* (agent calls, control flow) and the
*compute* (matmuls, Harmony, clustering) don't have to share one allocation's lifetime.

## Non-negotiable constraints (from discussion with the user)

- **Never assume the user has a Slurm allocation to spread work across.** Default behavior must
  be identical to today, with zero new dependencies imported, when nothing is configured.
- **The switch must be per-installation, not hardcoded.** A user with a single laptop-class node
  and a user with a warm multi-node pool both need this to work — the endpoint choice is a
  runtime config, not a compile-time assumption.
- **Must not disturb work already in flight.** All development happens in this worktree; the
  primary `projects/msp` checkout is untouched until a deliberate merge at a drain window (no
  round anywhere mid-stage in `msp`). This matters even though the feature is opt-in: editing any
  file under `msp/` changes `runtime_identity()`'s content hash for *everyone*, whether or not
  they use the new feature.

## Module breakdown

One new file, `msp/compute.py`, three pieces:

### 1. `ComputeEndpoint` protocol

Deliberately shaped like `concurrent.futures.Executor`, not invented from scratch:

```python
class ComputeEndpoint(Protocol):
    def submit(self, fn: Callable, *args, **kwargs) -> Future: ...
    def __enter__(self) -> "ComputeEndpoint": ...
    def __exit__(self, *exc) -> None: ...
```

Why this shape:
- `dask.distributed.Client` already implements `submit()` -> `Future` natively. A Dask backend is
  close to a passthrough, not an adapter layer.
- `Future.result()` (blocking) covers every call site we plan to touch first (one call, wait for
  it). Nothing here requires the caller to become async.
- The same shape scales to *later* needs (submitting several independent computations
  concurrently, e.g. parent-core DEG in different lineages) without an interface change — only
  more `submit()` calls from the caller. Not implemented now; the point is not designing ourselves
  into a corner for a feature we're deliberately deferring.

### 2. Backends

| Backend | This round | Later |
|---|---|---|
| `local` | **implemented** — `submit()` runs `fn(*args, **kwargs)` synchronously in-process, wraps the result in an already-resolved `concurrent.futures.Future`. Byte-for-byte, timing-for-timing identical to today's direct call. | — |
| `dask-local` | **implemented** (2026-09-12) — `distributed.LocalCluster(processes=True)`, multi-process on one node, no Slurm needed, torn down per `with` block | — |
| `dask` | **implemented** (2026-09-12) — attach to an already-running scheduler named by `MSP_DASK_SCHEDULER` (`tcp://host:port` or scheduler-file path). The shared warm pool of eca-rsi#8: pool outlives any step or run, this backend only opens/closes a client | — |

`dask-slurm` was originally reserved for a `dask_jobqueue.SLURMCluster` backend (scheduler here,
`sbatch` a worker job per step). Dropped without being built: eca-rsi#8's model is a pool
**shared across concurrent runs** with a lifetime much longer than one step, and an endpoint
scoped to one heavy step cannot own that pool. So the Slurm-specific part lives outside msp as a
launcher (`eca-rsi/container/dask-pool.sh`: scheduler on the current node, `ssh -f` a
container-wrapped `dask worker` into any allocation you hold, `status`, `stop`), and msp only
knows how to *attach*. `dask_jobqueue` is not a dependency.

### 3. `resolve_endpoint()`

```python
def resolve_endpoint() -> ComputeEndpoint:
    kind = os.environ.get("MSP_COMPUTE_ENDPOINT", "local")
    if kind == "local":
        return LocalEndpoint()
    if kind == "dask-local":
        return DaskLocalEndpoint()
    if kind == "dask":
        return DaskEndpoint()   # MSP_DASK_SCHEDULER names the pool
    raise ValueError(f"unknown MSP_COMPUTE_ENDPOINT={kind!r}")
```

Mirrors `resolve_agent_config()` / `resolve_model_pool()` in `agent-harness-bridge`: an env-var
driven factory, callers never import a concrete backend class directly. `dask`/`distributed` are
imported lazily inside the concrete backend classes, not at module top level — so a plain
`pip install msp-sc` (no extras) never touches dask, and `MSP_COMPUTE_ENDPOINT` unset never
attempts the import.

New optional dependency group: `msp-sc[dask]` (adds `dask[distributed]`, later `dask-jobqueue`).
Not part of the default install.

## Call-site contract

Exactly one call site touches this in the first pass: `integrate/pipeline.py::_embed()`
(Harmony integration).

```python
def _embed(ad, ...):
    with resolve_endpoint() as ep:
        fut = ep.submit(harmony_correct, embedding, batch_labels, ...)
        ad.obsm["X_pca_harmony"] = fut.result()
```

Rules for any function passed to `submit()`:
- **Pure, array-in/array-out.** Never pass an AnnData or other msp-internal object — only numpy
  arrays / plain data the function actually needs. This is what makes a future cross-node backend
  cheap: Dask's wire serialization only ever sees small, well-typed payloads, never a whole
  in-memory dataset.
- **No filesystem side effects assumed to be visible without a shared filesystem.** For the
  `dask-slurm` backend (later), inputs/outputs should be arrays passed through Dask's own
  data movement, not "write to path X, worker reads path X" — even though Sherlock's Lustre would
  happen to make that work, the design shouldn't depend on it silently.
- **Lifecycle is scoped to the call site.** `resolve_endpoint()` is called fresh at each site that
  needs it, used inside a `with` block, and torn down when that block exits — not held open across
  a whole round or the whole pipeline. Costs nothing for `local` (no real resource); keeps the
  eventual `dask-local`/`dask` backends from becoming an accidental long-lived shared
  service by construction.

## Explicit non-goals this round

- No changes outside `msp/`. `osp`, `zmip`, `eca-rsi`, `agent-harness-bridge` are untouched.
  `agent-harness-bridge`'s `HARNESS`/`AGENT_MODEL_POOL` and this `MSP_COMPUTE_ENDPOINT` are
  orthogonal switches; neither module imports the other.
- No `eca-rsi` `downstream.py` / `runtime_identity()` special-casing. `compute.py` is just another
  file under `msp/`, covered by the existing whole-package-hash rule.
- No second call site (ZMIP per-lineage dispatch) until the first one is proven.
- No nested submission (`worker_client()`/`secede()`/`rejoin()`) — relevant once a driver task
  needs to submit a sub-task from inside a worker; not needed for a single Harmony call.

## Spike findings (2026-09-11/12, outside the repo, `$SCRATCH/spikes/dask-compute-endpoint/`)

Used two live Slurm allocations as A (coordinator, `sh04-01n16`, bigmem) and B (worker,
`sh03-08n39`, normal) to de-risk the `dask-slurm` backend before building it.

**Bare-metal cross-node**: scheduler on A (`tcp://10.20.1.16:8786`), worker on B via
`ssh sh03-08n39 '... dask worker ...'`. Confirmed genuine remote execution (`socket.gethostname()`
returned from the worker call was `sh03-08n39.int`, not A). Real computation (1500×1500 matmul)
and an array round-trip (numpy array in, aggregate stats out) both correct.

**Apptainer**: rebuilt the venv *inside* `python312-slim.sif` (matches how the real `eca-ct`
environment is built), ran scheduler and worker each via `apptainer exec --bind /scratch,/oak,/home
$SIF ...`. Confirmed the worker executed inside the container, not merely visible through a bind
mount (`/etc/os-release` inside the executed task reported Debian 13 trixie, the container's OS —
not Sherlock's host CentOS 7). Same computation, correct result, comparable timing to bare metal.

**Operational gotcha, worth remembering for the real `dask-slurm` backend**: launching a remote
worker as `ssh node 'cmd &'` hung the whole call for the full timeout. `nohup`/background inside a
non-interactive ssh session does not fully detach — ssh still waits on inherited file descriptors
(stdin, even with stdout/stderr redirected). Fix: `ssh -f` (fork to background after auth, before
running the remote command). Any script that shells out to launch remote Dask workers needs this,
or it will silently deadlock the driver.

**Conclusion**: neither the network topology nor Apptainer are obstacles to `dask-jobqueue`'s
`SLURMCluster` model (scheduler here, workers spawned by generated `sbatch` scripts elsewhere) —
when that backend is eventually built, its generated worker launch command should shell into
`apptainer exec ... $VENV/bin/dask worker ...`, the same pattern `venvs/eca-ct/python` already
uses for every other kernel subprocess.

## `dask-local` build (2026-09-12) — two real gotchas hit and fixed

`DaskLocalEndpoint` (real `distributed.LocalCluster`, `processes=True`, torn down per `with`
block) is implemented and wired into the same `_embed()` Harmony call site as `local`. Verified
byte-for-byte identical `X_pca`/`X_pca_harmony`/`X_umap`/`obs` against both unmodified main and
the `local` backend, on the synthetic two-sample dataset (real Harmony path, not the single-batch
skip). Two things broke on the way there, both now fixed and worth remembering for any future
call site:

- **A driver script without `if __name__ == "__main__":` reruns itself in every worker.**
  `LocalCluster(processes=True)` spawns fresh interpreters that re-import the launching module.
  The verification script originally ran `write_samples()` at module top level; every spawned
  worker re-ran it, and several processes writing the same h5ad files at once hit HDF5 file-lock
  errors (`BlockingIOError: ... Resource temporarily unavailable`). Any script that constructs a
  `dask-local`/`dask` endpoint needs its real work behind a `__main__` guard — this is a
  property of *callers*, not of `compute.py` itself, but it will bite the first person who forgets
  it, so it is written down here.
- **A pandas `Categorical` column does not survive distributed's task-graph pickling.** Passing
  `ad.obs[[batch_col]]` (categorical dtype, the normal dtype for a batch/sample column) into
  `submit()` failed deep inside distributed's deserialization with
  `NotImplementedError` from `pandas.core.arrays.categorical.Categorical.__setstate__` — happening
  even with client and worker as the exact same interpreter, so it is a real serialization gap in
  the pandas/distributed pickling path, not an environment mismatch (despite dask's generic error
  text suggesting one). Fix: decategorize (`.astype({batch_col: "object"})`) before the value
  crosses a `ComputeEndpoint` boundary. This is now the general rule for every future call site,
  not just Harmony: **never pass a `Categorical`-dtype column through `submit()`** — convert to
  plain object/string dtype first, on the caller's side, before construting the arguments.

## `dask` (warm pool) build (2026-09-12) — cross-node parity is ulp-level, not bit-level

Pool: scheduler on the coordinator allocation (`sh04-01n16`, Intel Xeon 8462Y+), two worker
processes on the other allocation (`sh03-08n39`, AMD EPYC 7543), every process the container
interpreter with `PYTHONPATH` pointing at this worktree (workers must import the same `msp` as
the client — `submit(fn)` pickles module-level functions by reference). `_run_harmony` provably
executed on the remote node inside the container (task returned `sh03-08n39.int` / Debian 13).

Same synthetic two-sample pipeline, all under the container interpreter:

| pair | `X_pca` | `X_pca_harmony` | `X_umap` | `obs.csv` (labels) |
|---|---|---|---|---|
| main vs worktree `local` (same node) | identical | identical | identical | identical |
| main vs `dask-local` (same node) | identical | identical | identical | identical |
| main vs `dask` (Harmony on the AMD node) | identical | max abs diff 2.4e-7 | differs | identical |

The Harmony delta is float32 ulp noise from a different CPU vendor's BLAS kernels (same
`ncores`, same `random_state`); UMAP then amplifies it into visibly different coordinates while
every Leiden label stays the same. Two consequences worth stating plainly:

- **Bit-identity is a same-hardware property, not a backend property.** A heterogeneous pool
  gives numerically equivalent, not byte-equal, embeddings. Nothing in msp/ecarsi's identity or
  resume machinery compares embeddings (counts are compared, embeddings are not), so this does
  not break resume — but "re-run the round on the pool and diff the h5ad" is not a valid check.
- The earlier "`X_pca` differs" scare was the *host* interpreter (`dl2025`, no-BLAS numpy) vs the
  container one — two different numeric stacks, nothing to do with Dask. Always compare runs
  produced by the same interpreter.

Launcher gotchas already baked into `dask-pool.sh`: `ssh -f` (see spike findings), scheduler
file on `$SCRATCH` so nobody needs the scheduler's IP, `--nthreads 1` per worker process so
Harmony's own `ncores` threading is the only parallelism inside a task.

## Second and third call sites (2026-09-12): the graph stage and the DE stage

Two more `submit()` sites in msp, same array-in/array-out contract, each one task:

- **`_cluster` → `_run_cluster(X_pca_harmony, n_neighbors, resolutions)`**: neighbors + leiden at
  every resolution + UMAP on a bare AnnData carrying only the embedding. Returns obsp
  (connectivities / distances), every uns entry scanpy wrote, leiden partitions as
  `(codes, categories)`, and the UMAP; the driver puts them back so the object is exactly what
  scanpy-in-place would have produced.
- **`_cluster_annotations` → `_compute_de(X, var_names, labels, log1p, X_pca_harmony, keys)`**:
  phase 1 (neighbors on the survivors + PAGA + task list) and phase 2 (the thread-pooled
  wilcoxon runs) as one task; the driver writes the PAGA tables, logs the undersized clusters,
  and runs phase 3 unchanged. Results are keyed by leiden key, not object identity — the
  driver's objects are not the task's objects. This ships the survivors' log-normalized X
  (~1 GB at 60k cells) through the endpoint; marked `ponytail:` in the code — stage a file
  path if a remote pool ever makes the transfer the bottleneck.

Because zmip's per-lineage compute *is* `msp.integrate_adata`, every lineage subprocess gets all
three sites for free — no zmip change; it only needs `MSP_COMPUTE_ENDPOINT` / `MSP_DASK_SCHEDULER`
in its environment, which it inherits.

Parity against unmodified main (`compare_runs.py` in the spike dir: three embeddings, obs.csv,
all 22 CSVs under `msp_out/`, the h5ad's obs / obsp / obsm / uns):

| backend | differs from main in |
|---|---|
| `local` | nothing |
| `dask-local` | nothing |
| `dask` (Harmony, graph and DE all on the AMD node) | `X_pca_harmony`, `X_umap`, obsp (all downstream of the Harmony ulp delta); **every CSV identical**, including `deg_global_*`, `deg_local_*`, `paga_neighbors_*`, `stress_clusters`, and all cluster labels |

The scanpy warning "unsupported threading environment, rankdata in serial mode" seen in the
worker log is not pool-specific — unmodified main prints it too.

## Load-balancing test on a three-node pool (2026-09-12)

Pool: scheduler on `sh04-01n16` (Intel Xeon 8462Y+), 4 worker processes each on `sh03-08n39`
(AMD EPYC 7543) and `sh02-14n13` (a GPU allocation, Intel Xeon E5-2640 v4), all container-wrapped.

- **40 equal CPU-bound tasks**: wall 1.5 s; the faster AMD node took 24, the old E5 node 16
  (0.22 vs 0.33 s/task). Dask hands work to whoever frees up, so an uneven pool balances itself.
- **Four pipelines attached concurrently** (four client processes, one pool): 12 distinct tasks,
  `_run_cluster` and `_run_harmony` split 2/2 across the nodes, `_compute_de` 3/1; every CSV in
  every run identical to unmodified main. First task on a fresh worker process pays msp import
  + numba JIT (11–17 s on 150 cells), later tasks on the same process 0.4–0.6 s — a warm pool is
  warm per worker process, so `--nworkers` should not be far above the concurrent task count.
- **Found and fixed: dask deduplicates tasks by content.** `Client.submit` defaults to
  `pure=True`, keying a task on (function, argument contents); the first four-way run collapsed
  into roughly one computation shared across all four clients (task stream showed +1 task for
  +4 runs). Semantically fine for pure functions with identical inputs, but on a long-lived pool
  whose code may be updated mid-life it would serve a result computed by the previous code to a
  new client. Both Dask backends now submit with `pure=False`; identical concurrent runs each
  compute their own (test: two `submit(time.perf_counter_ns)` get distinct keys and values).
- **Bit-identity is per CPU model, not per vendor.** Harmony on the E5-2640 v4 also differs from
  the 8462Y+ at the ulp level (different SIMD generation, different BLAS kernel selection) even
  though both are Intel. Labels and all CSVs identical throughout.

Instrumentation lives in the spike dir: `loadbalance.py` (synthetic), `taskstream.py start|report`
(scheduler-side task stream, no pipeline code touched), `compare_runs.py`.

## GPU tier (2026-09-12, branch `compute-gpu`)

The pool can hold GPU workers; a heavy step can ask for one. Plumbing, all card-agnostic:

- `submit(fn, *args, tier="cpu"|"gpu", **kwargs)`. A tier is a dask worker *resource*:
  `dask-pool.sh worker <node> --gpu` starts one process per card with `--resources GPU=1` and
  `APPTAINER_NV=1` (the wrapper then passes `--nv`). `DaskEndpoint.has_tier("gpu")` asks the
  scheduler; `submit(tier="gpu")` checks it first, because dask would otherwise queue the task
  forever. `local` ignores the tier (in-process, the GPU is whatever the process sees);
  `dask-local` rejects it. Verified live: a gpu-tier task on the TITAN Xp node saw the card, a
  cpu-tier task on the same node's plain workers did not.
- **`MSP_COMPUTE_GPU=1`** (`compute.gpu_requested()`): each call site picks its rapids-singlecell
  implementation and submits with `tier="gpu"` — `_run_harmony_gpu` (`rsc.pp.harmony_integrate`,
  Harmony2), `_run_cluster_gpu` (`rsc.pp.neighbors` / `rsc.tl.leiden` / `rsc.tl.umap`, host arrays
  on return), `_compute_de(gpu=True)` (X moved to the GPU, `rsc.tl.rank_genes_groups` wilcoxon,
  sequential). Same inputs, same return shapes. Unset = the CPU path, byte for byte. Numerical
  differences are accepted (user's call, 2026-09-12); cugraph's Leiden is a different algorithm
  from igraph's, so partitions differ by construction (19 vs 26 clusters at r=1.0 on pbmc68k).

What the first GPU allocation (`sh02-14n13`, TITAN Xp, compute capability 6.1, driver 550 /
CUDA 12.4) taught:

- **RAPIDS does not run on Pascal** (needs compute capability ≥ 7.0 since 24.02), so
  rapids-singlecell — the actual prize: neighbors / UMAP / Leiden / DE on GPU — could not be
  tested. Needs a Volta-or-newer card: on Sherlock `-C GPU_CC:7.0` at minimum, in practice an
  Ampere/Hopper node (`-C "GPU_GEN:AMP|GPU_GEN:HPR"`).
- torch works, but the default wheel (2.14, cu130) refuses the 550 driver; `torch==2.6.0` from
  the cu124 index runs (sm_50…sm_90 in its arch list). sgemm 4000³ 0.21 s.
- **harmony-pytorch on this card is a loss**: 60k × 50, 4 batches, GPU 32 s (its own CPU path
  53 s) versus harmonypy 2.0's C++ CPU solver at 13 s on 66k. Harmony is not where the GPU pays;
  the graph stage and DE are, and those need RAPIDS.

### RTX 3090 (`sh03-15n01`, compute capability 8.6, node CPU 8-core EPYC 7502P), pbmc68k 68,579 × 14,771

| call site | CPU (this node) | GPU |
|---|---|---|
| `_run_harmony` (4 batches) | 4.7 s | 0.3 s warm; 26 s on a worker's first call (rsc import + JIT) |
| `_run_cluster` (neighbors + 3 Leiden + UMAP) | 101 s | 18.5 s |
| `_compute_de` (neighbors + PAGA + global + 26 locals) | 282 s | 14.1 s |

Top DE genes identical, scores agree to ~1e-6. Verified three ways: the functions directly on
the node, the whole synthetic pipeline in-process with the switch on (`local` backend), and
through a pool — GPU worker on the 3090 node, client on another node, all three tasks routed to
it by `tier="gpu"`.

**End to end through ecarsi (2026-09-12, chondroatlas 12_Yangetal, 12,880 cells × 22,492 genes, 3
samples)**: `eca-rsi run` with `MSP_COMPUTE_ENDPOINT=dask MSP_COMPUTE_GPU=1`, a one-worker GPU pool on
the 3090 node, driver on another node. Round-1 integrate: `_run_harmony_gpu` 2.2 s, `_run_cluster_gpu`
2.4 s, `_compute_de` (gpu) 22.5 s of compute on the worker; the whole integrate stage 2.5 min wall,
the rest being normalize / HVG / PCA / QC tables / figures on the driver. The agent stages (inspect
16 requests, 700k input tokens) dominate the round as before — which is the point of eca-rsi#8.

**Install path that works** (the rsc blog's nanobind rewrite explains the rest): the prebuilt
wheel is the separately named `rapids-singlecell-cu12[rapids]` from `pypi.nvidia.com`; plain
`rapids-singlecell` ≥ 0.15 on PyPI is sdist-only (CMake + nvcc + C++). RAPIDS must be pinned to
one release or pip backtracks through cudf versions, downloading a 700 MB `libcudf` wheel per
candidate. `container/build.sh` has it as `ECA_GPU=1` (`ECA_RAPIDS`, default 26.6), built once
as `venvs/eca-ct-gpu` — the production venv is untouched. `dask-pool.sh worker <node> --gpu`
runs that venv when `ECA_CT_GPU_ROOT` is set.

**Cross-venv pool**: the GPU venv carries numpy 2.4.6 and dask 2026.1.1 (RAPIDS pins) against
the production venv's 2.5.3 / 2026.8.0. Works (distributed prints a VersionMismatchWarning);
plain workers must never run the GPU venv, so `--gpu` is the only path that uses it. A GPU
worker also accepts cpu-tier tasks (dask resources restrict, they do not reserve) — so on a
mixed pool some cpu work may land on the GPU node's venv. Acceptable for now; if it matters,
give GPU workers `--nworkers 1` only and keep the node's plain workers on the production venv.

The earlier TITAN Xp probe venv (`$SCRATCH/spikes/gpu-venv`, torch) and the two RAPIDS trial
venvs (`gpu-venv-rapids`, `gpu-venv-rsc`) under `$SCRATCH/spikes/` are deletable.

## Rollout order

1. ~~`msp/compute.py`: `ComputeEndpoint` protocol + `LocalEndpoint` + `resolve_endpoint()`.~~ Done.
2. ~~Wire `_embed()`'s Harmony call through it. Verify byte-identical output vs. the un-wired
   path on a real dataset.~~ Done.
3. Merge to `msp` main at a drain window (no msp stage mid-round anywhere). Release as a normal
   msp version bump. **Deliberately deferred at the user's own call (2026-09-12: "不急吧,我们功能
   都没开发呢" — nothing worth merging yet) even though steps 1-2 were validated and a drain window
   existed at the time.**
4. ~~Build `dask-local` for real, validated the same way (byte-identical output, via a real local
   multi-process cluster).~~ Done ahead of the original order (originally step 4, gated on 1-3
   being merged first) at the user's explicit request to build the real backend now rather than
   wait. Two real bugs found and fixed in the process (see "dask-local build" above); not merged
   to `msp` main either, same reasoning as step 3.
5. ~~Build the warm-pool backend, reusing the spike's Apptainer-wrapped worker-launch pattern.~~
   Done as `dask` + `eca-rsi/container/dask-pool.sh` (see above); `dask-slurm` dropped.
6. ~~A second call site (Leiden / DEG in msp, or ZMIP per-lineage).~~ Done: graph stage and DE
   stage, which cover zmip's lineages too (see above).
7. ~~GPU implementations behind `tier="gpu"`.~~ Done on an RTX 3090 (see above); branch
   `compute-gpu` on top of `compute-endpoint`, neither merged.
8. Left on the driver: normalize / HVG / scale / PCA (seconds), standissect, QC tables, figures,
   and the agent calls. Nothing else in msp is minutes-long on 60k cells.

## Status

`msp/compute.py` has `ComputeEndpoint`, `LocalEndpoint`, `DaskLocalEndpoint`, `DaskEndpoint`,
`resolve_endpoint()`. Three call sites go through it: Harmony (`_run_harmony`), the graph stage
(`_run_cluster`) and the DE stage (`_compute_de`) — which is also every heavy step of a zmip lineage.
`local`/`dask-local` verified byte-identical against unmodified main; `dask` verified on a real
two-node pool (labels identical, embeddings ulp-equivalent, see above). Tests:
`tests/test_compute.py`, `tests/test_compute_dask.py` (skipped without `dask[distributed]`).
Pool launcher: `eca-rsi/container/dask-pool.sh`.
**Nothing in this worktree has been merged to `msp` main** — that merge is intentionally on hold,
not blocked by anything technical.
