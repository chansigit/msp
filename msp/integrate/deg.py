"""Cluster Annotations: PAGA neighbours, global and local DEG tables at the
r1.0 / r2.0 resolutions, and the dissociation-stress signature check."""

from __future__ import annotations

import logging
import os
from copy import deepcopy

import anndata as an
import numpy as np
import pandas as pd
import scanpy as sc

from ..compute import gpu_requested, resolve_endpoint
from ..deg_logging import rank_genes_groups

log = logging.getLogger(__name__)

# Conservative "dissociation stress" core panel (not osp's full ~130-gene
# DISSOCIATION_GENES_HS — that one includes ECM/lineage genes like DCN,
# LMNA, SERPINE1 that are real cell-identity markers in plenty of tissues).
# Two independent, mechanistically distinct, well-established acute-
# dissociation-stress axes: heat-shock/chaperone response, and AP-1/
# immediate-early transcription — both firing together is far more specific
# than one broad gene list. Human symbols; matched case-insensitively
# (dataset gene names uppercased before lookup) so mouse data works too.
STRESS_GENES_CORE = [
    "HSPA1A",
    "HSPA1B",
    "HSPA8",
    "HSPB1",
    "HSP90AA1",
    "HSP90AB1",
    "HSPH1",
    "HSPE1",
    "DNAJA1",
    "DNAJB1",
    "DNAJB4",
    "FOS",
    "FOSB",
    "JUN",
    "JUNB",
    "JUND",
    "EGR1",
    "EGR2",
    "ATF3",
    "NR4A1",
    "PPP1R15A",
    "ZFP36",
    "IER2",
    "IER3",
    "DUSP1",
]
STRESS_GENE_SET = set(STRESS_GENES_CORE)
STRESS_HIT_THRESHOLD = 3  # a cluster is "stress" if MORE than this many top genes hit
STRESS_CHECK_TOP_N = 10  # matches what the report displays, independent of top_n_de
MIN_DE_GROUP_SIZE = 10  # clusters smaller than this are excluded from DE comparisons
# (both global one-vs-rest and local vs-PAGA-neighbors views)


def _is_stress_gene(symbol) -> bool:
    su = str(symbol).upper()
    return su in STRESS_GENE_SET or su.startswith("MT-")


def _stress_hits(names) -> list[str]:
    return [n for n in names if _is_stress_gene(n)]


def _global_deg_workspace(ad):
    """Private mutable metadata with a shared, read-only expression matrix.

    Scanpy's Wilcoxon reads X and writes obs/uns metadata. Do not copy
    counts layers or graph arrays for every global task.
    """
    return an.AnnData(X=ad.X, obs=ad.obs.copy(), var=ad.var.copy(), uns=deepcopy(ad.uns))


def _compute_de(X, var_names, labels, log1p, rep, keys, gpu=False):
    """Phase 1 (neighbors + PAGA + task list) and phase 2 (the wilcoxon runs)
    of :func:`_cluster_annotations`, on a bare AnnData rebuilt from arrays so
    the DE stage can run on any ComputeEndpoint. ``labels`` maps each leiden
    key to ``(codes, categories)``. Returns per-key PAGA neighbour rows, the
    undersized clusters skipped, the plan, and results as
    ``(key, view, cluster, DataFrame)`` -- keyed by name, never by object
    identity, because the caller's objects are not these objects.

    ``gpu=True``: X lives on the GPU (rapids-singlecell), neighbors and the
    wilcoxon runs are rsc's, sequential (each takes well under a second;
    no point sharing one GPU between threads)."""
    from concurrent.futures import ThreadPoolExecutor

    from ..resources import available_cpus

    if gpu:
        import rapids_singlecell as rsc

        neighbors, rgg = rsc.pp.neighbors, (lambda a, key, **kw: rsc.tl.rank_genes_groups(a, key, **{k: v for k, v in kw.items() if k != "use_raw"}))
    else:
        neighbors, rgg = sc.pp.neighbors, rank_genes_groups

    obs = pd.DataFrame({k: pd.Categorical.from_codes(*labels[k]) for k in keys}, index=pd.RangeIndex(X.shape[0]).astype(str))
    ad_excl = an.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(var_names)), uns={"log1p": dict(log1p)})
    if rep is not None:
        ad_excl.obsm["X_pca_harmony"] = rep
    if gpu:
        rsc.get.anndata_to_GPU(ad_excl)
    log.info("== cluster annotations: neighbors on the excluded subset" + (" [gpu]" if gpu else ""))
    neighbors(ad_excl, use_rep="X_pca_harmony")

    # phase 1 (sequential, cheap): PAGA + neighbour tables + the task list
    plan, paga, skipped = [], {}, {}  # per key: dict(key, cats, valid_groups, top3)
    for key in keys:
        log.info(f"== cluster annotations on {key}")
        sc.tl.paga(ad_excl, groups=key)
        conn = ad_excl.uns["paga"]["connectivities"].toarray()
        cats = list(ad_excl.obs[key].cat.categories)

        top3, neighbor_rows = {}, []
        for i, c in enumerate(cats):
            order = np.argsort(conn[i])[::-1]
            picked = [cats[j] for j in order if j != i and conn[i, j] > 0][:3]
            top3[c] = picked
            for rank, nb in enumerate(picked, start=1):
                neighbor_rows.append(
                    {
                        "cluster": c,
                        "neighbor": nb,
                        "rank": rank,
                        "connectivity": round(float(conn[i, cats.index(nb)]), 4),
                    }
                )
        paga[key] = neighbor_rows

        # wilcoxon needs >=2 cells per group to run at all, but a cluster that
        # tiny (can survive this far when it's PAGA-connected enough not to get
        # merged) doesn't give trustworthy DE either; require MIN_DE_GROUP_SIZE
        sizes = ad_excl.obs[key].value_counts()
        valid_groups = [c for c in cats if sizes.get(c, 0) >= MIN_DE_GROUP_SIZE]
        skipped[key] = [c for c in cats if c not in valid_groups]
        if not valid_groups:
            continue
        plan.append({"key": key, "cats": cats, "valid": valid_groups, "top3": top3})

    # phase 2 (parallel): ad_excl stays read-only throughout the pool.
    # Global tasks share expression buffers but own metadata; locals own subsets.
    def run_global(item):
        key = item["key"]
        slot = f"_rgg_{key}"
        work = _global_deg_workspace(ad_excl)
        rgg(work, key, groups=item["valid"], method="wilcoxon", use_raw=False, pts=True, key_added=slot)
        gdf = sc.get.rank_genes_groups_df(work, group=None, key=slot)
        # Scanpy omits group when only one group qualifies for testing.
        if "group" not in gdf and len(item["valid"]) == 1:
            gdf.insert(0, "group", item["valid"][0])
        return gdf.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"})

    def run_local(item, c):
        key = item["key"]
        neighbors = item["top3"].get(c, [])
        if not neighbors:
            return None
        sub = ad_excl[ad_excl.obs[key].isin([c, *neighbors])].copy()
        if int((sub.obs[key] == c).sum()) < MIN_DE_GROUP_SIZE:
            return None
        rgg(sub, key, groups=[c], reference="rest", method="wilcoxon", use_raw=False, pts=True)
        ldf = sc.get.rank_genes_groups_df(sub, group=c)
        ldf = ldf.rename(columns={"pct_nz_group": "pct1", "pct_nz_reference": "pct2"})
        # rank_genes_groups_df drops the "group" column when `group` is a scalar
        # (only keeps it for group=None/list) — put it back for schema parity
        # with the global-view CSV and so the report can key off it
        ldf.insert(0, "group", c)
        ldf["neighbors"] = "|".join(neighbors)
        return ldf

    n_workers = 1 if gpu else max(1, min(available_cpus(), 8))
    log.info(f"== cluster annotations: DE on {n_workers} thread(s)")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for item in plan:
            futures.append((item["key"], "global", None, pool.submit(run_global, item)))
            for c in item["cats"]:
                futures.append((item["key"], "local", c, pool.submit(run_local, item, c)))
        results = [(key, view, c, f.result()) for key, view, c, f in futures]
    return {"paga": paga, "skipped": skipped, "plan": plan, "results": results}


def _cluster_annotations(ad, remove_mask, leiden_keys, resolutions, outdir, top_n_de=50):
    """Cluster Annotations: for the r1.0 and r2.0 leiden clusterings, two DE
    views per cluster — global (one-vs-rest, as before) and local (one-vs-
    its-top-3-PAGA-neighbors, pooled). remove_mask (see
    _preannotation_removal, the union of minor_sibling_qc fragments and
    cell-level doublet/ambient outliers) is excluded from BOTH views (and
    from the PAGA graph itself) — this is a computation-only exclusion, msp
    never drops cells from ad or the written h5ad.

    Each (key, cluster) is also checked for a dissociation-stress signature
    (STRESS_GENES_CORE / mitochondrial genes) among its top
    STRESS_CHECK_TOP_N genes, in each view separately — but if EITHER view
    hits the threshold, the whole (key, cluster) is recommend_removal,
    written to stress_clusters.csv. Same rule as everywhere else in msp:
    propose, never remove cells directly.

    The wilcoxon runs (2 global + one local per cluster per key, all
    independent of each other) go through a thread pool sized to the CPUs
    this process may actually use (msp.resources): on a 59k-cell object the
    stage dropped from ~5 min to ~2.3 min. Outputs are assembled in the
    same key/cluster order as the old sequential loop, so the CSVs are
    byte-identical; every run owns its mutable metadata, so global result
    writes cannot race with local subset copying."""
    keep_mask = ~remove_mask
    n_excluded = int((~keep_mask).sum())
    ad_excl = ad[keep_mask].copy()
    log.info(
        f"== cluster annotations: excluding {n_excluded} recommend_removal cells ({ad_excl.n_obs}/{ad.n_obs} remain)",
    )

    res_to_key = dict(zip(resolutions, leiden_keys, strict=True))
    target = [(r, res_to_key[r]) for r in (1.0, 2.0) if r in res_to_key]
    if not target:
        log.warning("== cluster annotations: neither r1.0 nor r2.0 in resolutions — skipping")
        return

    # phases 1+2 run as one ComputeEndpoint task on arrays only (see
    # pipeline._run_harmony); categoricals travel as (codes, categories).
    # ponytail: ships the whole log-normalized X of the survivors through the
    # endpoint (~1 GB at 60k cells); stage a file path instead if a remote pool
    # ever makes that transfer the bottleneck.
    keys = [key for _, key in target]
    labels = {k: (ad_excl.obs[k].cat.codes.to_numpy(), list(ad_excl.obs[k].cat.categories)) for k in keys}
    rep = ad_excl.obsm["X_pca_harmony"] if "X_pca_harmony" in ad_excl.obsm else None
    gpu = gpu_requested()
    with resolve_endpoint() as ep:
        out = ep.submit(
            _compute_de, ad_excl.X, list(ad_excl.var_names), labels, dict(ad_excl.uns.get("log1p", {})), rep, keys,
            gpu=gpu, tier="gpu" if gpu else "cpu",
        ).result()
    plan, results = out["plan"], out["results"]
    for key in keys:
        # Keep the existing columns even when this graph has no positive edges.
        pd.DataFrame(out["paga"][key], columns=["cluster", "neighbor", "rank", "connectivity"]).to_csv(
            os.path.join(outdir, f"paga_neighbors_{key}.csv"), index=False
        )
        if out["skipped"][key]:
            log.info(
                f"== cluster annotations: skipping global DE for undersized "
                f"(<{MIN_DE_GROUP_SIZE} cells) cluster(s) {out['skipped'][key]} in {key}",
            )

    # phase 3 (sequential): write tables + stress rows in the old loop's order
    stress_rows = []
    for item in plan:
        key = item["key"]
        gdf = next(res for k, view, c, res in results if k == key and view == "global")
        gdf.groupby("group", observed=True).head(top_n_de).reset_index(drop=True).to_csv(
            os.path.join(outdir, f"deg_global_{key}.csv"), index=False
        )
        for c, sub in gdf.groupby("group", observed=True).head(STRESS_CHECK_TOP_N).groupby("group", observed=True):
            hits = _stress_hits(sub["names"].tolist())
            stress_rows.append(
                {
                    "key": key,
                    "cluster": c,
                    "view": "global",
                    "n_hits": len(hits),
                    "hit_genes": "|".join(hits),
                    "stress": len(hits) > STRESS_HIT_THRESHOLD,
                }
            )
        local_rows = []
        for k, view, c, ldf in results:
            if k != key or view != "local" or ldf is None:
                continue
            hits = _stress_hits(ldf.head(STRESS_CHECK_TOP_N)["names"].tolist())
            stress_rows.append(
                {
                    "key": key,
                    "cluster": c,
                    "view": "local",
                    "n_hits": len(hits),
                    "hit_genes": "|".join(hits),
                    "stress": len(hits) > STRESS_HIT_THRESHOLD,
                }
            )
            local_rows.append(ldf.head(top_n_de))
        if local_rows:
            pd.concat(local_rows, ignore_index=True).to_csv(os.path.join(outdir, f"deg_local_{key}.csv"), index=False)

    if stress_rows:
        stress_df = pd.DataFrame(stress_rows)
        # a cluster is recommend_removal overall if EITHER its global or its
        # local view hit the stress threshold — not judged per-view
        overall = (
            stress_df.groupby(["key", "cluster"])["stress"]
            .any()
            .reset_index()
            .rename(columns={"stress": "recommend_removal"})
        )
        stress_df = stress_df.merge(overall, on=["key", "cluster"])
        stress_df.to_csv(os.path.join(outdir, "stress_clusters.csv"), index=False)
        n_removal = int(overall["recommend_removal"].sum())
        log.info(
            f"== cluster annotations: {n_removal}/{len(overall)} (key, cluster) pairs "
            f"recommend_removal (stress signature)",
        )
