"""Small synthetic multi-sample dataset for end-to-end tests.

Three populations with disjoint marker genes, two samples with a mild
batch effect, inherited QC columns and one sample-local leiden column —
the same shape an osp `clustered.h5ad` has, at a size (150 cells) the full
integration runs on in seconds.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

N_GENES = 60
N_POPULATIONS = 3
CELLS_PER_POPULATION = 25
SAMPLES = ("A", "B")


def make_sample(name, rng, batch_scale):
    n = N_POPULATIONS * CELLS_PER_POPULATION
    genes = [f"G{i}" for i in range(N_GENES)]
    counts = rng.poisson(1.0, (n, N_GENES)).astype(np.int32)
    population = np.repeat(np.arange(N_POPULATIONS), CELLS_PER_POPULATION)
    for p in range(N_POPULATIONS):
        markers = slice(p * 10, p * 10 + 10)
        counts[population == p, markers] += rng.poisson(8.0, (CELLS_PER_POPULATION, 10)).astype(np.int32)
    # a mild, sample-wide effect on a gene block harmony can absorb
    counts[:, 40:50] += rng.poisson(batch_scale, (n, 10)).astype(np.int32)
    total = counts.sum(axis=1)
    obs = pd.DataFrame(
        {
            "sample_id": pd.Categorical([name] * n),
            "leiden_r1.0": pd.Categorical(population.astype(str)),
            "population": pd.Categorical([f"pop{p}" for p in population]),
            "total_counts": total.astype(float),
            "n_genes_by_counts": (counts > 0).sum(axis=1).astype(float),
            "pct_counts_mt": rng.uniform(1, 6, n),
            "doublet_score": rng.uniform(0.01, 0.15, n),
            "_qc_action": pd.Categorical(["keep"] * (n - 2) + ["drop", "flag"]),
        },
        index=[f"{name}_cell{i}" for i in range(n)],
    )
    sample = ad.AnnData(counts.astype(np.float32), obs=obs, var=pd.DataFrame(index=genes))
    sample.layers["counts"] = counts.copy()
    return sample


def write_samples(root):
    """Two clustered.h5ad files under ``root``; returns their paths."""
    rng = np.random.default_rng(2024)
    paths = []
    for name, scale in zip(SAMPLES, (0.0, 2.0), strict=True):
        path = root / f"{name}.clustered.h5ad"
        make_sample(name, rng, scale).write_h5ad(path)
        paths.append(path)
    return paths
