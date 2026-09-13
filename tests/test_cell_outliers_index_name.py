"""_cell_level_outliers must not rename the shared obs index: inputs with an obs
column called ``cell`` (Tabula Muris Senis) could no longer be written by anndata
("DataFrame.index.name ('cell') is also used by a column")."""

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from msp.integrate.outliers import _cell_level_outliers


def test_outlier_table_does_not_rename_obs_index(tmp_path):
    n = 60
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "cell": [f"orig{i}" for i in range(n)],
            "leiden_r1.0": pd.Categorical(rng.choice(["0", "1"], n)),
            "pct_counts_mt": rng.uniform(0, 5, n),
            "doublet_score": rng.uniform(0, 0.2, n),
        },
        index=[f"c{i}" for i in range(n)],
    )
    ad = anndata.AnnData(X=sp.csr_matrix(np.ones((n, 3), dtype=np.float32)), obs=obs)
    _cell_level_outliers(ad, ["leiden_r1.0"], [1.0], str(tmp_path))
    assert ad.obs.index.name is None
    ad.write_h5ad(tmp_path / "ok.h5ad")  # the failure mode: ValueError from anndata
