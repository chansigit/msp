import weakref
import numpy as np
import pandas as pd
import anndata as an
from scipy import sparse
import pytest
from msp.agent_data import metadata, materialize, apply
from msp.checkpoint import data_identity, agent_identity


def test_disposable_expression_preserves_exact_data_and_working_labels(tmp_path):
    a = an.AnnData(sparse.csr_matrix(np.arange(60).reshape(10, 6), dtype=float),
                   obs=pd.DataFrame({'cluster': ['a']*5+['b']*5}, index=['001', 'NA', *map(str, range(8))]))
    a.layers['counts'] = a.X.copy()
    a.obsp['connectivities'] = sparse.eye(10, format='csr')
    a.obsm['X_umap'] = np.arange(20).reshape(10, 2)
    a.uns['neighbors'] = {'connectivities_key': 'connectivities'}
    path = tmp_path/'integrated.h5ad'
    a.write_h5ad(path)
    meta = metadata(path)
    assert meta.X is None and not meta.layers and not meta.obsp
    assert agent_identity(meta, tmp_path, [], __file__) == agent_identity(a, tmp_path, [], __file__)
    with materialize(meta) as full:
        assert data_identity(full, []) == data_identity(a, [])
        reference = weakref.ref(full.X.data)
        full.obs['refined'] = ['a,0']*3+['a,1']*2+['b']*5
        full.uns['refined'] = {'resolution': .5}
    assert full.X is None and not full.layers and not full.obsp
    assert reference() is None
    assert meta.obs['refined'].tolist() == ['a,0']*3+['a,1']*2+['b']*5
    with materialize(meta) as again:
        assert again.uns['refined']['resolution'] == .5
        np.testing.assert_array_equal(again.layers['counts'].toarray(), a.X.toarray())
    with pytest.raises(RuntimeError):
        with materialize(meta) as disposable:
            raise RuntimeError('tool failed')
    assert disposable.X is None
    # Outside users' input files remain unchanged by both successful/failed tools.
    assert data_identity(an.read_h5ad(path), []) == data_identity(a, [])
    path.touch()
    with pytest.raises(ValueError, match='changed'):
        with materialize(meta):
            pass
