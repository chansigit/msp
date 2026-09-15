"""Split execution preserves legacy evidence and publishes an immutable SQL view."""
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from synthetic_data import write_samples
from msp.integrate import integrate_adata, load_and_merge
from msp.integrate.deg import compute_deg_task, load_deg_input, write_deg_results
from msp.evidence import DegTables
from msp.steps import step_pending


def test_deferred_integration_and_readonly_database(tmp_path):
    inputs = write_samples(tmp_path)
    arguments = dict(batch_col='sample_id', species='human', resolutions=(.3,1.,2.),
                     n_top_genes=30,n_pcs=10,n_neighbors=10)
    legacy, split = tmp_path/'legacy', tmp_path/'split'
    for directory, deferred in ((legacy,False),(split,True)):
        integrate_adata(load_and_merge(inputs,'sample_id'),outdir=str(directory),defer_deg=deferred,**arguments)
    assert step_pending(split,'integrate') and not step_pending(legacy,'integrate')
    assert not list(split.glob('deg_global_*.csv'))
    data = load_deg_input(split/'deg_input')
    from scipy import sparse
    assert not (data.X.data if sparse.issparse(data.X) else data.X).flags.writeable
    plan = json.loads((split/'deg_plan.json').read_text())
    results=[]
    for item in plan['plan']:
        for cluster in [None,*item['valid']]:
            frame = compute_deg_task(data,item,cluster)
            results.append((item['key'],'global' if cluster is None else 'local',cluster,frame))
    write_deg_results({**plan,'results':results},plan['keys'],str(split))
    for original in legacy.glob('deg_*.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(original),pd.read_csv(split/original.name))
    with DegTables(split,'msp_leiden_r2.0') as tables:
        expected=tables.sql('SELECT count(*) AS n FROM deg')
        tables.write_database(tmp_path/'deg.sqlite',provenance={'version':'one'})
    for path in split.glob('*.csv'):
        path.unlink()
    with DegTables(database=tmp_path/'deg.sqlite',base_key='msp_leiden_r2.0') as tables:
        assert tables.sql('SELECT count(*) AS n FROM deg')==expected
        with pytest.raises(sqlite3.DatabaseError):
            tables.conn.execute('DELETE FROM deg')
        assert 'SQL error' in tables.sql('SELECT load_extension("missing")')
    assert not (tmp_path/'deg.sqlite').stat().st_mode & 0o222


def test_no_de_when_reference_population_is_empty():
    from msp.integrate.deg import prepare_deg
    from scipy.sparse import csr_matrix
    _,plan=prepare_deg(csr_matrix(np.ones((10,3))),['a','b','c'],
        {'key':(np.zeros(10,dtype=int),['only'])},{},np.arange(30).reshape(10,3),['key'])
    assert plan['plan']==[] and plan['skipped']=={'key':['only']}


def test_empty_deg_key_survives_database_publication(tmp_path):
    pd.DataFrame(columns=['group','names','logfoldchanges','pvals_adj','pct1','pct2']).to_csv(
        tmp_path/'deg_global_msp_leiden_r2.0.csv',index=False)
    with DegTables(tmp_path) as tables:
        assert tables.keys==['msp_leiden_r2.0']
        expected=tables.lookup(key='msp_leiden_r2.0',cluster='0')
        assert 'empty result' in expected and 'check_deg' not in expected
        tables.write_database(tmp_path/'deg.sqlite',provenance={'version':'empty'})
    with DegTables(database=tmp_path/'deg.sqlite') as tables:
        assert tables.lookup(key='msp_leiden_r2.0',cluster='0')==expected
        assert 'no precomputed tables' in tables.lookup(key='unknown',cluster='0')


def test_sparse_global_does_not_modify_mapped_input(tmp_path):
    import anndata as an
    from scipy.sparse import csr_matrix
    from msp.integrate.deg import save_deg_input
    counts=np.random.default_rng(42).poisson(1.,(30,8)).astype('float32')
    data=an.AnnData(csr_matrix(counts),obs=pd.DataFrame({'group':pd.Categorical(['a']*15+['b']*15)},index=[str(i) for i in range(30)]))
    data.uns['log1p']={'base':None}
    save_deg_input(data,tmp_path/'mapped')
    before={p.name:p.read_bytes() for p in (tmp_path/'mapped').glob('*.npy')}
    mapped=load_deg_input(tmp_path/'mapped')
    assert not mapped.X.data.flags.writeable
    frame=compute_deg_task(mapped,dict(key='group',valid=['a','b']))
    assert set(frame['group'])=={'a','b'}
    assert {p.name:p.read_bytes() for p in (tmp_path/'mapped').glob('*.npy')}==before
