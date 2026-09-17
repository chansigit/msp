"""Keep only cell metadata between agent tools; load expression for one operation."""
from contextlib import contextmanager, nullcontext
import importlib
import os
from pathlib import Path
import copy
import gc


def metadata(path):
    import anndata as ad
    import h5py
    from anndata.io import read_elem

    path = Path(path).resolve()
    stamp = path.stat()
    with h5py.File(path, 'r') as f:
        result = ad.AnnData(obs=read_elem(f['obs']), var=read_elem(f['var']),
                           uns=read_elem(f['uns']) if 'uns' in f else {})
    result._agent_source = (path, stamp.st_size, stamp.st_mtime_ns)
    return result


@contextmanager
def materialize(ad):
    """Preserve working cluster columns; never keep loaded arrays after a tool."""
    source = getattr(ad, '_agent_source', None)
    if source is None:
        yield ad
        return
    import anndata

    with work(ad):
        path, size, modified = source
        stamp = path.stat()
        if (stamp.st_size, stamp.st_mtime_ns) != (size, modified):
            raise ValueError(f'agent input changed during the session: {path}')
        full = anndata.read_h5ad(path)
        if not full.obs_names.equals(ad.obs_names) or not full.var_names.equals(ad.var_names):
            raise ValueError('agent metadata and expression identifiers differ')
        full.obs = ad.obs.copy()
        # Working metadata (including restored subcluster columns) is authoritative.
        full.uns = copy.deepcopy(ad.uns)
        try:
            yield full
        finally:
            ad.obs = full.obs.copy()
            ad.uns = copy.deepcopy(full.uns)
            # The caller's `with ... as full` local survives context exit. Clear the
            # disposable object itself so that reference cannot retain expression.
            full.X = None
            full.raw = None
            for values in (full.layers, full.obsm, full.obsp, full.varm, full.varp, full.uns):
                for key in list(values):
                    del values[key]
            del full
            gc.collect()


def work(ad=None):
    module = os.environ.get('ECA_DRIVER_BUDGET_MODULE')
    if not module:
        return nullcontext()
    budget = importlib.import_module(module)
    source = getattr(ad, '_agent_source', None)
    if source and hasattr(budget, 'matrix_working_bytes'):
        return budget.work(memory_bytes=budget.matrix_working_bytes(source[0]))
    return budget.work()


def _invoke(function, source, obs, uns, args, kwargs):
    """Worker operation: file carries expression; small metadata carries refinements."""
    path, size, modified = source
    ad = metadata(path)
    if ad._agent_source[1:] != (size, modified):
        raise ValueError('agent input changed before pool execution')
    ad.obs, ad.uns = obs, uns
    with materialize(ad) as full:
        result = function(full, *args, **kwargs)
    return result, ad.obs, ad.uns


def apply(function, ad, *args, **kwargs):
    from harness_bridge.control import safe_point
    safe_point()
    source = getattr(ad, '_agent_source', None)
    if source is None or os.environ.get('MSP_COMPUTE_ENDPOINT') != 'pool':
        with materialize(ad) as full:
            return function(full, *args, **kwargs)
    from ecarsi.pool.client import PoolEndpoint

    # The batch admission estimate already covers full local matrix work. Use
    # it conservatively on the worker too; standalone callers use cell count.
    memory = int(os.environ.get('ECA_DRIVER_MEMORY_BYTES', '0')) or max(2*2**30, ad.n_obs*4*2**20)
    module = os.environ.get('ECA_DRIVER_BUDGET_MODULE')
    if module:
        budget = importlib.import_module(module)
        if hasattr(budget, 'matrix_working_bytes'):
            memory = budget.matrix_working_bytes(source[0])
    from functools import partial
    root = Path(os.environ.get('ECA_POOL_DATA_ROOT', str(source[0].parent))).resolve()
    if not source[0].is_relative_to(root):
        raise ValueError('agent input is outside ECA_POOL_DATA_ROOT')
    invocation = partial(_invoke, function, source, ad.obs, ad.uns, args, kwargs)
    invocation.__module__, invocation.__name__ = function.__module__, function.__name__
    with PoolEndpoint(mode='pool') as pool:
        result, updated, updated_uns = pool.submit(invocation,
            needs=dict(cpus=1, memory=memory, seconds=max(60, ad.n_obs/20),
                       roots=[str(root)], modules=['msp'])).result()
    safe_point()
    if not updated.index.equals(ad.obs.index):
        raise ValueError('pool tool returned misaligned cell metadata')
    ad.obs, ad.uns = updated, updated_uns
    return result
