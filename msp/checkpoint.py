"""Private, atomically replaced agent progress; callers revalidate every decision."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def data_identity(ad, context, *, ignore_obs=()):
    """Hash actual ordered data, without serializing another full AnnData copy."""
    digest = hashlib.sha256()

    def add(value):
        digest.update(json.dumps(value, sort_keys=True, default=lambda x: x.tolist()).encode() + b"\0")

    def matrix(value):
        if value is None:
            add(None)
        elif sparse.issparse(value):
            add([value.format, value.shape])
            for part in (value.data, value.indices, value.indptr):
                matrix(part)
        else:
            value = np.asarray(value)
            add([value.shape, str(value.dtype)])
            for part in np.nditer(
                value, flags=["external_loop", "buffered", "zerosize_ok", "refs_ok"], order="C", buffersize=1 << 20
            ):
                if value.dtype.kind in "OUS":
                    add(part.tolist())
                else:
                    digest.update(part.tobytes())

    add(context)
    for frame in (ad.obs.drop(columns=list(ignore_obs), errors="ignore"), ad.var):
        add(list(frame.columns))
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    matrix(ad.X)
    for mapping in (ad.layers, ad.obsm, ad.obsp):
        for key in sorted(mapping, key=str):
            add(key)
            matrix(mapping[key])
    add(ad.uns.get("log1p"))
    add(ad.uns.get("neighbors"))
    return digest.hexdigest()


def agent_identity(ad, outdir, context, source_file, *, ignore_obs=()):
    digest = hashlib.sha256(data_identity(ad, context, ignore_obs=ignore_obs).encode())
    roots = {Path(__file__).parent, Path(source_file).parent}
    for root in sorted(roots):
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    # These are the host's precomputed evidence and the user's scientific context.
    for path in sorted(Path(outdir).glob("*")):
        if (
            path.suffix == ".csv"
            and not path.name.startswith("annotation_")
            or path.name
            in (
                "design_context.txt",
                "report_context.txt",
            )
        ):
            digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def load(path, identity):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        saved = json.loads(path.read_text())
        if saved["schema"] != 1 or saved["identity"] != identity or not isinstance(saved["progress"], dict):
            raise ValueError("input, evidence, settings or code changed")
        return saved["progress"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"cannot restore agent checkpoint {path}: {exc}; use a new output directory or --force"
        ) from exc


def save(path, identity, progress):
    write_json(path, {"schema": 1, "identity": identity, "progress": progress})


def restore_clustering(ad, state, columns, base_key, prefix):
    """Only accept cell-aligned refinements of one parent cluster per step."""
    n = state.get("n_sub")
    if type(n) is not int or n < 0 or state.get("key") != (f"{prefix}{n}" if n else base_key):
        raise ValueError("invalid checkpoint clustering state")
    if set(columns) != {f"{prefix}{i}" for i in range(1, n + 1)}:
        raise ValueError("missing checkpoint subcluster columns")
    previous = ad.obs[base_key].astype(str).tolist()
    for i in range(1, n + 1):
        key = f"{prefix}{i}"
        labels = columns[key]
        if not isinstance(labels, list) or len(labels) != ad.n_obs or not all(isinstance(x, str) for x in labels):
            raise ValueError("invalid checkpoint cell labels")
        changed = {old for old, new in zip(previous, labels, strict=True) if old != new}
        if len(changed) != 1 or any(
            new != old and not new.startswith(old + ",") for old, new in zip(previous, labels, strict=True)
        ):
            raise ValueError("checkpoint clustering is not a parent refinement")
        parent = next(iter(changed))
        if any(
            old == parent and (new == old or not new[len(old) + 1 :].isdigit())
            for old, new in zip(previous, labels, strict=True)
        ):
            raise ValueError("checkpoint split has invalid child identifiers")
        if len({new for old, new in zip(previous, labels, strict=True) if old == parent}) < 2:
            raise ValueError("checkpoint split has fewer than two children")
        ad.obs[key] = pd.Categorical(labels)
        previous = labels
