"""Bag aggregation (SDD 5.8 / 12A.1).  Per-game dense features -> one fixed-length vector per bag.

Output layout (stat-major): for stat in (mean, std, q25, q50, q75): all dense columns, then the mean of every
selected sparse column, then `bag_n`, then `is_white`.  Names are `<col>__<stat>`, `sp<idx>__mean`.
No sums are used, so N does not change the scale of any feature.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sps

from .errors import ConfigError, DataFormatError

STATS = ("mean", "std", "q25", "q50", "q75")
_Q = {"q25": 0.25, "q50": 0.5, "q75": 0.75}


@dataclass
class BagFeatureSpec:
    dense_names: tuple[str, ...]
    sparse_cols: np.ndarray                    # int32[Ks]
    stats: tuple[str, ...] = STATS
    names: tuple[str, ...] = ()                # output column names


def make_spec(dense_names: Sequence[str], sparse_cols: np.ndarray, vec_keys: Sequence[str] = ()) -> BagFeatureSpec:
    """`vec_keys` lists GameTable.vecs keys the caller wants to feed into Rank LightGBM; embeddings are refused
    (SDD 8 / 12A.1.1 point 8: embedding spaces of different folds are not comparable)."""
    if "emb" in vec_keys or any(str(k).startswith("emb") for k in vec_keys):
        raise ConfigError("vecs['emb'] must not be used as Rank LightGBM input (cross-fold embedding spaces differ)")
    cols = np.asarray(sparse_cols, dtype=np.int32)
    dn = tuple(dense_names)
    names = [f"{n}__{s}" for s in STATS for n in dn] + [f"sp{int(c)}__mean" for c in cols] + ["bag_n", "is_white"]
    return BagFeatureSpec(dense_names=dn, sparse_cols=cols, stats=STATS, names=tuple(names))


def fit_sparse_selector(sp: sps.csr_matrix, min_df: float = 0.005) -> np.ndarray:
    """Columns whose document frequency (fraction of rows with a non-zero) is >= min_df.  Label-free."""
    sp = sps.csr_matrix(sp)
    n = sp.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    df = np.bincount(sp.indices[sp.data != 0], minlength=sp.shape[1])
    return np.nonzero(df >= min_df * n * (1 - 1e-12))[0].astype(np.int32)


def _dense_stats(dense: np.ndarray) -> np.ndarray:
    d = np.asarray(dense, dtype=np.float64)   # float64 accumulation: bitwise row-order invariant in practice
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        out = [np.nanmean(d, axis=0), np.nanstd(d, axis=0)]
        out.append(np.nanquantile(d, 0.25, axis=0))
        out.append(np.nanquantile(d, 0.5, axis=0))
        out.append(np.nanquantile(d, 0.75, axis=0))
    return np.concatenate([np.asarray(o, dtype=np.float32) for o in out])


def _sparse_mean(sp_sel, n: int) -> np.ndarray:
    """sp_sel: rows of the bag restricted to the selected columns (csr or dense)."""
    if sps.issparse(sp_sel):
        sp_sel = sp_sel.toarray()
    if sp_sel.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    return sp_sel.astype(np.float64).mean(axis=0).astype(np.float32)


def _assemble(dense: np.ndarray, sp_sel, n: int, color: int) -> np.ndarray:
    tail = np.array([n, color], dtype=np.float32)
    return np.concatenate([_dense_stats(dense), _sparse_mean(sp_sel, n), tail])


def aggregate_bag(dense, sp, spec: BagFeatureSpec, color: int) -> np.ndarray:
    """`dense` is float32[N,D] and `sp` csr[N,*]; or `dense` is a sequence of GameTable slices (rows of one bag,
    same row order) in which case `sp` is ignored (the table that has a sparse block supplies it)."""
    if isinstance(dense, (list, tuple)):
        tabs = list(dense)
        d = np.hstack([np.asarray(t.dense, dtype=np.float32) for t in tabs])
        sps_ = [t.sparse for t in tabs if t.sparse is not None]
        sp = sps_[0] if sps_ else sps.csr_matrix((d.shape[0], 0), dtype=np.float32)
        dense = d
    dense = np.asarray(dense, dtype=np.float32)
    if dense.ndim != 2 or dense.shape[1] != len(spec.dense_names):
        raise DataFormatError(f"dense has {dense.shape[-1]} columns, spec expects {len(spec.dense_names)}")
    n = dense.shape[0]
    cols = spec.sparse_cols
    if cols.size:
        sp_sel = sps.csr_matrix(sp)[:, cols]
    else:
        sp_sel = np.zeros((n, 0), dtype=np.float32)
    return _assemble(dense, sp_sel, n, color)


def aggregate_vecs(vecs: np.ndarray) -> np.ndarray:
    """Per-dimension mean over the bag, then L2-normalised (Player query vector)."""
    m = np.asarray(vecs, dtype=np.float64).mean(axis=0)
    nrm = np.linalg.norm(m)
    return (m / nrm if nrm > 0 else m).astype(np.float32)


@dataclass
class L0Store:
    """Row store of per-game dense/sparse features addressed by uid."""
    uids: Sequence[str]
    dense: np.ndarray
    sparse: sps.csr_matrix | None
    names: tuple[str, ...]
    tables: tuple = ()

    def __post_init__(self):
        self.uids = list(self.uids)
        self.names = tuple(self.names)
        self._index = {u: i for i, u in enumerate(self.uids)}
        if len(self._index) != len(self.uids):
            raise DataFormatError("duplicate uid in L0Store")
        if self.sparse is not None:
            self.sparse = sps.csr_matrix(self.sparse)

    @classmethod
    def from_table(cls, table) -> "L0Store":
        return cls(list(table.uids), table.dense, table.sparse, table.names, (table,))

    @classmethod
    def from_batch(cls, batch, names: Sequence[str] | None = None) -> "L0Store":
        from .features_l0 import DENSE_NAMES
        return cls(batch.uids, batch.dense, batch.sparse, tuple(names or DENSE_NAMES))

    def rows(self, uids: Sequence[str]) -> np.ndarray:
        try:
            return np.fromiter((self._index[u] for u in uids), dtype=np.int64, count=len(uids))
        except KeyError as e:
            raise DataFormatError(f"uid not in store: {e.args[0]!r}") from None


def aggregate_many(bags: Sequence, store: L0Store, spec: BagFeatureSpec) -> np.ndarray:
    """Bags need `.uids` and `.color`.  Returns float32[Q, D_agg].  Same arithmetic as `aggregate_bag`."""
    if tuple(store.names) != tuple(spec.dense_names):
        raise DataFormatError("store column names differ from spec.dense_names")
    cols = spec.sparse_cols
    if cols.size:
        if store.sparse is None:
            raise DataFormatError("spec selects sparse columns but the store has no sparse block")
        sel = store.sparse[:, cols].tocsr()
    else:
        sel = None
    out = np.empty((len(bags), len(spec.names)), dtype=np.float32)
    for i, b in enumerate(bags):
        rows = store.rows(b.uids)
        sp_sel = sel[rows] if sel is not None else np.zeros((len(rows), 0), dtype=np.float32)
        out[i] = _assemble(store.dense[rows], sp_sel, len(rows), int(b.color))
    return out
