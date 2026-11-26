"""Per-game feature table shared by all routes (SDD 12A.1) and cross-fit helpers (12A.1.1)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sps

from .errors import DataFormatError
from .io import GameRecord

SCHEMA_VERSION = "1"
ROUTE_PREFIX = {"l0": "l0_", "A": "ka_", "B": "st_", "C": "sc_"}
NAN_SUFFIXES = ("_mean", "_median", "_std")


@dataclass
class GameTable:
    route: str
    uids: np.ndarray
    dense: np.ndarray
    names: tuple[str, ...]
    sparse: sps.csr_matrix | None = None
    vecs: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.uids = np.asarray(self.uids)
        self.names = tuple(self.names)
        self.dense = np.asarray(self.dense, dtype=np.float32)
        if self.sparse is not None:
            self.sparse = sps.csr_matrix(self.sparse)
        G = len(self.uids)
        if self.dense.ndim != 2 or self.dense.shape != (G, len(self.names)):
            raise DataFormatError(f"dense shape {self.dense.shape} vs {G} uids x {len(self.names)} names")
        if len(set(self.names)) != len(self.names):
            raise DataFormatError("duplicate column names")
        prefixes = tuple(ROUTE_PREFIX[r] for r in self.route.split("+") if r in ROUTE_PREFIX)
        if not prefixes:
            raise DataFormatError(f"unknown route {self.route!r}")
        bad = [n for n in self.names if not n.startswith(prefixes)]
        if bad:
            raise DataFormatError(f"column names {bad[:3]} do not carry the prefix of route {self.route!r}")
        if self.sparse is not None and self.sparse.shape[0] != G:
            raise DataFormatError("sparse rows differ from uid count")
        if np.isinf(self.dense).any():
            raise DataFormatError("inf in dense features")
        nan_cols = np.where(np.isnan(self.dense).any(axis=0))[0]
        for j in nan_cols:
            if not self.names[j].endswith(NAN_SUFFIXES):
                raise DataFormatError(f"NaN in column {self.names[j]!r} (allowed only for *_mean/_median/_std)")
        for k, v in self.vecs.items():
            if len(v) != G:
                raise DataFormatError(f"vecs[{k!r}] has {len(v)} rows, expected {G}")


class GameFeatureProvider(Protocol):
    route: str

    def compute(self, records: Sequence[GameRecord], fold: int | None) -> GameTable: ...


class L0Provider:
    route = "l0"

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def compute(self, records: Sequence[GameRecord], fold: int | None = None) -> GameTable:
        from .features_l0 import DENSE_NAMES, compute_l0_many
        b = compute_l0_many(records, n_jobs=self.n_jobs)
        return GameTable("l0", np.array(b.uids), b.dense, tuple("l0_" + n for n in DENSE_NAMES), b.sparse, {},
                         {"route": "l0", "model_id": "l0", "fold": fold, "config_hash": "", "external_resources": []})


# ------------------------------------------------------------------ I/O ----------------------------------------
def save_table(t: GameTable, path) -> None:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    arrays = [pa.array(t.uids.astype(str))] + [pa.array(t.dense[:, j], type=pa.float32()) for j in range(len(t.names))]
    tab = pa.Table.from_arrays(arrays, names=["uid"] + list(t.names))
    tab = tab.replace_schema_metadata({"schema_version": SCHEMA_VERSION})
    pq.write_table(tab, d / "dense.parquet")
    if t.sparse is not None:
        sps.save_npz(d / "sparse.npz", t.sparse)
    for k, v in t.vecs.items():
        np.save(d / f"vecs_{k}.npy", v)
    (d / "meta.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "route": t.route,
                                             "names": list(t.names), "vec_keys": sorted(t.vecs),
                                             "has_sparse": t.sparse is not None, "meta": t.meta}, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))


def load_table(path) -> GameTable:
    d = Path(path)
    info = json.loads((d / "meta.json").read_text())
    if info.get("schema_version") != SCHEMA_VERSION:
        raise DataFormatError(f"{d}: schema_version {info.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    tab = pq.read_table(d / "dense.parquet")
    names = tuple(info["names"])
    dense = np.column_stack([tab.column(n).to_numpy(zero_copy_only=False) for n in names]).astype(np.float32) \
        if names else np.zeros((tab.num_rows, 0), np.float32)
    sp = sps.load_npz(d / "sparse.npz").tocsr() if info["has_sparse"] else None
    vecs = {k: np.load(d / f"vecs_{k}.npy") for k in info["vec_keys"]}
    return GameTable(info["route"], np.array(tab.column("uid").to_pylist()), dense, names, sp, vecs, info["meta"])


def merge_tables(ts: Sequence[GameTable]) -> GameTable:
    """Inner-join by uid, in the row order of the first table.  A uid missing from any table is an error."""
    ts = list(ts)
    if not ts:
        raise DataFormatError("nothing to merge")
    base = [str(u) for u in ts[0].uids]
    parts, sparse, vecs, names, routes = [], None, {}, [], []
    for t in ts:
        pos = {str(u): i for i, u in enumerate(t.uids)}
        if len(pos) != len(t.uids):
            raise DataFormatError("duplicate uid within a table")
        if set(pos) != set(base):
            missing = sorted(set(base) ^ set(pos))[:3]
            raise DataFormatError(f"uid sets differ between tables, e.g. {missing}")
        order = np.array([pos[u] for u in base])
        parts.append(t.dense[order])
        names += list(t.names)
        routes.append(t.route)
        if t.sparse is not None:
            if sparse is not None:
                raise DataFormatError("more than one sparse block")
            sparse = t.sparse[order]
        for k, v in t.vecs.items():
            if k in vecs:
                raise DataFormatError(f"duplicate vecs key {k!r}")
            vecs[k] = v[order]
    route = "+".join(dict.fromkeys(routes))
    meta = {"route": route, "model_id": "+".join(str(t.meta.get("model_id", "")) for t in ts), "fold": None,
            "config_hash": "", "external_resources": sum((list(t.meta.get("external_resources", [])) for t in ts), [])}
    return GameTable(route, np.array(base), np.hstack(parts), tuple(names), sparse, vecs, meta)


# ------------------------------------------------------------------ cross-fit -------------------------------------
def _holdout(split: pd.DataFrame) -> set[str]:
    m = split["fold"] == -1
    if "is_holdout" in split.columns:
        m = m | split["is_holdout"].astype(bool)
    return set(split.loc[m, "player_id"])


def _train_players(meta: dict) -> list[set[str]]:
    if "train_players_per_fold" in meta:
        return [set(x) for x in meta["train_players_per_fold"]]
    return []


def assert_crossfit(table: GameTable, split: pd.DataFrame) -> None:
    """The network(s) behind `table` never saw the fold the rows belong to, nor the hold-out H."""
    H = _holdout(split)
    meta = table.meta
    fold = meta.get("fold")
    if fold == "oof" or fold == "avg":
        per = _train_players(meta)
        if not per:
            raise DataFormatError("cross-fit table without meta['train_players_per_fold']")
        for j, tp in enumerate(per):
            fj = set(split.loc[split["fold"] == j, "player_id"])
            if tp & fj and fold == "oof":
                raise DataFormatError(f"net {j} was trained on players of its own fold (e.g. {sorted(tp & fj)[:2]})")
            if tp & H:
                raise DataFormatError(f"net {j} was trained on hold-out players (e.g. {sorted(tp & H)[:2]})")
        return
    if not isinstance(fold, (int, np.integer)):
        raise DataFormatError("meta['fold'] must be an int, 'avg' or 'oof'")
    if "train_players" not in meta:
        raise DataFormatError("meta['train_players'] missing; cannot verify cross-fit")
    tp = set(meta["train_players"])
    fj = set(split.loc[split["fold"] == int(fold), "player_id"])
    if tp & fj:
        raise DataFormatError(f"table of fold {fold} was produced by a net trained on that fold (e.g. {sorted(tp & fj)[:2]})")
    if tp & H:
        raise DataFormatError(f"net trained on hold-out players (e.g. {sorted(tp & H)[:2]})")


def require_oof_table(table: GameTable) -> None:
    """OOF evaluation must not use features of a network trained on all data (learned routes B, C)."""
    learned = any(r in ("B", "C") for r in table.route.split("+"))
    if learned and table.meta.get("fold") != "oof":
        raise DataFormatError(f"route {table.route!r} table has fold={table.meta.get('fold')!r}; OOF evaluation needs the OOF table")


def build_oof_table(records: Sequence[GameRecord], split: pd.DataFrame, K: int,
                    train_net: Callable[[set, int], object], apply_net: Callable[[object, Sequence[GameRecord]], np.ndarray],
                    route: str, names: Sequence[str]) -> GameTable:
    """SDD 12A.1.1: net_j is trained on non-H players outside fold j; rows of fold-j players come from net_j, all
    other rows (H, test) are the column-wise mean of the K nets."""
    recs = list(records)
    fold_of = dict(zip(split["player_id"], split["fold"].astype(int)))
    H = _holdout(split)
    train_sets, nets = [], []
    for j in range(K):
        tp = {p for p, f in fold_of.items() if 0 <= f < K and f != j and p not in H}
        if tp & H or tp & {p for p, f in fold_of.items() if f == j}:
            raise DataFormatError("training set overlaps the fold or the hold-out")
        train_sets.append(tp)
        nets.append(train_net(set(tp), j))
    G, D = len(recs), len(names)
    dense = np.zeros((G, D), dtype=np.float32)
    row_fold = np.full(G, -1, dtype=np.int64)
    rf = np.array([fold_of.get(r.player_id, -1) if r.player_id not in H else -1 for r in recs], dtype=np.int64)
    rf[(rf >= K)] = -1
    for j in range(K):
        idx = np.where(rf == j)[0]
        if idx.size:
            dense[idx] = np.asarray(apply_net(nets[j], [recs[i] for i in idx]), dtype=np.float32)
            row_fold[idx] = j
    idx = np.where(rf < 0)[0]
    if idx.size:
        sub = [recs[i] for i in idx]
        dense[idx] = np.mean([np.asarray(apply_net(n, sub), dtype=np.float64) for n in nets], axis=0).astype(np.float32)
    meta = {"route": route, "model_id": "oof", "fold": "oof", "config_hash": "", "external_resources": [],
            "K": K, "row_fold": row_fold.tolist(), "train_players_per_fold": [sorted(s) for s in train_sets]}
    return GameTable(route, np.array([r.uid for r in recs]), dense, tuple(names), None, {}, meta)
