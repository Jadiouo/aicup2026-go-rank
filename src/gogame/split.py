"""Player-grouped, rank-stratified folds and the fixed Player hold-out H (SDD 5.9, 8.1)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SEED
from .errors import DataFormatError, GogameError
from .seeds import derive_seed

__all__ = ["PlayerLeakError", "assert_no_player_leak", "make_folds", "make_player_holdout", "make_split_table",
           "write_split", "read_split", "SPLIT_COLUMNS"]

SPLIT_COLUMNS = ["player_id", "rank_id", "fold", "is_holdout", "n_black"]


class PlayerLeakError(GogameError):
    """The same player appears on both sides of a train/eval boundary."""


def assert_no_player_leak(train_players, val_players) -> None:
    inter = set(train_players) & set(val_players)
    if inter:
        raise PlayerLeakError(f"{len(inter)} player(s) on both sides, e.g. {sorted(inter)[:3]}")


def _check_players(players: pd.DataFrame) -> None:
    if players["player_id"].duplicated().any():
        d = players.loc[players["player_id"].duplicated(), "player_id"].iloc[0]
        raise DataFormatError(f"player_id appears in more than one row (e.g. {d!r}); a player must have one rank")


def make_folds(players: pd.DataFrame, n_folds: int = 5, seed: int | None = None) -> pd.DataFrame:
    """players: player_id, rank_id. Returns player_id, rank_id, fold:int8 sorted by (rank_id, player_id).

    Per rank (ascending rank_id, one shared RNG): sort ids, permute, fold = position % n_folds.
    The caller must already have removed the hold-out players H.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    _check_players(players)
    rng = np.random.default_rng(derive_seed(SEED, "split") if seed is None else seed)
    parts = []
    for rid in sorted(players["rank_id"].unique()):
        ids = sorted(players.loc[players["rank_id"] == rid, "player_id"])
        if len(ids) < n_folds:
            warnings.warn(f"rank_id {rid}: only {len(ids)} players for {n_folds} folds; some folds get none", UserWarning)
        perm = rng.permutation(len(ids))
        shuffled = [ids[i] for i in perm]
        parts.append(pd.DataFrame({"player_id": shuffled, "rank_id": np.int8(rid),
                                   "fold": (np.arange(len(ids)) % n_folds).astype(np.int8)}))
    if not parts:
        return pd.DataFrame({"player_id": [], "rank_id": np.array([], dtype=np.int8), "fold": np.array([], dtype=np.int8)})
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["rank_id", "player_id"], kind="stable").reset_index(drop=True)


def make_player_holdout(players_meta: pd.DataFrame, min_black: int = 120, n_holdout: int = 640,
                        seed: int | None = None) -> set[str]:
    """Hold-out H: players with n_black >= min_black, drawn per rank in proportion to the heavy-player count
    (largest-remainder quotas, total = min(n_holdout, #heavy)); ~half of each rank when n_holdout ~ #heavy/2."""
    _check_players(players_meta)
    rng = np.random.default_rng(derive_seed(SEED, "player_holdout") if seed is None else seed)
    heavy = players_meta[players_meta["n_black"] >= min_black]
    if heavy.empty:
        return set()
    ranks = sorted(heavy["rank_id"].unique())
    pools = {r: sorted(heavy.loc[heavy["rank_id"] == r, "player_id"]) for r in ranks}
    total = sum(len(v) for v in pools.values())
    target = min(n_holdout, total)
    exact = {r: target * len(pools[r]) / total for r in ranks}
    quota = {r: int(np.floor(exact[r])) for r in ranks}
    left = target - sum(quota.values())
    order = sorted(ranks, key=lambda r: (-(exact[r] - quota[r]), r))
    for r in order[:left]:
        quota[r] += 1
    chosen: set[str] = set()
    for r in ranks:
        perm = rng.permutation(len(pools[r]))
        chosen.update(pools[r][i] for i in perm[:quota[r]])
    return chosen


def make_split_table(players_meta: pd.DataFrame, n_folds: int = 5, min_black: int = 120, n_holdout: int = 640,
                     seed: int | None = None, holdout_seed: int | None = None) -> pd.DataFrame:
    """The data/.../split.parquet frame: player_id, rank_id:int8, fold:int8 (-1 for H), is_holdout, n_black:int32."""
    h = make_player_holdout(players_meta, min_black, n_holdout, holdout_seed)
    rest = players_meta[~players_meta["player_id"].isin(h)][["player_id", "rank_id"]]
    folds = make_folds(rest, n_folds, seed)
    meta = players_meta[["player_id", "rank_id", "n_black"]]
    hdf = meta[meta["player_id"].isin(h)].assign(fold=-1)
    out = pd.concat([folds.merge(meta[["player_id", "n_black"]], on="player_id"), hdf[["player_id", "rank_id", "fold", "n_black"]]],
                    ignore_index=True)
    out["is_holdout"] = out["player_id"].isin(h)
    out["rank_id"] = out["rank_id"].astype("int8")
    out["fold"] = out["fold"].astype("int8")
    out["n_black"] = out["n_black"].astype("int32")
    out = out[SPLIT_COLUMNS].sort_values(["rank_id", "player_id"], kind="stable").reset_index(drop=True)
    return out


def write_split(df: pd.DataFrame, path: str | Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"schema_version"] = b"1"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)


def read_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if list(df.columns) != SPLIT_COLUMNS:
        raise DataFormatError(f"{path}: columns {list(df.columns)} != {SPLIT_COLUMNS}")
    if df["player_id"].duplicated().any():
        raise DataFormatError(f"{path}: duplicate player_id")
    if ((df["fold"] == -1) != df["is_holdout"]).any():
        raise DataFormatError(f"{path}: is_holdout and fold == -1 disagree")
    return df
