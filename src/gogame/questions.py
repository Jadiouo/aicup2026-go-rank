"""Training bags, validation questions, Player pool simulation (SDD 5.10). Never crosses a player boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd

from .constants import SEED
from .errors import ConfigError, DataFormatError
from .seeds import derive_seed
from .split import PlayerLeakError

__all__ = ["Bag", "PlayerIndex", "PoolSim", "build_player_index", "sample_train_bags", "build_val_questions",
           "build_player_pool_sim", "to_questions_frame", "bag_class_weights", "n_class_report", "flag_n_class_shift"]


@dataclass(frozen=True)
class Bag:
    qid: str
    player_id: str
    rank_id: int
    color: int          # 0=B, 1=W
    n: int
    uids: tuple[str, ...]


@dataclass
class PlayerIndex:
    games: dict[str, dict[int, list[str]]]   # player -> color -> sorted uids
    rank: dict[str, int]                     # player -> rank_id


@dataclass(frozen=True)
class PoolSim:
    draw: int
    lib_games: int
    players: tuple[str, ...]
    library: dict[str, tuple[str, ...]]      # player -> black-game uids (candidate library)
    queries: list[Bag]                       # black-only, disjoint from the player's library


def build_player_index(meta: pd.DataFrame) -> PlayerIndex:
    """meta: uid, player_id, rank_id and color_id (0=B,1=W) or color ('B'/'W')."""
    if "color_id" in meta.columns:
        col = meta["color_id"].astype(int)
    elif "color" in meta.columns:
        bad = set(meta["color"]) - {"B", "W"}
        if bad:
            raise DataFormatError(f"bad color values {sorted(bad)}")
        col = (meta["color"] == "W").astype(int)
    else:
        raise DataFormatError("meta needs color_id or color")
    games: dict[str, dict[int, list[str]]] = {}
    rank: dict[str, int] = {}
    for uid, pid, rid, c in zip(meta["uid"], meta["player_id"], meta["rank_id"], col):
        games.setdefault(pid, {0: [], 1: []})[int(c)].append(uid)
        if pid in rank and rank[pid] != int(rid):
            raise DataFormatError(f"player {pid} has more than one rank")
        rank[pid] = int(rid)
    for d in games.values():
        for c in (0, 1):
            d[c].sort()
    return PlayerIndex(games, rank)


def _pick(rng: np.random.Generator, uids: list[str], n: int) -> tuple[str, ...]:
    idx = np.sort(rng.choice(len(uids), size=n, replace=False))
    return tuple(uids[i] for i in idx)


def sample_train_bags(index: PlayerIndex, players: Iterable[str], bags_per_player_color: int = 8,
                      n_range: tuple[int, int] = (5, 20), seed: int | None = None,
                      n_sequence: Iterator[int] | None = None) -> list[Bag]:
    """N first (uniform on n_range, or taken from the injected n_sequence), then a (player, colour) group among those
    with >= N games, then N games without replacement. Never truncates. Duplicate game sets are redrawn (games only)."""
    lo, hi = n_range
    rng = np.random.default_rng(derive_seed(SEED, "train_bags") if seed is None else seed)
    groups: list[tuple[str, int, list[str]]] = []
    for p in sorted(set(players)):
        if p not in index.games:
            raise DataFormatError(f"player {p!r} not in index")
        for c in (0, 1):
            g = index.games[p][c]
            if len(g) >= lo:
                groups.append((p, c, g))
    sizes = np.array([len(g[2]) for g in groups])
    eligible = {n: np.flatnonzero(sizes >= n) for n in range(lo, hi + 1)}
    seq = iter(n_sequence) if n_sequence is not None else None
    bags: list[Bag] = []
    seen: set[frozenset[str]] = set()
    for i in range(bags_per_player_color * len(groups)):
        for _attempt in range(100):
            n = int(next(seq)) if seq is not None else int(rng.integers(lo, hi + 1))
            cand = eligible.get(n)
            if cand is not None and len(cand):
                break
        else:
            raise DataFormatError("no (player, colour) group has enough games for any drawn N")
        p, c, g = groups[int(cand[int(rng.integers(len(cand)))])]
        for _try in range(50):
            uids = _pick(rng, g, n)
            key = frozenset(uids)
            if key not in seen:
                seen.add(key)
                bags.append(Bag(f"tb_{len(bags):07d}", p, index.rank[p], c, n, uids))
                break
    return bags


def build_val_questions(index: PlayerIndex, players: Iterable[str], ns: Sequence[int] = (5, 10, 15, 20),
                        seed: int | None = None) -> list[Bag]:
    """At most one question per (player, N), single colour, N games without replacement.
    If both colours have >= N games the colour is drawn from a player-specific named seed."""
    base = derive_seed(SEED, "val_questions") if seed is None else seed
    out: list[Bag] = []
    for p in sorted(set(players)):
        if p not in index.games:
            raise DataFormatError(f"player {p!r} not in index")
        crng = np.random.default_rng(derive_seed(base, "val_color", p))
        for n in ns:
            ok = [c for c in (0, 1) if len(index.games[p][c]) >= n]
            if not ok:
                continue
            c = ok[0] if len(ok) == 1 else ok[int(crng.integers(2))]
            grng = np.random.default_rng(derive_seed(base, "val_games", p, str(n)))
            out.append(Bag(f"val_{p}_{n}", p, index.rank[p], c, n, _pick(grng, index.games[p][c], n)))
    return out


def build_player_pool_sim(index: PlayerIndex, holdout_players: Sequence[str], pool_size: int = 400, n_draws: int = 20,
                          lib_games: int = 100, ns: Sequence[int] = (5, 10, 15, 20), seed: int | None = None,
                          holdout_set: Iterable[str] | None = None) -> list[PoolSim]:
    """Candidates only from H. Per draw: pool_size players; lib_games black games each form the library, queries of N
    black games are taken from the remaining ones (disjoint). Players with fewer than lib_games+5 black games are
    excluded; a (player, N) with fewer than N remaining games gets no query. holdout_set (if given) is the
    authoritative H: any listed player outside it raises PlayerLeakError."""
    base = derive_seed(SEED, "pool_sim") if seed is None else seed
    hp = sorted(set(holdout_players))
    if holdout_set is not None:
        extra = sorted(set(hp) - set(holdout_set))
        if extra:
            raise PlayerLeakError(f"{len(extra)} non-hold-out player(s) in the pool, e.g. {extra[:3]}")
    missing = [p for p in hp if p not in index.games]
    if missing:
        raise DataFormatError(f"players not in index: {missing[:3]}")
    eligible = [p for p in hp if len(index.games[p][0]) >= lib_games + 5]
    if pool_size > len(eligible):
        raise ConfigError(f"pool_size={pool_size} exceeds the {len(eligible)} eligible hold-out players")
    sims: list[PoolSim] = []
    for d in range(1, n_draws + 1):
        drng = np.random.default_rng(derive_seed(base, "pool", str(d)))
        pool = sorted(eligible[i] for i in drng.choice(len(eligible), size=pool_size, replace=False))
        library: dict[str, tuple[str, ...]] = {}
        queries: list[Bag] = []
        for p in pool:
            prng = np.random.default_rng(derive_seed(base, "pool", str(d), p, str(lib_games)))
            black = index.games[p][0]
            perm = prng.permutation(len(black))
            lib = sorted(black[i] for i in perm[:lib_games])
            rest = sorted(black[i] for i in perm[lib_games:])
            library[p] = tuple(lib)
            for n in ns:
                if len(rest) >= n:
                    queries.append(Bag(f"pool{d}_{p}_{n}", p, index.rank[p], 0, n, _pick(prng, rest, n)))
        sims.append(PoolSim(d, lib_games, tuple(pool), library, queries))
    return sims


def to_questions_frame(bags: Sequence[Bag]) -> pd.DataFrame:
    cols = ["qid", "player_id", "rank_id", "color", "n"] + [f"uid_{i}" for i in range(1, 21)]
    rows = []
    for b in bags:
        if b.n > 20 or len(b.uids) != b.n:
            raise DataFormatError(f"bag {b.qid}: n={b.n} but {len(b.uids)} uids")
        rows.append([b.qid, b.player_id, b.rank_id, b.color, b.n] + list(b.uids) + [""] * (20 - b.n))
    df = pd.DataFrame(rows, columns=cols)
    for c, t in (("rank_id", "int8"), ("color", "int8"), ("n", "int8")):
        df[c] = df[c].astype(t)
    for c in cols[:2] + cols[5:]:
        df[c] = df[c].astype(object)
    return df


def bag_class_weights(bags: Sequence[Bag], mode: Literal["inv_players", "balanced"] = "inv_players",
                      players_per_rank: dict[int, int] | None = None) -> np.ndarray:
    """float64[B], mean 1. inv_players: 1/(#training players of the bag's rank); balanced: 1/(#bags of the rank)."""
    ranks = np.array([b.rank_id for b in bags])
    if mode == "inv_players":
        if players_per_rank is None:
            pp: dict[int, set] = {}
            for b in bags:
                pp.setdefault(b.rank_id, set()).add(b.player_id)
            players_per_rank = {r: len(s) for r, s in pp.items()}
        w = np.array([1.0 / players_per_rank[r] for r in ranks], dtype=np.float64)
    elif mode == "balanced":
        vals, cnt = np.unique(ranks, return_counts=True)
        c = dict(zip(vals.tolist(), cnt.tolist()))
        w = np.array([1.0 / c[r] for r in ranks], dtype=np.float64)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return w / w.mean() if len(w) else w


def n_class_report(bags: Sequence[Bag]) -> pd.DataFrame:
    """One row per N: n, total, c_0..c_9 (bag counts per rank_id), f_0..f_9 (shares; each row sums to 1)."""
    rows = []
    for n in sorted({b.n for b in bags}):
        cnt = np.zeros(10, dtype=np.int64)
        for b in bags:
            if b.n == n:
                cnt[b.rank_id] += 1
        tot = int(cnt.sum())
        rows.append({"n": n, "total": tot, **{f"c_{r}": int(cnt[r]) for r in range(10)},
                     **{f"f_{r}": cnt[r] / tot for r in range(10)}})
    cols = ["n", "total"] + [f"c_{r}" for r in range(10)] + [f"f_{r}" for r in range(10)]
    return pd.DataFrame(rows, columns=cols)


def flag_n_class_shift(report: pd.DataFrame, thr: float = 0.05) -> dict[int, bool]:
    """True for an N whose rank shares differ from the overall shares by more than thr (absolute) for any rank."""
    c = report[[f"c_{r}" for r in range(10)]].to_numpy().sum(axis=0)
    overall = c / c.sum()
    out = {}
    for _, row in report.iterrows():
        f = np.array([row[f"f_{r}"] for r in range(10)])
        out[int(row["n"])] = bool((np.abs(f - overall) > thr).any())
    return out
