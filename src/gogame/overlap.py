"""Train/test overlap detection and test-set distribution report (SDD 5.14, 10.2, 10.3)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .io import GameRecord, normalize_qid  # noqa: F401  (normalize_qid re-exported for scripts)

__all__ = ["game_key", "prefix_key", "build_train_index", "build_prefix_index", "exact_overlap", "prefix_overlap",
           "bag_centroid_shift", "library_centroid_check", "decide_overlap", "distribution_report", "OverlapStats",
           "PrefixStats", "ShiftStats", "OverlapDecision"]

_MOVE = re.compile(r";\s*([BW])\[([a-z]{0,2})\]")


def _tokens(sgf: str) -> list[str]:
    return [f"{c}[{'' if p == 'tt' else p}]" for c, p in _MOVE.findall(sgf)]


def game_key(sgf: str) -> str:
    """sha1 of the normalised move sequence 'B[pd];W[pq];...' ([tt] and [] are both a pass)."""
    return hashlib.sha1(";".join(_tokens(sgf)).encode()).hexdigest()


def prefix_key(sgf: str, k: int) -> str | None:
    """Hash of the first k plies (both sides); None if the game is shorter than k."""
    t = _tokens(sgf)
    if len(t) < k:
        return None
    return hashlib.sha1(";".join(t[:k]).encode()).hexdigest()[:20]


def build_train_index(sgfs: Iterable[str], colors: Iterable[int]) -> tuple[dict[str, list[int]], np.ndarray]:
    """game_key -> list of training row numbers, plus the colour (0=B/1=W) of every row."""
    idx: dict[str, list[int]] = {}
    for i, s in enumerate(sgfs):
        idx.setdefault(game_key(s), []).append(i)
    return idx, np.asarray(list(colors), dtype=np.int8)


def build_prefix_index(sgfs: Iterable[str], ks: Sequence[int] = (6, 10, 16, 24, 30)) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {k: set() for k in ks}
    for s in sgfs:
        t = _tokens(s)
        for k in ks:
            if len(t) >= k:
                out[k].add(hashlib.sha1(";".join(t[:k]).encode()).hexdigest()[:20])
    return out


@dataclass(frozen=True)
class OverlapStats:
    n_test: int
    n_exact: int
    n_same_color: int       # an identical game exists in train with the same annotated colour (player overlap evidence)
    n_other_side: int       # identical game only with the other colour (= the opponent's side of the same game)
    exact_frac: float
    same_color_frac: float


def exact_overlap(test_games: Sequence[GameRecord], train_index: dict[str, list[int]], train_colors: np.ndarray) -> OverlapStats:
    n = len(test_games)
    same = other = 0
    for g in test_games:
        rows = train_index.get(game_key(g.sgf))
        if not rows:
            continue
        if any(int(train_colors[r]) == g.color for r in rows):
            same += 1
        else:
            other += 1
    d = n or 1
    return OverlapStats(n, same + other, same, other, (same + other) / d, same / d)


@dataclass(frozen=True)
class PrefixStats:
    n_test: int
    n_match: dict[int, int]
    frac: dict[int, float]


def prefix_overlap(test_games: Sequence[GameRecord], prefix_index: dict[int, set[str]], k: int = 30,
                   ref_ks: Sequence[int] = (6, 10, 16, 24)) -> PrefixStats:
    """Share of test games whose first K plies equal some training game's first K plies. Decision uses K=30 only;
    ref_ks are reported for comparison (those present in the index)."""
    if k not in prefix_index:
        raise ValueError(f"prefix index was not built for k={k}")
    ks = sorted({k, *[r for r in ref_ks if r in prefix_index]})
    match = {kk: 0 for kk in ks}
    for g in test_games:
        for kk in ks:
            key = prefix_key(g.sgf, kk)
            if key is not None and key in prefix_index[kk]:
                match[kk] += 1
    n = len(test_games)
    return PrefixStats(n, match, {kk: (match[kk] / n if n else 0.0) for kk in ks})


@dataclass
class ShiftStats:
    n_test: int
    quantiles: dict[float, float]
    holdout_quantiles: dict[float, float]
    ks_stat: float
    ks_pvalue: float
    holdout_p99: float
    p99_frac: float
    max_sim: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    holdout_max_sim: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


_QS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, n, out=np.zeros_like(a), where=n > 0)


def _max_sim(vecs: np.ndarray, centroids_unit: np.ndarray, chunk: int = 4096) -> np.ndarray:
    v = _unit(vecs)
    out = np.empty(len(v))
    for i in range(0, len(v), chunk):
        out[i:i + chunk] = (v[i:i + chunk] @ centroids_unit.T).max(axis=1)
    return out


def _shift(test_vecs, train_centroids, holdout_vecs) -> ShiftStats:
    from scipy.stats import ks_2samp
    C = _unit(train_centroids)
    t = _max_sim(test_vecs, C)
    h = _max_sim(holdout_vecs, C)
    p99 = float(np.quantile(h, 0.99))
    ks = ks_2samp(t, h)
    return ShiftStats(len(t), {q: float(np.quantile(t, q)) for q in _QS}, {q: float(np.quantile(h, q)) for q in _QS},
                      float(ks.statistic), float(ks.pvalue), p99, float((t > p99).mean()) if len(t) else 0.0, t, h)


def bag_centroid_shift(test_bag_vecs: np.ndarray, train_centroids: np.ndarray, holdout_bag_vecs: np.ndarray) -> ShiftStats:
    """Max cosine of every test bag to the training-player centroids vs the same statistic for hold-out bags
    (players the model has never seen = the null). p99_frac = share of test bags above the hold-out 99th percentile."""
    return _shift(test_bag_vecs, train_centroids, holdout_bag_vecs)


def library_centroid_check(cand_centroids: np.ndarray, train_centroids: np.ndarray, holdout_centroids: np.ndarray) -> ShiftStats:
    """Same statistic for Player candidate-library centroids (one per candidate) vs hold-out player centroids."""
    return _shift(cand_centroids, train_centroids, holdout_centroids)


# ------------------------------------------------------------------------- decision (OR rule, SDD 10.2)
DEFAULT_THRESHOLDS = {"exact_same_color_frac_thr": 0.01, "prefix30_frac_thr": 0.01,
                      "nn_p99_frac_thr": 0.05, "library_p99_frac_thr": 0.05}
_EVIDENCE = ("exact_same_color_frac", "prefix30_frac", "nn_p99_frac", "library_p99_frac")


@dataclass(frozen=True)
class OverlapDecision:
    overlap: bool
    label: str                       # "待人工確認" or "否"
    triggered: dict[str, bool]
    evidence: dict[str, float | None]
    thresholds: dict[str, float]
    not_evaluated: tuple[str, ...]


def decide_overlap(stats: Mapping[str, float | None], cfg: Mapping[str, float] | None = None) -> OverlapDecision:
    """Overlap = ANY of: same-colour exact share >= 1%, 30-ply prefix share >= 1%, bag nn p99 share >= 5%,
    library nn p99 share >= 5% (thresholds are inferred defaults, to be confirmed by the user). Missing stats are
    reported in not_evaluated and never trigger."""
    thr = {**DEFAULT_THRESHOLDS, **(cfg or {})}
    trig, ev, ne = {}, {}, []
    for key in _EVIDENCE:
        v = stats.get(key)
        ev[key] = v
        if v is None:
            ne.append(key)
            trig[key] = False
        else:
            trig[key] = bool(v >= thr[key + "_thr"])
    hit = any(trig.values())
    return OverlapDecision(hit, "待人工確認" if hit else "否", trig, ev, thr, tuple(ne))


# ------------------------------------------------------------------------- distribution report (SDD 10.3)
def distribution_report(test_df: pd.DataFrame, task: str, cand_df: pd.DataFrame | None = None,
                        train_plies: np.ndarray | None = None) -> dict:
    """Facts about a test-question frame (raw strings, as read by io.read_questions_csv)."""
    from .board import replay_summary
    from .sgf import parse_sgf

    n_dist = {int(k): int(v) for k, v in test_df["num_games"].astype(int).value_counts().sort_index().items()}
    c1 = test_df["color_1"].value_counts()
    qids = list(test_df["question_id"])
    n_pref = sum(1 for q in qids if q.startswith(("pub_", "priv_")))
    plies: list[int] = []
    flags = {"n_pass": 0, "n_bad_size": 0, "n_has_setup": 0, "n_nonalt": 0, "n_bad_coord": 0, "n_first_not_black": 0}
    illegal = {"suicide": 0, "occupied": 0, "ko_violation": 0}
    extra_copies = q_with_dup = 0
    colors_seen: set[str] = set()
    for _, row in test_df.iterrows():
        keys = []
        for k in range(1, 21):
            s = row[f"sgf_{k}"]
            if s == "":
                continue
            colors_seen.add(row[f"color_{k}"])
            keys.append(game_key(s))
            pg = parse_sgf(s)
            plies.append(len(pg.moves))
            f = pg.flags
            flags["n_pass"] += f.n_pass
            flags["n_bad_size"] += int(f.bad_size)
            flags["n_has_setup"] += int(f.has_setup)
            flags["n_nonalt"] += int(f.nonalt)
            flags["n_bad_coord"] += f.bad_coord
            flags["n_first_not_black"] += int(f.first_not_black)
            for kk, v in replay_summary(pg).counts.items():
                illegal[kk] += v
        e = len(keys) - len(set(keys))
        extra_copies += e
        q_with_dup += int(e > 0)
    pl = np.asarray(plies, dtype=float)
    ply = ({"min": int(pl.min()), "max": int(pl.max()), "mean": float(pl.mean()),
            **{f"p{int(q * 100):02d}": float(np.quantile(pl, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
           if len(pl) else {})
    rep: dict = {
        "task": task,
        "n_questions": int(len(test_df)),
        "n_games": len(plies),
        "n_dist": n_dist,
        "color1_dist": {c: int(c1.get(c, 0)) for c in ("B", "W")},
        "all_black": colors_seen <= {"B"},
        "ply": ply,
        "dup_games": {"n_extra_copies": extra_copies, "n_questions_with_dup": q_with_dup,
                      "frac": extra_copies / len(plies) if plies else 0.0},
        "flags": flags,
        "illegal": illegal,
        "qid_prefix": {"n_prefixed": n_pref, "n_total": len(qids),
                       "state": "all" if n_pref == len(qids) else "none" if n_pref == 0 else "mixed"},
    }
    if train_plies is not None and len(pl):
        from scipy.stats import ks_2samp
        r = ks_2samp(pl, np.asarray(train_plies, dtype=float))
        rep["ks_vs_train"] = {"stat": float(r.statistic), "pvalue": float(r.pvalue)}
    if cand_df is not None:
        cnt = cand_df.groupby("player_id").size()
        rep["candidates"] = {"n_players": int(len(cnt)), "games_min": int(cnt.min()) if len(cnt) else 0,
                             "games_max": int(cnt.max()) if len(cnt) else 0, "n_not_100": int((cnt != 100).sum()),
                             "all_black": bool((cand_df["color"] == "B").all())}
    return rep
