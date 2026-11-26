"""Scoring functions (SDD 7) and paired bootstrap (SDD 8.4)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .constants import SEED
from .seeds import derive_seed

E1 = math.exp(-1.0)
N_RANKS = 10


def rank_score_matrix() -> np.ndarray:
    """s[j, k] = 1 if j==k; e^-1 if |j-k|==1; else 0 (SDD 7.1). float64[10,10]."""
    d = np.abs(np.arange(N_RANKS)[:, None] - np.arange(N_RANKS)[None, :])
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def _check_rank_ids(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.issubdtype(a.dtype, np.integer):
        if not np.all(np.equal(np.mod(a, 1), 0)):
            raise ValueError(f"{name} must be integer rank ids")
    a = a.astype(np.int64)
    if a.min() < 0 or a.max() >= N_RANKS:
        raise ValueError(f"{name} out of range 0..{N_RANKS - 1}")
    return a


def rank_scores(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = _check_rank_ids(pred, "pred")
    truth = _check_rank_ids(truth, "truth")
    if pred.shape != truth.shape:
        raise ValueError("pred and truth shapes differ")
    d = np.abs(pred - truth)
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0)).astype(np.float64)


def rank_score(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(rank_scores(pred, truth).mean())


def player_scores(top5: np.ndarray, truth: np.ndarray) -> np.ndarray:
    top5 = np.asarray(top5)
    truth = np.asarray(truth)
    if top5.ndim != 2 or top5.shape[1] != 5:
        raise ValueError("top5 must have shape [Q,5]")
    if top5.shape[0] == 0:
        raise ValueError("empty input")
    if truth.shape != (top5.shape[0],):
        raise ValueError("truth must have shape [Q]")
    out = np.zeros(top5.shape[0], dtype=np.float64)
    for i in range(top5.shape[0]):
        row = list(top5[i])
        if len(set(row)) != 5:
            raise ValueError(f"duplicate ids in top5 of question {i}")
        for r, pid in enumerate(row):
            if pid == truth[i]:
                out[i] = math.exp(-r)
                break
    return out


def player_score(top5: np.ndarray, truth: np.ndarray) -> float:
    return float(player_scores(top5, truth).mean())


def balanced_weights(rank_ids: np.ndarray) -> np.ndarray:
    """Per-question weight proportional to 1/(questions of that rank), mean 1."""
    rank_ids = np.asarray(rank_ids)
    if rank_ids.size == 0:
        raise ValueError("empty input")
    _, inv, counts = np.unique(rank_ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.mean()


def by_group(scores: np.ndarray, keys: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"key": np.asarray(keys), "score": np.asarray(scores, dtype=np.float64)})
    g = df.groupby("key", sort=True)["score"].agg(["mean", "count"]).rename(columns={"count": "n"})
    return g.reset_index()


@dataclass
class BootResult:
    delta: float
    ci_low: float
    ci_high: float
    p_gt: dict[float, float]
    n_groups: int
    n_boot: int
    weights_mode: Literal["raw", "balanced"]


def paired_bootstrap(
    sa: np.ndarray,
    sb: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int | None = None,
    weights: np.ndarray | None = None,
    thresholds: tuple[float, ...] = (0.0, 0.015),
) -> BootResult:
    """Group-level paired bootstrap of the (weighted) mean score difference sa - sb."""
    sa = np.asarray(sa, dtype=np.float64)
    sb = np.asarray(sb, dtype=np.float64)
    groups = np.asarray(groups)
    if not (sa.shape == sb.shape == groups.shape) or sa.ndim != 1:
        raise ValueError("sa, sb, groups must be 1-D arrays of equal length")
    if sa.size == 0:
        raise ValueError("empty input")
    if weights is None:
        w = np.ones_like(sa)
        mode: Literal["raw", "balanced"] = "raw"
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != sa.shape:
            raise ValueError("weights shape mismatch")
        mode = "balanced"
    if seed is None:
        seed = derive_seed(SEED, "bootstrap")
    _, gidx = np.unique(groups, return_inverse=True)
    P = int(gidx.max()) + 1
    delta_g = np.bincount(gidx, weights=w * (sa - sb), minlength=P)
    n_g = np.bincount(gidx, weights=w, minlength=P)
    rng = np.random.default_rng(seed)
    c = rng.multinomial(P, np.full(P, 1.0 / P), size=n_boot).astype(np.float64)  # [B,P]
    db = (c @ delta_g) / (c @ n_g)
    return BootResult(
        delta=float(delta_g.sum() / n_g.sum()),
        ci_low=float(np.quantile(db, 0.025)),
        ci_high=float(np.quantile(db, 0.975)),
        p_gt={float(t): float(np.mean(db > t)) for t in thresholds},
        n_groups=P,
        n_boot=n_boot,
        weights_mode=mode,
    )
