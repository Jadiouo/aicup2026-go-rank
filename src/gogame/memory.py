"""Memory baseline (SDD 10.1): nearest training-player centroid by cosine on z-scored bag vectors.

Works on any feature matrix (it does not depend on L0). models.py re-exports it (SDD 5.11 location).
"""
from __future__ import annotations

import numpy as np

__all__ = ["MemoryBaseline"]


def _l2(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, n, out=np.zeros_like(a), where=n > 0)


class MemoryBaseline:
    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self.keys_: list = []
        self.ranks_: np.ndarray | None = None

    @property
    def n_centroids(self) -> int:
        return 0 if self.centroids_ is None else len(self.centroids_)

    def _z(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
        return (X - self.mean_) / self.std_ if self.standardize else X

    def fit(self, X: np.ndarray, player_ids: np.ndarray, rank_ids: np.ndarray, colors: np.ndarray | None = None) -> "MemoryBaseline":
        """One centroid per player (per colour when `colors` is given): mean of the z-scored bag vectors, L2-normalised."""
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
        pid = np.asarray(player_ids).astype(str)
        rid = np.asarray(rank_ids)
        if not (len(X) == len(pid) == len(rid)):
            raise ValueError("X, player_ids and rank_ids must have the same length")
        self.mean_ = X.mean(axis=0) if self.standardize else np.zeros(X.shape[1])
        sd = X.std(axis=0) if self.standardize else np.ones(X.shape[1])
        self.std_ = np.where(sd > 0, sd, 1.0)
        Z = self._z(X)
        keys = list(zip(pid, np.asarray(colors).astype(int))) if colors is not None else list(pid)
        order: dict = {}
        for i, k in enumerate(keys):
            order.setdefault(k, []).append(i)
        self.keys_ = sorted(order)
        cents, ranks = [], []
        player_rank: dict[str, int] = {}
        for k in self.keys_:
            rows = order[k]
            rs = set(rid[rows].tolist())
            if len(rs) != 1:
                raise ValueError(f"centroid {k!r} has several rank ids {sorted(rs)}")
            p = k[0] if isinstance(k, tuple) else k
            if player_rank.setdefault(p, next(iter(rs))) != next(iter(rs)):
                raise ValueError(f"player {p!r} has several rank ids")
            cents.append(Z[rows].mean(axis=0))
            ranks.append(next(iter(rs)))
        self.centroids_ = _l2(np.vstack(cents))
        self.ranks_ = np.array(ranks, dtype=np.int64)
        return self

    def similarities(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("fit first")
        return _l2(self._z(X)) @ self.centroids_.T

    def predict(self, X: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(rank_id[Q], max_sim[Q]). k>1: vote among the k nearest centroids; ties -> larger summed cosine, then lower rank."""
        if k < 1:
            raise ValueError("k must be >= 1")
        sim = self.similarities(X)
        k = min(k, sim.shape[1])
        top = np.argsort(-sim, axis=1, kind="stable")[:, :k]
        preds = np.empty(len(sim), dtype=self.ranks_.dtype)
        for q in range(len(sim)):
            votes: dict[int, list[float]] = {}
            for j in top[q]:
                v = votes.setdefault(int(self.ranks_[j]), [0, 0.0])
                v[0] += 1
                v[1] += float(sim[q, j])
            preds[q] = min(votes, key=lambda r: (-votes[r][0], -votes[r][1], r))
        return preds, sim.max(axis=1)
