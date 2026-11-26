"""E04v2 Player reranker (E05 frozen protocol): L0 cosine Top-K candidates -> LightGBM lambdarank -> Top-5.

Shared by scripts/experiments/e05/run.py (evaluation) and scripts/predict.py (inference) so both compute the
very same feature matrix.  Inputs are already z-scored float32 L0 game vectors (dense + sparse columns).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

K = 20
N_DENSE = 31
N_SPARSE_SEL = 256
TOP = 5
LGB_ROUNDS = 400


def lgb_params(seed: int, num_threads: int = 6) -> dict:
    """E05 frozen LightGBM parameters (scripts/experiments/e05/run.py LGB)."""
    return dict(objective="lambdarank", learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, max_bin=63,
                feature_fraction=0.8, eval_at=[5], deterministic=True, force_row_wise=True,
                num_threads=num_threads, verbose=-1, seed=int(seed))


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def candidate_features(lib_z: Sequence[np.ndarray], q_z: Sequence[np.ndarray], sel: np.ndarray, k: int = K,
                       n_dense: int = N_DENSE) -> dict:
    """lib_z[i]: (n_i, D) z-scored games of candidate i; q_z[j]: (N_j, D) games of query j.

    Returns cos (nq,P), top (nq,k) candidate indices by cosine (stable), ns, and X (nq*k, F) float32.
    Candidate order = lib_z order (callers sort player ids).  Operation order is that of E05 `pool_features`.
    """
    P, nq = len(lib_z), len(q_z)
    k = min(k, P)
    Craw = np.stack([z.mean(axis=0) for z in lib_z])
    Qraw = np.stack([z.mean(axis=0) for z in q_z])
    ns = np.array([len(z) for z in q_z])
    cos = unit(Qraw) @ unit(Craw).T
    order = np.argsort(-cos, axis=1, kind="stable")
    top = order[:, :k]
    res = {"cos": cos, "top": top, "ns": ns, "k": k}
    ct = np.take_along_axis(cos, top, axis=1)
    mt, st = ct.mean(1, keepdims=True), ct.std(1, keepdims=True) + 1e-9
    ma, sa = cos.mean(1, keepdims=True), cos.std(1, keepdims=True) + 1e-9
    gap1 = ct[:, :1] - ct
    gapp = np.concatenate([np.zeros((nq, 1)), ct[:, :-1] - ct[:, 1:]], axis=1)
    rank = np.broadcast_to(np.arange(1, k + 1), (nq, k))
    cosD = unit(Qraw[:, :n_dense]) @ unit(Craw[:, :n_dense]).T
    sp = sel[n_dense:]
    cosS = unit(Qraw[:, sp]) @ unit(Craw[:, sp]).T
    cD = np.take_along_axis(cosD, top, axis=1)
    cS = np.take_along_axis(cosS, top, axis=1)
    # game-level kNN features in the selected space
    sizes = {len(z) for z in lib_z}
    kmax = np.empty((nq, k)); kmean = np.empty((nq, k)); ktop3 = np.empty((nq, k))
    if len(sizes) == 1:                                                              # E05 path (equal library sizes)
        LIB = np.stack([unit(z[:, sel]) for z in lib_z])                             # P,m,D
        for i in range(nq):
            Gq = unit(q_z[i][:, sel])                                                # N,D
            s = np.einsum("nd,kmd->knm", Gq, LIB[top[i]], optimize=True)             # k,N,m
            kmax[i] = s.max(axis=2).mean(axis=1)
            kmean[i] = s.mean(axis=(1, 2))
            ktop3[i] = np.sort(s, axis=2)[:, :, -3:].mean(axis=(1, 2))
    else:                                                                            # unequal libraries: per candidate
        LIBL = [unit(z[:, sel]) for z in lib_z]
        for i in range(nq):
            Gq = unit(q_z[i][:, sel])
            for j, c in enumerate(top[i]):
                s = Gq @ LIBL[c].T                                                   # N,m
                kmax[i, j] = s.max(axis=1).mean()
                kmean[i, j] = s.mean()
                ktop3[i, j] = np.sort(s, axis=1)[:, -3:].mean()
    q_sel, c_sel = Qraw[:, sel], Craw[:, sel]
    pair = c_sel[top]                                                                # nq,k,D
    diff = np.abs(q_sel[:, None, :] - pair)
    prod = q_sel[:, None, :] * pair
    scal = np.stack([ct, rank, gap1, gapp, (ct - mt) / st, (ct - ma) / sa, np.broadcast_to(ns[:, None], (nq, k)),
                     cD, cS, kmax, kmean, ktop3], axis=2)
    X = np.concatenate([scal, diff, prod], axis=2).astype(np.float32)
    assert np.isfinite(X).all()
    res["X"] = X.reshape(nq * k, -1)
    return res


def top5_indices(top: np.ndarray, score: np.ndarray | None, k: int | None = None) -> np.ndarray:
    """Top-5 candidate indices: cosine order (score None, = M01) or reranked by `score` (nq*k)."""
    if score is None:
        return top[:, :TOP]
    nq, kk = top.shape
    o = np.argsort(-np.asarray(score).reshape(nq, kk), axis=1, kind="stable")[:, :TOP]
    return np.take_along_axis(top, o, axis=1)


def fratio_sparse_select(z_black: np.ndarray, pid: np.ndarray, n_dense: int = N_DENSE,
                         n_sel: int = N_SPARSE_SEL) -> np.ndarray:
    """Column index set (dense 0..n_dense-1 + top n_sel sparse columns by between/within player variance ratio)
    from the black games (z-scored) of the training players."""
    Zf = z_black[:, n_dense:]
    codes, inv = np.unique(pid, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    m1 = np.zeros((len(codes), Zf.shape[1]))
    m2 = np.zeros_like(m1)
    for a in range(0, len(Zf), 20000):
        b = Zf[a:a + 20000].astype(np.float64)
        np.add.at(m1, inv[a:a + 20000], b)
        np.add.at(m2, inv[a:a + 20000], b * b)
    mean = m1 / cnt[:, None]
    within = (m2 / cnt[:, None] - mean ** 2).mean(axis=0)
    between = mean.var(axis=0)
    F = between / (within + 1e-9)
    sel_sparse = np.argsort(-F, kind="stable")[:n_sel]
    return np.r_[np.arange(n_dense), n_dense + sel_sparse].astype(np.int32)
