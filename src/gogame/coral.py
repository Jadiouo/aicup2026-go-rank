"""CORAL ordinal logistic regression (CPU, closed convex problem, L-BFGS).  Used by E06 M04.

P(y > k | x) = sigmoid(w.x + b_k), b_0 > b_1 > ... > b_{K-2} (monotone via softplus gaps), one shared slope w.
Same objective as scripts/experiments/e03/runner.coral_proba (ridge 0.01 on the slope, deterministic, zero init),
plus per-sample weights and a train_prior so that `predict_proba` matches the RankModel interface
(raw probabilities, or prior-corrected toward a uniform target).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import apply_prior_correction

CLASSES = 10


def _softplus(x):
    return np.logaddexp(0.0, x)


def _expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass
class CoralModel:
    mu: np.ndarray
    sd: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    train_prior: np.ndarray
    info: dict = field(default_factory=dict)
    best_iteration: int = 0

    def _z(self, X):
        x = np.asarray(X, dtype=np.float64)
        return np.nan_to_num((x - self.mu) / self.sd, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_proba(self, X, target_prior=None, raw: bool = False) -> np.ndarray:
        z = (self._z(X) @ self.slope)[:, None] + self.intercept[None, :]
        c = _expit(z)                                       # P(y > k), k = 0..K-2, non-increasing in k
        p = np.column_stack((1.0 - c[:, 0], c[:, :-1] - c[:, 1:], c[:, -1]))
        p = np.clip(p, 0.0, None)
        p = p / p.sum(axis=1, keepdims=True)
        if not raw:
            p = apply_prior_correction(p, self.train_prior, target_prior)
        return p.astype(np.float32)


def _standardize_chunked(X, chunk: int):
    """Low-memory equivalent of the default path: X stays in its own dtype (e.g. float32), mu/sd are accumulated in float64
    over row chunks and only the float64 Z is materialised (no float64 copy of X, no (X-mu)/sd temporaries)."""
    n, d = X.shape
    s1, cnt = np.zeros(d), np.zeros(d)
    for i in range(0, n, chunk):
        x = np.asarray(X[i:i + chunk], dtype=np.float64)
        ok = ~np.isnan(x)
        s1 += np.where(ok, x, 0.0).sum(axis=0)
        cnt += ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nan_to_num(s1 / cnt, nan=0.0)
    s2 = np.zeros(d)
    for i in range(0, n, chunk):
        x = np.asarray(X[i:i + chunk], dtype=np.float64)
        s2 += np.where(np.isnan(x), 0.0, (x - mu) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.nan_to_num(np.sqrt(s2 / cnt), nan=1.0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Z = np.empty((n, d))
    for i in range(0, n, chunk):
        Z[i:i + chunk] = np.nan_to_num((np.asarray(X[i:i + chunk], dtype=np.float64) - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return Z, mu, sd


def fit_coral(X, y, w=None, ridge: float = 0.01, maxiter: int = 500, n_classes: int = CLASSES, chunk: int | None = None) -> CoralModel:
    """chunk=None: original behaviour (float64 copy of X).  chunk=int: same maths, row-chunked standardisation (E10)."""
    from scipy.optimize import minimize
    if chunk:
        Z, mu, sd = _standardize_chunked(X, int(chunk))
        n, d = Z.shape
    else:
        X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if not chunk:
        n, d = X.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    w = w / w.mean()
    if not chunk:
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            mu = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)
            sd = np.nan_to_num(np.nanstd(X, axis=0), nan=1.0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        Z = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    K1 = n_classes - 1
    target = (y[:, None] > np.arange(K1)[None, :]).astype(np.float64)

    def unpack(theta):
        slope = theta[:d]
        gaps = _softplus(theta[d + 1:])
        return slope, theta[d] - np.r_[0.0, np.cumsum(gaps)]

    def objective(theta):
        slope, icpt = unpack(theta)
        z = (Z @ slope)[:, None] + icpt[None, :]
        err = (_expit(z) - target) * w[:, None]
        loss = ((_softplus(z) - target * z) * w[:, None]).sum() / n + ridge * slope @ slope
        g_slope = Z.T @ err.sum(axis=1) / n + 2 * ridge * slope
        b = err.sum(axis=0) / n                                # dL/d icpt_k
        gap_grad = -_expit(theta[d + 1:]) * np.cumsum(b[::-1])[::-1][1:]
        return loss, np.r_[g_slope, b.sum(), gap_grad]

    theta0 = np.zeros(d + K1)
    theta0[d + 1:] = -1.0
    fit = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-9, "gtol": 1e-6})
    slope, icpt = unpack(fit.x)
    prior = np.bincount(y, weights=w, minlength=n_classes).astype(np.float64)
    prior = prior / prior.sum()
    return CoralModel(mu, sd, slope, icpt, prior,
                      info={"optimizer_success": bool(fit.success), "n_iterations": int(fit.nit),
                            "message": str(fit.message), "n_features": int(d)},
                      best_iteration=int(fit.nit))
