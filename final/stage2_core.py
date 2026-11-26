"""Rank final stage-2 core (v12 / v13).  Trimmed, line-for-line extract of
  scripts/experiments/e117/e117_core.py  (E117: second-stage multinomial LR on the six probability blocks; class Stage2, make_X, combine, sample_weights)
  scripts/experiments/e124/e124_core.py  (E124: shift-invariant kernel second stage; class ShiftS2 + L-BFGS objective)
with the validation-framework imports (e105_core / e103 / e43 ...) removed, so that the last steps of the Rank pipeline run from three small files:
  val_preds.npz (4,568 validation questions, six probability blocks + labels), test_inputs.npz (800 test questions, same six blocks), stage2_fit.pkl (E117 LR fitted on val).
The pickle stage2_fit.pkl was written by e117_core, so make_rank_final.py registers this module as `e117_core` in sys.modules before unpickling.
CPU only.
"""
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

TAG = "e117_rank_stage2"; NB = 4000; EPS = 1e-12
KEYS = ["P_ALL", "P_REF", "FF", "P_C1", "P_C2", "P_C3"]
LR_C = [.01, .1, 1.]; GBM_T = [100, 300]; RHOS = [.5, 1.]
CONFIGS = [("LR", c) for c in LR_C] + [("GBM", t) for t in GBM_T]
SEED = 117; N_LEVELS = [5, 10, 15, 20]
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
THRESH_DELTA = .002; THRESH_FRAC = .85

K_HALF = 2; C_SHIFT = 30.; RHO = .5
FITLOG = []

def make_X(Pk, ns):
    """feature matrix: log of the six probability blocks + one-hot N.  Pk: dict key -> (n,10)."""
    ns = np.asarray(ns); assert set(np.unique(ns)) <= set(N_LEVELS)
    L = [np.log(np.asarray(Pk[k], np.float64) + EPS) for k in KEYS]
    return np.hstack(L + [np.stack([(ns == n).astype(np.float64) for n in N_LEVELS], 1)])


def sample_weights(y, ns):
    """1/(count of the same (N, rank)) within the training set, mean 1."""
    w = np.zeros(len(y))
    for n in N_LEVELS:
        m = ns == n
        for r in range(10):
            k = m & (y == r)
            if k.any(): w[k] = 1.0 / k.sum()
    return w / w.mean()


# ---------------------------------------------------------------------------------------------------------------- models
class Stage2:
    """fit all CONFIGS on (X, y, w) once; predict the 10-class probabilities of every config.  GBM: one 300-tree fit, the 100-tree model is its first 100 iterations (identical to a 100-tree fit)."""
    def __init__(self): self.m = {}

    def fit(self, X, y, w):
        assert set(np.unique(y)) == set(range(10)), "a training set without all 10 ranks"
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-9; Z = (X - self.mu) / self.sd
        for c in LR_C:
            self.m[("LR", c)] = LogisticRegression(C=c, max_iter=3000, tol=1e-6).fit(Z, y, sample_weight=w)
        g = lgb.LGBMClassifier(objective="multiclass", num_leaves=4, learning_rate=.03, n_estimators=max(GBM_T), n_jobs=2, random_state=SEED, verbose=-1, deterministic=True, force_row_wise=True)
        g.fit(X, y, sample_weight=w); self.m[("GBM", "all")] = g
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd; out = {}
        for c in LR_C:
            m = self.m[("LR", c)]; assert list(m.classes_) == list(range(10)); out[("LR", c)] = m.predict_proba(Z)
        g = self.m[("GBM", "all")]; assert list(g.classes_) == list(range(10))
        for t in GBM_T: out[("GBM", t)] = g.predict_proba(X, num_iteration=t)
        return out


def combine(P_all, P_s2, rho):
    """normalise(P_ALL^(1-rho) * P_S2^rho); rho = 0 -> P_ALL (same code path)."""
    z = (1 - rho) * np.log(np.asarray(P_all, np.float64) + EPS) + rho * np.log(np.asarray(P_s2, np.float64) + EPS)
    z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------------------ generic weighted-NLL + L2 (kernel only) objective
def make_objective(fwd, bwd, y, w, pen_mask, C):
    n = len(y); idx = np.arange(n)
    def f(theta):
        lg = fwd(theta); m = lg.max(1, keepdims=True); e = np.exp(lg - m); s = e.sum(1, keepdims=True); lse = (m + np.log(s))[:, 0]
        nll = float((w * (lse - lg[idx, y])).sum())
        G = e / s; G[idx, y] -= 1.0; G *= w[:, None]
        g = bwd(G); tp = theta[pen_mask]
        g[pen_mask] += tp / C
        return nll + float((tp ** 2).sum()) / (2 * C), g
    return f


def run_lbfgs(f, n_par, tag):
    t0 = time.time()
    r = minimize(f, np.zeros(n_par), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, maxfun=40000, maxcor=30, ftol=1e-15, gtol=1e-7))
    g = f(r.x)[1]
    FITLOG.append(dict(tag=tag, success=bool(r.success), message=str(r.message), nit=int(r.nit), nfev=int(r.nfev), fun=float(r.fun), grad_inf=float(np.abs(g).max()), sec=round(time.time() - t0, 2)))
    return r.x


# ------------------------------------------------------------------------------------------------------------ shift-invariant model
def log_stack(Pk, groups):
    return np.stack([np.log(np.asarray(Pk[k], np.float64) + EPS) for k in groups], 1)          # (n, G, 10)


def kernel_design(Z, k):
    """Z: (n,G,10) standardised.  Returns T (n,10,G*(2k+1)) with T[i,c,g*(2k+1)+j] = Z[i,g,clip(c+j-k,0,9)]."""
    n, G, _ = Z.shape; idx = np.clip(np.arange(10)[:, None] + np.arange(2 * k + 1)[None, :] - k, 0, 9)      # (10, 2k+1)
    T = Z[:, :, idx]                                                                           # (n,G,10,2k+1)
    return np.ascontiguousarray(T.transpose(0, 2, 1, 3).reshape(n, 10, G * (2 * k + 1)))


def kernel_logit(Z, a, k):
    """the kernel part of the logit alone; a: (G,2k+1)."""
    return kernel_design(Z, k).reshape(-1, a.size).dot(a.reshape(-1)).reshape(len(Z), 10)


def n_index(ns):
    ns = np.asarray(ns); ni = np.full(len(ns), -1)
    for i, n in enumerate(N_LEVELS): ni[ns == n] = i
    assert (ni >= 0).all(); return ni


def shift_objective(T, ni, y, w, C):
    n, _, P = T.shape; T2 = T.reshape(n * 10, P)
    def fwd(th): return (T2 @ th[:P]).reshape(n, 10) + th[P:].reshape(4, 10)[ni]
    def bwd(G):
        g = np.zeros(P + 40); g[:P] = T2.T @ G.reshape(-1)
        gb = np.zeros((4, 10))
        for i in range(4): gb[i] = G[ni == i].sum(0)
        g[P:] = gb.reshape(-1); return g
    pm = np.zeros(P + 40, bool); pm[:P] = True
    return make_objective(fwd, bwd, y, w, pm, C), P + 40


class ShiftS2:
    def __init__(self, k=K_HALF, C=C_SHIFT, groups=tuple(KEYS)): self.k, self.C, self.groups = k, C, list(groups)

    def fit(self, Pk, ns, y, w, tag="shift"):
        assert set(np.unique(y)) == set(range(10))
        L = log_stack(Pk, self.groups); self.mu = float(L.mean()); self.sd = float(L.std() + 1e-9)
        T = kernel_design((L - self.mu) / self.sd, self.k); f, npar = shift_objective(T, n_index(ns), np.asarray(y), np.asarray(w, np.float64), self.C)
        th = run_lbfgs(f, npar, tag); P = len(self.groups) * (2 * self.k + 1)
        self.a = th[:P].reshape(len(self.groups), 2 * self.k + 1); self.B = th[P:].reshape(4, 10); return self

    def logits(self, Pk, ns):
        Z = (log_stack(Pk, self.groups) - self.mu) / self.sd
        return kernel_logit(Z, self.a, self.k) + self.B[n_index(ns)]

    def predict(self, Pk, ns):
        lg = self.logits(Pk, ns); lg -= lg.max(1, keepdims=True); p = np.exp(lg); return p / p.sum(1, keepdims=True)
