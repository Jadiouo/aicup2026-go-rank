"""E124 core: second stage with a shift-invariant (along the rank axis) kernel.  Plan (frozen): experiments/E124_rank_stage2_shift/PLAN.md.  CPU only.
logit_c = sum_b sum_{d=-k..k} a[b,d] * z_b[clip(c+d,0,9)] + bias[N,c];  objective = sum_i w_i * NLL_i + (1/(2C)) * sum a^2  (kernel only, bias unpenalised); L-BFGS from zero (convex).
z = (log(P+EPS) - mu)/sd with ONE scalar mu, sd over all training log-probability columns.  Fixed: k=2, C=30, rho=.5.
Read-only imports from e117_core (data, features, weights, combine, scoring).  Nothing here reads H players, h_eval_ledger.jsonl or any test answer.
"""
import os, sys, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/experiments/e117"))
import numpy as np
import e117_core as K117                                                                       # read-only
from e117_core import (load_all, make_X, sample_weights, combine, full_sc, bal_full, boot, diff, ci, decide_ev, rank_scores, bal, KEYS, N_LEVELS, EPS, RANKS, NS)   # noqa
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

OUT = ROOT / ".scratch/e124"
TAG = "e124_rank_stage2_shift"; NB = 4000
K_HALF = 2; C_SHIFT = 30.; RHO = .5
FITLOG = []


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sha16(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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


# ------------------------------------------------------------------------------------------------------------ the same optimiser, unconstrained "full matrix" form (= multinomial LR), for the dry check against sklearn
class FullLRS2:
    """E117's LR (per-column standardisation of X = make_X, C, intercept unpenalised) solved with OUR objective/L-BFGS code."""
    def __init__(self, C=1.): self.C = C

    def fit(self, X, y, w, tag="fullLR"):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-9; Z = (X - self.mu) / self.sd; F = Z.shape[1]; y = np.asarray(y)
        def fwd(th): return Z @ th[:F * 10].reshape(F, 10) + th[F * 10:]
        def bwd(G): return np.concatenate([(Z.T @ G).reshape(-1), G.sum(0)])
        pm = np.zeros(F * 10 + 10, bool); pm[:F * 10] = True
        th = run_lbfgs(make_objective(fwd, bwd, y, np.asarray(w, np.float64), pm, self.C), F * 10 + 10, tag)
        self.W = th[:F * 10].reshape(F, 10); self.b = th[F * 10:]; return self

    def predict(self, X):
        lg = ((X - self.mu) / self.sd) @ self.W + self.b; lg -= lg.max(1, keepdims=True); p = np.exp(lg); return p / p.sum(1, keepdims=True)


def sk_lr_predict(Xtr, ytr, wtr, Xte, C):
    """sklearn LR exactly as e117_core.Stage2 fits it (per-column standardisation with TRAINING mean/sd, max_iter 3000, tol 1e-6)."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=3000, tol=1e-6).fit((Xtr - mu) / sd, ytr, sample_weight=wtr); assert list(m.classes_) == list(range(10))
    return m.predict_proba((Xte - mu) / sd)


# ------------------------------------------------------------------------------------------------------------ nested (outer 5 parts = D["part"]) predictions
def nested_shift(D, y_train, groups=tuple(KEYS), k=K_HALF, C=C_SHIFT, tag="shift"):
    part, ns = D["part"], D["ns"]; out = np.zeros((len(ns), 10))
    for j in range(5):
        tr = part != j; te = ~tr; assert not (set(D["pl"][tr]) & set(D["pl"][te]))
        sub = lambda m: {kk: D["Pk"][kk][m] for kk in set(groups)}
        m = ShiftS2(k, C, groups).fit(sub(tr), ns[tr], y_train[tr], sample_weights(y_train[tr], ns[tr]), tag=f"{tag}_part{j}")
        out[te] = m.predict(sub(te), ns[te])
    return out


def nested_lr(D, X, y_train, Cs, impl="sk"):
    part, ns = D["part"], D["ns"]; out = {C: np.zeros((len(ns), 10)) for C in Cs}
    for j in range(5):
        tr = part != j; te = ~tr; assert not (set(D["pl"][tr]) & set(D["pl"][te])); w = sample_weights(y_train[tr], ns[tr])
        for C in Cs:
            out[C][te] = sk_lr_predict(X[tr], y_train[tr], w, X[te], C) if impl == "sk" else FullLRS2(C).fit(X[tr], y_train[tr], w, tag=f"fullLR_C{C}_part{j}").predict(X[te])
    return out
