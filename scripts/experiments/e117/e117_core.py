"""E117 core: second-stage classifier on the components of the final Rank probabilities.  Plan (frozen): experiments/E117_rank_stage2/PLAN.md.  CPU only.
Everything frozen by the plan is a module constant here and is not changed:
  inputs   : log of P_ALL, P_REF, FF, P_C1, P_C2, P_C3 (6 x 10) + one-hot of N (5/10/15/20)   (no other feature; LR inputs are standardised with TRAINING mean/sd = preprocessing, not a feature)
  models   : (i) multinomial LR, L2, C in {.01,.1,1};  (ii) LightGBM multiclass, num_leaves 4, learning_rate .03, n_estimators in {100,300}
  weights  : every training question gets 1/(# training questions with the same N and the same rank), rescaled to mean 1 (balanced inside every N, which is what the score does)
  output   : normalise(P_ALL^(1-rho) * P_S2^rho), rho in {.5, 1}; EV decode
  nesting  : outer = E105/E103 player-level 5 parts (e105_core.load_val()['part']); inner = 3-fold player-grouped (StratifiedKFold on players x rank) inside the 4 training parts,
             the (model, hyper-parameter, rho) with the best inner-OOF balanced score is used on the outer part.
Read-only imports from e105_core (which imports e103/e90/e43/e35 helpers).  Nothing here reads H players, h_eval_ledger.jsonl or any test answer.
"""
import os, sys, json, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/experiments/e105"))
import numpy as np
import e105_core as C5                                                                     # read-only
from e105_core import ROOT as _R, VAL_PREDS, boot, diff, ci, decide_ev, rank_scores, bal, NS      # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

OUT = ROOT / ".scratch/e117"
TAG = "e117_rank_stage2"; NB = 4000; EPS = 1e-12
KEYS = ["P_ALL", "P_REF", "FF", "P_C1", "P_C2", "P_C3"]
LR_C = [.01, .1, 1.]; GBM_T = [100, 300]; RHOS = [.5, 1.]
CONFIGS = [("LR", c) for c in LR_C] + [("GBM", t) for t in GBM_T]
SEED = 117; N_LEVELS = [5, 10, 15, 20]
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
THRESH_DELTA = .002; THRESH_FRAC = .85


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sha16(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------- data
def load_all():
    D = C5.load_val(); v = np.load(VAL_PREDS)
    D["Pk"] = {k: np.asarray(v[k], np.float64) for k in KEYS}
    assert np.array_equal(D["Pk"]["P_ALL"], D["P_ALL"]) and np.array_equal(D["Pk"]["P_REF"], D["P_REF"])
    for k in KEYS: assert D["Pk"][k].shape == (4568, 10) and np.isfinite(D["Pk"][k]).all() and np.allclose(D["Pk"][k].sum(1), 1, atol=1e-6), k
    return D


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


def bal_sub(P, y, ns):
    """balanced score (N equal, rank equal inside N) of the EV decision of P on a subset (same arithmetic as e43_common.bal)."""
    return float(bal(rank_scores(decide_ev(P), y), y, ns))


def inner_oof(X, y, ns, pl, strat, seed=SEED):
    """3-fold player-grouped OOF probabilities of every config on the given rows (StratifiedKFold on players x rank).  Asserts the inner train/eval players are disjoint."""
    players = np.unique(pl); prank = np.array([strat[pl == p][0] for p in players]); assert all((strat[pl == p] == strat[pl == p][0]).all() for p in players)     # strat = the player's TRUE rank, only to stratify the player split
    oof = {c: np.zeros((len(y), 10)) for c in [("LR", c) for c in LR_C] + [("GBM", t) for t in GBM_T]}
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(players, prank):
        tr = np.isin(pl, players[a]); te = np.isin(pl, players[b]); assert not (set(pl[tr]) & set(pl[te])) and (tr ^ te).all()
        m = Stage2().fit(X[tr], y[tr], sample_weights(y[tr], ns[tr])); pr = m.predict(X[te])
        for c in oof: oof[c][te] = pr[c]
    return oof


def select(X, y, ns, pl, P_all, strat):
    """inner selection on the given (training) rows.  Returns the inner score table {(model, hp, rho): score}, the best hp for every (model, rho), the overall best (model, hp, rho)."""
    oof = inner_oof(X, y, ns, pl, strat)
    tab = {(c[0], c[1], rho): bal_sub(combine(P_all, oof[c], rho), y, ns) for c in oof for rho in RHOS}
    best_by = {}
    for mdl in ("LR", "GBM"):
        for rho in RHOS:
            cands = [(k, v) for k, v in tab.items() if k[0] == mdl and k[2] == rho]; best_by[(mdl, rho)] = max(cands, key=lambda kv: kv[1])[0]    # first max in CONFIGS order
    best = max(tab.items(), key=lambda kv: kv[1])[0]
    return tab, best_by, best


def nested(D, X, y_train, Pk=None, tag=""):
    """outer 5 parts; returns per-arm OOF probabilities {arm: (n,10)} for arms LR@.5, LR@1, GBM@.5, GBM@1 (inner-chosen hp) and 'CHOSEN' (inner-chosen model+hp+rho), plus per-fold logs.
    y_train = labels used for fitting / selection (true labels, or shuffled for the dry run); evaluation labels are NOT used inside."""
    part, ns, pl, P_all = D["part"], D["ns"], D["pl"], D["Pk"]["P_ALL"] if Pk is None else Pk["P_ALL"]
    arms = {f"{m}@{r:g}": np.zeros((len(ns), 10)) for m in ("LR", "GBM") for r in RHOS}; arms["CHOSEN"] = np.zeros((len(ns), 10)); folds = []
    for j in range(5):
        tr = part != j; te = ~tr; assert not (set(pl[tr]) & set(pl[te])), "train/eval players overlap"
        tab, best_by, best = select(X[tr], y_train[tr], ns[tr], pl[tr], P_all[tr], D.get('y_strat', D['y'])[tr])
        m = Stage2().fit(X[tr], y_train[tr], sample_weights(y_train[tr], ns[tr])); pr = m.predict(X[te])
        for (mdl, rho), k in best_by.items(): arms[f"{mdl}@{rho:g}"][te] = combine(P_all[te], pr[(k[0], k[1])], rho)
        arms["CHOSEN"][te] = combine(P_all[te], pr[(best[0], best[1])], best[2])
        folds.append(dict(part=j, n_train=int(tr.sum()), n_eval=int(te.sum()), chosen=dict(model=best[0], hp=best[1], rho=best[2]), chosen_inner_score=tab[best],
                          best_by_model_rho={f"{a}@{r:g}": dict(hp=k[1], inner_score=tab[k]) for (a, r), k in best_by.items()}, inner_table={f"{k[0]}|{k[1]}|{k[2]:g}": v for k, v in tab.items()}))
        log(tag, "outer part", j, "chosen", best, "inner %.4f" % tab[best])
    return arms, folds


# ---------------------------------------------------------------------------------------------------------------- scoring helpers (full-size arrays for e35 boot)
def full_sc(D, P, y=None):
    y = D["y"] if y is None else y; s = np.zeros(D["ny"]); s[D["idx4"]] = rank_scores(decide_ev(P), y); return s


def frame_with_truth(D, y):
    from types import SimpleNamespace
    t = D["F"].truth.copy(); t[D["idx4"]] = y; return SimpleNamespace(pl=D["F"].pl, ns=D["F"].ns, truth=t, fold=D["F"].fold)


def bal_full(D, s, F=None):
    F = D["F"] if F is None else F; return float(bal(s, F.truth, F.ns, D["m4"]))
