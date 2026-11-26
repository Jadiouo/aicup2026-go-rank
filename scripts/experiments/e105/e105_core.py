"""E105 core: per-rank bias correction of the final 10-class probabilities P of v11 (E103 ALL arm).  Plan (frozen): experiments/E105_rank_class_bias/PLAN.md.
P' = normalise(P * exp(b)), b in R^10 with sum 0, fitted by multinomial negative log-likelihood (no regulariser, no hyper-parameter).  B2: log P' = log(P) / T + b  (information only).
Scoring / bootstrap functions are those of E103 (scripts/experiments/e103/e103_core.py re-exports r0_common -> e35.score.boot / diff / ci, e43_common.bal); imported READ-ONLY.
Nothing here reads H players or h_eval_ledger.jsonl.  CPU only.
"""
import os, sys, json, hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[3]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts/experiments/e90"), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from r0_common import bal, NS, boot, diff, ci, rank_scores, decide_ev, nrm          # same names e103_core imports from r0_common
from scripts.oneoff.e43_common import Q

OUT = ROOT / ".scratch/e105"
TAG = "e105_class_bias"; NB = 4000; EPS = 1e-12                                       # EPS = the 1e-12 floor of e103_core.fuse
VAL_PREDS = ROOT / ".scratch/e103/val_preds.npz"                                        # E103 formal-run output (P_ALL, P_REF, qid, pred_*)


# ---------------------------------------------------------------------------------------------------------------- data
def load_val():
    """fold-4 questions in the E103 order: y, ns, pl, F (full-size frame for boot), P_ALL, P_REF, pred_ALL, pred_REF, part (the 5 player-level parts of E103's c1_nested outer split)."""
    from sklearn.model_selection import StratifiedKFold
    Qf = Q(); fold = Qf.fold; idx4 = np.flatnonzero(fold == 4); ny = len(fold)
    v = np.load(VAL_PREDS); qid = Qf.q.qid.to_numpy()[idx4].astype(str); assert (v["qid"].astype(str) == qid).all(), "val_preds qid order != fold-4 order"
    y = Qf.truth[idx4]; ns = Qf.ns[idx4]; pl = Qf.pl[idx4]; assert (v["yq_not_for_scoring_here"] == y).all() and (v["nq"] == ns).all()
    players = np.unique(pl); prank = Qf.aux["player_rank"][players].astype(int); assert (prank[np.searchsorted(players, pl)] == y).all()
    part = np.full(len(idx4), -1)
    for j, (_, tep) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(players, prank)): part[np.isin(pl, players[tep])] = j     # = c1_nested's outer split (E103 ctx["part"])
    assert (part >= 0).all()
    F = SimpleNamespace(pl=Qf.pl, ns=Qf.ns, truth=Qf.truth, fold=fold)
    return dict(y=y, ns=ns, pl=pl, idx4=idx4, ny=ny, m4=(fold == 4), F=F, part=part, P_ALL=np.asarray(v["P_ALL"], np.float64), P_REF=np.asarray(v["P_REF"], np.float64),
                pred_ALL=v["pred_ALL"], pred_REF=v["pred_REF"], Qf=Qf)


def full_scores(D, p):
    """question scores of the EV decision of p (fold-4 rows) as a full-size array (other rows 0, masked out by m4) - same as e103 `R["sc"](full(P))` on the fold-4 rows."""
    s = np.zeros(D["ny"]); s[D["idx4"]] = rank_scores(decide_ev(p), D["y"]); return s


def bal4(D, sc_full):
    return float(bal(sc_full, D["F"].truth, D["F"].ns, D["m4"]))


# ---------------------------------------------------------------------------------------------------------------- fits
def _nll_grad(theta, L, y, with_T):
    a = float(np.exp(theta[10])) if with_T else 1.0
    Z = a * L + theta[:10]; lse = logsumexp(Z, axis=1); lp = Z - lse[:, None]; n = len(y)
    nll = -lp[np.arange(n), y].mean(); G = np.exp(lp); G[np.arange(n), y] -= 1.; G /= n
    g = np.zeros_like(theta); g[:10] = G.sum(0)
    if with_T: g[10] = (G * L).sum() * a
    return nll, g


def fit_bias(P, y, with_T=False):
    """B1 (with_T False): b only.  B2: b and T (log P / T).  Multinomial NLL, L-BFGS-B, no regulariser.  Returns (b centred to sum 0, T or 1.0, nll)."""
    L = np.log(np.asarray(P, np.float64) + EPS); th0 = np.zeros(11 if with_T else 10)
    r = minimize(_nll_grad, th0, args=(L, y, with_T), jac=True, method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-15, gtol=1e-9))
    assert r.success or r.nit > 0, r
    b = r.x[:10] - r.x[:10].mean(); T = float(np.exp(-r.x[10])) if with_T else 1.0
    return b, T, float(r.fun), int(r.nit)


def apply_bias(P, b, T=1.0):
    z = np.log(np.asarray(P, np.float64) + EPS) / T + b; z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def oof_apply(P, y, part, with_T=False):
    """player-level 5-fold: fit on 4 parts, correct the 5th; returns corrected P (all rows) and the 5 fitted (b, T)."""
    out = np.zeros_like(P); fits = []
    for j in range(5):
        tr = part != j; te = ~tr; b, T, nll, nit = fit_bias(P[tr], y[tr], with_T); out[te] = apply_bias(P[te], b, T); fits.append(dict(part=j, b=b.tolist(), T=T, nll_train=nll, nit=nit))
    return out, fits


# ---------------------------------------------------------------------------------------------------------------- reporting helpers
def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def confusion(y, pred):
    M = np.zeros((10, 10), int); np.add.at(M, (y, pred), 1); return M


def reweight10_boot(D, d_q):
    """Delta of per-question score difference d_q (fold-4 rows) with every TRUE rank weighted 10 % (unweighted mean inside each rank, then mean over the 10 ranks), point + player-level bootstrap
    (same resample matrix construction as e35.score.boot: multinomial over players, seed sd('bootstrap', tag), NB draws)."""
    pl, y = D["pl"], D["y"]; u = np.unique(pl); P = len(u); g = np.searchsorted(u, pl)
    Cm = np.random.default_rng(sd_("bootstrap", TAG + "_w10")).multinomial(P, np.full(P, 1.0 / P), size=NB).astype(np.float64)
    num = np.stack([np.bincount(g, weights=d_q * (y == r), minlength=P) for r in range(10)]); cnt = np.stack([np.bincount(g, weights=(y == r).astype(float), minlength=P) for r in range(10)])
    est = float(np.mean(num.sum(1) / cnt.sum(1)))
    with np.errstate(invalid="ignore", divide="ignore"): bs = np.mean((Cm @ num.T) / (Cm @ cnt.T), axis=1)
    return dict(delta=est, **ci(bs), frac_boot_gt_0=float((bs > 0).mean()))


def sd_(*a):
    from scripts.experiments.e35.common import sd                                               # the `sd` e35.score.boot uses
    return sd(*a)
