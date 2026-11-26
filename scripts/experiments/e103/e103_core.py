"""E103 shared core (validation chain e103_val.py and test path e103_make.py use THESE functions, so the two paths cannot drift apart).
Plan (frozen): experiments/E103_rank_small_signal_merge/PLAN.md.  Imports e96 / e101 / e102 / e92 / r0_common READ-ONLY.  Nothing here reads H players or h_eval_ledger.

Fixed setting (v10d): gamma 0, tau 0 (no threshold), alpha .1, lam 2.0; library = small identity network finh_full.  Rank index 0..9 = D,C,B,A,1D..6D; eff 0 = questioned player BLACK.

Wiring of the three components (the ambiguity the task asked to resolve; chosen so that every component alone reproduces its old number):
  per voting game g (hd in (0,1)) of a question, on the QUESTIONED player's rank axis:
     t_lib = shift_raw(lib_lift) (C3 off)  |  E101 site_lift(lib_lift) (C3 on; hd == 1 rows only, hd == 0 untouched, per-game renormalised to mass 1)
     t_nn  = shift_raw(nn_lift)  (C2 on; the neighbour vote keeps ITS OWN E96 half/half conversion in every case = 'C2 的定先換算不動'; zero rows = no neighbour vote)
     x     = normalise(t_lib + w * t_nn)
  shift_raw = e96 shift_lift without its last renormalisation.  Why: shift_lift is not mass preserving, so E102's shift_lift(normalise(lib + w nn)) equals normalise(shift_raw(lib) + w shift_raw(nn)) and NOT the sum of two
  renormalised shift_lift outputs (first attempt did that and missed the E102 number by 2 answers).  With C2 and C3 off this is e96 shift_lift exactly; with C3 alone it is E101's F_site exactly; with C2 alone it is E102's T3 exactly.
  V = normalised sum of x over the voting games of the question (uniform .1 if none), then P = fuse(base, V, lam, alpha) = normalise(base * (V + alpha)^lam) with base = F_E79 (C1 off) or the C1 fusion probabilities (C1 on).
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts/experiments/e102"), str(ROOT / "scripts/experiments/e101"), str(ROOT / "scripts/experiments/e96"), str(ROOT / "scripts/experiments/e92"),
                str(ROOT / "scripts/experiments/e90"), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]
import numpy as np
import e96_score as S96
import e101_score as E1
import e102_score as E2
E2.S = S96                                                    # e102 helper functions refer to a module-level `S`
from r0_common import rebuild, bal, NS, boot, diff, ci, rank_scores, decide_ev, nrm
from e92_score import fit_predict, CS, ZS

OUT = ROOT / ".scratch/e103"
GM, LO, HI, AL, LAM = E1.GM, E1.LO, E1.HI, E1.AL, E1.LAM     # 0, 0, 9, .1, 2.0
BETA, W_NN, K_NN = 40., .5, 50                                # C2 fixed: beta 40, w .5, 50 neighbours
K_WHITE, K_BLACK = E1.K_WHITE, E1.K_BLACK
T_TAB = E2.T_TAB


def sc_q(p, yy):
    return rank_scores(decide_ev(p), yy)


# ============================================================================================================================ question structure (validation or test)
def slots_from_rows(rows, hd_g, eff_g):
    """rows [Q,S] = index of the game in the per-game arrays (or -1).  Returns slot-level dict(idx, valid, vote, eff, hd); vote = valid & hd in (0,1) (exactly E102.slots / e96 question_votes with hd_ok (0,1))."""
    valid = rows >= 0; idx = np.where(valid, rows, 0)
    hd = np.where(valid, hd_g[idx], -1).astype(np.int64); eff = np.where(valid, eff_g[idx], 0).astype(np.int64)
    return dict(idx=idx, valid=valid, vote=valid & np.isin(hd, (0, 1)), eff=eff, hd=hd)


# ============================================================================================================================ vote conversion + fusion (the shared path)
def shift_raw(x, eff, hd):
    """E96 shift_lift WITHOUT its final per-row renormalisation (same two formulas as e96_score.shift_lift; shift_lift is NOT mass preserving: sum = 1 + .5 (x9 - x0), so shift_lift(a + w b) != shift_lift(a) + w shift_lift(b);
    the raw operator IS linear, and E102's votes_slot = normalise(shift_raw(lib) + w shift_raw(nn)) exactly).  hd == 1 rows only; zero rows stay zero."""
    out = x.copy(); k = np.arange(10); blk = (hd == 1) & (eff == 0); wht = (hd == 1) & (eff == 1)
    out[blk] = .5 * x[blk] + .5 * x[blk][:, np.minimum(k + 1, 9)]; out[wht] = .5 * x[wht] + .5 * x[wht][:, np.maximum(k - 1, 0)]
    return out


def slot_x(sl, lib_g, nn_g=None, w=0., site=False):
    """-> x [Q,S,10] on the questioned player's rank axis (rows of non-voting slots are zero) and nn_present [Q,S].
    term_lib = shift_raw(lib) (C3 off) | E101 site_lift(lib) (C3 on, per-game normalised, hd == 1 only);  term_nn = shift_raw(nn) (C2's own E96 conversion, always);  x = normalise(term_lib + w term_nn)."""
    Q_, S_ = sl["idx"].shape; eff = sl["eff"].reshape(-1); hd = sl["hd"].reshape(-1); vote = sl["vote"].reshape(-1)
    lib = lib_g[sl["idx"]].reshape(-1, 10)
    if site: xl, _ = E1.site_lift(lib, eff, hd)
    else: xl = shift_raw(lib, eff, hd)
    x = xl.copy(); nn_present = np.zeros(len(x), bool)
    if nn_g is not None and w > 0:
        nn = nn_g[sl["idx"]].reshape(-1, 10) * vote[:, None]; nn_present = nn.sum(1) > 0; x = xl + w * shift_raw(nn, eff, hd)
    s = x.sum(1); ok = s > 0; x[ok] /= s[ok, None]; x[~ok] = 0.
    x = x * vote[:, None]
    return x.reshape(Q_, S_, 10), nn_present.reshape(Q_, S_)


def votes(sl, lib_g, nn_g=None, w=0., site=False):
    x, nnp = slot_x(sl, lib_g, nn_g, w, site)
    V = x.sum(1); tot = V.sum(1); has = tot > 0
    Vn = np.full(V.shape, .1); Vn[has] = V[has] / tot[has, None]
    return Vn, has


def fuse(base, V, lam=LAM, alpha=AL):
    z = np.log(base + 1e-12) + lam * np.log(V + alpha); z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def candidate_P(base, sl, lib_g, nn_g=None, w=W_NN, site=False):
    V, has = votes(sl, lib_g, nn_g, w if nn_g is not None else 0., site)
    return fuse(base, V), V, has


# ============================================================================================================================ C1: nested OOF fusion (copy of e92_score.pipeline keeping the fused probabilities) and the all-question fit
def c1_nested(R, X):
    """Line-by-line copy of scripts/experiments/e92/e92_score.py `pipeline` up to FF (the nested, calibrated fusion of F_E79 with the logistic regression on X); returns FF, PS, chosen.  No bootstrap here."""
    from sklearn.model_selection import StratifiedKFold
    y, ns, fold, F79, Qf = R["y"], R["ns"], R["fold"], R["F79"], R["Qf"]
    idx4 = np.flatnonzero(fold == 4); yq = y[idx4]; nq = ns[idx4]; plq = Qf.pl[idx4]; f79q = F79[idx4]
    aux = Qf.aux; players = np.unique(plq); prank = aux["player_rank"][players].astype(int)
    assert (prank[np.searchsorted(players, plq)] == yq).all()
    PS = np.zeros((len(idx4), 10)); FF = np.zeros((len(idx4), 10)); chosen = []
    outer = StratifiedKFold(5, shuffle=True, random_state=0)
    for j, (trp, tep) in enumerate(outer.split(players, prank)):
        tr = np.isin(plq, players[trp]); te = np.isin(plq, players[tep]); assert not (tr & te).any() and (tr | te).all()
        tri = np.flatnonzero(tr); trpl = np.unique(plq[tr]); trrank = aux["player_rank"][trpl].astype(int)
        oof = {C: np.zeros((len(idx4), 10)) for C in CS}
        for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(trpl, trrank):
            ia = tri[np.isin(plq[tri], trpl[a])]; ib = tri[np.isin(plq[tri], trpl[b])]; assert not np.isin(plq[ia], plq[ib]).any()
            for C in CS: oof[C][ib] = fit_predict(X[ia], yq[ia], X[ib], C)
        sC = {C: bal(sc_q(oof[C][tri], yq[tri]), yq[tri], nq[tri]) for C in CS}
        Cb = None
        for C in CS:
            if Cb is None or sC[C] > sC[Cb] + 1e-12: Cb = C
        sZ = {z: bal(sc_q(nrm((1 - z) * f79q[tri] + z * oof[Cb][tri]), yq[tri]), yq[tri], nq[tri]) for z in ZS}
        zb = None
        for z in ZS:
            if zb is None or sZ[z] > sZ[zb] + 1e-12: zb = z
        tei = np.flatnonzero(te); PS[tei] = fit_predict(X[tri], yq[tri], X[tei], Cb); FF[tei] = nrm((1 - zb) * f79q[tei] + zb * PS[tei])
        chosen.append(dict(part=j, C=Cb, z=zb))
    assert (FF.sum(1) > 0).all() and (PS.sum(1) > 0).all()
    return FF, PS, chosen


def c1_fit_all(R, X):
    """The inner program of e98_score / e92 pipeline run ONCE on all fold-4 questions (3-fold inner OOF over players stratified by rank: C from .1/1/10, z from 0..0.6), then the logistic regression is refitted on ALL
    questions.  Returns dict(C, z, mu, sd, coef, intercept, inner_scores...) - the test-time model (PLAN section 1, C1 'test')."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    y, ns, fold, F79, Qf = R["y"], R["ns"], R["fold"], R["F79"], R["Qf"]
    idx4 = np.flatnonzero(fold == 4); yq = y[idx4]; nq = ns[idx4]; plq = Qf.pl[idx4]; f79q = F79[idx4]; aux = Qf.aux
    tri = np.arange(len(idx4)); trpl = np.unique(plq); trrank = aux["player_rank"][trpl].astype(int)
    oof = {C: np.zeros((len(idx4), 10)) for C in CS}
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(trpl, trrank):
        ia = tri[np.isin(plq[tri], trpl[a])]; ib = tri[np.isin(plq[tri], trpl[b])]; assert not np.isin(plq[ia], plq[ib]).any()
        for C in CS: oof[C][ib] = fit_predict(X[ia], yq[ia], X[ib], C)
    sC = {C: bal(sc_q(oof[C][tri], yq[tri]), yq[tri], nq[tri]) for C in CS}; Cb = None
    for C in CS:
        if Cb is None or sC[C] > sC[Cb] + 1e-12: Cb = C
    sZ = {z: bal(sc_q(nrm((1 - z) * f79q[tri] + z * oof[Cb][tri]), yq[tri]), yq[tri], nq[tri]) for z in ZS}; zb = None
    for z in ZS:
        if zb is None or sZ[z] > sZ[zb] + 1e-12: zb = z
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
    clf = LogisticRegression(C=Cb, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-4).fit((X - mu) / sd, yq)
    assert (clf.classes_ == np.arange(10)).all()
    return dict(C=float(Cb), z=float(zb), mu=mu, sd=sd, coef=clf.coef_.copy(), intercept=clf.intercept_.copy(), inner_S_by_C={str(k): float(v) for k, v in sC.items()},
                inner_fused_by_z={str(k): float(v) for k, v in sZ.items()}, n_questions=int(len(idx4)), n_players=int(len(trpl)))


def c1_predict(fit, X):
    """softmax(standardised X @ coef.T + b) = LogisticRegression.predict_proba of the stored fit."""
    zz = ((X - fit["mu"]) / fit["sd"]) @ fit["coef"].T + fit["intercept"]; zz -= zz.max(1, keepdims=True); p = np.exp(zz); return p / p.sum(1, keepdims=True)


def c1_base(fit, F_base, X):
    """test-time C1 base = nrm((1 - z) F_E79 + z P_LR)."""
    return nrm((1 - fit["z"]) * F_base + fit["z"] * c1_predict(fit, X))


def save_c1_fit(fit, fn_json, fn_npz):
    meta = {k: v for k, v in fit.items() if k not in ("mu", "sd", "coef", "intercept")}
    Path(fn_json).write_text(json.dumps(meta, indent=1, default=float)); np.savez(fn_npz, mu=fit["mu"], sd=fit["sd"], coef=fit["coef"], intercept=fit["intercept"], C=fit["C"], z=fit["z"])


def load_c1_fit(fn_npz):
    z = np.load(fn_npz); return dict(mu=z["mu"], sd=z["sd"], coef=z["coef"], intercept=z["intercept"], C=float(z["C"]), z=float(z["z"]))


# ============================================================================================================================ C2 helpers
def bank_T(meta):
    """filtered (hd <= 1) bank rows and their table-rank distributions T [M,10] (float32), exactly as e102 evaluate()."""
    keep = np.flatnonzero(meta["hd"] <= 1)
    T = T_TAB[meta["hd"][keep].astype(int), meta["eff"][keep].astype(int), meta["owner_rank"][keep].astype(int)].astype(np.float32)
    return keep, T


def nn_game_lift(nn_row, nn_cos, T_bank, beta=BETA):
    return E2.nn_lift_games(nn_row, nn_cos, T_bank, beta, len(nn_row))
