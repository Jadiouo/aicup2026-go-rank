"""E92 scoring (PLAN experiments/E92_rank_target_net/PLAN.md; copy of scripts/experiments/e90/r0_score.py with the new-leg features replaced; same questions, F_E79 baseline,
nested design, bootstrap and reading rule).  A post-hoc multinomial logistic regression on per-question features predicts the rank of unseen fold-4 players; fused with F_E79
(rebuilt as in r0_common, asserted .6997).  No network is trained here.  Two arms from the SAME checkpoint's export R_r0.npz (e92_embed.py r0):
  MAIN (a)      : per-question mean of the bag's per-game 10-class softmax probabilities (10-d)            -> classification / reading rule
  SECONDARY (b) : per-question mean of the bag's per-game 128-d vectors, L2-normalised (128-d, as E90 0R)   -> information only
  dry-noise / dry-signal : full pipeline on FAKE features (never reads R_r0.npz) -> .scratch/e92/result_dryrun_{noise,signal}.json  (may be re-run)
  run                    : the formal computation with R_r0.npz -> .scratch/e92/result.json.  RUN ONCE: refuses if result.json exists; also needs E92_FORMAL=1 in the environment
                           (so that nobody runs it by accident - it is the commander's step).
  PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/experiments/e92/e92_score.py <mode>

Details (inherited unchanged from r0_score.py, fixed before any score was seen):
 - per-dimension standardisation (mean/std) fitted ONLY on the training questions of each fit (inner or outer), applied to the predicted questions; std < 1e-12 -> 1.
 - split: players of the fold-4 validation set, 5 parts = StratifiedKFold(5, shuffle=True, random_state=0) over players, stratified by player rank.  Inner C-selection: StratifiedKFold(3, ...).
 - classifier: sklearn LogisticRegression(lbfgs, multinomial, L2, class_weight="balanced", max_iter 2000, tol 1e-4); C chosen on the grid .1/1/10 (ties -> smaller) by the project metric (bal, EV decode)
   of the inner-3-fold OOF probabilities; z (grid 0,.1,..,.6; ties -> smaller) by bal of nrm((1-z)F_E79 + z P_inner); refit on all training questions, predict the held-out part, fuse with the chosen z.
 - bootstrap: scripts/experiments/e35/score.py boot/diff (player-level, paired, 4,000 draws, seed sd("bootstrap", TAG)).
 - reading rule (MAIN arm): positive = delta(F - F_E79) >= +.005 and 95% lower bound > 0; negative = point estimate <= 0; otherwise unclear.
"""
import json, sys, warnings, hashlib, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "scripts/experiments/e90"))     # read-only import of r0_common
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from r0_common import rebuild, bal, NS, nrm, boot, diff, ci, rank_scores, decide_ev
OUT = ROOT / ".scratch/e92"; OUT.mkdir(parents=True, exist_ok=True)
CS = (.1, 1., 10.); ZS = (0., .1, .2, .3, .4, .5, .6); TAG = "e92_r0"
_conv = {"n": 0}


def fit_predict(Xtr, ytr, Xte, C):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
    clf = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); clf.fit((Xtr - mu) / sd, ytr)
        _conv["n"] += sum(issubclass(x.category, ConvergenceWarning) for x in w)
    assert (clf.classes_ == np.arange(10)).all(), clf.classes_
    return clf.predict_proba((Xte - mu) / sd)


def pipeline(R, X):
    y, ns, fold, sc, F79, A1P, Qf = R["y"], R["ns"], R["fold"], R["sc"], R["F79"], R["A1P"], R["Qf"]
    idx4 = np.flatnonzero(fold == 4); yq = y[idx4]; nq = ns[idx4]; plq = Qf.pl[idx4]; f79q = F79[idx4]
    aux = Qf.aux; players = np.unique(plq); prank = aux["player_rank"][players].astype(int)
    assert (prank[np.searchsorted(players, plq)] == yq).all()
    PS = np.zeros((len(idx4), 10)); FF = np.zeros((len(idx4), 10)); chosen = []
    outer = StratifiedKFold(5, shuffle=True, random_state=0)
    part_of_player = np.zeros(len(players), int)
    for j, (trp, tep) in enumerate(outer.split(players, prank)):
        part_of_player[tep] = j
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
        chosen.append(dict(part=j, n_train_players=int(len(trpl)), n_test_players=int(len(np.unique(plq[te]))), n_train_q=int(len(tri)), n_test_q=int(len(tei)),
                           C=Cb, z=zb, inner_S_score_by_C={str(k): v for k, v in sC.items()}, inner_fused_score_by_z={str(k): v for k, v in sZ.items()}))
    assert (FF.sum(1) > 0).all() and (PS.sum(1) > 0).all()
    # full-length arrays (rows outside fold 4 are neutral and never in the mask)
    def full(p):
        o = np.full((len(y), 10), .1); o[idx4] = p; return o
    m4 = fold == 4
    Sc = {"F_E79": sc(F79), "S": sc(full(PS)), "F": sc(full(FF)), "A1P": sc(A1P)}
    r = boot(Sc, R["F"], m4, "balanced", TAG)
    res = dict(n_players=int(len(players)), n_questions=int(len(idx4)), scores={k: dict(score=r[k][0], per_N={str(n): v for n, v in r[k][1].items()}, **ci(r[k][2])) for k in r},
               delta_F_minus_F_E79=diff(r, "F", "F_E79"))
    d = res["delta_F_minus_F_E79"]
    res["classification"] = "positive" if (d["delta"] >= .005 and d["lo"] > 0) else ("negative" if d["delta"] <= 0 else "unclear")
    res["info"] = dict(S_minus_A1P=diff(r, "S", "A1P"), S_minus_F_E79=diff(r, "S", "F_E79"), chosen_per_part=chosen, n_convergence_warnings=_conv["n"],
                       z_chosen=[c["z"] for c in chosen], C_chosen=[c["C"] for c in chosen], mean_selected_z=float(np.mean([c["z"] for c in chosen])),
                       perN_delta={str(n): float(r["F"][1][n] - r["F_E79"][1][n]) for n in NS}, agree_S_vs_F_E79_decision=float((decide_ev(PS) == decide_ev(f79q)).mean()),
                       n_part_per_player_check=np.bincount(part_of_player).tolist())
    return res


def sc_q(p, yy):
    return rank_scores(decide_ev(p), yy)


def question_features(R, path):
    """per fold-4 question (Qf order restricted to fold 4): arm a = mean per-game softmax probabilities (10-d), arm b = L2-normalised mean per-game vector (128-d)."""
    Qf = R["Qf"]; idx4 = np.flatnonzero(R["fold"] == 4); rows = Qf.rows[idx4]
    Z = np.load(path); rid = Z["rid"].astype(np.int64); gv = Z["gv_all"].astype(np.float64); lg = Z["logits"].astype(np.float64)
    assert (np.diff(rid) > 0).all() and np.isfinite(gv).all() and np.isfinite(lg).all() and np.abs(np.linalg.norm(gv, axis=1) - 1).max() < 1e-3 and lg.shape == (len(rid), 10)
    need = np.unique(rows[rows >= 0]); assert len(need) == len(rid) and (need == rid).all(), "export games != the games used by the fold-4 questions"
    pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    A = np.zeros((len(idx4), 10)); B = np.zeros((len(idx4), gv.shape[1]))
    for i, r in enumerate(rows):
        r = r[r >= 0]; j = np.searchsorted(rid, r); A[i] = pr[j].mean(0); v = gv[j].mean(0); B[i] = v / np.linalg.norm(v)
    assert np.allclose(A.sum(1), 1)
    return dict(a=A, b=B)


def fake_features(R, kind):
    idx4 = np.flatnonzero(R["fold"] == 4); rng = np.random.default_rng(12345); n = len(idx4); y = R["y"][idx4]
    A = rng.dirichlet(np.ones(10), size=n); B = rng.normal(size=(n, 128))
    if kind == "signal":                                  # PLANTED rank signal (uses labels) - only to check the pipeline can detect a gain
        A[np.arange(n), y] += .12; A /= A.sum(1, keepdims=True); B[np.arange(n), y] += 1.5
    return dict(a=A, b=B / np.linalg.norm(B, axis=1, keepdims=True))


def run_arms(R, F):
    out = {}
    for arm, X in (("a", F["a"]), ("b", F["b"])):
        _conv["n"] = 0; out[arm] = pipeline(R, X)
    return out


def main():
    mode = sys.argv[1]; assert mode in ("dry-noise", "dry-signal", "run")
    if mode == "run":
        assert not (OUT / "result.json").exists(), "result.json exists: the formal computation runs ONCE - refusing"
        assert os.environ.get("E92_FORMAL") == "1", "the formal run is the commander's step: set E92_FORMAL=1 deliberately"
    R = rebuild(); print("repro", R["repro"], flush=True)
    if mode == "run":
        sp = OUT / "R_r0.npz"; F = question_features(R, sp); extra = dict(export_sha16=hashlib.sha256(sp.read_bytes()).hexdigest()[:16], script_sha16=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]); fn = OUT / "result.json"
    else:
        F = fake_features(R, mode.split("-")[1]); extra = dict(FAKE_FEATURES=mode); fn = OUT / f"result_dryrun_{mode.split('-')[1]}.json"
    arms = run_arms(R, F); res = dict(main=arms["a"], secondary=arms["b"], arm_main="mean per-game 10-class probabilities", arm_secondary="mean per-game 128-d vector (L2-normalised)", classification=arms["a"]["classification"],
                                       repro=R["repro"], mode=mode, **extra)
    with open(fn, "x" if mode == "run" else "w") as f: json.dump(res, f, indent=1, default=float)
    for nm, r in (("MAIN a", arms["a"]), ("SECONDARY b", arms["b"])):
        s = r["scores"]; d = r["delta_F_minus_F_E79"]
        print(mode, nm, {k: round(v["score"], 4) for k, v in s.items()}, f"delta {d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]", r["classification"], "C", r["info"]["C_chosen"], "z", r["info"]["z_chosen"], "warn", r["info"]["n_convergence_warnings"])


if __name__ == "__main__":
    main()
