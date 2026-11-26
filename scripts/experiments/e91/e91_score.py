"""E91 scoring (PLAN_draft.md): do opponent-identity votes from the S network raise Rank on the fold-4 validation questions?

  dry-noise / dry-signal / dry-weak : the whole pipeline on FAKE per-game vectors (never reads any real S vector) -> .scratch/e91/result_dryrun_<kind>.json (may be re-run)
        noise  : both sides random unit vectors                      -> every difference must be ~0
        signal : 25% of the games' OPPONENT vector = a library player's head vector (same rank as the questioned player w.p. .9)  -> gain must be measured,
                 and must sit in the high-confidence bin only
        weak   : same with 4% of the games (about the identity rate the E85 / scan estimates give)  -> shows what the bootstrap can resolve
  run : the formal computation with the real vectors (.scratch/e91/S_opp.npz = opponent side, .scratch/e90/stage0r/S_r0.npz = own side).  RUN ONCE: refuses if
        result.json exists, and refuses unless .scratch/e91/PLAN_FROZEN exists (written by the commander after the plan review).  NOT to be run by the author of this file.
  PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/experiments/e91/e91_score.py <mode>

Reused read-only from E90 stage 0R (scripts/experiments/e90/r0_common.py = rebuild of F_E79 asserting .6837/.6913/.6997; r0_score.py: question set, 5 player parts,
nested selection, bootstrap).  Everything that is decided here is written in PLAN_draft.md; nothing is tuned on a score.
"""
import json, sys, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "scripts/experiments/e90")]
import numpy as np
from sklearn.model_selection import StratifiedKFold
from r0_common import rebuild, bal, NS, nrm, boot, diff, ci, rank_scores, decide_ev
from e91_embed import head_stats, HEAD

OUT = ROOT / ".scratch/e91"; TAG = "e91_rank_opp"
S_OWN = ROOT / ".scratch/e90/stage0r/S_r0.npz"; S_OPP = OUT / "S_opp.npz"; GAMES = ROOT / ".scratch/e90/stage0r/r0_games.npz"
GAMMAS = (0., 1., 2.); TAU_Q = (None, .80, .95); ALPHAS = (.02, .1, .5); LAMS = (0., .5, 1., 2.)      # main grid: tau = 0 or the .80 / .95 quantile of own-side p_max
BIN_Q = (0., .50, .95, 1.01)                                                                          # control bins of p_max: low / mid / high (quantiles of own-side p_max)
GATE_PI = .01; PLANT = dict(signal=.25, weak=.04, noise=0.)


def sc_q(p, yy): return rank_scores(decide_ev(p), yy)


# ---------------------------------------------------------------------------------------------------------------- per-game head readout
def game_readout(gv, W, cls_rank):
    st = head_stats(gv, W); pi = np.bincount(cls_rank, minlength=10) / len(cls_rank)
    hist = st["rank_hist"].astype(np.float64); lift = hist / pi; lift /= lift.sum(1, keepdims=True)           # divide out the library's rank prior, renormalise
    return dict(pmax=st["pmax"].astype(np.float64), lift=lift, top1_rank=cls_rank[st["top20_cls"][:, 0].astype(np.int64)].astype(np.int64))


def question_votes(rows, rid, hd, ro, gamma, lo, hi):
    """Per question: V = sum over its hd==0 games with lo <= p_max < hi of p_max**gamma * lift, normalised; uniform when no game qualifies."""
    valid = rows >= 0; idx = np.searchsorted(rid, np.where(valid, rows, rid[0])); assert (rid[idx[valid]] == rows[valid]).all()
    ok = valid & (hd[idx] == 0) & (ro["pmax"][idx] >= lo) & (ro["pmax"][idx] < hi)
    w = np.where(ok, ro["pmax"][idx] ** gamma, 0.); V = np.einsum("qg,qgk->qk", w, ro["lift"][idx]); tot = V.sum(1); has = tot > 0
    Vn = np.full(V.shape, .1); Vn[has] = V[has] / tot[has, None]
    return Vn, has


def fuse(f79, Vn, lam, alpha):
    z = np.log(f79 + 1e-12) + lam * np.log(Vn + alpha); z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def grid_main(thr):   # (gamma, lo, hi, alpha, lam); lam = 0 is the single "no vote" entry
    g = [(0., 0., 9., 1., 0.)]
    for lam in LAMS[1:]:
        for gm in GAMMAS:
            for tq in TAU_Q:
                for al in ALPHAS: g.append((gm, 0. if tq is None else thr[tq], 9., al, lam))
    return g


def grid_bin(lo, hi):
    g = [(0., lo, hi, 1., 0.)]
    for lam in LAMS[1:]:
        for gm in GAMMAS:
            for al in ALPHAS: g.append((gm, lo, hi, al, lam))
    return g


def nested(ctx, ro, combos, perm=None):
    """5-part nested selection over players (same parts as stage 0R): the combo is chosen on the other 4 parts' questions (bal, EV decode), applied to the held-out part.
    ties (within 1e-12) -> smaller lam, then earlier in the list.  perm: optional row permutation of the vote matrix (shuffle controls)."""
    f79q, yq, nq, rows, rid, hd, part = ctx["f79q"], ctx["yq"], ctx["nq"], ctx["rows"], ctx["rid"], ctx["hd"], ctx["part"]
    order = sorted(range(len(combos)), key=lambda i: (combos[i][4], i)); cache = {}; scq = []
    for gm, lo, hi, al, lam in combos:
        k = (gm, lo, hi)
        if k not in cache: Vn, has = question_votes(rows, rid, hd, ro, gm, lo, hi); cache[k] = (Vn[perm] if perm is not None else Vn, has)
        scq.append(sc_q(fuse(f79q, cache[k][0], lam, al), yq))
    P = np.zeros_like(f79q); chosen = []
    for j in range(5):
        tr = part != j; te = part == j; best = None
        for i in order:
            s = bal(scq[i], yq, nq, tr)
            if best is None or s > best[0] + 1e-12: best = (s, i)
        gm, lo, hi, al, lam = combos[best[1]]; Vn, has = cache[(gm, lo, hi)]; P[te] = fuse(f79q[te], Vn[te], lam, al)
        chosen.append(dict(part=j, gamma=gm, lo=lo, hi=hi, alpha=al, lam=lam, train_score=best[0], frac_questions_with_vote_in_part=float(has[te].mean())))
    return P, chosen


# ---------------------------------------------------------------------------------------------------------------- gate 0 and diagnostics (no F_E79, no fusion)
def gate0(ctx, ro_opp, ro_own, thr95):
    """pi_hat = P_opp(p_max >= Q95_own) - P_own(p_max >= Q95_own), games with hd == 0, player-level bootstrap.  Premise: an opponent outside the library behaves like an
    unseen player (= the questioned players, who are all outside the library), so the EXCESS of high-confidence opponent games is the share recognised as a library player."""
    hd = ctx["hd_game"]; m = hd == 0; pl = ctx["game_player"][m]; up, inv = np.unique(pl, return_inverse=True)
    a = np.bincount(inv, weights=(ro_opp["pmax"][m] >= thr95), minlength=len(up)); b = np.bincount(inv, weights=(ro_own["pmax"][m] >= thr95), minlength=len(up)); n = np.bincount(inv).astype(float)
    pi = a.sum() / n.sum() - b.sum() / n.sum()
    C = np.random.default_rng(91).multinomial(len(up), np.full(len(up), 1 / len(up)), size=2000).astype(np.float64); bs = (C @ a - C @ b) / (C @ n)
    return dict(n_games_hd0=int(m.sum()), share_opp_high=float(a.sum() / n.sum()), share_own_high=float(b.sum() / n.sum()), pi_hat=float(pi), **ci(bs), gate_threshold=GATE_PI,
                passes=bool(pi >= GATE_PI and np.quantile(bs, .025) > 0))


def diag_rank_match(ctx, ro_opp, ro_own, thr):
    """Per own-side p_max quantile bin: share of games and rate of 'top-1 library player has the same rank as the questioned player' vs the rate expected by chance (labels used,
    NO fusion).  Excess (opp - own) in the high bin = identity; in the low bins = style reading shared with the own side."""
    m = ctx["hd_game"] == 0; tr = ctx["true_rank_g"][m]; edges = [0.] + [thr[q] for q in (.50, .80, .95)] + [9.]; out = []
    for side, ro in (("opp", ro_opp), ("own", ro_own)):
        pm, t1 = ro["pmax"][m], ro["top1_rank"][m]
        for a, b in zip(edges[:-1], edges[1:]):
            s = (pm >= a) & (pm < b)
            if s.sum() == 0: continue
            pt = np.bincount(tr[s], minlength=10) / s.sum(); pq = np.bincount(t1[s], minlength=10) / s.sum()
            out.append(dict(side=side, lo=float(a), hi=float(b), share=float(s.mean()), n=int(s.sum()), same_rank_rate=float((t1[s] == tr[s]).mean()), chance=float((pt * pq).sum())))
    return out


# ---------------------------------------------------------------------------------------------------------------- fake vectors for the dry runs
def fake_vectors(kind, W, cls_rank, true_rank_g, rng):
    G = len(true_rank_g); unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    own = unit(rng.normal(size=(G, 128))).astype(np.float32); opp = unit(rng.normal(size=(G, 128))).astype(np.float32)
    phi = PLANT[kind]
    if phi > 0:                                                   # PLANTED identity signal (uses the questioned player's true rank) - dry run only
        Wn = unit(W); by_rank = [np.flatnonzero(cls_rank == k) for k in range(10)]
        plant = np.flatnonzero(rng.random(G) < phi)
        for g in plant:
            k = int(true_rank_g[g]) if rng.random() < .9 else int(rng.integers(10))
            j = rng.choice(by_rank[k]); v = Wn[j] + .5 * rng.normal(size=128) / np.sqrt(128); opp[g] = (v / np.linalg.norm(v)).astype(np.float32)
    return own, opp


def build_ctx(R, game):
    Qf = R["Qf"]; idx4 = np.flatnonzero(R["fold"] == 4); aux = Qf.aux
    yq, nq, plq = R["y"][idx4], R["ns"][idx4], Qf.pl[idx4]; players = np.unique(plq); prank = aux["player_rank"][players].astype(int)
    part_of_player = np.zeros(len(players), int)
    for j, (_, tep) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(players, prank)): part_of_player[tep] = j
    part = part_of_player[np.searchsorted(players, plq)]
    return dict(idx4=idx4, f79q=R["F79"][idx4], yq=yq, nq=nq, plq=plq, rows=Qf.rows[idx4], rid=game["rid"].astype(np.int64), hd=game["hd"].astype(np.int64), hd_game=game["hd"],
                part=part, game_player=game["player"], true_rank_g=aux["player_rank"][game["player"]].astype(np.int64))


def run(mode):
    assert mode in ("dry-noise", "dry-signal", "dry-weak", "run")
    if mode == "run":
        assert (OUT / "PLAN_FROZEN").exists(), "PLAN_FROZEN not found: the plan has not been frozen by the commander - refusing"
        assert not (OUT / "result.json").exists(), "result.json exists: the formal computation runs ONCE - refusing"
    R = rebuild(); print("repro", R["repro"], flush=True)
    g = np.load(GAMES); game = {k: g[k] for k in g.files}; h = np.load(HEAD); W, cls_rank = h["W"], h["cls_rank"].astype(np.int64)
    ctx = build_ctx(R, game); rng = np.random.default_rng(910)
    if mode == "run":
        so, sp = np.load(S_OWN), np.load(S_OPP)
        for s in (so, sp):
            assert (s["rid"].astype(np.int64) == ctx["rid"]).all() and np.isfinite(s["gv_all"]).all() and np.abs(np.linalg.norm(s["gv_all"], axis=1) - 1).max() < 1e-3
        own, opp = so["gv_all"], sp["gv_all"]; extra = dict(S_opp_sha16=hashlib.sha256(S_OPP.read_bytes()).hexdigest()[:16], S_own_sha16=hashlib.sha256(S_OWN.read_bytes()).hexdigest()[:16],
                                                           script_sha16=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]); fn = OUT / "result.json"
    else:
        kind = mode.split("-")[1]; own, opp = fake_vectors(kind, W, cls_rank, ctx["true_rank_g"], rng); extra = dict(FAKE_VECTORS=mode); fn = OUT / f"result_dryrun_{kind}.json"
    ro_own, ro_opp = game_readout(own, W, cls_rank), game_readout(opp, W, cls_rank); m0 = game["hd"] == 0
    thr = {q: float(np.quantile(ro_own["pmax"][m0], q)) for q in (.50, .80, .95)}                # thresholds from OWN-side p_max (unseen players), label free
    res = dict(mode=mode, repro=R["repro"], thresholds_pmax_own_quantiles=thr, **extra)
    res["gate0"] = gate0(ctx, ro_opp, ro_own, thr[.95]); res["diag_rank_match"] = diag_rank_match(ctx, ro_opp, ro_own, thr)
    print("gate0", json.dumps(res["gate0"]), flush=True)
    if mode == "run" and (OUT / "S_libval.npz").exists():       # optional info (not in any decision): recognition rate of the library's own held-out games at the same threshold -> f_hat = pi_hat / r_in
        lv = np.load(OUT / "S_libval.npz"); st = head_stats(lv["gv_all"], W); hdv = np.load(ROOT / ".scratch/e89/data/s_games.npz")["hd"][lv["ids"]]; mv = hdv == 0
        r_in = float((st["pmax"][mv] >= thr[.95]).mean()); res["libval_info"] = dict(n_games=int(mv.sum()), r_in_at_Q95=r_in, f_hat_in_library=float(res["gate0"]["pi_hat"] / r_in) if r_in > 0 else None)
        print("libval", res["libval_info"], flush=True)
    idx4, yq, nq = ctx["idx4"], ctx["yq"], ctx["nq"]; nqn = len(yq)
    arms = {"F_opp": (ro_opp, grid_main(thr), None), "F_own": (ro_own, grid_main(thr), None),
            "F_opp_lo": (ro_opp, grid_bin(0., thr[.50]), None), "F_opp_mid": (ro_opp, grid_bin(thr[.50], thr[.95]), None), "F_opp_hi": (ro_opp, grid_bin(thr[.95], 9.), None),
            "F_own_hi": (ro_own, grid_bin(thr[.95], 9.), None)}
    perm = np.arange(nqn); perm_strat = np.arange(nqn)
    for n in NS:                                                                                   # shuffle controls: question level, within N (full) / within (N, true rank) (strat)
        ii = np.flatnonzero(nq == n); perm[ii] = rng.permutation(ii)
        for k in range(10):
            jj = np.flatnonzero((nq == n) & (yq == k)); perm_strat[jj] = rng.permutation(jj)
    arms["F_shuf"] = (ro_opp, grid_main(thr), perm); arms["F_shuf_samerank"] = (ro_opp, grid_main(thr), perm_strat)
    def full(p):
        o = np.full((len(R["y"]), 10), .1); o[idx4] = p; return o
    Sc = {"F_E79": R["sc"](R["F79"])}; chosen = {}
    for nm, (ro, combos, pm) in arms.items():
        P, ch = nested(ctx, ro, combos, pm); Sc[nm] = R["sc"](full(P)); chosen[nm] = ch
        print(nm, "bal fold4", round(bal(Sc[nm], R["y"], R["ns"], R["fold"] == 4), 4), "lam", [c["lam"] for c in ch], flush=True)
    r = boot(Sc, R["F"], R["fold"] == 4, "balanced", TAG)
    res["scores"] = {k: dict(score=r[k][0], per_N={str(n): v for n, v in r[k][1].items()}, **ci(r[k][2])) for k in r}
    res["delta_vs_F_E79"] = {k: diff(r, k, "F_E79") for k in r if k != "F_E79"}
    res["delta_pairs"] = {"opp_minus_own": diff(r, "F_opp", "F_own"), "opp_hi_minus_own_hi": diff(r, "F_opp_hi", "F_own_hi"), "opp_minus_shuf": diff(r, "F_opp", "F_shuf"),
                          "opp_minus_shuf_samerank": diff(r, "F_opp", "F_shuf_samerank")}
    res["chosen_per_part"] = chosen
    pm = ctx["part"]; m4 = R["fold"] == 4; part_full = np.full(len(R["y"]), -1); part_full[idx4] = pm
    res["per_part_delta_F_opp"] = [float(bal(Sc["F_opp"], R["y"], R["ns"], part_full == j) - bal(Sc["F_E79"], R["y"], R["ns"], part_full == j)) for j in range(5)]
    res["per_N_delta_F_opp"] = {str(n): float(r["F_opp"][1][n] - r["F_E79"][1][n]) for n in NS}
    d = res["delta_vs_F_E79"]["F_opp"]; dh = res["delta_vs_F_E79"]["F_opp_hi"]; dsh = res["delta_vs_F_E79"]["F_shuf"]; dpo = res["delta_pairs"]["opp_minus_own"]
    dps = res["delta_pairs"]["opp_minus_shuf"]; ident = dpo["lo"] > 0 and dh["delta"] >= .5 * d["delta"] and dps["lo"] > 0
    res["classification"] = ("negative" if d["delta"] <= 0 else ("positive_identity" if (d["delta"] >= .005 and d["lo"] > 0 and ident) else ("positive_unattributed" if (d["delta"] >= .005 and d["lo"] > 0) else "unclear")))
    res["classification_inputs"] = dict(delta=d["delta"], lo=d["lo"], opp_minus_own_lo=dpo["lo"], hi_delta=dh["delta"], half_main_delta=.5 * d["delta"], shuf_delta=dsh["delta"], opp_minus_shuf_lo=dps["lo"], identity_conditions_met=bool(ident),
                                        gate0_pass=res["gate0"]["passes"])
    with open(fn, "x" if mode == "run" else "w") as f: json.dump(res, f, indent=1, default=float)
    print(mode, {k: round(v["score"], 4) for k, v in res["scores"].items()}); print("delta vs F_E79", {k: round(v["delta"], 4) for k, v in res["delta_vs_F_E79"].items()}); print("pairs", {k: round(v["delta"], 4) for k, v in res["delta_pairs"].items()})
    print("classification", res["classification"], res["classification_inputs"])


if __name__ == "__main__":
    run(sys.argv[1])
