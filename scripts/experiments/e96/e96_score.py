"""E96 scoring: COPY of scripts/experiments/e91/e91_score.py (untouched) for the VALIDATION identity network `finh_v4` (finh_full recipe, roster without any fold-4 player).
Algorithm unchanged (nested 5-part selection over players, player-level paired bootstrap, arms F_opp / F_own / F_opp_lo|mid|hi / F_own_hi / F_shuf / F_shuf_samerank, gate 0, diag_rank_match, rng streams).
Changes, all of them:
  1. paths -> .scratch/e96 (S_own_v4.npz, S_opp_v4.npz, head_v4.npz, result_v4.json, PLAN_FROZEN); the head readout is generalised from 5,700 to C classes (`head_stats` below = e91_embed.head_stats with cls_rank passed in).
  2. NEW pre-registered extension arm F_opp_x: votes also from hd == 1 games (opponent side), the opponent's rank vote is shifted by one rank on the own-rank axis: own colour black (eff 0) -> opponent is one rank
     higher -> likelihood over own rank k = lift[min(k+1, 9)]; own colour white (eff 1) -> lift[max(k-1, 0)]; rows renormalised.  Same weights / thresholds (own hd==0 quantiles) / grid / nested selection.  Extra columns only;
     it never changes any other arm and draws no random numbers.
  3. NEW `e96_decision` block = the rule written in experiments/E96_rank_opp_anchor/PLAN.md; `final_params_candidate` (in-sample best combo on all 4,568 questions, E91 `fit` rule) for F_opp and F_opp_x.
  4. dry runs use the legacy head (.scratch/e91/head.npz, 5,700 classes) by default so that the numbers can be compared with .scratch/e91/result_dryrun_*.json; E96_DRY_HEAD=v4 uses head_v4.npz if present.
     extra dry mode `dry-signal-x` (plants hd == 1 opponent games with the shifted rank) to test F_opp_x; `rehearse` = the formal code path on FAKE files in .scratch/e96/rehearsal (never writes result_v4.json).
  5. `run` additionally asserts: head roster finh_v4 / no fold-4 class, S files' checkpoint sha == head's sha, rid alignment.
  dry-noise / dry-signal / dry-weak / dry-signal-x : FAKE vectors -> .scratch/e96/result_dryrun_<kind>.json (may be re-run)
  rehearse : formal path on fake S files in .scratch/e96/rehearsal -> result_rehearsal.json (may be re-run)
  run : formal computation with the real vectors; RUN ONCE: refuses if result_v4.json exists and unless .scratch/e96/PLAN_FROZEN exists.  NOT to be run by the author of this file.
  PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/experiments/e96/e96_score.py <mode>
Original E91 docstring: E91 scoring (PLAN_draft.md): do opponent-identity votes from the S network raise Rank on the fold-4 validation questions?  Reused read-only from E90 stage 0R (r0_common.py, r0_score.py).
"""
import json, sys, os, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "scripts/experiments/e90")]
import numpy as np
from sklearn.model_selection import StratifiedKFold
from r0_common import rebuild, bal, NS, nrm, boot, diff, ci, rank_scores, decide_ev

OUT = ROOT / ".scratch/e96"; TAG = "e96_rank_opp"
S_OWN = OUT / "S_own_v4.npz"; S_OPP = OUT / "S_opp_v4.npz"; HEAD = OUT / "head_v4.npz"; GAMES = ROOT / ".scratch/e90/stage0r/r0_games.npz"
REH = OUT / "rehearsal"; LEGACY_HEAD = ROOT / ".scratch/e91/head.npz"; FOLD4 = OUT / "fold4_pids.json"
GAMMAS = (0., 1., 2.); TAU_Q = (None, .80, .95); ALPHAS = (.02, .1, .5); LAMS = (0., .5, 1., 2.)      # main grid: tau = 0 or the .80 / .95 quantile of own-side p_max
BIN_Q = (0., .50, .95, 1.01)                                                                          # control bins of p_max: low / mid / high (quantiles of own-side p_max)
GATE_PI = .01; PLANT = dict(signal=.25, weak=.04, noise=0., **{"signal-x": .25})
SCALE = 30.0


def sc_q(p, yy): return rank_scores(decide_ev(p), yy)


# ---------------------------------------------------------------------------------------------------------------- per-game head readout
def head_stats(gv, W, cls_rank, chunk=2000):
    """= e91_embed.head_stats with the class->rank table passed in (C classes instead of the hard-coded 5,700).  gv [n,128] unit vectors -> top20_cls, top20_p, pmax, rank_hist [n,10]; softmax(30 cos), T = 1."""
    C = len(cls_rank); cls_rank = cls_rank.astype(np.int64)
    Wn = (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32); n = len(gv)
    t_c = np.zeros((n, 20), np.int16); t_p = np.zeros((n, 20), np.float32); hist = np.zeros((n, 10), np.float32)
    onehot = np.zeros((C, 10), np.float32); onehot[np.arange(C), cls_rank] = 1
    for a in range(0, n, chunk):
        lg = SCALE * (gv[a:a + chunk].astype(np.float32) @ Wn.T); lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
        top = np.argpartition(-p, 19, axis=1)[:, :20]; tp = np.take_along_axis(p, top, 1); o = np.argsort(-tp, 1)
        t_c[a:a + chunk] = np.take_along_axis(top, o, 1); t_p[a:a + chunk] = np.take_along_axis(tp, o, 1); hist[a:a + chunk] = p @ onehot
    return dict(top20_cls=t_c, top20_p=t_p, pmax=t_p[:, 0].copy(), rank_hist=hist)


def game_readout(gv, W, cls_rank):
    st = head_stats(gv, W, cls_rank); pi = np.bincount(cls_rank, minlength=10) / len(cls_rank)
    hist = st["rank_hist"].astype(np.float64); lift = hist / pi; lift /= lift.sum(1, keepdims=True)           # divide out the library's rank prior, renormalise
    return dict(pmax=st["pmax"].astype(np.float64), lift=lift, top1_rank=cls_rank[st["top20_cls"][:, 0].astype(np.int64)].astype(np.int64))


def question_votes(rows, rid, hd, ro, gamma, lo, hi, hd_ok=(0,)):
    """Per question: V = sum over its hd in hd_ok (E91: hd==0 only) games with lo <= p_max < hi of p_max**gamma * lift, normalised; uniform when no game qualifies."""
    valid = rows >= 0; idx = np.searchsorted(rid, np.where(valid, rows, rid[0])); assert (rid[idx[valid]] == rows[valid]).all()
    ok = valid & np.isin(hd[idx], hd_ok) & (ro["pmax"][idx] >= lo) & (ro["pmax"][idx] < hi)
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


def nested(ctx, ro, combos, perm=None, hd_ok=(0,)):
    """5-part nested selection over players (same parts as stage 0R): the combo is chosen on the other 4 parts' questions (bal, EV decode), applied to the held-out part.
    ties (within 1e-12) -> smaller lam, then earlier in the list.  perm: optional row permutation of the vote matrix (shuffle controls)."""
    f79q, yq, nq, rows, rid, hd, part = ctx["f79q"], ctx["yq"], ctx["nq"], ctx["rows"], ctx["rid"], ctx["hd"], ctx["part"]
    order = sorted(range(len(combos)), key=lambda i: (combos[i][4], i)); cache = {}; scq = []
    for gm, lo, hi, al, lam in combos:
        k = (gm, lo, hi)
        if k not in cache: Vn, has = question_votes(rows, rid, hd, ro, gm, lo, hi, hd_ok); cache[k] = (Vn[perm] if perm is not None else Vn, has)
        scq.append(sc_q(fuse(f79q, cache[k][0], lam, al), yq))
    P = np.zeros_like(f79q); chosen = []
    for j in range(5):
        tr = part != j; te = part == j; best = None
        for i in order:
            s = bal(scq[i], yq, nq, tr)
            if best is None or s > best[0] + 1e-12: best = (s, i)
        gm, lo, hi, al, lam = combos[best[1]]; Vn, has = cache[(gm, lo, hi)]; P[te] = fuse(f79q[te], Vn[te], lam, al)
        chosen.append(dict(part=j, gamma=gm, lo=lo, hi=hi, alpha=al, lam=lam, train_score=best[0], frac_questions_with_vote_in_part=float(has[te].mean())))
    return P, chosen, dict(order=order, scq=scq, cache=cache)


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
def fake_vectors(kind, W, cls_rank, true_rank_g, rng, eff_g=None, hd_g=None):
    G = len(true_rank_g); unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    own = unit(rng.normal(size=(G, 128))).astype(np.float32); opp = unit(rng.normal(size=(G, 128))).astype(np.float32)
    phi = PLANT[kind]
    if phi > 0:                                                   # PLANTED identity signal (uses the questioned player's true rank) - dry run only
        Wn = unit(W); by_rank = [np.flatnonzero(cls_rank == k) for k in range(10)]
        plant = np.flatnonzero(rng.random(G) < phi)
        for g in plant:
            k = int(true_rank_g[g]) if rng.random() < .9 else int(rng.integers(10))
            if kind == "signal-x" and hd_g[g] == 1: k = int(np.clip(k + (1 if eff_g[g] == 0 else -1), 0, 9))     # E96: a hd==1 opponent sits one rank above (own black) / below (own white); same rng draws as 'signal'
            j = rng.choice(by_rank[k]); v = Wn[j] + .5 * rng.normal(size=128) / np.sqrt(128); opp[g] = (v / np.linalg.norm(v)).astype(np.float32)
    return own, opp


def shift_lift(lift, eff, hd):
    """E96 F_opp_x: opponent rank vote -> likelihood over the QUESTIONED player's rank for hd == 1 games.  eff 0 (own black) -> opponent half a rank higher on average: L(k) = .5*lift[k] + .5*lift[min(k+1, 9)];
    eff 1 (own white) -> L(k) = .5*lift[k] + .5*lift[max(k-1, 0)]; rows renormalised.  hd == 0 rows are untouched; other hd are never used (weight 0)."""
    x = lift.copy(); k = np.arange(10); blk = (hd == 1) & (eff == 0); wht = (hd == 1) & (eff == 1)
    x[blk] = .5 * lift[blk] + .5 * lift[blk][:, np.minimum(k + 1, 9)]; x[wht] = .5 * lift[wht] + .5 * lift[wht][:, np.maximum(k - 1, 0)]   # 11/18 11:10: half no-shift + half one-rank shift (measured mean offset ~.3-.48 rank)
    m = hd == 1; x[m] /= x[m].sum(1, keepdims=True); return x


def build_ctx(R, game):
    Qf = R["Qf"]; idx4 = np.flatnonzero(R["fold"] == 4); aux = Qf.aux
    yq, nq, plq = R["y"][idx4], R["ns"][idx4], Qf.pl[idx4]; players = np.unique(plq); prank = aux["player_rank"][players].astype(int)
    part_of_player = np.zeros(len(players), int)
    for j, (_, tep) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(players, prank)): part_of_player[tep] = j
    part = part_of_player[np.searchsorted(players, plq)]
    return dict(idx4=idx4, f79q=R["F79"][idx4], yq=yq, nq=nq, plq=plq, rows=Qf.rows[idx4], rid=game["rid"].astype(np.int64), hd=game["hd"].astype(np.int64), hd_game=game["hd"],
                part=part, game_player=game["player"], true_rank_g=aux["player_rank"][game["player"]].astype(np.int64))


def load_real(d, ctx, expect_classes):
    """Real (or rehearsal) inputs from directory d: head, own and opponent vectors, with the identity checks (E96 additions)."""
    h = np.load(d / "head_v4.npz", allow_pickle=True); W, cls_rank = h["W"], h["cls_rank"].astype(np.int64)
    assert str(h["roster"]) == "finh_v4" and int(h["n_classes"]) == W.shape[0] == len(cls_rank) == len(h["cls_pid"]), "head is not a finh_v4 class table"
    if expect_classes: assert int(h["n_classes"]) == expect_classes, (int(h["n_classes"]), expect_classes)
    f4 = json.loads(FOLD4.read_text()); assert not (set(h["cls_pid"].astype(str).tolist()) & set(f4["pid"])), "a fold-4 player is a class of the head - STOP"
    assert (np.bincount(cls_rank, minlength=10) > 0).all(); out = {}
    for side in ("own", "opp"):
        f = d / f"S_{side}_v4.npz"; sv = np.load(f); sj = json.loads(Path(str(f) + ".json").read_text())
        assert sj["ckpt_sha16"] == str(h["ckpt_sha16"]), f"S_{side}_v4 was made with another checkpoint than the head"
        assert (sv["rid"].astype(np.int64) == ctx["rid"]).all() and sv["gv_all"].shape == (len(ctx["rid"]), 128) and np.isfinite(sv["gv_all"]).all() and np.abs(np.linalg.norm(sv["gv_all"], axis=1) - 1).max() < 1e-3
        out[side] = sv["gv_all"]; out[side + "_sha16"] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    out["head_sha16"] = hashlib.sha256((d / "head_v4.npz").read_bytes()).hexdigest()[:16]; out["W"], out["cls_rank"] = W, cls_rank; return out


def make_rehearsal():
    """FAKE files in the format of the real ones (6,039 random classes, random-rank labels, weak planted signal) -> .scratch/e96/rehearsal.  Only tests the formal code path."""
    REH.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(5); C = 6039
    W = rng.normal(size=(C, 128)).astype(np.float32); cls_rank = (np.arange(C) % 10).astype(np.int8); g = np.load(GAMES); game = {k: g[k] for k in g.files}
    tr = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=True)["player_rank"][game["player"]].astype(np.int64)
    own, opp = fake_vectors("weak", W, cls_rank.astype(np.int64), tr, rng)
    np.savez(REH / "head_v4.npz", W=W, cls_rank=cls_rank, cls_pid=np.array([f"fake{i}" for i in range(C)]), cls_code=np.arange(C), ckpt_sha16="rehearsal", roster="finh_v4", n_classes=C, n_h_classes=0)
    for side, v in (("own", own), ("opp", opp)):
        np.savez(REH / f"S_{side}_v4.npz", gv_all=v, rid=game["rid"]); (REH / f"S_{side}_v4.npz.json").write_text(json.dumps(dict(ckpt_sha16="rehearsal", FAKE=True)))
    print("rehearsal files written to", REH)


def select_all(ctx, combos, info):
    """E91 `fit` rule on ALL 4,568 questions (in-sample): best balanced score, ties -> smaller lam then grid order.  Uses the scores nested() already computed."""
    yq, nq = ctx["yq"], ctx["nq"]; sc = [bal(x, yq, nq) for x in info["scq"]]; best = None
    for i in info["order"]:
        if best is None or sc[i] > best[0] + 1e-12: best = (sc[i], i)
    i = best[1]; gm, lo, hi, al, lam = combos[i]; top = sorted(range(len(combos)), key=lambda j: -sc[j])[:8]
    return dict(rule="in-sample best of the 82-entry grid on all fold-4 questions (not nested; the nested delta stays the performance estimate)", gamma=gm, tau_value=lo, alpha=al, lam=lam, score_all_questions=best[0],
                score_no_vote_entry=sc[0], is_no_vote_entry=bool(lam == 0), frac_questions_with_vote=float(info["cache"][(gm, lo, hi)][1].mean()),
                top8=[dict(gamma=combos[j][0], tau_value=combos[j][1], alpha=combos[j][3], lam=combos[j][4], score=sc[j]) for j in top])


def decide_e96(res):
    """Rule frozen in experiments/E96_rank_opp_anchor/PLAN.md section 4 (literal)."""
    d = res["delta_vs_F_E79"]["F_opp"]; parts = res["per_part_delta_F_opp"]; sh = res["delta_vs_F_E79"]["F_shuf"]
    leak = sh["lo"] > 0; pos_parts = all(x > 0 for x in parts)
    if leak: v = "STOP_pipeline_leak_suspected"
    elif d["delta"] >= .02 and d["lo"] > 0 and pos_parts: v = "established_make_v10d"
    elif d["lo"] > 0: v = "small_gain_make_v10d_with_note"          # 11/18 11:10 (before any E96 result existed): also covers delta >= .02 with lo > 0 but some part <= 0 (PLAN section 4, rule 3)
    else: v = "no_file_keep_E91_setting"
    return dict(verdict=v, delta=d["delta"], lo=d["lo"], hi=d["hi"], per_part_delta=parts, all_five_parts_positive=bool(pos_parts), F_shuf_delta=sh["delta"], F_shuf_lo=sh["lo"], leak_flag=bool(leak),
                note_parts="delta >= .02 and lo > 0 but some part <= 0 -> small gain (rule 3); parts: %s" % parts if (d["delta"] >= .02 and d["lo"] > 0 and not pos_parts and not leak) else None)


def run(mode):
    assert mode in ("dry-noise", "dry-signal", "dry-weak", "dry-signal-x", "rehearse", "run", "make-rehearsal")
    if mode == "make-rehearsal": return make_rehearsal()
    if mode == "run":
        assert (OUT / "PLAN_FROZEN").exists(), "PLAN_FROZEN not found: the plan has not been frozen by the commander - refusing"
        assert not (OUT / "result_v4.json").exists(), "result_v4.json exists: the formal computation runs ONCE - refusing"
    R = rebuild(); print("repro", R["repro"], flush=True)
    g = np.load(GAMES); game = {k: g[k] for k in g.files}
    ctx = build_ctx(R, game); rng = np.random.default_rng(910)
    if mode in ("run", "rehearse"):
        d = OUT if mode == "run" else REH; real = load_real(d, ctx, 6039 if mode == "run" else None); W, cls_rank = real["W"], real["cls_rank"]; own, opp = real["own"], real["opp"]
        extra = dict(S_opp_sha16=real["opp_sha16"], S_own_sha16=real["own_sha16"], head_sha16=real["head_sha16"], n_classes=int(len(cls_rank)), script_sha16=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16])
        fn = OUT / "result_v4.json" if mode == "run" else OUT / "result_rehearsal.json"
        if mode == "rehearse": extra["FAKE_FILES_REHEARSAL"] = True
    else:
        kind = mode[4:]; hp = OUT / "head_v4.npz" if (os.environ.get("E96_DRY_HEAD") == "v4" and (OUT / "head_v4.npz").exists()) else LEGACY_HEAD
        h = np.load(hp, allow_pickle=True); W, cls_rank = h["W"], h["cls_rank"].astype(np.int64)
        own, opp = fake_vectors(kind, W, cls_rank, ctx["true_rank_g"], rng, game["eff"], game["hd"]); extra = dict(FAKE_VECTORS=mode, head_used=str(hp), n_classes=int(len(cls_rank))); fn = OUT / f"result_dryrun_{kind}.json"
    ro_own, ro_opp = game_readout(own, W, cls_rank), game_readout(opp, W, cls_rank); m0 = game["hd"] == 0
    thr = {q: float(np.quantile(ro_own["pmax"][m0], q)) for q in (.50, .80, .95)}                # thresholds from OWN-side p_max (unseen players), label free
    res = dict(mode=mode, repro=R["repro"], thresholds_pmax_own_quantiles=thr, **extra)
    res["gate0"] = gate0(ctx, ro_opp, ro_own, thr[.95]); res["diag_rank_match"] = diag_rank_match(ctx, ro_opp, ro_own, thr)
    print("gate0", json.dumps(res["gate0"]), flush=True)
    ro_x = dict(pmax=ro_opp["pmax"], lift=shift_lift(ro_opp["lift"], game["eff"], game["hd"]), top1_rank=ro_opp["top1_rank"])
    idx4, yq, nq = ctx["idx4"], ctx["yq"], ctx["nq"]; nqn = len(yq)
    arms = {"F_opp": (ro_opp, grid_main(thr), None, (0,)), "F_own": (ro_own, grid_main(thr), None, (0,)),
            "F_opp_lo": (ro_opp, grid_bin(0., thr[.50]), None, (0,)), "F_opp_mid": (ro_opp, grid_bin(thr[.50], thr[.95]), None, (0,)), "F_opp_hi": (ro_opp, grid_bin(thr[.95], 9.), None, (0,)),
            "F_own_hi": (ro_own, grid_bin(thr[.95], 9.), None, (0,))}
    perm = np.arange(nqn); perm_strat = np.arange(nqn)
    for n in NS:                                                                                   # shuffle controls: question level, within N (full) / within (N, true rank) (strat)
        ii = np.flatnonzero(nq == n); perm[ii] = rng.permutation(ii)
        for k in range(10):
            jj = np.flatnonzero((nq == n) & (yq == k)); perm_strat[jj] = rng.permutation(jj)
    arms["F_shuf"] = (ro_opp, grid_main(thr), perm, (0,)); arms["F_shuf_samerank"] = (ro_opp, grid_main(thr), perm_strat, (0,))
    arms["F_opp_x"] = (ro_x, grid_main(thr), None, (0, 1))                                          # E96 pre-registered extension (extra column; no random numbers)
    def full(p):
        o = np.full((len(R["y"]), 10), .1); o[idx4] = p; return o
    Sc = {"F_E79": R["sc"](R["F79"])}; chosen = {}; infos = {}
    for nm, (ro, combos, pm, hok) in arms.items():
        P, ch, infos[nm] = nested(ctx, ro, combos, pm, hok); Sc[nm] = R["sc"](full(P)); chosen[nm] = ch
        print(nm, "bal fold4", round(bal(Sc[nm], R["y"], R["ns"], R["fold"] == 4), 4), "lam", [c["lam"] for c in ch], flush=True)
    r = boot(Sc, R["F"], R["fold"] == 4, "balanced", TAG)
    res["scores"] = {k: dict(score=r[k][0], per_N={str(n): v for n, v in r[k][1].items()}, **ci(r[k][2])) for k in r}
    res["delta_vs_F_E79"] = {k: diff(r, k, "F_E79") for k in r if k != "F_E79"}
    res["delta_pairs"] = {"opp_minus_own": diff(r, "F_opp", "F_own"), "opp_hi_minus_own_hi": diff(r, "F_opp_hi", "F_own_hi"), "opp_minus_shuf": diff(r, "F_opp", "F_shuf"),
                          "opp_minus_shuf_samerank": diff(r, "F_opp", "F_shuf_samerank"), "oppx_minus_opp": diff(r, "F_opp_x", "F_opp")}
    res["chosen_per_part"] = chosen
    pm = ctx["part"]; m4 = R["fold"] == 4; part_full = np.full(len(R["y"]), -1); part_full[idx4] = pm
    res["per_part_delta_F_opp"] = [float(bal(Sc["F_opp"], R["y"], R["ns"], part_full == j) - bal(Sc["F_E79"], R["y"], R["ns"], part_full == j)) for j in range(5)]
    res["per_part_delta_F_opp_x"] = [float(bal(Sc["F_opp_x"], R["y"], R["ns"], part_full == j) - bal(Sc["F_E79"], R["y"], R["ns"], part_full == j)) for j in range(5)]
    res["per_N_delta_F_opp"] = {str(n): float(r["F_opp"][1][n] - r["F_E79"][1][n]) for n in NS}
    res["per_N_delta_F_opp_x"] = {str(n): float(r["F_opp_x"][1][n] - r["F_E79"][1][n]) for n in NS}
    res["frac_questions_with_vote_F_opp_x_vs_F_opp"] = dict(F_opp_x=float(np.mean([c["frac_questions_with_vote_in_part"] for c in chosen["F_opp_x"]])), F_opp=float(np.mean([c["frac_questions_with_vote_in_part"] for c in chosen["F_opp"]])))
    d = res["delta_vs_F_E79"]["F_opp"]; dh = res["delta_vs_F_E79"]["F_opp_hi"]; dsh = res["delta_vs_F_E79"]["F_shuf"]; dpo = res["delta_pairs"]["opp_minus_own"]
    dps = res["delta_pairs"]["opp_minus_shuf"]; ident = dpo["lo"] > 0 and dh["delta"] >= .5 * d["delta"] and dps["lo"] > 0
    res["classification"] = ("negative" if d["delta"] <= 0 else ("positive_identity" if (d["delta"] >= .005 and d["lo"] > 0 and ident) else ("positive_unattributed" if (d["delta"] >= .005 and d["lo"] > 0) else "unclear")))
    res["classification_inputs"] = dict(delta=d["delta"], lo=d["lo"], opp_minus_own_lo=dpo["lo"], hi_delta=dh["delta"], half_main_delta=.5 * d["delta"], shuf_delta=dsh["delta"], opp_minus_shuf_lo=dps["lo"], identity_conditions_met=bool(ident),
                                        gate0_pass=res["gate0"]["passes"])
    res["e96_decision"] = decide_e96(res)
    res["final_params_candidate"] = {"F_opp": select_all(ctx, arms["F_opp"][1], infos["F_opp"]), "F_opp_x": select_all(ctx, arms["F_opp_x"][1], infos["F_opp_x"]), "thresholds_own_pmax": thr}
    with open(fn, "x" if mode == "run" else "w") as f: json.dump(res, f, indent=1, default=float)
    print(mode, {k: round(v["score"], 4) for k, v in res["scores"].items()}); print("delta vs F_E79", {k: round(v["delta"], 4) for k, v in res["delta_vs_F_E79"].items()}); print("pairs", {k: round(v["delta"], 4) for k, v in res["delta_pairs"].items()})
    print("classification", res["classification"], res["classification_inputs"]); print("e96_decision", json.dumps(res["e96_decision"], default=float))


if __name__ == "__main__":
    run(sys.argv[1])
