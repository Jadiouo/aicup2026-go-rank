"""E91 -> submission candidate: carry the validated opponent-vote fusion over to the 10,000 Rank test games.  NO score is ever computed on the test side (there are no labels).

  /usr/bin/python3 e91_final.py prep-test                      # opp/own game files of the 10,000 Rank test games (CPU, seconds) + file checks
  /usr/bin/python3 e91_final.py check-test [--n 200]           # input planes of the test opp side vs src/gogame/board.py (K2) and colour symmetry (K3); numba -> system python
  .venv/bin/python  e91_final.py fit      [--tag A]            # choose ONE (gamma,tau,alpha,lam) on all 4,568 validation questions (not nested) -> .scratch/e91/final_params.json
  .venv/bin/python  e91_final.py pi-test  [--tag A] [--dry]    # label-free transfer check pi_hat_test (Public / Private separately) -> .scratch/e91/pi_test.json
  .venv/bin/python  e91_final.py make     [--tag A] [--name rank_v10a_對手票] [--dry]   # test probabilities of v9 + votes -> submission csv (+ lam=0 == v9 assert)
  /usr/bin/python3 e91_final.py embed-test --ckpt PATH --tag TAG   # later checkpoints (E90 final_model.pt): export vectors + head for the test games, files carry the tag

Nothing here modifies the e89/e90/e91 programs; they are imported read-only.  The `--dry` paths use FAKE vectors and write only *dryrun* files (never data/submissions).
Tag A = checkpoint A (the one the validation questions were measured with); file names of tag A are the plain ones (S_test_opp.npz, final_params.json, ...).
"""
import os, sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / ".scratch/e91"
os.environ.setdefault("NUMBA_CACHE_DIR", str(OUT / "numba_cache")); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(ROOT / "scripts/experiments/e90")]
import numpy as np

T_GAMES = ROOT / ".scratch/e90/data/t_games.npz"; T_UIDS = ROOT / ".scratch/e90/data/t_games_uids.npz"; BAGS = ROOT / ".scratch/e29/test/test_bags.npz"
ROW0, ROW1 = 14200, 24200                      # Rank test games inside t_games.npz
TEST_OPP = OUT / "test_opp_games.npz"; TEST_OWN = OUT / "test_own_games.npz"; TEST_UIDS = OUT / "test_uids.npz"


def sfx(tag): return "" if tag in (None, "", "A") else f"_{tag}"


def split_of(uids):
    """'pub_q_0001#3' -> 0 (Public), 'priv_q_0001#3' -> 1 (Private)."""
    s = np.array([u.split("_q_")[0] for u in uids]); assert set(s) <= {"pub", "priv"}; return (s == "priv").astype(np.int8)


# ------------------------------------------------------------------------------------------------------------------------------ prep-test
def prep_test():
    import e91_prep as P
    d = P.load(T_GAMES); uids = np.load(T_UIDS)["uids"]; bags = np.load(BAGS, allow_pickle=True)
    assert len(uids) == len(d["eff"]) == 24200 and (np.diff(d["rid"]) > 0).all() and "side" not in d
    ur = uids[ROW0:ROW1]; assert len(ur) == 10000 and (ur == bags["uids"]).all(), "uid order of t_games rows 14200-24199 != test_bags uids"
    assert all("#" in u and "_q_" in u for u in ur)
    idx = np.arange(ROW0, ROW1); g = P.subset(d, idx)
    assert (g["rid"] == d["rid"][ROW0:ROW1]).all() and set(np.unique(g["hd"])) <= {0, 1} and set(np.unique(g["eff"])) <= {0, 1}
    assert np.array_equal(g["moves"], d["moves"][d["off"][ROW0]:d["off"][ROW1]]) and g["off"][-1] == len(g["moves"])
    outs = {TEST_OPP: P.flip_side(g), TEST_OWN: dict(g, side=np.zeros(len(g["eff"]), np.int8))}
    for p, o in outs.items():
        assert not p.exists(), f"{p} exists - refuse to overwrite"
        back = P.flip_side(o)
        for k in g:                                                         # double flip == source; own file == source
            assert np.array_equal(back[k] if int(o["side"][0]) == 1 else o[k], g[k]), f"double flip / own file differs in {k}"
        assert np.array_equal(o["moves"], g["moves"]) and np.array_equal(o["off"], g["off"])
        npos = P.own_positions(o, o["eff"]); other = P.own_positions(o, 1 - o["eff"]); assert (npos > 0).all(), "a game has no position for this side"
        st = dict(file=str(p), n_games=int(len(o["eff"])), side=int(o["side"][0]), n_positions_this_side=int(npos.sum()), n_positions_other_side=int(other.sum()), n_games_hd0=int((o["hd"] == 0).sum()),
                  n_games_hd1=int((o["hd"] == 1).sum()), min_positions=int(npos.min()), double_flip_equals_source=True, rows="t_games rows 14200..24199 = test_bags uids order")
        np.savez(p, **o); Path(str(p) + ".json").write_text(json.dumps(st, indent=1)); print(json.dumps(st, indent=1))
    if not TEST_UIDS.exists(): np.savez(TEST_UIDS, uids=ur, rid=g["rid"], split=split_of(ur), hd=g["hd"], eff=g["eff"])
    print("prep-test OK: public", int((split_of(ur) == 0).sum()), "private", int((split_of(ur) == 1).sum()))


# ------------------------------------------------------------------------------------------------------------------------------ check-test (planes vs board engine)
def check_test(n):
    sys.modules["coverage"] = None
    sys.path[:0] = [str(ROOT / ".scratch/e89/pylib"), str(ROOT / "src"), str(ROOT / "scripts/experiments/e89")]
    import fastboard as fb
    from gogame.board import iter_own_positions, liberty_map
    from gogame.planes import encode_position
    from gogame.sgf import GameFlags, ParsedGame
    O = np.load(TEST_OPP); W0 = np.load(TEST_OWN)
    st = dict(K1_flip_equals_own=bool(np.array_equal((1 - O["eff"]).astype(np.int8), W0["eff"]) and np.array_equal(O["moves"], W0["moves"]) and (O["side"] == 1).all() and (W0["side"] == 0).all()))
    mvs, off, eff, hd = O["moves"], O["off"], O["eff"], O["hd"]
    rng = np.random.default_rng(0); picks = np.sort(np.r_[rng.choice(np.flatnonzero(hd == 0), n * 3 // 4, replace=False), rng.choice(np.flatnonzero(hd == 1), n - n * 3 // 4, replace=False)])
    mm = dict(K2_grid=0, K2_meta=0, K2_lib_own=0, K2_lib_opp=0, K2_planes=0, K3_arrays=0, K3_planes=0, K3_flag=0, K5_same_as_own=0); npos = dict(K2=0, K3=0)
    FL = GameFlags(False, False, False, 0, False, 0); f = lambda t: -1 if t is None else t[0] * 19 + t[1]
    for g in picks:
        mv = mvs[off[g]:off[g + 1]]; e = int(eff[g]); h = int(hd[g])
        col = (mv >= 362).astype(np.int16); pos = mv - 362 * col
        arr = np.stack([col, np.where(pos == 361, -1, pos // 19), np.where(pos == 361, -1, pos % 19)], 1).astype(np.int16)
        setup = np.array([(0, p // 19, p % 19) for p in fb.HD_PTS[h].tolist()], np.int16).reshape(-1, 3); game = ParsedGame(arr, setup, FL)
        sel = fb.valid_ordinals(mv, e); out = fb.encode(mv, e, h, sel); grid, lib, po, pw, ko, move, ply, oi = out
        X = fb.build_planes_np(grid, lib, po, pw, ko, move, ply, oi, e); ref = {p.own_idx: p for p in iter_own_positions(game, e)}
        for k in range(len(sel)):
            p = ref[int(oi[k])]; npos["K2"] += 1
            if not np.array_equal(p.grid_own.reshape(-1), grid[k]): mm["K2_grid"] += 1
            if (f(p.prev_opp), f(p.prev_own), f(p.ko_point), f(p.move), p.ply) != (po[k], pw[k], ko[k], move[k], ply[k]): mm["K2_meta"] += 1
            R = encode_position(p, include_move=True).astype(np.float32)
            if not np.array_equal(R[6], X[k, 3] * 1 + X[k, 4] * 2 + X[k, 5] * 3 + X[k, 6] * 4): mm["K2_lib_own"] += 1
            lo = np.clip(liberty_map(p.grid_own, 2), 0, 4).astype(np.float32)
            if not np.array_equal(lo, X[k, 7] * 1 + X[k, 8] * 2 + X[k, 9] * 3 + X[k, 10] * 4): mm["K2_lib_opp"] += 1
            if not (np.array_equal(R[0], X[k, 0]) and np.array_equal(R[1], X[k, 1]) and np.array_equal(R[2], X[k, 2]) and np.array_equal(R[3], X[k, 11]) and np.array_equal(R[4], X[k, 12])
                    and np.array_equal(R[5], X[k, 13]) and np.array_equal(R[7], X[k, 15]) and np.array_equal(R[8], X[k, 14])): mm["K2_planes"] += 1
        selo = fb.valid_ordinals(mv, 1 - e); outo = fb.encode(mv, 1 - e, h, selo)
        if len(selo) == len(sel) and np.array_equal(outo[0], grid): mm["K5_same_as_own"] += 1
        if h <= 1:
            mvs_sw = np.where(mv >= 362, mv - 362, mv + 362).astype(np.int16); sel2 = fb.valid_ordinals(mvs_sw, 1 - e); out2 = fb.encode(mvs_sw, 1 - e, h, sel2); npos["K3"] += len(sel)
            if len(sel2) != len(sel) or not all(np.array_equal(x, y) for x, y in zip(out, out2)): mm["K3_arrays"] += 1
            X2 = fb.build_planes_np(*out2, 1 - e); keep = [i for i in range(18) if i != 15]
            if not np.array_equal(X[:, keep], X2[:, keep]): mm["K3_planes"] += 1
            if not (np.array_equal(X[:, 15], 1 - X2[:, 15]) and set(np.unique(X[:, 15])) <= {0.0, 1.0}): mm["K3_flag"] += 1
    res = dict(check="E91 test-side opponent input", games=int(len(picks)), positions_checked=npos, mismatches=mm, **st); res["pass"] = st["K1_flip_equals_own"] and all(v == 0 for v in mm.values())
    (OUT / "check_test_planes.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); sys.exit(0 if res["pass"] else 1)



# ------------------------------------------------------------------------------------------------------------------------------ shared helpers
def load_head(tag):
    """(W [C,128], cls_rank [C]) of the head that belongs to the vectors of `tag` (A: .scratch/e91/head.npz; others: head_<tag>.npz written by embed-test)."""
    h = np.load(OUT / ("head.npz" if tag in (None, "", "A") else f"head_{tag}.npz")); return h["W"].astype(np.float32), h["cls_rank"].astype(np.int64)


def readout(gv, W, cls_rank, chunk=2000):
    """per game: p_max of softmax(30 cos(gv, W_j)) and the rank 'lift' (rank histogram divided by the library's rank prior, renormalised).  Same maths as e91_score.game_readout."""
    Wn = (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32); n = len(gv); C = len(W)
    onehot = np.zeros((C, 10), np.float32); onehot[np.arange(C), cls_rank] = 1; pi = np.bincount(cls_rank, minlength=10) / C; assert (pi > 0).all(), "a rank has no library player"
    pmax = np.zeros(n); hist = np.zeros((n, 10))
    for a in range(0, n, chunk):
        lg = 30.0 * (gv[a:a + chunk].astype(np.float32) @ Wn.T); lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
        pmax[a:a + chunk] = p.max(1); hist[a:a + chunk] = p @ onehot
    lift = hist / pi; lift /= lift.sum(1, keepdims=True); return dict(pmax=pmax, lift=lift)


def test_meta():
    """per game: uid, split (0 public / 1 private), hd; per question: qid, n, rows, split, question index of every game.  Asserts that bag rows and uids agree."""
    u = np.load(TEST_UIDS); uids = u["uids"]; bags = np.load(BAGS, allow_pickle=True); qids = bags["qids"].astype(str); rows = bags["rows"]
    assert (bags["uids"] == uids).all()
    gq = np.full(len(uids), -1)
    for q in range(len(qids)):
        r = rows[q][rows[q] >= 0]; assert len(r) == bags["n"][q] and all(uids[i].startswith(qids[q] + "#") for i in r); assert (gq[r] == -1).all(); gq[r] = q
    assert (gq >= 0).all()
    return dict(uids=uids, gsplit=u["split"].astype(int), hd=u["hd"].astype(int), qids=qids, n=bags["n"], rows=rows, qsplit=split_of(qids).astype(int), gq=gq)


def vectors(tag, dry):
    """(own gv, opp gv, W, cls_rank, info).  dry: FAKE random unit vectors (4% of the opponent games planted as a library player + noise) - never reads S_test_*."""
    W, cr = load_head(tag)
    if dry:
        rng = np.random.default_rng(1234); n = 10000; unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        own = unit(rng.normal(size=(n, 128))).astype(np.float32); opp = unit(rng.normal(size=(n, 128))).astype(np.float32); Wn = unit(W)
        for g in np.flatnonzero(rng.random(n) < .04): opp[g] = unit((Wn[rng.integers(len(W))] + .5 * rng.normal(size=128) / np.sqrt(128))[None])[0]
        return own, opp, W, cr, dict(FAKE_VECTORS=True)
    fo, fp = OUT / f"S_test_own{sfx(tag)}.npz", OUT / f"S_test_opp{sfx(tag)}.npz"
    assert fo.exists() and fp.exists(), f"{fo} / {fp} missing: run .scratch/e91/gpu_job_test.sh (tag A) or embed-test first"
    so, sp = np.load(fo), np.load(fp); rid = np.load(TEST_OWN)["rid"].astype(np.int64)
    for s in (so, sp): assert (s["rid"].astype(np.int64) == rid).all() and len(s["gv_all"]) == 10000 and np.isfinite(s["gv_all"]).all() and np.abs(np.linalg.norm(s["gv_all"], axis=1) - 1).max() < 1e-3
    cos = (so["gv_all"] * sp["gv_all"]).sum(1); assert np.median(cos) < .9, "own and opponent vectors are (nearly) identical - the side flip did not take effect"
    return so["gv_all"], sp["gv_all"], W, cr, dict(S_own=str(fo), S_opp=str(fp), cos_own_opp_median=float(np.median(cos)))


def thresholds(ro_own, meta):
    """label-free thresholds: quantiles of OWN-side p_max over hd==0 games of the SAME split (public / private never mixed)."""
    return {s: {q: float(np.quantile(ro_own["pmax"][(meta["gsplit"] == s) & (meta["hd"] == 0)], q)) for q in (.50, .80, .95)} for s in (0, 1)}


# ------------------------------------------------------------------------------------------------------------------------------ fit
def fit(tag):
    assert tag in ("A",), "fit uses the validation vectors of checkpoint A only; later checkpoints reuse final_params.json (their fold-4 questions are in their training set)"
    import e91_score as E
    from r0_common import bal
    fn = OUT / "final_params.json"; assert not fn.exists(), f"{fn} exists - refuse to overwrite"
    R = E.rebuild(); g = np.load(E.GAMES); game = {k: g[k] for k in g.files}; h = np.load(E.HEAD); W, cls_rank = h["W"], h["cls_rank"].astype(np.int64); ctx = E.build_ctx(R, game)
    so, sp = np.load(E.S_OWN), np.load(E.S_OPP)
    for s in (so, sp): assert (s["rid"].astype(np.int64) == ctx["rid"]).all() and np.isfinite(s["gv_all"]).all()
    ro_own, ro_opp = E.game_readout(so["gv_all"], W, cls_rank), E.game_readout(sp["gv_all"], W, cls_rank); m0 = game["hd"] == 0
    thr = {q: float(np.quantile(ro_own["pmax"][m0], q)) for q in (.50, .80, .95)}; combos = E.grid_main(thr); assert len(combos) == 82
    yq, nq, f79q = ctx["yq"], ctx["nq"], ctx["f79q"]; cache = {}; sc = []
    for gm, lo, hi, al, lam in combos:
        k = (gm, lo, hi)
        if k not in cache: cache[k] = E.question_votes(ctx["rows"], ctx["rid"], ctx["hd"], ro_opp, gm, lo, hi)
        sc.append(bal(E.sc_q(E.fuse(f79q, cache[k][0], lam, al), yq), yq, nq))
    order = sorted(range(82), key=lambda i: (combos[i][4], i)); best = None
    for i in order:
        if best is None or sc[i] > best[0] + 1e-12: best = (sc[i], i)
    i = best[1]; gm, lo, hi, al, lam = combos[i]; assert i > 0 and lam > 0, "the best entry is 'no vote' (lam = 0): the votes do not help on the validation questions"
    tq = [None, .80, .95][[0., thr[.80], thr[.95]].index(lo)]
    top = sorted(range(82), key=lambda j: -sc[j])[:8]
    res = dict(rule="one combo of the 82-entry E87 grid chosen on ALL 4,568 fold-4 validation questions (balanced per-N score, EV decode; ties -> smaller lam, then grid order); NOT nested: the score is an in-sample figure, the nested +.0158 stays the performance estimate",
               gamma=gm, tau_q=tq, tau_value_validation_own_quantile=lo, alpha=al, lam=lam, score_all_questions=best[0], score_F_E79_all_questions=sc[0], delta_in_sample=best[0] - sc[0], n_questions=int(len(yq)),
               top8=[dict(gamma=combos[j][0], tau_value=combos[j][1], alpha=combos[j][3], lam=combos[j][4], score=sc[j]) for j in top], validation_thresholds_own_pmax=thr,
               frac_questions_with_vote=float(cache[(gm, lo, hi)][1].mean()), repro=R["repro"], S_opp_sha16=__import__("hashlib").sha256(Path(E.S_OPP).read_bytes()).hexdigest()[:16],
               nested_choices_in_result_json=[dict(part=c["part"], gamma=c["gamma"], alpha=c["alpha"], lam=c["lam"], lo=c["lo"]) for c in json.loads((OUT / "result.json").read_text())["chosen_per_part"]["F_opp"]])
    with open(fn, "x") as f: json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("nested_choices_in_result_json",)}, indent=1, default=float))


def load_params():
    p = json.loads((OUT / "final_params.json").read_text()); return p


# ------------------------------------------------------------------------------------------------------------------------------ pi-test
def pi_test(tag, dry):
    meta = test_meta(); own, opp, W, cr, info = vectors(tag, dry); ro_own, ro_opp = readout(own, W, cr), readout(opp, W, cr); thr = thresholds(ro_own, meta); nq = len(meta["qids"])
    if tag == "A" and not dry:                                              # the numpy readout equals the one the validation used
        import e91_score as E; chk = E.game_readout(own, W, cr); assert np.abs(chk["lift"] - ro_own["lift"]).max() < 1e-4 and np.abs(chk["pmax"] - ro_own["pmax"]).max() < 1e-4
    res = dict(tag=tag, **info, rule="pi_hat_test = P_opp(p_max >= Q95_own) - P_own(p_max >= Q95_own) over hd==0 games; Q95 = 95% quantile of OWN-side p_max of the SAME split; resampling unit = question (2000 draws, seed 91)")
    try: res["validation_gate0_for_reference"] = {k: v for k, v in json.loads((OUT / "result.json").read_text())["gate0"].items() if k in ("pi_hat", "lo", "hi", "share_opp_high", "share_own_high")}
    except Exception as e: res["validation_gate0_for_reference"] = str(e)
    for s, nm in ((0, "public"), (1, "private")):
        m = (meta["gsplit"] == s) & (meta["hd"] == 0); q = meta["gq"][m]; t = thr[s][.95]
        a = np.bincount(q, weights=(ro_opp["pmax"][m] >= t), minlength=nq); b = np.bincount(q, weights=(ro_own["pmax"][m] >= t), minlength=nq); n = np.bincount(q, minlength=nq).astype(float)
        qs = np.flatnonzero(n > 0); a, b, n = a[qs], b[qs], n[qs]; pi = a.sum() / n.sum() - b.sum() / n.sum()
        C = np.random.default_rng(91).multinomial(len(qs), np.full(len(qs), 1 / len(qs)), size=2000).astype(np.float64); bs = (C @ a - C @ b) / (C @ n)
        res[nm] = dict(n_games_hd0=int(m.sum()), n_questions=int(len(qs)), thr_Q50_Q80_Q95_own=[thr[s][x] for x in (.5, .8, .95)], share_opp_high=float(a.sum() / n.sum()), share_own_high=float(b.sum() / n.sum()),
                       pi_hat_test=float(pi), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
    fn = OUT / (f"pi_test{sfx(tag)}.json" if not dry else f"pi_test_dryrun{sfx(tag)}.json"); fn.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------------------------------ make
def make(tag, name, dry):
    import pandas as pd
    import e91_score as E
    from gogame.constants import RANKS
    from gogame.models import decide_ev
    from gogame.submission import write_rank_submission
    from scripts.experiments.e37.predict_test import sha, K, DATA
    nrm = lambda p: p / p.sum(1, keepdims=True); P = load_params(); gm, tq, al, lam = P["gamma"], P["tau_q"], P["alpha"], P["lam"]; assert lam > 0
    name = name or ("rank_v10a_對手票" if tag == "A" else f"rank_v10_{tag}_對手票")
    out_csv = (OUT / f"dryrun_{name}.csv") if dry else (ROOT / "data/submissions" / f"{name}.csv")
    stats_fn = OUT / (f"final_stats_dryrun{sfx(tag)}.json" if dry else f"final_stats{sfx(tag)}.json")
    assert not out_csv.exists() and (dry or not stats_fn.exists()), f"{out_csv} / {stats_fn} exists - refuse to overwrite"
    meta = test_meta(); qids = meta["qids"].tolist(); nqn = meta["n"]; rows = meta["rows"]
    # ---- v9 test-side probabilities (same formula as scripts/oneoff/e79_fuse.py) and the lam = 0 assert
    fw = json.loads((ROOT / "experiments/E79_rank_pos_attn/results/final.json").read_text())["final_weights_all_oof"]["F_E79"]; w, v, u = fw["w_A"], fw["v_M03"], fw["u_A1P"]; assert (w, v, u) == (.9, .5, .75)
    p03, p04 = np.load(ROOT / ".scratch/e66/final/test_m03.npy"), np.load(ROOT / ".scratch/e66/final/test_m04.npy"); s2, sp_ = np.load(ROOT / ".scratch/e66/final/set_A1T2.npy"), np.load(ROOT / ".scratch/e79/final/set_A1P.npy")
    assert p03.shape == p04.shape == (800, 10) and s2.shape == sp_.shape == (5, 3, 800, 10)
    F = nrm((1 - w) * nrm(v * p03 + (1 - v) * p04) + w * nrm((1 - u) * nrm(s2.mean((0, 1))) + u * nrm(sp_.mean((0, 1)))))
    v9 = pd.read_csv(ROOT / "data/submissions/rank_v9_逐局面注意力.csv"); assert list(v9["question_id"]) == qids; r9 = np.array([RANKS.index(x) for x in v9["pred_rank"]])
    assert (decide_ev(F) == r9).all(), "decode of the rebuilt v9 test probabilities != rank_v9 csv: not the v9 probabilities - STOP"
    # ---- votes
    own, opp, W, cr, info = vectors(tag, dry); ro_own, ro_opp = readout(own, W, cr), readout(opp, W, cr); thr = thresholds(ro_own, meta)
    V = np.full((800, 10), .1); has = np.zeros(800, bool); rid = np.arange(10000)
    for s in (0, 1):
        lo = 0. if tq is None else thr[s][tq]; qm = np.flatnonzero(meta["qsplit"] == s)
        Vs, hs = E.question_votes(rows[qm], rid, meta["hd"], ro_opp, gm, lo, 9.); V[qm] = Vs; has[qm] = hs
    assert (np.abs(V.sum(1) - 1) < 1e-9).all()
    F0 = E.fuse(F, V, 0., al); assert (decide_ev(F0) == r9).all(), "lam = 0 does not reproduce v9 answers question by question"
    print("ASSERT OK: lam=0 reproduces v9 (800/800 answers identical)", flush=True)
    Pf = E.fuse(F, V, lam, al); pred = decide_ev(Pf); assert pred.shape == (800,) and pred.min() >= 0 and pred.max() < K
    write_rank_submission(qids, pred, out_csv)
    import check_submission
    rc = check_submission.main([str(out_csv), "--task", "rank", "--data-dir", str(DATA)]); assert rc == 0, "check_submission failed"
    sp_q = meta["qsplit"]; same = pred == r9
    st = dict(tag=tag, dry=dry, file=str(out_csv), sha256=sha(out_csv), check_submission_rc=rc, params=dict(gamma=gm, tau_q=tq, alpha=al, lam=lam), vectors=info, lam0_reproduces_v9="800/800", n_questions=800,
              same_answer_as_v9=float(same.mean()), n_changed=int((~same).sum()), frac_questions_with_vote=float(has.mean()), thresholds_own_pmax_by_split={"public": thr[0], "private": thr[1]},
              pred_share_v9=np.bincount(r9, minlength=K).tolist(), pred_share_new=np.bincount(pred, minlength=K).tolist(), mean_rank_shift=float((pred - r9).mean()),
              per_split={nm: dict(n=int((sp_q == s).sum()), same_as_v9=float(same[sp_q == s].mean()), n_changed=int((~same[sp_q == s]).sum()), frac_with_vote=float(has[sp_q == s].mean()),
                                  pred_share_v9=np.bincount(r9[sp_q == s], minlength=K).tolist(), pred_share_new=np.bincount(pred[sp_q == s], minlength=K).tolist()) for s, nm in ((0, "public"), (1, "private"))},
              per_N_same={int(n): float(same[nqn == n].mean()) for n in (5, 10, 15, 20)}, v9_sha256=sha(ROOT / "data/submissions/rank_v9_逐局面注意力.csv"))
    stats_fn.write_text(json.dumps(st, indent=1, default=float)); print(json.dumps(st, indent=1, default=float))


# ------------------------------------------------------------------------------------------------------------------------------ embed-test (other checkpoints)
def embed_test(ckpt, tag, limit):
    """vectors of both test sides with checkpoint `ckpt` (A's ckpt_latest.pt or an E90 final_model.pt) -> S_test_{opp,own}_<tag>.npz + head_<tag>.npz.  Wraps e90/embed.py (monkeypatches CKPT / Net only)."""
    sys.modules["coverage"] = None; sys.path[:0] = [str(ROOT / ".scratch/e89/pylib"), str(ROOT / "scripts/experiments/e89")]
    import torch, embed as e90, train_s as ts
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); Wm = ck["model"]["W"].float().numpy(); C = Wm.shape[0]; sha = e90.sha16(ckpt)
    if "players" in ck:                                                    # E90 final_model.pt: class j <-> player code players[j] -> rank
        aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=True); n_aux = len(aux["player_ids"])
        sj = OUT / "split_ranks.json"                                          # system python has no pyarrow: dump split.parquet once through the project venv
        if not sj.exists():
            import subprocess
            subprocess.run([str(ROOT / ".venv/bin/python"), "-c", "import pandas as pd,json,sys;s=pd.read_parquet('data/splits/split.parquet');json.dump(dict(pid=s.player_id.astype(str).tolist(),rank=s.rank_id.astype(int).tolist(),h=s.is_holdout.astype(bool).tolist()),open(sys.argv[1],'w'))", str(sj)], check=True, cwd=ROOT)
        sp = json.loads(sj.read_text()); Hset = sorted(p for p, h in zip(sp["pid"], sp["h"]) if h); rk = dict(zip(sp["pid"], sp["rank"]))
        pr = aux["player_rank"].astype(int); pid = aux["player_ids"].astype(str)
        assert all(pr[i] == rk[pid[i]] for i in range(n_aux) if pid[i] in rk), "aux player_rank != split rank_id"
        codes = ck["players"].astype(np.int64); assert len(codes) == C and len(np.unique(codes)) == C and codes.max() < n_aux + len(Hset)
        cls_rank = np.array([pr[c] if c < n_aux else rk[Hset[c - n_aux]] for c in codes], np.int64); n_h = int((codes >= n_aux).sum())
    else:
        assert sha == e90.SHA16, "no `players` in the checkpoint and it is not checkpoint A"; h = np.load(OUT / "head.npz"); assert np.array_equal(h["W"], Wm); cls_rank = h["cls_rank"].astype(np.int64); n_h = 0
    hf = OUT / f"head_{tag}.npz"
    if not hf.exists(): np.savez(hf, W=Wm, cls_rank=cls_rank, ckpt_sha16=sha, n_classes=C, n_h_classes=n_h)
    print("head", hf, "classes", C, "rank counts", np.bincount(cls_rank, minlength=10).tolist(), "H classes", n_h, flush=True)
    Orig = ts.Net; ts.Net = lambda n, r, d: Orig(C, r, d); e90.CKPT = Path(ckpt); e90.SHA16 = sha
    for side in ("opp", "own"):
        src = OUT / f"test_{side}_games.npz"; dst = OUT / f"S_test_{side}_{tag}.npz"
        if limit:
            import e91_prep as P; d = P.load(src); small = OUT / f"test_{side}_games_lim{limit}.npz"; (None if small.exists() else np.savez(small, **P.subset(d, np.arange(limit)))); src = small
            dst = OUT / f"S_test_{side}_{tag}_lim{limit}.npz"
        if dst.exists(): print("exists", dst); continue
        sys.argv = ["embed.py", str(src), str(dst), "--chunk", "500", "--threads", "2"] + (["--cpu"] if limit else []); e90.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["prep-test", "check-test", "fit", "pi-test", "make", "embed-test"])
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--tag", default="A"); ap.add_argument("--ckpt"); ap.add_argument("--name"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "prep-test": prep_test()
    elif a.mode == "check-test": check_test(a.n)
    elif a.mode == "fit": fit(a.tag)
    elif a.mode == "pi-test": pi_test(a.tag, a.dry)
    elif a.mode == "make": make(a.tag, a.name, a.dry)
    else:
        assert a.ckpt and a.tag not in ("", "A"), "embed-test needs --ckpt and a tag other than A (A's files are the plain ones made by gpu_job_test.sh)"; embed_test(a.ckpt, a.tag, a.limit)
