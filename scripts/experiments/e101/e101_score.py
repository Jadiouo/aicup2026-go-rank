"""E101 Rank: opponent votes of handicap-1 (hd == 1, 'teaching' / fixed-first-move) games converted with the STATION RULE (transition kernels) instead of the E96 'half no shift / half one-rank shift'.
Plan (frozen, never edited): experiments/E101_rank_hd1_site_rule/PLAN.md.  Reuses scripts/experiments/e96/e96_score.py read-only (rebuild, build_ctx, load_real, game_readout, question_votes, fuse, nested, grids, shift_lift).
Rank index 0..9 = D, C, B, A, 1D, 2D, 3D, 4D, 5D, 6D.   eff 0 = the questioned player has BLACK, eff 1 = WHITE (same convention as e96 shift_lift).
K[k, j] = share of "the opponent sits at rank j when the questioned player is rank k"; NOT normalised (mass falling outside the 10-rank library is dropped).  L(k) = sum_j K[k, j] * lift[j]; then per-game renormalisation;
a game whose L is all zero casts no vote.  hd == 0 games are untouched; hd >= 2 games never vote (hd_ok = (0, 1)).
  unit   : print K and the unit-vector sanity check (no data)                     -> .scratch/e101/unit_check.json
  repro  : reference arm F_opp_x must reproduce .scratch/e96/posthoc_x_fixed.json (< 1e-9); NO F_site score computed  -> .scratch/e101/repro_check.json
  run    : repro assertion first, then the formal F_site scoring ONCE            -> .scratch/e101/result.json (refuses to overwrite)
  CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 env PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/experiments/e101/e101_score.py <mode>
Never touches result_v4.json, H, h_eval_ledger, data/submissions.  Writes only .scratch/e101/.
"""
import os, sys, json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts/experiments/e96"), str(ROOT / "scripts/experiments/e90"), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]
import numpy as np

OUT = ROOT / ".scratch/e101"
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
GM, LO, HI, AL, LAM = 0., 0., 9., .1, 2.0           # fixed setting (same as e96_posthoc.py)
TWO3, ONE3 = 2. / 3., 1. / 3.


def make_K():
    """Rows: own rank k; columns: opponent rank j.  Literal transcription of PLAN.md 'Do' section."""
    Kw = np.zeros((10, 10)); Kb = np.zeros((10, 10))
    # own WHITE: opponent one station-level LOWER.  2D-6D (5..9) -> k-1 ; 1D (4) -> A (3) ; A,B,C (3,2,1) -> same band 2/3 + one band lower 1/3 ; D (0) -> same band 2/3 (the other 1/3 is outside the library)
    for k in range(5, 10): Kw[k, k - 1] = 1.
    Kw[4, 3] = 1.
    for k in (3, 2, 1): Kw[k, k] = TWO3; Kw[k, k - 1] = ONE3
    Kw[0, 0] = TWO3
    # own BLACK: opponent one station-level HIGHER.  1D-5D (4..8) -> k+1 ; 6D (9) -> outside library (row of zeros) ; A (3) -> same band 2/3 + 1D 1/3 ; D,C,B (0,1,2) -> same band 2/3 + one band higher 1/3
    for k in range(4, 9): Kb[k, k + 1] = 1.
    Kb[3, 3] = TWO3; Kb[3, 4] = ONE3
    for k in (0, 1, 2): Kb[k, k] = TWO3; Kb[k, k + 1] = ONE3
    return Kw, Kb


K_WHITE, K_BLACK = make_K()


def site_lift(lift, eff, hd):
    """hd == 1 rows: L = lift @ K.T (own white -> K_WHITE, own black -> K_BLACK), then per-game renormalisation; a row whose L is exactly all-zero stays zero (no vote).  Other rows are returned unchanged.
    Returns (x, zero_row_mask over all games)."""
    x = lift.copy(); blk = (hd == 1) & (eff == 0); wht = (hd == 1) & (eff == 1)
    x[blk] = lift[blk] @ K_BLACK.T; x[wht] = lift[wht] @ K_WHITE.T
    m = hd == 1; s = x[m].sum(1); zero = np.zeros(len(lift), bool); zero[np.flatnonzero(m)[s <= 0]] = True
    sm = x[m]; ok = s > 0; sm[ok] /= s[ok, None]; sm[~ok] = 0.; x[m] = sm
    return x, zero


def unit_check():
    f = lambda v: [round(float(a), 6) for a in v]
    out = dict(K_WHITE=K_WHITE.tolist(), K_BLACK=K_BLACK.tolist(), rank_index=RANKS, cases={})
    def L(j, eff):
        lift = np.zeros((1, 10)); lift[0, j] = 1.; x, z = site_lift(lift, np.array([eff]), np.array([1])); return x[0], bool(z[0])
    # derivation by hand (from the PLAN definition): L(k) = K[k, j] normalised over k
    cases = {
        "opp=3D (j=6), own WHITE": (6, 1, "only k=7 (4D) has K[k,6]=1  -> L = 1 at 4D"),
        "opp=3D (j=6), own BLACK": (6, 0, "only k=5 (2D) has K[k,6]=1  -> L = 1 at 2D"),
        "opp=A (j=3), own WHITE": (3, 1, "K[4,3]=1 (1D), K[3,3]=2/3 (A) -> L = 0.6 at 1D, 0.4 at A"),
        "opp=A (j=3), own BLACK": (3, 0, "K[3,3]=2/3 (A), K[2,3]=1/3 (B) -> L = 2/3 at A, 1/3 at B  (k=4..: K[k,3]=0)"),
        "opp=6D (j=9), own WHITE": (9, 1, "no k has K[k,9] > 0 (would need own rank 7D) -> all-zero row, game casts no vote"),
        "opp=6D (j=9), own BLACK": (9, 0, "K[8,9]=1 (5D) -> L = 1 at 5D"),
        "opp=D (j=0), own BLACK": (0, 0, "K[0,0]=2/3 (D) only (no rank below D) -> L = 1 at D"),
        "opp=D (j=0), own WHITE": (0, 1, "K[0,0]=2/3 (D), K[1,0]=1/3 (C) -> L = 2/3 at D, 1/3 at C"),
    }
    for name, (j, eff, derived) in cases.items():
        x, z = L(j, eff); out["cases"][name] = dict(program_L=f(x), zero_row=z, derived_by_hand=derived,
                                                     L_nonzero={RANKS[i]: round(float(x[i]), 6) for i in range(10) if x[i] > 0})
    # directional assertions demanded by the task
    xw, _ = L(6, 1); xb, _ = L(6, 0); assert xw.argmax() == 7 and abs(xw[7] - 1) < 1e-12 and xb.argmax() == 5 and abs(xb[5] - 1) < 1e-12
    xa, _ = L(3, 1); assert abs(xa[4] - .6) < 1e-12 and abs(xa[3] - .4) < 1e-12
    xa, _ = L(3, 0); assert abs(xa[3] - 2 / 3) < 1e-12 and abs(xa[2] - 1 / 3) < 1e-12
    _, z = L(9, 1); assert z
    # hd != 1 rows untouched
    rng = np.random.default_rng(0); lf = rng.dirichlet(np.ones(10), 50); ef = rng.integers(0, 2, 50); hdf = rng.integers(0, 4, 50); x, _ = site_lift(lf, ef, hdf)
    assert np.array_equal(x[hdf != 1], lf[hdf != 1]); assert np.allclose(x[hdf == 1].sum(1), 1)
    out["passed"] = True
    return out


def load_everything():
    import e96_score as S
    from r0_common import rebuild
    R = rebuild(); print("repro", R["repro"], flush=True)
    g = np.load(S.GAMES); game = {k: g[k] for k in g.files}; ctx = S.build_ctx(R, game)
    real = S.load_real(S.OUT, ctx, 6039); W, cls_rank = real["W"], real["cls_rank"]
    ro_opp = S.game_readout(real["opp"], W, cls_rank)
    return S, R, game, ctx, ro_opp, real


def reference_check(S, R, game, ctx, ro_opp):
    """Fixed-setting F_opp_x (shift_lift) and F_opp, F_E79: scores on all 4,568 and on the 2,801 'any hd1 game' questions; must equal posthoc_x_fixed.json to < 1e-9."""
    from r0_common import boot
    ro_x = dict(pmax=ro_opp["pmax"], lift=S.shift_lift(ro_opp["lift"], game["eff"], game["hd"]), top1_rank=ro_opp["top1_rank"])
    idx4, rows, rid, hd = ctx["idx4"], ctx["rows"], ctx["rid"], ctx["hd"]
    def full(p):
        o = np.full((len(R["y"]), 10), .1); o[idx4] = p; return o
    V0, _ = S.question_votes(rows, rid, hd, ro_opp, GM, LO, HI, (0,)); V1, _ = S.question_votes(rows, rid, hd, ro_x, GM, LO, HI, (0, 1))
    Sc = {"F_E79": R["sc"](R["F79"]), "F_opp": R["sc"](full(S.fuse(ctx["f79q"], V0, LAM, AL))), "F_opp_x": R["sc"](full(S.fuse(ctx["f79q"], V1, LAM, AL)))}
    valid = rows >= 0; ii = np.searchsorted(rid, np.where(valid, rows, rid[0])); q_hd1 = (valid & (hd[ii] == 1)).any(1)
    msub = np.zeros(len(R["y"]), bool); msub[idx4[q_hd1]] = True; m4 = R["fold"] == 4
    ra = boot(Sc, R["F"], m4, "balanced", S.TAG); rs = boot(Sc, R["F"], msub, "balanced", S.TAG)
    ref = json.loads((ROOT / ".scratch/e96/posthoc_x_fixed.json").read_text()); rows_out = []; ok = True
    for arm in ("F_E79", "F_opp", "F_opp_x"):
        for scope, r, pj in (("all4568", ra, ref["scores"][arm]), ("hd1_2801", rs, ref["subset_questions_with_any_hd1_game"]["scores"][arm])):
            lo, hi = np.quantile(r[arm][2], .025), np.quantile(r[arm][2], .975)
            for key, val in (("score", r[arm][0]), ("lo", float(lo)), ("hi", float(hi))):
                d = abs(val - pj[key]); rows_out.append(dict(arm=arm, scope=scope, key=key, got=float(val), posthoc=pj[key], absdiff=float(d))); ok &= bool(d < 1e-9)
    return dict(all_within_1e9=ok, max_absdiff=max(r["absdiff"] for r in rows_out), n_checks=len(rows_out), rows=rows_out), Sc, msub, q_hd1, ro_x


def main(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "unit":
        u = unit_check(); fn = OUT / "unit_check.json"
        if not fn.exists(): fn.write_text(json.dumps(u, indent=1))
        print("K_WHITE (own white; row = own rank k, col = opponent rank j; ranks %s)" % RANKS); print(np.round(K_WHITE, 4)); print("K_BLACK"); print(np.round(K_BLACK, 4))
        for n, c in u["cases"].items(): print(n, "->", c["L_nonzero"], "| zero_row" if c["zero_row"] else "", "| by hand:", c["derived_by_hand"])
        return
    assert mode in ("repro", "run")
    fn_res = OUT / "result.json"
    if mode == "run": assert not fn_res.exists(), "result.json exists: the formal scoring runs ONCE - refusing"
    assert (OUT / "PLAN_FROZEN").exists()
    sha0 = hashlib.sha256((ROOT / ".scratch/e96/result_v4.json").read_bytes()).hexdigest()[:16]; print("result_v4.json sha16 before:", sha0, flush=True)
    unit_check()                                                                       # assertions must hold before touching data
    S, R, game, ctx, ro_opp, real = load_everything()
    chk, Sc, msub, q_hd1, ro_x = reference_check(S, R, game, ctx, ro_opp)
    chk["result_v4_sha16_before"] = sha0
    print("REPRO F_opp_x vs posthoc: all < 1e-9 =", chk["all_within_1e9"], "max absdiff", chk["max_absdiff"], flush=True)
    if mode == "repro":
        fn = OUT / "repro_check.json"
        if not fn.exists(): fn.write_text(json.dumps(chk, indent=1))
        return
    if not chk["all_within_1e9"]: print("REPRODUCTION FAILED - STOP, no F_site score computed"); sys.exit(3)
    from r0_common import bal, NS, boot, diff, ci
    # ------------------------------------------------------------------------------------------ F_site
    lift_site, zero_game = site_lift(ro_opp["lift"], game["eff"], game["hd"]); ro_s = dict(pmax=ro_opp["pmax"], lift=lift_site, top1_rank=ro_opp["top1_rank"])
    idx4, yq, nq, rows, rid, hd = ctx["idx4"], ctx["yq"], ctx["nq"], ctx["rows"], ctx["rid"], ctx["hd"]; ny = len(R["y"]); m4 = R["fold"] == 4
    def full(p):
        o = np.full((ny, 10), .1); o[idx4] = p; return o
    Vs, hs = S.question_votes(rows, rid, hd, ro_s, GM, LO, HI, (0, 1)); P_site = S.fuse(ctx["f79q"], Vs, LAM, AL)
    Sc["F_site"] = R["sc"](full(P_site))
    ro_own = S.game_readout(real["own"], real["W"], real["cls_rank"]); thr = {q: float(np.quantile(ro_own["pmax"][game["hd"] == 0], q)) for q in (.50, .80, .95)}
    combos = S.grid_main(thr); ro_x_ = ro_x
    Px, chx, _ = S.nested(ctx, ro_x_, combos, None, (0, 1)); Ps, chs, _ = S.nested(ctx, ro_s, combos, None, (0, 1))
    Sc["F_opp_x_nested"] = R["sc"](full(Px)); Sc["F_site_nested"] = R["sc"](full(Ps))
    ref96 = json.loads((ROOT / ".scratch/e96/result_v4.json").read_text()); nested_check = dict(F_opp_x_nested_got=float(bal(Sc["F_opp_x_nested"], R["y"], R["ns"], m4)), result_v4_F_opp_x=ref96["scores"]["F_opp_x"]["score"])
    nested_check["absdiff"] = abs(nested_check["F_opp_x_nested_got"] - nested_check["result_v4_F_opp_x"])
    print("nested F_opp_x vs result_v4:", nested_check, flush=True)
    # ------------------------------------------------------------------------------------------ bootstrap, main + info
    def pack(r, names): return {k: dict(score=r[k][0], per_N={str(n): v for n, v in r[k][1].items()}, **ci(r[k][2])) for k in names}
    sc_names = list(Sc); ra = boot(Sc, R["F"], m4, "balanced", S.TAG); rs = boot(Sc, R["F"], msub, "balanced", S.TAG)
    pair = lambda r, a, b: dict(diff(r, a, b), per_N={str(n): float(r[a][1][n] - r[b][1][n]) for n in r[a][1]})
    main = pair(rs, "F_site", "F_opp_x")
    verdict = "established_make_v10e" if (main["delta"] >= .003 and main["lo"] > 0) else ("small_gain_make_v10e_with_note" if main["lo"] > 0 else "not_adopted_keep_v10d")
    res = dict(mode="run", NOTE="E101 formal scoring (run once). Fixed setting gamma 0, tau 0, alpha .1, lam 2.0 for F_opp_x and F_site; nested arms use the E96 82-entry grid, hd_ok (0,1).", plan="experiments/E101_rank_hd1_site_rule/PLAN.md",
               setting=dict(gamma=GM, tau_lo=LO, alpha=AL, lam=LAM), K_WHITE=K_WHITE.tolist(), K_BLACK=K_BLACK.tolist(), repro_check=dict(all_within_1e9=chk["all_within_1e9"], max_absdiff=chk["max_absdiff"], n_checks=chk["n_checks"]),
               nested_consistency_vs_result_v4=nested_check, script_sha16=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16], n_bootstrap=4000, n_questions_fold4=int(len(yq)), n_questions_hd1_subset=int(q_hd1.sum()), n_players_hd1_subset=int(len(np.unique(ctx["plq"][q_hd1]))))
    res["primary_criterion"] = dict(subset="questions with >= 1 hd==1 game (2,801)", delta_F_site_minus_F_opp_x=main, verdict=verdict, rule="delta >= .003 and lo > 0 -> established (v10e); lo > 0 and delta < .003 -> small gain; else not adopted (keep v10d)")
    res["scores_hd1_subset"] = pack(rs, sc_names); res["scores_all4568"] = pack(ra, sc_names)
    res["info_all4568_delta_F_site_minus_F_opp_x"] = pair(ra, "F_site", "F_opp_x")
    res["info_hd1_subset_delta_vs_F_E79"] = {a: diff(rs, a, "F_E79") for a in ("F_opp", "F_opp_x", "F_site")}; res["info_all4568_delta_vs_F_E79"] = {a: diff(ra, a, "F_E79") for a in ("F_opp", "F_opp_x", "F_site")}
    res["info_nested"] = dict(all4568=dict(F_site_nested_minus_F_opp_x_nested=pair(ra, "F_site_nested", "F_opp_x_nested")), hd1_subset=dict(F_site_nested_minus_F_opp_x_nested=pair(rs, "F_site_nested", "F_opp_x_nested")),
                              chosen_per_part=dict(F_opp_x=chx, F_site=chs))
    part_full = np.full(ny, -1); part_full[idx4] = ctx["part"]
    def part_delta(mask): return [float(bal(Sc["F_site"], R["y"], R["ns"], (part_full == j) & mask) - bal(Sc["F_opp_x"], R["y"], R["ns"], (part_full == j) & mask)) for j in range(5)]
    res["info_per_part_delta"] = dict(all4568=part_delta(m4), hd1_subset=part_delta(msub))
    # by the questioned player's TRUE rank
    ytrue = np.full(ny, -1); ytrue[idx4] = yq; by = {}
    for k in range(10):
        for nm, base in (("hd1_subset", msub), ("all4568", m4)):
            mk = base & (ytrue == k); n = int(mk.sum())
            if n == 0: continue
            r = boot({a: Sc[a] for a in ("F_site", "F_opp_x")}, R["F"], mk, "balanced", S.TAG)
            by.setdefault(nm, {})[RANKS[k]] = dict(n_questions=n, n_players=int(len(np.unique(ctx["plq"][(mk[idx4])]))), mean_hit_F_opp_x=float(r["F_opp_x"][0]), mean_hit_F_site=float(r["F_site"][0]), **diff(r, "F_site", "F_opp_x"))
    res["info_by_true_rank"] = by
    # counts
    valid = rows >= 0; ii = np.searchsorted(rid, np.where(valid, rows, rid[0])); slot_hd = np.where(valid, hd[ii], -1); slot_zero = np.where(valid, zero_game[ii], False)
    nz_games = int(zero_game.sum()); hd1_games = int((game["hd"] == 1).sum())
    zero_by_color = {"own_black(eff0)": int((zero_game & (game["eff"] == 0)).sum()), "own_white(eff1)": int((zero_game & (game["eff"] == 1)).sum())}
    res["counts"] = dict(n_games_total=int(len(game["hd"])), n_games_hd1=hd1_games, share_games_hd1=hd1_games / len(game["hd"]), n_hd1_games_zero_row_no_vote=nz_games, zero_row_by_color=zero_by_color,
                         n_hd1_games_with_site_L_sum_lt_1e12=None, n_bag_slots_fold4_valid=int(valid.sum()), n_bag_slots_hd1=int((slot_hd == 1).sum()), share_bag_slots_hd1=float((slot_hd == 1).sum() / valid.sum()),
                         n_bag_slots_hd1_zero_row=int(((slot_hd == 1) & slot_zero).sum()), share_hd1_bag_slots_zero_row=float(((slot_hd == 1) & slot_zero).sum() / max(1, (slot_hd == 1).sum())),
                         n_questions_any_hd1=int(q_hd1.sum()), share_questions_any_hd1=float(q_hd1.mean()),
                         frac_questions_with_vote=dict(F_site=float(hs.mean())), eff_values=sorted(map(int, np.unique(game["eff"]))), hd_values=sorted(map(int, np.unique(game["hd"]))))
    # tiny-sum diagnostic (how many hd1 games have raw sum(L) < 1e-12 before renormalisation but > 0)
    raw_b = ro_opp["lift"] @ K_BLACK.T; raw_w = ro_opp["lift"] @ K_WHITE.T; rs_ = np.where(game["eff"] == 0, raw_b.sum(1), raw_w.sum(1)); m1 = game["hd"] == 1
    res["counts"]["n_hd1_games_with_site_L_sum_lt_1e12"] = int((rs_[m1] < 1e-12).sum())
    # hd1 own-colour mix, for the reader
    res["counts"]["hd1_games_by_color"] = {"own_black(eff0)": int(((game["hd"] == 1) & (game["eff"] == 0)).sum()), "own_white(eff1)": int(((game["hd"] == 1) & (game["eff"] == 1)).sum())}
    sha1 = hashlib.sha256((ROOT / ".scratch/e96/result_v4.json").read_bytes()).hexdigest()[:16]; res["result_v4_sha16_before_after"] = [sha0, sha1]; assert sha0 == sha1
    with open(fn_res, "x") as f: json.dump(res, f, indent=1, default=float)
    print("PRIMARY", json.dumps(res["primary_criterion"], default=float)); print("ALL4568 delta", json.dumps(res["info_all4568_delta_F_site_minus_F_opp_x"], default=float)); print("result_v4.json sha16 after:", sha1)


if __name__ == "__main__":
    main(sys.argv[1])
