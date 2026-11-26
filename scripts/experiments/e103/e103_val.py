"""E103 validation chain (fold 4, 4,568 questions).  Plan (frozen): experiments/E103_rank_small_signal_merge/PLAN.md section 2.  Shared maths: e103_core.py.
Modes (all CPU):
  repro        : reproduce the old numbers one component at a time (REF .7363 vs posthoc_x_fixed.json; C1 only, no votes = E98 F .7081; C2 only .7407; C3 only 2,801 .7421 / all Delta +.0024) -> .scratch/e103/repro_val.json
                 does NOT compute ALL, leave-one-out or C1-with-votes.
  dry-noise    : full wiring with NOISE C1 features, SHUFFLED C2 neighbour votes; reports Delta vs REF only for noise worlds                      -> .scratch/e103/dry_val_noise.json
  dry-signal   : same but C1 features with a PLANTED signal (labels used; dry only) -> Delta must be clearly > 0                               -> .scratch/e103/dry_val_signal.json
  formal       : E103_FORMAL=1 required.  Repro assertion first (stops if any number is off), then REF, C1, C2, C3, ALL, three leave-one-out arms, paired player-level bootstrap (4,000), PLAN verdict.
                 Written ONCE with open(..., "x") to .scratch/e103/result_val.json (+ val_preds.npz with the per-question answers for the rehearsal in e103_make.py).
  CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 env PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/experiments/e103/e103_val.py <mode>
"""
import os, sys, json, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import e103_core as C
from e103_core import ROOT, OUT, S96, E1, E2

TAG = "e103_small_signal_merge"; NB = 4000
EXPORT = ROOT / ".scratch/e98/exports/ft4_s0_val12.npz"; E98_RES = ROOT / ".scratch/e98/result_ft4_val12.json"; POSTHOC = ROOT / ".scratch/e96/posthoc_x_fixed.json"
E101_RES = ROOT / ".scratch/e101/result.json"; E102_RES = ROOT / ".scratch/e102/result.json"; NNC = ROOT / ".scratch/e102/nn_cache/real/t3_game_top50.npz"
TOL = 1e-9


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================================================ setup
def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    S, R, game, ctx, ro_opp, real = E1.load_everything()
    sl_e2 = E2.slots(ctx, game); rid = ctx["rid"]; rows = ctx["rows"]; valid = rows >= 0
    pos = np.where(valid, np.searchsorted(rid, np.where(valid, rows, rid[0])), -1)
    sl = C.slots_from_rows(pos, game["hd"], game["eff"])
    for k in ("idx", "valid", "vote", "eff", "hd"): assert np.array_equal(sl[k][sl["valid"]] if k != "valid" else sl[k], sl_e2[k][sl["valid"]] if k != "valid" else sl_e2[k]), k
    assert np.array_equal(sl["idx"][sl["vote"]], sl_e2["idx"][sl_e2["vote"]])
    ny = len(R["y"]); idx4 = ctx["idx4"]; m4 = R["fold"] == 4
    def full(p):
        o = np.full((ny, 10), .1); o[idx4] = p; return o
    X = np.load(EXPORT); qid = R["Qf"].q.qid.to_numpy()[R["fold"] == 4].astype(str); assert (X["qid"].astype(str) == qid).all(), "export qid != fold-4 questions"
    Xa = np.asarray(X["q_probs"], np.float64); assert Xa.shape == (len(qid), 10) and np.isfinite(Xa).all() and np.allclose(Xa.sum(1), 1, atol=1e-4)
    return dict(R=R, game=game, ctx=ctx, ro_opp=ro_opp, real=real, sl=sl, sl_e2=sl_e2, full=full, m4=m4, Xa=Xa, lib=ro_opp["lift"])


def load_nn(G):
    """C2 neighbour votes from the E102 cache (top-50 neighbours of each queried validation game in the v4 bank, hd <= 1 rows), beta 40.  Returns g_nn [Gn,10] (zero rows where no neighbour vote), qg, T_bank."""
    sl = G["sl_e2"]; qg = np.unique(sl["idx"][sl["vote"]]); Gn = len(G["game"]["hd"])
    z = np.load(NNC); key = str(z["key"]).split("|"); nn_row, nn_cos = z["nn_row"], z["nn_cos"]
    meta = np.load(E2.OUT / "bank_meta.npz"); keep, T_bank = C.bank_T({k: meta[k] for k in ("hd", "eff", "owner_rank")})
    sj = {s: json.loads(Path(str(E2.OUT / f"S_bank_{s}_v4.npz") + ".json").read_text()) for s in ("own", "opp")}
    assert key[1] == "t3" and key[2] == E2.sha_arr(G["real"]["opp"][qg]) and key[4] == str(len(keep)) and key[5] == f"k{C.K_NN}", key
    assert nn_row.shape == (len(qg), C.K_NN) and nn_row.max() < len(keep) and (nn_cos[:, :-1] >= nn_cos[:, 1:] - 1e-6).all()
    g_nn = np.zeros((Gn, 10)); g_nn[qg] = C.nn_game_lift(nn_row, nn_cos, T_bank, C.BETA)
    return g_nn, qg, T_bank, dict(n_queried_games=int(len(qg)), n_bank_used=int(len(keep)), cache_key_head=key[0], n_games_with_nn_vote=int((g_nn.sum(1) > 0).sum()))


def arms_P(base0, base1, G, g_nn):
    """all eight arms as fused question probabilities (fixed setting).  names: REF C1 C2 C3 ALL noC1 noC2 noC3 (+ C1_novote = base1 itself)."""
    sl, lib = G["sl"], G["lib"]; P = {}
    def A(base, nn, site): return C.candidate_P(base, sl, lib, nn, C.W_NN, site)[0]
    P["REF"] = A(base0, None, False); P["C1"] = A(base1, None, False); P["C2"] = A(base0, g_nn, False); P["C3"] = A(base0, None, True)
    P["ALL"] = A(base1, g_nn, True); P["noC1"] = A(base0, g_nn, True); P["noC2"] = A(base1, None, True); P["noC3"] = A(base1, g_nn, False)
    return P


# ============================================================================================================================ repro
def repro_table(G, g_nn, FF, chosen, write=True):
    R, ctx, sl, lib, full, m4 = G["R"], G["ctx"], G["sl"], G["lib"], G["full"], G["m4"]; S = S96; f79q = ctx["f79q"]; out = {}
    rg = np.random.default_rng(3); xl_ = rg.dirichlet(np.ones(10), 500); ef_ = rg.integers(0, 2, 500); hf_ = rg.integers(0, 3, 500); a_ = C.shift_raw(xl_, ef_, hf_); m_ = hf_ == 1
    a_[m_] /= a_[m_].sum(1, keepdims=True); assert np.abs(a_ - S.shift_lift(xl_, ef_, hf_)).max() < 1e-15, "shift_raw != e96 shift_lift before its renormalisation"
    # ---- REF
    ro_x = dict(pmax=G["ro_opp"]["pmax"], lift=S.shift_lift(G["ro_opp"]["lift"], G["game"]["eff"], G["game"]["hd"]), top1_rank=G["ro_opp"]["top1_rank"])
    V0, _ = S.question_votes(ctx["rows"], ctx["rid"], ctx["hd"], ro_x, C.GM, C.LO, C.HI, (0, 1)); P_orig = S.fuse(f79q, V0, C.LAM, C.AL)
    P_ref, Vref, _ = C.candidate_P(f79q, sl, lib)
    Sc = {"REF": R["sc"](full(P_ref)), "REF_orig": R["sc"](full(P_orig))}
    ph = json.loads(POSTHOC.read_text()); out["REF"] = dict(expected=ph["scores"]["F_opp_x"]["score"], got=float(C.bal(Sc["REF"], R["y"], R["ns"], m4)), max_abs_dP_vs_e96_path=float(np.abs(P_ref - P_orig).max()),
                                                         n_answer_diff_vs_e96_path=int((C.decide_ev(P_ref) != C.decide_ev(P_orig)).sum()),
                                                         per_N_expected=ph["scores"]["F_opp_x"]["per_N"])
    # ---- C1 only, no votes (= E98 F, arm a, 12 positions)
    e98 = json.loads(E98_RES.read_text())["fusion_main"]; Sc["C1_novote"] = R["sc"](full(FF))
    out["C1_novote"] = dict(expected=e98["scores"]["F"]["score"], got=float(C.bal(Sc["C1_novote"], R["y"], R["ns"], m4)), C_expected=e98["info"]["C_chosen"], C_got=[c["C"] for c in chosen],
                            z_expected=e98["info"]["z_chosen"], z_got=[c["z"] for c in chosen])
    # ---- C2 only (beta 40, w .5)
    P2, _, _ = C.candidate_P(f79q, sl, lib, g_nn, C.W_NN, False); Sc["C2"] = R["sc"](full(P2))
    r2 = json.loads(E102_RES.read_text()); E2.S = S96; Vs, _ = E2.votes_slot(S96, G["sl_e2"], G["ro_opp"]["lift"], np.where(G["sl_e2"]["vote"][..., None], g_nn[G["sl_e2"]["idx"]], 0.), C.W_NN); P2e = S96.fuse(f79q, Vs, C.LAM, C.AL)
    out["C2"] = dict(expected=r2["T3_each_combo_balanced_score_all_fold4"]["beta40_w0.5"], got=float(C.bal(Sc["C2"], R["y"], R["ns"], m4)), max_abs_dP_vs_e102_votes_slot=float(np.abs(P2 - P2e).max()),
                     n_answer_diff_vs_e102=int((C.decide_ev(P2) != C.decide_ev(P2e)).sum()), REF_expected_in_e102=r2["scores"]["REF"]["score"])
    # ---- C3 only (site rule): all 4,568 and the 2,801 questions with an hd == 1 game
    P3, _, _ = C.candidate_P(f79q, sl, lib, None, C.W_NN, True); Sc["C3"] = R["sc"](full(P3))
    r1 = json.loads(E101_RES.read_text()); hd1 = ((sl["hd"] == 1) & sl["valid"]).any(1); msub = np.zeros(len(R["y"]), bool); msub[ctx["idx4"][hd1]] = True
    assert int(hd1.sum()) == 2801
    out["C3"] = dict(expected_all4568=r1["scores_all4568"]["F_site"]["score"], got_all4568=float(C.bal(Sc["C3"], R["y"], R["ns"], m4)), expected_hd1_2801=r1["scores_hd1_subset"]["F_site"]["score"],
                     got_hd1_2801=float(C.bal(Sc["C3"], R["y"], R["ns"], msub)), expected_delta_all=r1["info_all4568_delta_F_site_minus_F_opp_x"]["delta"],
                     got_delta_all=float(C.bal(Sc["C3"], R["y"], R["ns"], m4) - C.bal(Sc["REF"], R["y"], R["ns"], m4)), expected_REF_hd1_2801=r1["scores_hd1_subset"]["F_opp_x"]["score"], got_REF_hd1_2801=float(C.bal(Sc["REF"], R["y"], R["ns"], msub)))
    ok = (abs(out["REF"]["got"] - out["REF"]["expected"]) < TOL and out["REF"]["max_abs_dP_vs_e96_path"] < 1e-9 and abs(out["C1_novote"]["got"] - out["C1_novote"]["expected"]) < TOL
          and out["C1_novote"]["C_got"] == out["C1_novote"]["C_expected"] and out["C1_novote"]["z_got"] == out["C1_novote"]["z_expected"]
          and abs(out["C2"]["got"] - out["C2"]["expected"]) < TOL and out["C2"]["max_abs_dP_vs_e102_votes_slot"] < 1e-9
          and abs(out["C3"]["got_all4568"] - out["C3"]["expected_all4568"]) < TOL and abs(out["C3"]["got_hd1_2801"] - out["C3"]["expected_hd1_2801"]) < TOL and abs(out["C3"]["got_delta_all"] - out["C3"]["expected_delta_all"]) < TOL
          and abs(out["C3"]["got_REF_hd1_2801"] - out["C3"]["expected_REF_hd1_2801"]) < TOL)
    out["all_reproduced_within_1e9"] = bool(ok)
    if write: (OUT / "repro_val.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def show_repro(o):
    print("REPRODUCTION (expected / got):")
    print("  REF (F_opp_x fixed)   ", o["REF"]["expected"], o["REF"]["got"], "| max|dP| vs e96 path", o["REF"]["max_abs_dP_vs_e96_path"], "answer diff", o["REF"]["n_answer_diff_vs_e96_path"])
    print("  C1 only, no votes (F)  ", o["C1_novote"]["expected"], o["C1_novote"]["got"], "| C", o["C1_novote"]["C_got"], "z", o["C1_novote"]["z_got"], "(expected same:", o["C1_novote"]["C_got"] == o["C1_novote"]["C_expected"], o["C1_novote"]["z_got"] == o["C1_novote"]["z_expected"], ")")
    print("  C2 only (b40 w.5)      ", o["C2"]["expected"], o["C2"]["got"], "| max|dP| vs e102 votes_slot", o["C2"]["max_abs_dP_vs_e102_votes_slot"], "answer diff", o["C2"]["n_answer_diff_vs_e102"])
    print("  C3 only, all 4568      ", o["C3"]["expected_all4568"], o["C3"]["got_all4568"], "| delta vs REF", o["C3"]["expected_delta_all"], o["C3"]["got_delta_all"])
    print("  C3 only, 2801 hd1 Qs   ", o["C3"]["expected_hd1_2801"], o["C3"]["got_hd1_2801"], "| REF on 2801", o["C3"]["expected_REF_hd1_2801"], o["C3"]["got_REF_hd1_2801"])
    print("  ALL REPRODUCED:", o["all_reproduced_within_1e9"])


# ============================================================================================================================ dry runs
def dry(kind):
    G = setup(); R, ctx, sl, full, m4 = G["R"], G["ctx"], G["sl"], G["full"], G["m4"]; yq, nq = ctx["yq"], ctx["nq"]; f79q = ctx["f79q"]
    g_nn, qg, T_bank, info = load_nn(G)
    rng = np.random.default_rng(12345); n = len(yq)
    Xn = rng.dirichlet(np.ones(10), size=n)
    if kind == "signal": Xn[np.arange(n), yq] += .12; Xn /= Xn.sum(1, keepdims=True)                                   # PLANTED (labels used) - dry run only
    FF, _, ch = C.c1_nested(R, Xn); assert FF.shape == (n, 10) and np.allclose(FF.sum(1), 1)
    # shuffled C2: permute the neighbour-vote rows among the queried games (null world: votes unrelated to the game)
    rs = np.random.default_rng(1020); g_sh = np.zeros_like(g_nn); g_sh[qg] = g_nn[qg][rs.permutation(len(qg))]
    # wiring checks (no scores): shapes, alignment, normalisation, switches
    x, nnp = C.slot_x(sl, G["lib"], g_sh, C.W_NN, True); assert x.shape == (*sl["idx"].shape, 10) and np.allclose(x[sl["vote"]].sum(1), 1) and (x[~sl["vote"]] == 0).all()
    Va, _ = C.votes(sl, G["lib"]); Vb, _ = C.votes(sl, G["lib"], g_sh, 0.); assert np.array_equal(Va, Vb), "w = 0 must equal no neighbour vote"
    Vc, _ = C.votes(sl, G["lib"], np.zeros_like(g_sh), C.W_NN); assert np.abs(Va - Vc).max() < 1e-12, "all-zero neighbour votes must equal no neighbour vote"
    Vs, _ = C.votes(sl, G["lib"], None, 0., False); assert np.allclose(Vs.sum(1), 1)
    log("wiring checks OK: x shape", x.shape, "| slots with nn vote:", int(nnp.sum()), "| questions with vote:", int(C.votes(sl, G["lib"])[1].sum()))
    Pr = {}; sc = lambda p: R["sc"](full(p))
    base0 = f79q
    Pr["REF"] = C.candidate_P(base0, sl, G["lib"])[0]
    if kind == "noise":
        Pr["noise_C1+shufC2_C3off"] = C.candidate_P(FF, sl, G["lib"], g_sh, C.W_NN, False)[0]
        Pr["noise_C1+shufC2_C3on"] = C.candidate_P(FF, sl, G["lib"], g_sh, C.W_NN, True)[0]
        Pr["C3_alone_for_reference"] = C.candidate_P(base0, sl, G["lib"], None, C.W_NN, True)[0]
    else:
        Pr["signalC1+shufC2_C3off"] = C.candidate_P(FF, sl, G["lib"], g_sh, C.W_NN, False)[0]
        Pr["signalC1_only_C3off"] = C.candidate_P(FF, sl, G["lib"])[0]
    Sc = {k: sc(p) for k, p in Pr.items()}; r = C.boot(Sc, R["F"], m4, "balanced", TAG)
    res = dict(kind=kind, C1_chosen=ch, wiring="OK", info=info, scores={k: dict(score=r[k][0], **C.ci(r[k][2])) for k in r}, delta_vs_REF={k: C.diff(r, k, "REF") for k in r if k != "REF"})
    (OUT / f"dry_val_{kind}.json").write_text(json.dumps(res, indent=1, default=float))
    print("DRY", kind, "(noise / shuffled votes; NOT the real ALL):")
    for k, d in res["delta_vs_REF"].items(): print(f"   {k:28s} delta vs REF {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  boot>0 {d['frac_boot_gt_0']:.3f}")
    print("   C1 (C, z) per part", [(c["C"], c["z"]) for c in ch])


# ============================================================================================================================ formal
def formal():
    assert os.environ.get("E103_FORMAL") == "1", "the formal run is a deliberate step: set E103_FORMAL=1"
    fn = OUT / "result_val.json"; assert not fn.exists(), "result_val.json exists: the formal chain runs ONCE - refusing"
    sha_v4 = hashlib.sha256((ROOT / ".scratch/e96/result_v4.json").read_bytes()).hexdigest()[:16]
    G = setup(); R, ctx, sl, full, m4 = G["R"], G["ctx"], G["sl"], G["full"], G["m4"]; yq, nq = ctx["yq"], ctx["nq"]; f79q = ctx["f79q"]
    g_nn, qg, T_bank, info = load_nn(G)
    FF, PS, ch = C.c1_nested(R, G["Xa"])
    rep = repro_table(G, g_nn, FF, ch, write=True); show_repro(rep)
    if not rep["all_reproduced_within_1e9"]: print("REPRODUCTION FAILED - STOP, nothing merged"); sys.exit(3)
    P = arms_P(f79q, FF, G, g_nn); names = ["REF", "C1", "C2", "C3", "ALL", "noC1", "noC2", "noC3"]
    Sc = {k: R["sc"](full(P[k])) for k in names}; Sc["C1_novote"] = R["sc"](full(FF))
    r = C.boot(Sc, R["F"], m4, "balanced", TAG)
    pn = lambda a, b: {str(n): float(r[a][1][n] - r[b][1][n]) for n in r[a][1]}
    part_full = np.full(len(R["y"]), -1); part_full[ctx["idx4"]] = ctx["part"]
    def pair(a, b):
        d = C.diff(r, a, b); d["per_N"] = pn(a, b)
        d["per_part"] = [float(C.bal(Sc[a], R["y"], R["ns"], part_full == j) - C.bal(Sc[b], R["y"], R["ns"], part_full == j)) for j in range(5)]; return d
    res = dict(mode="formal", plan="experiments/E103_rank_small_signal_merge/PLAN.md", plan_sha16=hashlib.sha256((ROOT / "experiments/E103_rank_small_signal_merge/PLAN.md").read_bytes()).hexdigest()[:16],
               script_sha16=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16], core_sha16=hashlib.sha256((Path(__file__).parent / "e103_core.py").read_bytes()).hexdigest()[:16],
               fixed=dict(gamma=C.GM, tau="none", alpha=C.AL, lam=C.LAM, beta=C.BETA, w=C.W_NN, k_nn=C.K_NN), n_questions=int(len(yq)), n_players=int(len(np.unique(ctx["plq"]))), n_bootstrap=NB, boot_tag=TAG,
               repro=rep, nn_info=info, c1_nested_chosen=ch,
               scores={k: dict(score=r[k][0], per_N={str(n): v for n, v in r[k][1].items()}, **C.ci(r[k][2])) for k in r},
               delta_vs_REF={k: pair(k, "REF") for k in ("C1", "C2", "C3", "ALL", "noC1", "noC2", "noC3")},
               leave_one_out_ALL_minus=dict(C1=pair("ALL", "noC1"), C2=pair("ALL", "noC2"), C3=pair("ALL", "noC3")),
               note_C1_novote=dict(score=r["C1_novote"][0], delta_vs_REF=C.diff(r, "C1_novote", "REF")["delta"], delta_vs_F_E79_info=float(r["C1_novote"][0] - C.bal(R["sc"](R["F79"]), R["y"], R["ns"], m4))))
    pr = {k: C.decide_ev(P[k]) for k in names}; r_val = float((pr["ALL"] != pr["REF"]).mean())
    res["r_val"] = dict(value=r_val, n_questions_changed=int((pr["ALL"] != pr["REF"]).sum()), of=int(len(yq)), by_N={str(n): float((pr["ALL"] != pr["REF"])[nq == n].mean()) for n in C.NS},
                        shift_ALL_minus_REF=dict(plus1=int(((pr["ALL"] - pr["REF"]) == 1).sum()), minus1=int(((pr["ALL"] - pr["REF"]) == -1).sum()), other=int((np.abs(pr["ALL"] - pr["REF"]) > 1).sum())))
    d_all = res["delta_vs_REF"]["ALL"]; loo = res["leave_one_out_ALL_minus"]
    rule1 = d_all["delta"] > 0; rule3 = [k for k, v in loo.items() if v["hi"] < 0]
    verdict = ("RULE_3_STOP_report_to_user(component drags: leave-one-out CI upper < 0 for %s)" % rule3) if rule3 else ("RULE_1_make_candidate(ALL - REF > 0)" if rule1 else "RULE_2_no_file(ALL - REF <= 0)")
    res["plan_verdict"] = dict(rule1_ALL_minus_REF_gt_0=bool(rule1), rule2_ALL_minus_REF_le_0=bool(not rule1), rule3_components_with_loo_upper_lt_0=rule3, verdict=verdict,
                               rule_text="1: ALL-REF > 0 -> make candidate (no CI requirement); 2: ALL-REF <= 0 -> no file, report; 3: any leave-one-out difference with 95% upper < 0 -> stop and report (takes precedence for the action)")
    res["result_v4_sha16_before_after"] = [sha_v4, hashlib.sha256((ROOT / ".scratch/e96/result_v4.json").read_bytes()).hexdigest()[:16]]; assert res["result_v4_sha16_before_after"][0] == res["result_v4_sha16_before_after"][1]
    with open(fn, "x") as f: json.dump(res, f, indent=1, default=float)
    with open(OUT / "val_preds.npz", "xb") as f: np.savez(f, qid=G["R"]["Qf"].q.qid.to_numpy()[R["fold"] == 4].astype(str), **{f"pred_{k}": v for k, v in pr.items()}, **{f"P_{k}": P[k] for k in names}, FF=FF, nq=nq, yq_not_for_scoring_here=yq)
    print("scores", {k: round(v["score"], 4) for k, v in res["scores"].items()})
    for k, v in res["delta_vs_REF"].items(): print(f"  {k:5s} - REF  {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] boot>0 {v['frac_boot_gt_0']:.3f}  perN {({a: round(b, 4) for a, b in v['per_N'].items()})}")
    for k, v in loo.items(): print(f"  ALL - no{k} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] boot>0 {v['frac_boot_gt_0']:.3f}")
    print("r_val", res["r_val"]); print("VERDICT", res["plan_verdict"]["verdict"]); print("written", fn)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "repro":
        G = setup(); g_nn, qg, T_bank, info = load_nn(G); FF, PS, ch = C.c1_nested(G["R"], G["Xa"]); o = repro_table(G, g_nn, FF, ch); show_repro(o)
    elif mode in ("dry-noise", "dry-signal"): dry(mode[4:])
    elif mode == "formal": formal()
    else: raise SystemExit("modes: repro dry-noise dry-signal formal")
