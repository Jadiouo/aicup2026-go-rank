"""E124 validation chain (fold 4, 4,568 questions).  Plan (frozen): experiments/E124_rank_stage2_shift/PLAN.md.  Maths: e124_core.py.  CPU only.
Modes (dry checks never print or store a shift-model validation score):
  dry-repro : (1) ALL / REF reproduction                                                        -> .scratch/e124/dry_repro.json
  dry-lr    : (2) sklearn LR C=1 nested == E117 .7541068, 0 answers differ from val_preds_e117 P_LR@0.5   -> dry_lr.json
  dry-opt   : (3) own optimiser as full-matrix LR vs sklearn LR; finite-difference gradients; convergence -> dry_opt.json
  dry-shift : (4) shift-invariance unit test + synthetic recovery (synthetic data only)           -> dry_shift.json
  formal    : E124_FORMAL=1, ONCE: .scratch/e124/result_val.json (open "x")
"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import e124_core as K
from e124_core import OUT, ROOT, log

EXPECT = dict(ALL=.746988768570657, REF=.7363474548692714)
E117_RES = ROOT / ".scratch/e117/result_val.json"; E117_VP = ROOT / ".scratch/e117/val_preds_e117.npz"
SHUF_SEEDS = (0, 1, 2)


def jdump(name, o):
    with open(OUT / name, "w") as f: json.dump(o, f, indent=1, default=float)


def dry_repro(D):
    got = dict(ALL=K.bal_full(D, K.full_sc(D, D["P_ALL"])), REF=K.bal_full(D, K.full_sc(D, D["P_REF"])))
    ans = bool((K.decide_ev(D["P_ALL"]) == D["pred_ALL"]).all() and (K.decide_ev(D["P_REF"]) == D["pred_REF"]).all())
    o = dict(check="(1) reproduce REF/ALL", expected=EXPECT, got=got, exact_equal={k: bool(got[k] == EXPECT[k]) for k in got}, answers_equal_stored_pred=ans, n_questions=int(len(D["y"])), n_players=int(len(np.unique(D["pl"]))), part_sizes=np.bincount(D["part"]).tolist())
    o["pass_"] = bool(all(o["exact_equal"].values()) and ans and o["n_questions"] == 4568 and o["n_players"] == 1420); jdump("dry_repro.json", o); print("DRY-REPRO", got, "PASS", o["pass_"]); return o


def dry_lr(D):
    X = K.make_X(D["Pk"], D["ns"]); y = D["y"]
    P = K.nested_lr(D, X, y, [1.])[1.]; Pc = K.combine(D["P_ALL"], P, .5); sc = K.bal_full(D, K.full_sc(D, Pc)); ans = K.decide_ev(Pc)
    e117 = json.loads(E117_RES.read_text())["scores"]["LR@0.5"]["score"]; vp = np.load(E117_VP); ans117 = K.decide_ev(np.asarray(vp["P_LR@0.5"], np.float64))
    # also the stored E117 answers
    nd = int((ans != ans117).sum()); pdiff = float(np.abs(Pc - np.asarray(vp["P_LR@0.5"], np.float64)).max())
    o = dict(check="(2) sklearn LR C=1 nested reproduces E117 LR@0.5", score=sc, e117_result_val_LR_at_0p5=e117, expected_task=.7541068, abs_diff_vs_e117=abs(sc - e117), n_answers_differ_vs_val_preds_e117=nd, max_abs_prob_diff=pdiff)
    o["pass_"] = bool(abs(sc - e117) < 1e-6 and abs(sc - .7541068) < 5e-7 and nd == 0); jdump("dry_lr.json", o); print("DRY-LR", sc, "e117", e117, "answers differ", nd, "maxprobdiff %.2e" % pdiff, "PASS", o["pass_"]); return o


def fd_check(f, n_par, seed, h=1e-5):
    rng = np.random.default_rng(seed); th = rng.normal(0, .3, n_par); _, g = f(th); gf = np.zeros(n_par)
    for i in range(n_par):
        e = np.zeros(n_par); e[i] = h; gf[i] = (f(th + e)[0] - f(th - e)[0]) / (2 * h)
    return float(np.abs(g - gf).max() / (np.abs(g).max() + 1e-12)), float(np.linalg.norm(g - gf) / (np.linalg.norm(g) + np.linalg.norm(gf)))


def synth(n, G, seed):
    rng = np.random.default_rng(seed); Pk = {f"g{i}": rng.dirichlet(np.ones(10) * .5, n) for i in range(G)}
    ns = rng.choice(K.N_LEVELS, n); y = rng.integers(0, 10, n); return Pk, ns, y


def dry_opt(D):
    out = {}
    # gradient finite differences on SYNTHETIC data (no validation score involved)
    for G, k in ((6, 2), (7, 2), (6, 1)):
        Pk, ns, y = synth(300, G, 7 + G + k); groups = list(Pk); L = K.log_stack(Pk, groups); Z = (L - L.mean()) / L.std()
        w = np.random.default_rng(3).uniform(.3, 2, 300); f, npar = K.shift_objective(K.kernel_design(Z, k), K.n_index(ns), y, w, 3.)
        out[f"fd_shift_G{G}_k{k}"] = dict(zip(("max_rel", "norm_rel"), fd_check(f, npar, 11)))
    Pk, ns, y = synth(200, 3, 5); X = K.make_X({**{kk: Pk[f"g{i}"] for i, kk in enumerate(K.KEYS[:3])}, **{kk: Pk["g0"] for kk in K.KEYS[3:]}}, ns)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9); F = Z.shape[1]; w = np.random.default_rng(4).uniform(.3, 2, 200)
    fwd = lambda th: Z @ th[:F * 10].reshape(F, 10) + th[F * 10:]; bwd = lambda G_: np.concatenate([(Z.T @ G_).reshape(-1), G_.sum(0)]); pm = np.zeros(F * 10 + 10, bool); pm[:F * 10] = True
    out["fd_fullLR"] = dict(zip(("max_rel", "norm_rel"), fd_check(K.make_objective(fwd, bwd, y, w, pm, 1.), F * 10 + 10, 12)))
    fd_ok = all(v["max_rel"] < 1e-5 for v in out.values())
    # own optimiser as full-matrix LR vs sklearn LR on the real nested chain (LR scores only)
    X = K.make_X(D["Pk"], D["ns"]); y = D["y"]; K.FITLOG.clear()
    Ps = K.nested_lr(D, X, y, [1.], "sk")[1.]; Po = K.nested_lr(D, X, y, [1.], "own")[1.]
    Cs, Co = K.combine(D["P_ALL"], Ps, .5), K.combine(D["P_ALL"], Po, .5)
    ssk, sow = K.bal_full(D, K.full_sc(D, Cs)), K.bal_full(D, K.full_sc(D, Co)); nd = int((K.decide_ev(Cs) != K.decide_ev(Co)).sum())
    out["own_fullLR_vs_sklearn"] = dict(score_sklearn=ssk, score_own=sow, abs_diff=abs(ssk - sow), n_answers_differ=nd, max_abs_prob_diff_raw=float(np.abs(Ps - Po).max()))
    out["lbfgs_fullLR_fits"] = list(K.FITLOG); conv = all(f["success"] for f in K.FITLOG)
    out["criteria"] = dict(fd_rel_lt_1e_5=bool(fd_ok), score_diff_le_0p0005=bool(abs(ssk - sow) <= .0005), answers_differ_le_3=bool(nd <= 3), lbfgs_reported_converged=bool(conv))
    out["pass_"] = bool(all(out["criteria"].values())); jdump("dry_opt.json", dict(check="(3) own optimiser", **out))
    print("DRY-OPT fd", {k: "%.1e" % v["max_rel"] for k, v in out.items() if k.startswith("fd_")}, "own-vs-sk score diff %.2e answers differ %d" % (abs(ssk - sow), nd), "conv", [(f["success"], "%.1e" % f["grad_inf"]) for f in K.FITLOG], "PASS", out["pass_"]); return out


def dry_shift():
    out = {}; rng = np.random.default_rng(124); k = 2; G = 6
    Z = rng.normal(size=(40, G, 10)); a = rng.normal(size=(G, 2 * k + 1)); Zs = np.zeros_like(Z); Zs[:, :, 1:] = Z[:, :, :-1]; Zs[:, :, 0] = rng.normal(size=(40, G))
    l0 = K.kernel_logit(Z, a, k); l1 = K.kernel_logit(Zs, a, k); cs = list(range(k, 9 - k)); err = max(float(np.abs(l1[:, c + 1] - l0[:, c]).max()) for c in cs)
    out["kernel_level_shift_by_one"] = dict(classes_tested=cs, max_abs_err=err)
    # model level (bias zero, fixed mu/sd): shift the log-probability inputs one step along the rank axis
    Pk, ns, y = synth(60, 6, 21); m = K.ShiftS2(groups=list(Pk)); m.mu, m.sd = -2.3, 1.1; m.a = rng.normal(size=(6, 5)); m.B = np.zeros((4, 10))
    Pk2 = {kk: np.concatenate([rng.random((60, 1)) + .1, v[:, :-1]], 1) for kk, v in Pk.items()}
    q0, q1 = m.logits(Pk, ns), m.logits(Pk2, ns); err2 = max(float(np.abs(q1[:, c + 1] - q0[:, c]).max()) for c in cs)
    out["model_level_shift_by_one"] = dict(max_abs_err=err2)
    # boundary behaviour is NOT invariant (clip): report only that the test is non-trivial
    out["boundary_class_differs_as_expected"] = bool(np.abs(l1[:, 1 + 0] - l0[:, 0]).max() > 1e-3)
    # synthetic recovery: labels drawn from a known kernel model, large C => fit recovers it (shape + convergence + optimiser sanity)
    Pk, ns, _ = synth(6000, 6, 33); gs = list(Pk); mt = K.ShiftS2(groups=gs); mt.mu, mt.sd = -2.3, 1.1; mt.a = rng.normal(0, .35, (6, 5)); mt.B = rng.normal(0, .3, (4, 10))
    P = mt.predict(Pk, ns); cum = P.cumsum(1); ys = np.minimum((rng.random(6000)[:, None] > cum).sum(1), 9)
    K.FITLOG.clear(); mf = K.ShiftS2(groups=gs, C=1e6).fit(Pk, ns, ys, np.ones(6000), tag="synthetic")
    # the fit standardises with ITS OWN mu/sd; rescale to compare: a_fit * (1/sd_fit) vs a_true * (1/sd_true) is the coefficient on log P
    ca_t, ca_f = mt.a / mt.sd, mf.a / mf.sd
    out["synthetic_recovery"] = dict(shape_a=list(mf.a.shape), shape_B=list(mf.B.shape), lbfgs=K.FITLOG[-1], max_abs_err_coef_on_logP=float(np.abs(ca_t - ca_f).max()), max_abs_coef_true=float(np.abs(ca_t).max()))
    out["criteria"] = dict(kernel_shift_err_lt_1e_12=bool(err < 1e-12), model_shift_err_lt_1e_12=bool(err2 < 1e-12), boundary_nontrivial=out["boundary_class_differs_as_expected"],
                          shapes_ok=bool(mf.a.shape == (6, 5) and mf.B.shape == (4, 10)), synthetic_converged=bool(K.FITLOG[-1]["success"]), synthetic_recovery_close=bool(out["synthetic_recovery"]["max_abs_err_coef_on_logP"] < .25 * out["synthetic_recovery"]["max_abs_coef_true"]))
    out["pass_"] = bool(all(out["criteria"].values())); jdump("dry_shift.json", dict(check="(4) shift invariance + synthetic", **out))
    print("DRY-SHIFT", {k: v for k, v in out["criteria"].items()}, "errs", err, err2, "synthetic", out["synthetic_recovery"]["max_abs_err_coef_on_logP"], "PASS", out["pass_"]); return out


# ------------------------------------------------------------------------------------------------------------ formal
def shuffle_within_N(y, ns, seed):
    rng = np.random.default_rng(seed); ys = y.copy()
    for n in K.N_LEVELS:
        i = np.flatnonzero(ns == n); ys[i] = y[rng.permutation(i)]
    return ys


def formal(D):
    assert os.environ.get("E124_FORMAL") == "1", "formal run is deliberate: set E124_FORMAL=1"
    fn = OUT / "result_val.json"; assert not fn.exists(), "result_val.json exists: the formal run is ONCE"
    for f in ("dry_repro.json", "dry_lr.json", "dry_opt.json", "dry_shift.json", "dry_make.json"): assert (OUT / f).exists(), f"{f} missing: dry checks first"; assert json.loads((OUT / f).read_text())["pass_"], f"{f} not passed"
    y, ns, pl, part = D["y"], D["ns"], D["pl"], D["part"]; X = K.make_X(D["Pk"], ns); K.FITLOG.clear()
    P_sh = K.nested_shift(D, y, tag="shift"); fit_shift = list(K.FITLOG); K.FITLOG.clear()
    LRP = K.nested_lr(D, X, y, [1., 3., 10., 30.]); Pc = lambda P: K.combine(D["P_ALL"], P, K.RHO)
    arms = {"SHIFT": Pc(P_sh), "LR_C1": Pc(LRP[1.]), "LR_C3": Pc(LRP[3.]), "LR_C10": Pc(LRP[10.]), "LR_C30": Pc(LRP[30.])}
    pred = {"ALL": K.decide_ev(D["P_ALL"]), **{a: K.decide_ev(v) for a, v in arms.items()}}
    S = {"ALL": K.full_sc(D, D["P_ALL"]), **{a: K.full_sc(D, v) for a, v in arms.items()}}
    r = K.boot(S, D["F"], D["m4"], "balanced", K.TAG)
    dl = lambda a, b: dict(**K.diff(r, a, b), per_N={str(n): float(r[a][1][n] - r[b][1][n]) for n in r[a][1]})
    pairs = {f"SHIFT-{b}": dl("SHIFT", b) for b in ("LR_C1", "LR_C3", "LR_C10", "LR_C30", "ALL")}; pairs.update({f"{a}-ALL": dl(a, "ALL") for a in ("LR_C1", "LR_C3", "LR_C10", "LR_C30")})
    idx4 = D["idx4"]; dq = lambda a, b: (S[a] - S[b])[idx4]
    # per-question equal weight difference, player-cluster bootstrap
    u, g = np.unique(pl, return_inverse=True); Pn = len(u); Cb = np.random.default_rng(124).multinomial(Pn, np.full(Pn, 1.0 / Pn), size=K.NB).astype(np.float64); cnt = np.bincount(g, minlength=Pn).astype(float)
    def pq(a, b):
        d = dq(a, b); s = np.bincount(g, weights=d, minlength=Pn); bs = (Cb @ s) / (Cb @ cnt); return dict(delta=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), frac_boot_gt_0=float((bs > 0).mean()))
    per_q = {f"SHIFT-{b}": pq("SHIFT", b) for b in ("LR_C1", "LR_C3", "ALL")}
    def part_deltas(a, b):
        d = dq(a, b); raw, bl = [], []
        for j in range(5):
            m = np.zeros(D["ny"], bool); m[idx4[part == j]] = True; raw.append(float(d[part == j].mean())); bl.append(float(K.bal(S[a] - S[b], D["F"].truth, D["F"].ns, m)))
        return dict(raw_question_mean=raw, balanced=bl)
    by_part = {f"SHIFT-{b}": part_deltas("SHIFT", b) for b in ("LR_C1", "LR_C3", "ALL")}
    by_class = {f"SHIFT-{b}": [float(dq("SHIFT", b)[y == c].mean()) for c in range(10)] for b in ("LR_C1", "LR_C3", "ALL")}
    def changes(a, b):
        pa, pn = pred[b], pred[a]; ch = pa != pn; d = pn - pa
        return dict(n_changed=int(ch.sum()), plus1=int((d == 1).sum()), minus1=int((d == -1).sum()), beyond1=int((np.abs(d) > 1).sum()), by_N={str(n): int((ch & (ns == n)).sum()) for n in K.N_LEVELS},
                    wrong_to_hit=int(((pa != y) & (pn == y)).sum()), hit_to_wrong=int(((pa == y) & (pn != y)).sum()))
    chg = {f"SHIFT_vs_{b}": changes("SHIFT", b) for b in ("ALL", "LR_C1", "LR_C3")}
    # ---- control 3: labels shuffled inside every N
    sALL = K.bal_full(D, S["ALL"]); ctl3 = {}
    for sd in SHUF_SEEDS:
        ys = shuffle_within_N(y, ns, sd); assert (ys != y).mean() > .5; K.FITLOG.clear()
        Pz = K.combine(D["P_ALL"], K.nested_shift(D, ys, tag=f"shuf{sd}"), K.RHO); sz = K.bal_full(D, K.full_sc(D, Pz))
        ctl3[str(sd)] = dict(score=sz, delta_vs_ALL=sz - sALL, all_converged=bool(all(f["success"] for f in K.FITLOG))); log("control3 seed", sd, "delta vs ALL %+.4f" % (sz - sALL))
    # ---- control 4: P_ALL copy as a 7th group
    K.FITLOG.clear(); P7 = K.combine(D["P_ALL"], K.nested_shift(D, y, groups=tuple(K.KEYS) + ("P_ALL",), tag="dup"), K.RHO); s7 = K.bal_full(D, K.full_sc(D, P7)); fit_dup = list(K.FITLOG)
    ctl4 = dict(score_7groups=s7, score_6groups=r["SHIFT"][0], change=s7 - r["SHIFT"][0], n_answers_differ=int((K.decide_ev(P7) != pred["SHIFT"]).sum()))
    # ---- criterion 5: test side, full fit on all 4,568 questions, vs the v12 file
    import e124_make as M
    st = M.test_side(D, write=None); n_diff = st["n_differ_from_v12"]
    d1, d2 = pairs["SHIFT-LR_C1"], pairs["SHIFT-LR_C3"]
    crit = {"1_shift_minus_A_ge_.005_and_ci_lo_gt_0": bool(d1["delta"] >= .005 and d1["lo"] > 0), "2_shift_minus_B_ge_.002_and_boot_pos_ge_.85": bool(d2["delta"] >= .002 and d2["frac_boot_gt_0"] >= .85),
            "3_shuffle_all_three_le_.001": bool(all(v["delta_vs_ALL"] <= .001 for v in ctl3.values())), "4_dup_abs_change_le_.001": bool(abs(ctl4["change"]) <= .001), "5_test_differ_from_v12_le_95": bool(n_diff <= 95)}
    verdict = "E124_VAL PASS" if all(crit.values()) else "E124_VAL FAIL (" + ", ".join(k for k, v in crit.items() if not v) + ")"
    res = dict(mode="formal", plan="experiments/E124_rank_stage2_shift/PLAN.md", plan_sha16=K.sha16(ROOT / "experiments/E124_rank_stage2_shift/PLAN.md"), core_sha16=K.sha16(Path(K.__file__)), run_sha16=K.sha16(Path(__file__)), make_sha16=K.sha16(Path(M.__file__)), boot_tag=K.TAG, n_bootstrap=K.NB,
               fixed=dict(k=K.K_HALF, C=K.C_SHIFT, rho=K.RHO), n_questions=int(len(y)), n_players=int(len(np.unique(pl))),
               scores={a: dict(score=r[a][0], per_N={str(n): v for n, v in r[a][1].items()}, **K.ci(r[a][2])) for a in S}, paired_deltas=pairs, per_question_equal_deltas=per_q, delta_by_outer_part=by_part, per_true_rank_delta=dict(ranks=K.RANKS, n=[int((y == c).sum()) for c in range(10)], **by_class),
               changes=chg, control3_shuffle=ctl3, control4_duplicate_P_ALL=ctl4, test_side=st, criteria=crit, verdict=verdict, shift_fits_lbfgs=fit_shift, dup_fits_lbfgs=fit_dup,
               pred_counts={k: np.bincount(v, minlength=10).tolist() for k, v in pred.items()})
    with open(fn, "x") as f: json.dump(res, f, indent=1, default=float)
    print("scores", {a: round(r[a][0], 5) for a in S})
    for k_, v in pairs.items(): print(f"  {k_:16s} {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] boot>0 {v['frac_boot_gt_0']:.3f} perN {({kk: round(x, 4) for kk, x in v['per_N'].items()})}")
    print("per-question:", {k_: (round(v['delta'], 4), round(v['lo'], 4), round(v['hi'], 4)) for k_, v in per_q.items()}); print("by part:", {k_: [round(x, 4) for x in v['balanced']] for k_, v in by_part.items()})
    print("changes:", chg); print("control3:", {k_: round(v['delta_vs_ALL'], 4) for k_, v in ctl3.items()}, "control4:", ctl4); print("test side n differ from v12:", n_diff); print("criteria", crit)
    if all(crit.values()):
        with open(OUT / "val_preds_e124.npz", "xb") as f: np.savez(f, qid=D["Qf"].q.qid.to_numpy()[idx4].astype(str), P_SHIFT=arms["SHIFT"], P_LR_C1=arms["LR_C1"], P_LR_C3=arms["LR_C3"], P_ALL=D["P_ALL"])
    print("written", fn); print(verdict)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""; OUT.mkdir(parents=True, exist_ok=True); D = K.load_all()
    if mode == "dry-repro": sys.exit(0 if dry_repro(D)["pass_"] else 3)
    elif mode == "dry-lr": sys.exit(0 if dry_lr(D)["pass_"] else 3)
    elif mode == "dry-opt": sys.exit(0 if dry_opt(D)["pass_"] else 3)
    elif mode == "dry-shift": sys.exit(0 if dry_shift()["pass_"] else 3)
    elif mode == "formal": formal(D)
    else: raise SystemExit("modes: dry-repro dry-lr dry-opt dry-shift formal")
