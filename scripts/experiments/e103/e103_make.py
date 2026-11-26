"""E103 test-side candidate files (v11 family) with the three components switchable.  Plan (frozen): experiments/E103_rank_small_signal_merge/PLAN.md section 3.  Shared maths with the validation chain: e103_core.py.
  C1 : base = nrm((1 - z) F_E79 + z * LR(q_probs of the E98 ft4 export))  -- (C, z, LR) fitted ONCE on all 4,568 validation questions (`fit-c1`, saved to .scratch/e103/c1_fit.{json,npz})
  C2 : neighbour votes (top-50, beta 40, w .5) of every test game's OPPONENT-side vector against the TEST-version bank (all training games hd <= 1)
  C3 : E101 site rule for the hd == 1 library votes
  all off -> v10d exactly (800 / 800).  NEVER writes data/submissions (out dir must be under .scratch/).
Subcommands
  fit-c1                                      fit the C1 test-time model on the 4,568 validation questions (no scores printed)
  make  --name N [--c1 NPZ] [--c2 DIR] [--c3]  test-side candidate -> .scratch/e103/final/N.csv + stats_N.json
  rehearse                                    fold-4 validation questions as 'test': the same make-path functions (C1 from ft4_s0_val12.npz fitted on all questions, C2 with the v4 bank through a FRESH kNN, C3) -> compares ANSWERS with the
                                              validation chain (val_preds.npz); no scores
  CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 env PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/experiments/e103/e103_make.py <sub> ...
"""
import os, sys, json, argparse, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/experiments/e91"))
import numpy as np
import e103_core as C
from e103_core import ROOT, OUT, S96, E1, E2

FINAL = OUT / "final"; C1_NPZ = OUT / "c1_fit.npz"; C1_JSON = OUT / "c1_fit.json"; VAL_EXPORT = ROOT / ".scratch/e98/exports/ft4_s0_val12.npz"
V10D = ROOT / "data/submissions/rank_v10d_對手票含定先盤.csv"; V9 = ROOT / "data/submissions/rank_v9_逐局面注意力.csv"
HEAD_B = ROOT / ".scratch/e91/b/head_b.npz"


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================================================ C2 bank (test version or the v4 validation bank) and the neighbour query
def load_bank(d, opp_name, meta_name, expect_ckpt_sha16):
    d = Path(d); f = d / opp_name; fj = Path(str(f) + ".json"); fm = d / meta_name
    for p in (f, fj, fm): assert p.exists(), f"{p} missing"
    sj = json.loads(fj.read_text()); assert sj["ckpt_sha16"] == expect_ckpt_sha16, f"bank checkpoint {sj['ckpt_sha16']} != head {expect_ckpt_sha16}"
    M = np.load(fm); z = np.load(f); rid = z["rid"]; gv = z["gv_all"]
    assert len(rid) == len(M["rid"]) == len(gv) and (rid.astype(np.int64) == M["rid"].astype(np.int64)).all(), "bank rid != bank_meta rid"
    assert gv.ndim == 2 and gv.shape[1] == 128 and gv.dtype == np.float32
    sub = np.random.default_rng(0).choice(len(gv), min(20000, len(gv)), replace=False); assert np.isfinite(gv[sub]).all() and np.abs(np.linalg.norm(gv[sub], axis=1) - 1).max() < 1e-3
    keep, T = C.bank_T({k: M[k] for k in ("hd", "eff", "owner_rank")})
    opp_k = np.ascontiguousarray(gv[keep]); del gv
    return dict(opp_k=opp_k, T=T, keep=keep, n_total=int(len(rid)), n_used=int(len(keep)), file=str(f), sha16=hashlib.sha256(f.read_bytes()).hexdigest()[:16], ckpt_sha16=sj["ckpt_sha16"])


def query_bank(bank, opp_vec, sl):
    """neighbour vote per game: top-50 (cosine) of the game's opponent-side vector in bank['opp_k'], beta 40.  Only games that vote in some question are queried.  -> g_nn [G,10], qg, nn_row, nn_cos"""
    qg = np.unique(sl["idx"][sl["vote"]]); t = time.time(); nn_i, nn_c = E2.knn(np.ascontiguousarray(opp_vec[qg]), bank["opp_k"], C.K_NN); log("kNN %d queries x %d rows: %.0fs" % (len(qg), len(bank["opp_k"]), time.time() - t))
    g = np.zeros((len(opp_vec), 10)); g[qg] = C.nn_game_lift(nn_i, nn_c, bank["T"], C.BETA)
    return g, qg, nn_i, nn_c


# ============================================================================================================================ C1 export
def load_c1_export(path, qids):
    Z = np.load(path); assert "qid" in Z.files and "q_probs" in Z.files, f"{path}: keys {Z.files}"
    q = Z["qid"].astype(str); X = np.asarray(Z["q_probs"], np.float64)
    assert len(q) == len(qids) and set(q.tolist()) == set(qids), "export qid set != test questions"
    pos = {k: i for i, k in enumerate(q.tolist())}; X = X[[pos[k] for k in qids]]
    assert X.shape == (len(qids), 10) and np.isfinite(X).all() and np.allclose(X.sum(1), 1, atol=1e-4)
    info = dict(file=str(path), sha16=hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16], ckpt_sha16=str(Z["ckpt_sha16"]) if "ckpt_sha16" in Z.files else None, step=int(Z["step"]) if "step" in Z.files else None,
                keys=list(Z.files), same_order_as_questions=bool((q == np.array(qids)).all()))
    if "split" in Z.files: info["split_key_present"] = True
    return X, info


def describe_probs(X):
    ent = -(X * np.log(X + 1e-12)).sum(1); am = X.argmax(1)
    return dict(n=int(len(X)), mean_entropy=float(ent.mean()), mean_max_prob=float(X.max(1).mean()), argmax_share=(np.bincount(am, minlength=10) / len(X)).tolist(), mean_prob=X.mean(0).tolist(), mean_ev_rank=float((X * np.arange(10)).sum(1).mean()))


# ============================================================================================================================ the candidate (same function for the test and the rehearsal)
def build(inp, c1_fit=None, X_c1=None, g_nn=None, c3=False):
    base = inp["Fp"] if c1_fit is None else C.c1_base(c1_fit, inp["Fp"], X_c1)
    P, V, has = C.candidate_P(base, inp["sl"], inp["lib"], g_nn, C.W_NN, c3)
    return P, dict(base=base, V=V, has=has)


def decide(P):
    from gogame.models import decide_ev
    return decide_ev(P)


# ============================================================================================================================ test inputs
def load_test_inputs():
    import pandas as pd
    import e91_final as F, e91_final_b as FB
    from gogame.constants import RANKS
    nrm = lambda p: p / p.sum(1, keepdims=True)
    meta = F.test_meta(); qids = meta["qids"].tolist(); rows = meta["rows"]; hd = meta["hd"]; assert set(np.unique(hd)) <= {0, 1}
    u = np.load(F.TEST_UIDS); eff = u["eff"].astype(np.int64); eo = np.load(F.TEST_OWN)["eff"].astype(np.int64); ep = np.load(ROOT / ".scratch/e91/test_opp_games.npz")["eff"].astype(np.int64)
    assert (eff == eo).all() and (eff == 1 - ep).all() and (u["hd"].astype(int) == hd).all() and set(np.unique(eff)) <= {0, 1}
    fw = json.loads((ROOT / "experiments/E79_rank_pos_attn/results/final.json").read_text())["final_weights_all_oof"]["F_E79"]; w, v, uu = fw["w_A"], fw["v_M03"], fw["u_A1P"]; assert (w, v, uu) == (.9, .5, .75)
    p03, p04 = np.load(ROOT / ".scratch/e66/final/test_m03.npy"), np.load(ROOT / ".scratch/e66/final/test_m04.npy"); s2, sp_ = np.load(ROOT / ".scratch/e66/final/set_A1T2.npy"), np.load(ROOT / ".scratch/e79/final/set_A1P.npy")
    Fp = nrm((1 - w) * nrm(v * p03 + (1 - v) * p04) + w * nrm((1 - uu) * nrm(s2.mean((0, 1))) + uu * nrm(sp_.mean((0, 1)))))
    v9 = pd.read_csv(V9); assert list(v9["question_id"]) == qids; r9 = np.array([RANKS.index(x) for x in v9["pred_rank"]]); assert (decide(Fp) == r9).all(), "rebuilt v9 test probabilities do not decode to rank_v9.csv - STOP"
    v10d = pd.read_csv(V10D); assert list(v10d["question_id"]) == qids; r10d = np.array([RANKS.index(x) for x in v10d["pred_rank"]])
    own, opp, W, cr, info = FB.get_source("b"); ro_opp = F.readout(opp, W, cr)
    sl = C.slots_from_rows(rows, hd, eff)
    return dict(meta=meta, qids=qids, rows=rows, hd=hd, eff=eff, Fp=Fp, r9=r9, r10d=r10d, opp=opp, lib=ro_opp["lift"], sl=sl, nq=meta["n"], qsplit=meta["qsplit"], source_info=info, head_ckpt_sha16=str(info["ckpt_sha16"]))


# ============================================================================================================================ subcommands
def fit_c1():
    R = C.rebuild(); print("repro", R["repro"], flush=True)
    idx4 = np.flatnonzero(R["fold"] == 4); X_ = np.load(VAL_EXPORT); qid = R["Qf"].q.qid.to_numpy()[idx4].astype(str); assert (X_["qid"].astype(str) == qid).all()
    X = np.asarray(X_["q_probs"], np.float64); assert X.shape == (len(idx4), 10)
    fit = C.c1_fit_all(R, X)
    from e92_score import fit_predict
    P_chk = fit_predict(X, R["y"][idx4], X, fit["C"]); P_mine = C.c1_predict(fit, X); fit["check_max_abs_diff_vs_sklearn_predict_proba"] = float(np.abs(P_chk - P_mine).max()); assert fit["check_max_abs_diff_vs_sklearn_predict_proba"] < 1e-9
    fit["export"] = dict(file=str(VAL_EXPORT), sha16=hashlib.sha256(VAL_EXPORT.read_bytes()).hexdigest()[:16], ckpt_sha16=str(X_["ckpt_sha16"]), step=int(X_["step"]))
    fit["script_sha16"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    assert not C1_NPZ.exists() and not C1_JSON.exists(), "c1_fit exists - refusing to overwrite"
    C.save_c1_fit(fit, C1_JSON, C1_NPZ)
    print("C1 fit: C", fit["C"], "z", fit["z"], "| inner S by C", fit["inner_S_by_C"], "| inner fused by z", fit["inner_fused_by_z"], "| sklearn check", fit["check_max_abs_diff_vs_sklearn_predict_proba"]); print("written", C1_JSON, C1_NPZ)


def make(a):
    import check_submission
    from gogame.constants import RANKS
    from gogame.submission import write_rank_submission
    from scripts.experiments.e37.predict_test import sha, K, DATA
    assert a.name.replace("_", "").isalnum(); FINAL.mkdir(parents=True, exist_ok=True); out_csv = FINAL / f"{a.name}.csv"; stats_fn = FINAL / f"stats_{a.name}.json"
    assert "submissions" not in out_csv.resolve().parts
    assert not out_csv.exists() and not stats_fn.exists(), "output exists - refusing to overwrite"
    inp = load_test_inputs(); qids = inp["qids"]; qs = inp["qsplit"]; nqn = inp["nq"]; r10d = inp["r10d"]
    # sanity: lam = 0 reproduces v9 (as in e99_rank_final)
    P0 = C.fuse(inp["Fp"], np.full((800, 10), .1), 0., C.AL); assert (decide(P0) == inp["r9"]).all(); log("ASSERT OK: lam=0 reproduces v9 (800/800)")
    st = dict(name=a.name, components=dict(C1=bool(a.c1), C2=bool(a.c2), C3=bool(a.c3)), params=dict(gamma=C.GM, tau="none", alpha=C.AL, lam=C.LAM, beta=C.BETA, w=C.W_NN, k_nn=C.K_NN))
    fit = X_c1 = g_nn = None
    if a.c1:
        fit = C.load_c1_fit(C1_NPZ); fj = json.loads(C1_JSON.read_text()); X_c1, einfo = load_c1_export(a.c1, qids); st["C1"] = dict(fit_C=fit["C"], fit_z=fit["z"], fit_file=str(C1_NPZ), export=einfo, fit_n_questions=fj["n_questions"])
        if einfo["ckpt_sha16"] != fj["export"]["ckpt_sha16"] and not a.allow_other_ckpt: raise SystemExit(f"STOP: C1 export ckpt {einfo['ckpt_sha16']} != the checkpoint the C1 fit was made on {fj['export']['ckpt_sha16']} (use --allow-other-ckpt only after the PLAN section 3.5 checks)")
        val_X = np.asarray(np.load(VAL_EXPORT)["q_probs"], np.float64)
        st["C1"]["descriptives_test_export"] = describe_probs(X_c1); st["C1"]["descriptives_validation_export"] = describe_probs(val_X)
        st["C1"]["descriptives_note"] = "description only, no hard threshold (PLAN section 3.4): compare mean entropy / mean max prob / mean predicted-rank distribution between test (800 questions, N in 5..20) and validation (4,568 questions)"
    if a.c2:
        bank = load_bank(a.c2, "S_bank_opp_full.npz", "bank_meta_full.npz", inp["head_ckpt_sha16"]); log("bank used rows", bank["n_used"], "of", bank["n_total"])
        g_nn, qg, nn_i, nn_c = query_bank(bank, inp["opp"], inp["sl"]); assert nn_i.max() < bank["n_used"]
        st["C2"] = dict(bank_file=bank["file"], bank_sha16=bank["sha16"], bank_rows_total=bank["n_total"], bank_rows_used_hd_le_1=bank["n_used"], n_queried_games=int(len(qg)), n_games_with_nn_vote=int((g_nn[qg].sum(1) > 0).sum()),
                        top1_cos_quantiles={str(q): float(np.quantile(nn_c[:, 0], q)) for q in (.05, .5, .95)}, neighbour_top1_table_rank_distribution=(np.bincount(bank["T"][nn_i[:, 0]].argmax(1)[bank["T"][nn_i[:, 0]].sum(1) > 0], minlength=10) / max(1, int((bank["T"][nn_i[:, 0]].sum(1) > 0).sum()))).tolist())
    P, aux = build(inp, fit, X_c1, g_nn, a.c3); pred = decide(P); assert pred.shape == (800,) and pred.min() >= 0 and pred.max() < K
    write_rank_submission(qids, pred, out_csv); rc = check_submission.main([str(out_csv), "--task", "rank", "--data-dir", str(DATA)]); assert rc == 0, "check_submission failed"
    ch = pred != r10d; d = pred - r10d
    shiftd = lambda m: dict(plus1=int((d[m] == 1).sum()), minus1=int((d[m] == -1).sum()), plus2=int((d[m] == 2).sum()), minus2=int((d[m] == -2).sum()), beyond2=int((np.abs(d[m]) > 2).sum()))
    r_val = json.loads((OUT / "result_val.json").read_text())["r_val"]["value"] if (OUT / "result_val.json").exists() else None; r_test = float(ch.mean())
    st.update(file=str(out_csv), sha256=sha(out_csv), check_submission_rc=rc, v10d_sha256=sha(V10D), csv_bytes_identical_to_v10d=bool(out_csv.read_bytes() == V10D.read_bytes()),
              n_changed_vs_v10d=int(ch.sum()), n_changed_public=int(ch[qs == 0].sum()), n_changed_private=int(ch[qs == 1].sum()), n_public=int((qs == 0).sum()), n_private=int((qs == 1).sum()), share_changed_vs_v10d=r_test,
              shift_vs_v10d=dict(all=shiftd(np.ones(800, bool)), public=shiftd(qs == 0), private=shiftd(qs == 1)), mean_rank_shift_vs_v10d=float(d.mean()), n_changed_by_N={int(n): int(ch[nqn == n].sum()) for n in (5, 10, 15, 20)},
              pred_rank_counts=dict(new=np.bincount(pred, minlength=K).tolist(), v10d=np.bincount(r10d, minlength=K).tolist()), rank_names=list(RANKS),
              frac_questions_with_vote=float(aux["has"].mean()),
              drift_check=dict(r_val=r_val, r_test=r_test, limit_1p5_r_val=(1.5 * r_val if r_val is not None else None), exceeds=(bool(r_test > 1.5 * r_val) if r_val is not None else None),
                               rule="PLAN section 3.3: share differing from v10d on the test file > 1.5 x r_val (validation: share of questions where ALL differs from REF) -> do NOT hand over, investigate first; only meaningful for the all-on file"))
    stats_fn.write_text(json.dumps(st, indent=1, default=float)); print(json.dumps(st, indent=1, default=float))


def rehearse(a):
    """fold-4 validation questions treated as the test set, through the make-path functions.  Compares ANSWERS only."""
    import e103_val as V
    G = V.setup(); R, ctx = G["R"], G["ctx"]; fn = OUT / "val_preds.npz"; assert fn.exists(), "val_preds.npz missing: the formal validation chain has not been run"
    vp = np.load(fn); qid = R["Qf"].q.qid.to_numpy()[R["fold"] == 4].astype(str); assert (vp["qid"] == qid).all()
    inp = dict(Fp=ctx["f79q"], sl=G["sl"], lib=G["lib"], opp=G["real"]["opp"])
    # the 'test' export: same file as the validation chain's input, loaded by the make-path loader (qid alignment code path)
    X_c1, einfo = load_c1_export(VAL_EXPORT, qid.tolist()); assert np.array_equal(X_c1, G["Xa"])
    fit = C.load_c1_fit(C1_NPZ); fj = json.loads(C1_JSON.read_text())
    head_sha = str(np.load(S96.HEAD, allow_pickle=True)["ckpt_sha16"])
    bank = load_bank(E2.OUT, "S_bank_opp_v4.npz", "bank_meta.npz", head_sha); log("v4 bank used rows", bank["n_used"])
    g_nn, qg, nn_i, nn_c = query_bank(bank, inp["opp"], inp["sl"])
    g_cache, qg_c, T_c, info_c = V.load_nn(G); z = np.load(V.NNC)
    nn_same = float((np.sort(nn_i, 1) == np.sort(z["nn_row"], 1)).all(1).mean()); cos_diff = float(np.abs(nn_c - z["nn_cos"]).max())
    res = dict(nn_fresh_vs_e102_cache=dict(n_queries=int(len(qg)), share_queries_identical_neighbour_set=nn_same, max_abs_cos_diff=cos_diff, max_abs_diff_game_lift=float(np.abs(g_nn - g_cache).max())))
    arms = dict(REF=(None, False, False), C1=(True, False, False), C2=(False, True, False), C3=(False, False, True), ALL=(True, True, True), noC1=(False, True, True), noC2=(True, False, True), noC3=(True, True, False))
    out = {}; Pm = {}
    for nm, (c1, c2, c3) in arms.items():
        P, aux = build(inp, fit if c1 else None, X_c1 if c1 else None, g_nn if c2 else None, c3); Pm[nm] = P; pm = decide(P); pv = vp[f"pred_{nm}"]
        out[nm] = dict(n_answers_differ_make_vs_val_chain=int((pm != pv).sum()), share=float((pm != pv).mean()), of=int(len(pm)))
    # C1 base itself: all-fit (make path) vs nested OOF (validation chain)
    base_make = C.c1_base(fit, inp["Fp"], X_c1); FF = vp["FF"]; dec = lambda p: C.decide_ev(p)
    c1d = dict(fit_C=fit["C"], fit_z=fit["z"], nested_z_per_part=json.loads((OUT / "result_val.json").read_text())["c1_nested_chosen"],
               n_base_answers_differ=int((dec(base_make) != dec(FF)).sum()), share_base_answers_differ=float((dec(base_make) != dec(FF)).mean()), mean_abs_diff_base_prob=float(np.abs(base_make - FF).mean()), max_abs_diff_base_prob=float(np.abs(base_make - FF).max()),
               mean_max_prob_fit_all=float(base_make.max(1).mean()), mean_max_prob_nested=float(FF.max(1).mean()))
    res.update(arms=out, c1_fit_all_vs_nested=c1d, c1_in_sample_note="the make path fits the logistic regression on ALL 4,568 questions and here predicts those same questions (in-sample), the validation chain predicts each fifth with a model fitted on the other four (nested OOF) -> C1, ALL, noC2, noC3 are NOT expected to be identical; REF, C2, C3, noC1 (no C1) must be identical",
               must_be_identical=["REF", "C2", "C3", "noC1"], identical_ok={k: out[k]["n_answers_differ_make_vs_val_chain"] == 0 for k in ("REF", "C2", "C3", "noC1")}, script_sha16=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16])
    fo = OUT / "rehearsal_make.json"; assert not fo.exists(), "rehearsal_make.json exists"; fo.write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit-c1"); sub.add_parser("rehearse")
    m = sub.add_parser("make"); m.add_argument("--name", required=True); m.add_argument("--c1"); m.add_argument("--c2"); m.add_argument("--c3", action="store_true"); m.add_argument("--allow-other-ckpt", action="store_true")
    a = ap.parse_args()
    if a.cmd == "fit-c1": fit_c1()
    elif a.cmd == "make": make(a)
    else: rehearse(a)
