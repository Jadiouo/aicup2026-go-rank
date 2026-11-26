"""E117 test-side file (only after the validation PASS).  v11b with ONE component added: the stage-2 classifier (e117_core) fitted on ALL 4,568 validation questions (inner 3-fold player-grouped selection of
(model, hp, rho), then one fit on all rows), applied to the test questions' P_ALL, P_REF, FF, P_C1, P_C2, P_C3 (+ N).  Everything upstream is e103_make's own code (imported READ-ONLY, not modified; the v11b command is
`e103_make.py make --name v11_retrain --c1 .scratch/e103/exports/final_s0_test12.npz --c2 .scratch/e103/bank --c3 --allow-other-ckpt`).  Output only under .scratch/e117/final/ (never data/submissions).
  CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 env PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/experiments/e117/e117_make.py {fit|inputs|make}
  fit    : select + fit the stage-2 model on all 4,568 validation questions            -> .scratch/e117/final/stage2_fit.pkl, stage2_fit.json   (refuses to overwrite)
  inputs : the six test-side probability blocks (+N, qid, qsplit); asserts P_ALL decodes to v11b 800/800  -> .scratch/e117/final/test_inputs.npz   (refuses to overwrite)
  make   : apply stage 2, write rank_v11b_stage2.csv + stats, byte-compare the rho=0 file with v11b, format check
"""
import os, sys, json, argparse, hashlib, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiments/e103")); sys.path.insert(0, str(HERE))
import numpy as np
import e117_core as K
from e117_core import OUT, log

FINAL = OUT / "final"
V11B = ROOT / "data/submissions/rank_v11b_小訊號合併_全資料重訓.csv"
C1_EXPORT = ROOT / ".scratch/e103/exports/final_s0_test12.npz"; BANK = ROOT / ".scratch/e103/bank"


def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit_all():
    pk = FINAL / "stage2_fit.pkl"; pj = FINAL / "stage2_fit.json"; assert not pk.exists() and not pj.exists(), "stage2_fit exists - refusing to overwrite"
    D = K.load_all(); X = K.make_X(D["Pk"], D["ns"]); y, ns, pl = D["y"], D["ns"], D["pl"]
    tab, best_by, best = K.select(X, y, ns, pl, D["Pk"]["P_ALL"], y)
    m = K.Stage2().fit(X, y, K.sample_weights(y, ns)); FINAL.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(model=m, best=best), open(pk, "xb"))
    info = dict(chosen=dict(model=best[0], hp=best[1], rho=best[2]), inner_score_chosen=tab[best], inner_table={f"{k[0]}|{k[1]}|{k[2]:g}": v for k, v in tab.items()}, n_train=int(len(y)), n_players=int(len(np.unique(pl))),
                core_sha16=K.sha16(Path(K.__file__)), val_preds_sha16=K.sha16(K.VAL_PREDS))
    with open(pj, "x") as f: json.dump(info, f, indent=1, default=float)
    print("chosen on all 4,568:", best, "inner %.4f" % tab[best]); print({k: round(v, 4) for k, v in info["inner_table"].items()})


def test_inputs():
    import e103_make as M, e103_core as C
    from gogame.constants import RANKS
    import pandas as pd
    fn = FINAL / "test_inputs.npz"; assert not fn.exists(), "test_inputs.npz exists"
    inp = M.load_test_inputs(); qids = inp["qids"]; assert len(qids) == 800
    fit = C.load_c1_fit(M.C1_NPZ); fj = json.loads(M.C1_JSON.read_text())
    X_c1, einfo = M.load_c1_export(C1_EXPORT, qids)
    st_b = json.loads((ROOT / ".scratch/e103/final/stats_v11_retrain.json").read_text())                    # the stats of the file v11b was made from: same C1 export / bank / checkpoint?
    assert st_b["C1"]["export"]["sha16"] == einfo["sha16"] and st_b["C1"]["export"]["ckpt_sha16"] == einfo["ckpt_sha16"], "C1 export differs from the one v11b was made with"
    bank = M.load_bank(BANK, "S_bank_opp_full.npz", "bank_meta_full.npz", inp["head_ckpt_sha16"]); assert bank["sha16"] == st_b["C2"]["bank_sha16"], "bank differs from the v11b one"
    g_nn, qg, nn_i, nn_c = M.query_bank(bank, inp["opp"], inp["sl"])
    del bank
    P_ALL, aux = M.build(inp, fit, X_c1, g_nn, True); FF = aux["base"]
    P_REF = M.build(inp, None, None, None, False)[0]; P_C1 = M.build(inp, fit, X_c1, None, False)[0]; P_C2 = M.build(inp, None, None, g_nn, False)[0]; P_C3 = M.build(inp, None, None, None, True)[0]
    v = pd.read_csv(V11B); assert list(v["question_id"]) == list(qids); r11b = np.array([RANKS.index(x) for x in v["pred_rank"]])
    d = M.decide(P_ALL); assert (d == r11b).all(), f"P_ALL decodes to v11b on {(d == r11b).sum()}/800 only - STOP"
    # the other keys must be consistent with their definitions: P_REF decodes to v10d (all components off)
    assert (M.decide(P_REF) == inp["r10d"]).all(), "P_REF does not decode to v10d"
    with open(fn, "xb") as f: np.savez(f, qid=np.array(qids), nq=np.asarray(inp["nq"]), qsplit=np.asarray(inp["qsplit"]), P_ALL=P_ALL, P_REF=P_REF, FF=FF, P_C1=P_C1, P_C2=P_C2, P_C3=P_C3, r11b=r11b, r10d=inp["r10d"])
    print("P_ALL decodes to v11b 800/800; P_REF decodes to v10d 800/800; wrote", fn, "sha16", K.sha16(fn))


def make(a):
    import check_submission
    from gogame.constants import RANKS
    from gogame.submission import write_rank_submission
    from gogame.models import decide_ev
    from scripts.experiments.e37.predict_test import DATA
    out_csv = FINAL / "rank_v11b_stage2.csv"; stats_fn = FINAL / "stats_rank_v11b_stage2.json"; assert not out_csv.exists() and not stats_fn.exists(), "output exists - refusing to overwrite"
    assert "submissions" not in out_csv.resolve().parts
    Z = np.load(FINAL / "test_inputs.npz"); qids = [str(q) for q in Z["qid"]]; nq = Z["nq"]; qs = Z["qsplit"]; r11b = Z["r11b"]
    Pk = {k: np.asarray(Z[k], np.float64) for k in K.KEYS}; assert all(np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-6) for p in Pk.values())
    pf = pickle.load(open(FINAL / "stage2_fit.pkl", "rb")); m, best = pf["model"], pf["best"]; rho = best[2]
    pr = m.predict(K.make_X(Pk, nq)); Ps = pr[(best[0], best[1])]
    # guard 1: rho = 0 is the current file, byte for byte
    p0 = decide_ev(K.combine(Pk["P_ALL"], Ps, 0.0)); assert (p0 == r11b).all()
    z_csv = FINAL / "_rho0_check.csv"; write_rank_submission(qids, p0, z_csv); rho0_identical = bool(z_csv.read_bytes() == V11B.read_bytes()); z_csv.unlink()
    assert rho0_identical, "rho=0 file is not byte-identical to v11b - STOP"
    P = K.combine(Pk["P_ALL"], Ps, rho); pred = decide_ev(P); assert pred.shape == (800,) and pred.min() >= 0 and pred.max() <= 9
    write_rank_submission(qids, pred, out_csv); rc = check_submission.main([str(out_csv), "--task", "rank", "--data-dir", str(DATA)]); assert rc == 0, "check_submission failed"
    ch = pred != r11b; d = pred - r11b
    shiftd = lambda mk: dict(n=int(mk.sum()), changed=int(ch[mk].sum()), plus1=int((d[mk] == 1).sum()), minus1=int((d[mk] == -1).sum()), beyond1=int((np.abs(d[mk]) > 1).sum()))
    # distribution comparison, test (800) vs validation (4,568): mean entropy and quantiles of the max probability, every input key and the stage-2 output
    D = K.load_all(); vP = dict(D["Pk"]); vX = K.make_X(vP, D["ns"]); vPs = None
    ent = lambda p: -(p * np.log(p + 1e-12)).sum(1); qs_ = (.05, .25, .5, .75, .95)
    dist = {k: dict(test=dict(mean_entropy=float(ent(Pk[k]).mean()), max_prob_q={str(q): float(np.quantile(Pk[k].max(1), q)) for q in qs_}, mean_prob=Pk[k].mean(0).tolist()),
                    val=dict(mean_entropy=float(ent(vP[k]).mean()), max_prob_q={str(q): float(np.quantile(vP[k].max(1), q)) for q in qs_}, mean_prob=vP[k].mean(0).tolist())) for k in K.KEYS}
    vA = np.load(OUT / "val_preds_e117.npz"); share_val = float((vA["pred_CHOSEN"] != vA["pred_ALL"]).mean())
    dist["stage2_probs_test_vs_val_note"] = "val stage-2 output = nested OOF CHOSEN arm (val_preds_e117.npz), test = model fitted on all 4,568"
    dist["P_final"] = dict(test=dict(mean_entropy=float(ent(P).mean()), max_prob_q={str(q): float(np.quantile(P.max(1), q)) for q in qs_}), val_nested=dict(mean_entropy=float(ent(vA["P_CHOSEN"]).mean()), max_prob_q={str(q): float(np.quantile(vA["P_CHOSEN"].max(1), q)) for q in qs_}))
    st = dict(name="rank_v11b_stage2", file=str(out_csv), sha256=sha256(out_csv), v11b_file=str(V11B), v11b_sha256=sha256(V11B), check_submission_rc=rc, rho0_byte_identical_to_v11b=rho0_identical, chosen_on_all_4568=dict(model=best[0], hp=best[1], rho=rho),
              stage2_fit_json=json.loads((FINAL / "stage2_fit.json").read_text()), n_changed_vs_v11b=int(ch.sum()), share_changed_vs_v11b=float(ch.mean()), share_changed_validation_nested_vs_ALL=share_val, ratio_test_over_val_share=float(ch.mean() / share_val),
              public=shiftd(qs == 0), private=shiftd(qs == 1), all=shiftd(np.ones(800, bool)), n_changed_by_N={str(n): int(ch[nq == n].sum()) for n in K.N_LEVELS}, n_questions_by_N={str(n): int((nq == n).sum()) for n in K.N_LEVELS},
              pred_rank_counts=dict(new=np.bincount(pred, minlength=10).tolist(), v11b=np.bincount(r11b, minlength=10).tolist()), rank_names=list(RANKS), distribution_test_vs_val=dist, test_inputs_sha16=K.sha16(FINAL / "test_inputs.npz"),
              make_sha16=K.sha16(Path(__file__)), core_sha16=K.sha16(Path(K.__file__)))
    with open(stats_fn, "x") as f: json.dump(st, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in st.items() if k not in ("distribution_test_vs_val", "stage2_fit_json")}, indent=1, default=float))
    for k in K.KEYS + ["P_final"]:
        t = dist[k]["test"]; v = dist[k].get("val", dist[k].get("val_nested")); print(f"{k:8s} entropy test {t['mean_entropy']:.3f} val {v['mean_entropy']:.3f} | max-prob q05/q50/q95 test {t['max_prob_q']['0.05']:.3f}/{t['max_prob_q']['0.5']:.3f}/{t['max_prob_q']['0.95']:.3f} val {v['max_prob_q']['0.05']:.3f}/{v['max_prob_q']['0.5']:.3f}/{v['max_prob_q']['0.95']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fit", "inputs", "make"]); a = ap.parse_args()
    FINAL.mkdir(parents=True, exist_ok=True)
    {"fit": lambda: fit_all(), "inputs": lambda: test_inputs(), "make": lambda: make(a)}[a.cmd]()
