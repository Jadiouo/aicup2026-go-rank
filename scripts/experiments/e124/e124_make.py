"""E124 test-side path (real production path of E117: test_inputs.npz -> stage 2 -> combine -> EV decode -> write_rank_submission).  Output only under .scratch/e124/ (never data/submissions).
  dry   : check (5): rho=0 gives v11b byte for byte (with the shift model); stage 2 = E117's LR from stage2_fit.pkl gives v12 byte for byte    -> .scratch/e124/dry_make.json
  final : only after result_val.json says E124_VAL PASS: write .scratch/e124/final/rank_v12_shift.csv + stats + kernel json
"""
import os, sys, json, pickle, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts/experiments/e117"))
import numpy as np
import e124_core as K
from e124_core import OUT, log, sha256

FINAL = OUT / "final"; DRY = OUT / "dry"
V11B = ROOT / "data/submissions/rank_v11b_小訊號合併_全資料重訓.csv"; V12 = ROOT / "data/submissions/rank_v12_第二階段分類器.csv"
TEST_IN = ROOT / ".scratch/e117/final/test_inputs.npz"; E117_FIT = ROOT / ".scratch/e117/final/stage2_fit.pkl"


def load_test():
    Z = np.load(TEST_IN); Pk = {k: np.asarray(Z[k], np.float64) for k in K.KEYS}
    assert all(np.isfinite(p).all() and p.shape == (800, 10) and np.allclose(p.sum(1), 1, atol=1e-6) for p in Pk.values())
    return [str(q) for q in Z["qid"]], Pk, Z["nq"], Z["qsplit"], Z["r11b"]


def read_answers(path, qids):
    import pandas as pd
    from gogame.constants import RANKS
    v = pd.read_csv(path); assert list(v["question_id"]) == list(qids), "qid order differs"
    return np.array([RANKS.index(x) for x in v["pred_rank"]])


def fit_shift_all(D):
    return K.ShiftS2().fit(D["Pk"], D["ns"], D["y"], K.sample_weights(D["y"], D["ns"]), tag="shift_all4568")


def write_csv(qids, pred, path):
    from gogame.submission import write_rank_submission
    write_rank_submission(qids, pred, path)


def test_side(D, write=None):
    """fit the shift model on all 4,568 validation questions, apply to the test inputs; compare with the v12 file (no test answers are involved)."""
    from gogame.models import decide_ev
    qids, Pk, nq, qs, r11b = load_test(); r12 = read_answers(V12, qids); K.FITLOG.clear(); m = fit_shift_all(D); fl = K.FITLOG[-1]
    Ps = m.predict(Pk, nq); P = K.combine(Pk["P_ALL"], Ps, K.RHO); pred = decide_ev(P); assert pred.shape == (800,) and pred.min() >= 0 and pred.max() <= 9
    ch = pred != r12; d = pred - r12
    shiftd = lambda mk: dict(n=int(mk.sum()), changed=int(ch[mk].sum()), plus1=int((d[mk] == 1).sum()), minus1=int((d[mk] == -1).sum()), beyond1=int((np.abs(d[mk]) > 1).sum()))
    st = dict(n_differ_from_v12=int(ch.sum()), n_differ_from_v11b=int((pred != r11b).sum()), public=shiftd(qs == 0), private=shiftd(qs == 1), all=shiftd(np.ones(800, bool)), by_N={str(n): shiftd(nq == n) for n in K.N_LEVELS},
              pred_rank_counts=dict(candidate=np.bincount(pred, minlength=10).tolist(), v12=np.bincount(r12, minlength=10).tolist(), v11b=np.bincount(r11b, minlength=10).tolist()),
              lbfgs=fl, kernel_a=m.a.tolist(), bias_B=m.B.tolist(), bias_range=[float(m.B.min()), float(m.B.max())], mu=m.mu, sd=m.sd)
    if write:
        write_csv(qids, pred, write); st["file"] = str(write); st["sha256"] = sha256(write)
    return st


def dry():
    import check_submission
    from gogame.models import decide_ev
    from scripts.experiments.e37.predict_test import DATA
    DRY.mkdir(parents=True, exist_ok=True); D = K.load_all(); qids, Pk, nq, qs, r11b = load_test(); res = {}
    # (5a) rho = 0 with the SHIFT stage 2 -> v11b, byte for byte, through the real writer
    m = fit_shift_all(D); Ps = m.predict(Pk, nq); p0 = decide_ev(K.combine(Pk["P_ALL"], Ps, 0.0)); f0 = DRY / "shift_rho0.csv"
    if f0.exists(): f0.unlink()
    write_csv(qids, p0, f0); res["rho0_shift_vs_v11b"] = dict(answers_equal_r11b=bool((p0 == r11b).all()), bytes_identical=bool(f0.read_bytes() == V11B.read_bytes()), sha256=sha256(f0), v11b_sha256=sha256(V11B))
    # (5b) stage 2 = E117's LR (stage2_fit.pkl) -> v12, byte for byte
    import e117_core as K117                                                                    # noqa  (the pickle refers to e117_core.Stage2)
    pf = pickle.load(open(E117_FIT, "rb")); m17, best = pf["model"], pf["best"]; Ps17 = m17.predict(K117.make_X(Pk, nq))[(best[0], best[1])]
    p12 = decide_ev(K.combine(Pk["P_ALL"], Ps17, best[2])); f12 = DRY / "e117_lr_path.csv"
    if f12.exists(): f12.unlink()
    write_csv(qids, p12, f12); res["e117_lr_vs_v12"] = dict(best=[best[0], best[1], best[2]], bytes_identical=bool(f12.read_bytes() == V12.read_bytes()), sha256=sha256(f12), v12_sha256=sha256(V12))
    # the same path with rho = 0 and the E117 LR also gives v11b
    p0b = decide_ev(K.combine(Pk["P_ALL"], Ps17, 0.0)); fb = DRY / "e117_lr_rho0.csv"
    if fb.exists(): fb.unlink()
    write_csv(qids, p0b, fb); res["e117_lr_rho0_vs_v11b_bytes_identical"] = bool(fb.read_bytes() == V11B.read_bytes())
    rc = check_submission.main([str(f12), "--task", "rank", "--data-dir", str(DATA)]); res["check_submission_rc_on_v12_reproduction"] = rc
    res["pass_"] = bool(res["rho0_shift_vs_v11b"]["bytes_identical"] and res["e117_lr_vs_v12"]["bytes_identical"] and res["e117_lr_rho0_vs_v11b_bytes_identical"] and rc == 0)
    with open(OUT / "dry_make.json", "w") as f: json.dump(dict(check="(5) real production path", **res), f, indent=1, default=float)
    print("DRY-MAKE", json.dumps(res, default=float)[:900]); return res


def final():
    import check_submission
    from scripts.experiments.e37.predict_test import DATA
    rv = json.loads((OUT / "result_val.json").read_text()); assert rv["verdict"] == "E124_VAL PASS", "formal verdict is not PASS - no candidate"
    assert json.loads((OUT / "dry_make.json").read_text())["pass_"]
    FINAL.mkdir(parents=True, exist_ok=True); out_csv = FINAL / "rank_v12_shift.csv"; stats_fn = FINAL / "stats_rank_v12_shift.json"; assert not out_csv.exists() and not stats_fn.exists(), "exists - refusing to overwrite"
    assert "submissions" not in out_csv.resolve().parts
    D = K.load_all(); st = test_side(D, write=out_csv); rc = check_submission.main([str(out_csv), "--task", "rank", "--data-dir", str(DATA)]); assert rc == 0
    dm = json.loads((OUT / "dry_make.json").read_text()); st.update(check_submission_rc=rc, byte_check_rho0_vs_v11b=dm["rho0_shift_vs_v11b"], byte_check_e117_lr_vs_v12=dm["e117_lr_vs_v12"], v12_file=str(V12), v12_sha256=sha256(V12),
                                                                 n_differ_from_v12_formal=rv["test_side"]["n_differ_from_v12"], make_sha16=K.sha16(Path(__file__)), core_sha16=K.sha16(Path(K.__file__)))
    with open(stats_fn, "x") as f: json.dump(st, f, indent=1, default=float)
    with open(FINAL / "kernel_and_bias.json", "x") as f: json.dump(dict(kernel_a_6x5=st["kernel_a"], groups=K.KEYS, offsets=list(range(-K.K_HALF, K.K_HALF + 1)), bias_B_4x10=st["bias_B"], bias_range=st["bias_range"], mu=st["mu"], sd=st["sd"], N_levels=K.N_LEVELS), f, indent=1)
    print(json.dumps({k: v for k, v in st.items() if k not in ("kernel_a", "bias_B")}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["dry", "final"]); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    {"dry": dry, "final": final}[a.cmd]()
