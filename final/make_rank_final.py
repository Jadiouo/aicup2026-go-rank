"""Rank final step: six probability blocks -> second stage -> EV decode -> submission CSV (v13n).  CPU only, ~1 minute.

  v13  = shift-invariant kernel stage 2 (E124, ShiftS2 fitted on ALL validation questions) combined with P_ALL (rho = .5), EV decode
  v12  = E117 multinomial-LR stage 2 (stage2_fit.pkl, fitted once, selected LR/C/rho stored inside), EV decode
  v13n = v13 for every question except num_games == 5, where the v12 answer is used          <- the file that was uploaded as the final Rank result

Inputs  (all under --work-dir, default $WORK_DIR or <repo>/.scratch):
  e103/val_preds.npz          4,568 validation questions: P_ALL, P_REF, FF, P_C1, P_C2, P_C3 (n,10), nq, yq_not_for_scoring_here  (labels used to fit stage 2)
  e117/final/test_inputs.npz  800 test questions: qid, nq, qsplit, the same six blocks (+ r11b, r10d for cross-checks)
  e117/final/stage2_fit.pkl   E117 stage-2 LR (pickled by e117_core.Stage2; resolved through stage2_core)
Output  : --out CSV (question_id,pred_rank).  With --reference FILE the script also compares byte for byte and prints sha256.

  python final/make_rank_final.py --out /tmp/rank_v13n.csv [--reference path/to/uploaded.csv]
"""
import os, sys, argparse, hashlib, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src")]
import numpy as np
import stage2_core as K
sys.modules["e117_core"] = K                      # stage2_fit.pkl refers to e117_core.Stage2
from gogame.models import decide_ev
from gogame.submission import write_rank_submission


def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR", str(ROOT / ".scratch")))
    ap.add_argument("--out", required=True); ap.add_argument("--reference", help="uploaded CSV to compare with (optional)")
    ap.add_argument("--rho", type=float, default=K.RHO)
    a = ap.parse_args(); W = Path(a.work_dir); out = Path(a.out)
    if out.exists(): raise SystemExit(f"refusing to overwrite {out}")
    V = np.load(W / "e103/val_preds.npz"); Pv = {k: np.asarray(V[k], np.float64) for k in K.KEYS}; yv, nv = V["yq_not_for_scoring_here"], V["nq"]
    T = np.load(W / "e117/final/test_inputs.npz"); Pt = {k: np.asarray(T[k], np.float64) for k in K.KEYS}; qids = [str(q) for q in T["qid"]]; nq = T["nq"]
    assert all(np.isfinite(p).all() and p.shape == (800, 10) and np.allclose(p.sum(1), 1, atol=1e-6) for p in Pt.values())
    assert all(p.shape == (4568, 10) and np.allclose(p.sum(1), 1, atol=1e-6) for p in Pv.values())
    # v13: E124 shift-invariant stage 2 fitted on all validation questions, applied to the test blocks
    m = K.ShiftS2().fit(Pv, nv, yv, K.sample_weights(yv, nv), tag="shift_all4568")
    p13 = decide_ev(K.combine(Pt["P_ALL"], m.predict(Pt, nq), a.rho))
    # v12: E117 LR stage 2 from the pickle
    pf = pickle.load(open(W / "e117/final/stage2_fit.pkl", "rb")); m17, best = pf["model"], pf["best"]
    p12 = decide_ev(K.combine(Pt["P_ALL"], m17.predict(K.make_X(Pt, nq))[(best[0], best[1])], best[2]))
    # v13n: N = 5 questions keep the v12 answer
    pred = np.where(nq == 5, p12, p13); assert pred.shape == (800,) and pred.min() >= 0 and pred.max() <= 9
    out.parent.mkdir(parents=True, exist_ok=True); write_rank_submission(qids, pred, out)
    print(f"v13n written: {out}\n  sha256 {sha256(out)}\n  differs from v13 on {(pred != p13).sum()} questions, from v12 on {(pred != p12).sum()}; N=5 questions {(nq == 5).sum()}; lbfgs {K.FITLOG[-1]['message'] if K.FITLOG else ''}")
    if a.reference:
        eq = Path(a.reference).read_bytes() == out.read_bytes(); print(f"  reference {a.reference}\n  reference sha256 {sha256(a.reference)}\n  BYTE-IDENTICAL: {eq}"); return 0 if eq else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
