"""E79 test-side fuse (only after PLAN section 3 passed).  F = nrm((1-w)*nrm(v*P_M03T2 + (1-v)*P_M04T) + w*nrm((1-u)*P_A1T2 + u*P_A1P)); (w,v,u) = final weights on ALL OOF (results/final.json final_weights_all_oof.F_E79).
P_A1T2 / P_A1P = row-normalised mean over 5 folds x 3 seeds.  EV decode, write_rank_submission, check_submission.
  .venv/bin/python scripts/oneoff/e79_fuse.py --m03 NPY --m04 NPY --set-a1t2 NPY --set-a1p NPY --out CSV --report J --ref v8 CSV"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from scripts.experiments.e37.predict_test import sha, K, DATA, E29T
nrm = lambda p: p / p.sum(1, keepdims=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); [ap.add_argument(f"--{x}", required=True) for x in ("m03", "m04", "set-a1t2", "set-a1p", "out", "report")]; ap.add_argument("--ref", action="append", nargs=2, metavar=("NAME", "CSV")); a = ap.parse_args()
    import pandas as pd
    from gogame.constants import RANKS
    from gogame.models import decide_ev
    from gogame.submission import write_rank_submission
    fw = json.loads((ROOT / "experiments/E79_rank_pos_attn/results/final.json").read_text())["final_weights_all_oof"]["F_E79"]; w, v, u = fw["w_A"], fw["v_M03"], fw["u_A1P"]
    bg = np.load(E29T / "test_bags.npz"); qids = bg["qids"].tolist(); nn_ = bg["n"]; p03, p04 = np.load(a.m03), np.load(a.m04); s2, sp = np.load(a.set_a1t2), np.load(a.set_a1p)
    assert p03.shape == p04.shape == (800, 10) and s2.shape == sp.shape == (5, 3, 800, 10)
    pa2, pap = nrm(s2.mean((0, 1))), nrm(sp.mean((0, 1))); p = nrm((1 - w) * nrm(v * p03 + (1 - v) * p04) + w * nrm((1 - u) * pa2 + u * pap)); pred = decide_ev(p); f = Path(a.out); write_rank_submission(qids, pred, f)
    out = {"file": str(f), "sha256": sha(f), "w_A": w, "v_M03T2": v, "u_A1P": u, "n_classes_predicted": int(len(np.unique(pred))), "pred_share": np.bincount(pred, minlength=K).tolist(),
           "leg_decision_agree": {"m03": float((decide_ev(nrm(p03)) == pred).mean()), "m04": float((decide_ev(nrm(p04)) == pred).mean()), "a1t2": float((decide_ev(pa2) == pred).mean()), "a1p": float((decide_ev(pap) == pred).mean())},
           "a1p_vs_a1t2_decision_agree": float((decide_ev(pap) == decide_ev(pa2)).mean())}
    for name, ref in (a.ref or []):
        rdf = pd.read_csv(ref); assert list(rdf["question_id"]) == qids; r = np.array([RANKS.index(x) for x in rdf["pred_rank"]])
        out[f"vs_{name}"] = {"file": ref, "sha256": sha(ref), "identical_rows": int((pred == r).sum()), "agree": float((pred == r).mean()), "per_N_agree": {int(n): float((pred[nn_ == n] == r[nn_ == n]).mean()) for n in (5, 10, 15, 20)}, "mean_pred_minus_ref": float((pred - r).mean())}
    import check_submission
    out["check_submission_rc"] = check_submission.main([str(f), "--task", "rank", "--data-dir", str(DATA)]); Path(a.report).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); assert out["check_submission_rc"] == 0
