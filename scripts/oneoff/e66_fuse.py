"""E66 test-side fuse.  F = nrm((1-w)*nrm(v*P_M03T2 + (1-v)*P_M04T) + w*P_A1T2), EV decode; (w, v) = final weights chosen on ALL OOF (results/final.json final_weights_all_oof.F_E66).
Same arithmetic as e66_score.py `mix`/`D` and e55 fuse (A1T2 test probs = mean over 5 folds x 3 seeds, row-normalised).  Reports agreement with v7 and writes the CSV through write_rank_submission + check_submission.
  .venv/bin/python scripts/oneoff/e66_fuse.py --m03 NPY --m04 NPY --set NPY --out CSV --report J --ref v7 CSV"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from scripts.experiments.e37.predict_test import sha, K, DATA, E29T
nrm = lambda p: p / p.sum(1, keepdims=True)
def fuse_probs(p03, p04, pset, w, v): return nrm((1 - w) * nrm(v * p03 + (1 - v) * p04) + w * pset)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--m03", required=True); ap.add_argument("--m04", required=True); ap.add_argument("--set", required=True); ap.add_argument("--out", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--w", type=float); ap.add_argument("--v", type=float); ap.add_argument("--ref", action="append", nargs=2, metavar=("NAME", "CSV")); a = ap.parse_args()
    import pandas as pd
    from gogame.constants import RANKS
    from gogame.models import decide_ev
    from gogame.submission import write_rank_submission
    fw = json.loads((ROOT / "experiments/E66_rank_combo/results/final.json").read_text())["final_weights_all_oof"]["F_E66"]; w = a.w if a.w is not None else fw["w_A1"]; v = a.v if a.v is not None else fw["v_M03"]
    bg = np.load(E29T / "test_bags.npz"); qids = bg["qids"].tolist(); nn_ = bg["n"]; p03, p04 = np.load(a.m03), np.load(a.m04); sp = np.load(a.set); assert p03.shape == p04.shape == (800, 10) and sp.shape == (5, 3, 800, 10)
    pset = nrm(sp.mean((0, 1))); p = fuse_probs(p03, p04, pset, w, v); pred = decide_ev(p); f = Path(a.out); write_rank_submission(qids, pred, f)
    out = {"file": str(f), "sha256": sha(f), "w_A1T2": w, "v_M03T2": v, "n_classes_predicted": int(len(np.unique(pred))), "pred_share": np.bincount(pred, minlength=K).tolist(),
           "leg_decision_agree": {"m03_vs_final": float((decide_ev(nrm(p03)) == pred).mean()), "m04_vs_final": float((decide_ev(nrm(p04)) == pred).mean()), "a1t2_vs_final": float((decide_ev(pset) == pred).mean())}}
    for name, ref in (a.ref or []):
        rdf = pd.read_csv(ref); assert list(rdf["question_id"]) == qids; r = np.array([RANKS.index(x) for x in rdf["pred_rank"]])
        out[f"vs_{name}"] = {"file": ref, "sha256": sha(ref), "identical_rows": int((pred == r).sum()), "agree": float((pred == r).mean()), "per_N_agree": {int(n): float((pred[nn_ == n] == r[nn_ == n]).mean()) for n in (5, 10, 15, 20)}, "mean_pred_minus_ref": float((pred - r).mean())}
    import check_submission
    out["check_submission_rc"] = check_submission.main([str(f), "--task", "rank", "--data-dir", str(DATA)]); Path(a.report).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); assert out["check_submission_rc"] == 0
