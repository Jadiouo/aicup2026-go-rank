"""E48 test side: e37 predict_test generalised to F=139/140 and a fusion weight w.  Same arithmetic as e37 fuse at w=.5 (must reproduce rank_v6 bytes).
  valcheck --models DIR --aux AUX --x X --F 140 --out J     (python3+torch, CPU)
  infer    --models DIR --x TEST_X --F 140 --out NPY        (python3+torch, CPU)
  fuse     --dse NPY --set NPY --w 0.5 --out CSV --report J [--ref NAME CSV]   (.venv)"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from scripts.experiments.e37.predict_test import _torch_env, sha, K, DATA, E29T

def _models(torch, sm, d, k, F):
    W = json.loads((d / f"select_ds_fold{k}.json").read_text())["chosen_W"]; ms = []
    for s in (0, 1, 2):
        ck = torch.load(d / "models" / f"ds_W{W}_fold{k}_s{s}.pt", map_location="cpu", weights_only=False)
        assert ck["kind"] == "ds" and ck["W"] == W and ck["fold"] == k and ck["seed"] == s and ck["F"] == F
        m = sm.build("ds", F, W, 0.2); m.load_state_dict(ck["state"]); m.eval(); ms.append(m)
    nz = np.load(d / f"norm_fold{k}.npz"); assert len(nz["mu"]) == F
    return ms, {x: nz[x] for x in ("lo", "hi", "mu", "sd")}, W

def cmd_valcheck(a):
    torch, sm, p29 = _torch_env(); d = Path(a.models); aux = np.load(a.aux, allow_pickle=False); Xb = np.load(a.x, mmap_mode="r"); res = {}
    for k in range(5):
        rows = aux[f"val_rows_f{k}"]; qid = aux[f"val_qid_f{k}"]; uniq = np.unique(rows[rows >= 0]); Xs = np.asarray(Xb[uniq], np.float32)
        remap = np.full(rows.shape, -1, np.int64); mk = rows >= 0; remap[mk] = np.searchsorted(uniq, rows[mk])
        ms, norm, W = _models(torch, sm, d, k, a.F); Pp = p29._predict(torch, ms, norm, Xs, remap)
        for s in range(3):
            z = np.load(d / f"setprobs_ds_W{W}_fold{k}_s{s}.npz", allow_pickle=False); assert (z["qid"] == qid).all(); df = np.abs(Pp[s] - z["p"].astype(np.float64))
            res[f"ds_f{k}_s{s}"] = {"max_abs_diff": float(df.max()), "decode_equal_share": float((np.argmax(Pp[s] @ p29._S(), 1) == np.argmax(z["p"].astype(np.float64) @ p29._S(), 1)).mean())}
        print(k, {kk: v for kk, v in res.items() if kk.startswith(f"ds_f{k}_")}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1)); worst = max(v["max_abs_diff"] for v in res.values()); wd = min(v["decode_equal_share"] for v in res.values())
    print("WORST max_abs_diff", worst, "WORST decode_equal_share", wd)
    if worst > 1e-3 or wd < 0.999: sys.exit("valcheck FAILED")

def cmd_infer(a):
    torch, sm, p29 = _torch_env(); d = Path(a.models); X = np.load(a.x); assert X.shape == (10000, a.F) and X.dtype == np.float32, X.shape
    rows = np.load(E29T / "test_bags.npz")["rows"].astype(np.int64); allp = []
    for k in range(5):
        ms, norm, W = _models(torch, sm, d, k, a.F); allp.append(p29._predict(torch, ms, norm, X, rows))
    allp = np.stack(allp); np.save(a.out, allp); print("set", allp.shape, "row-sum dev", float(np.abs(allp.sum(-1) - 1).max()), "x sha", sha(a.x)[:12], "out sha", sha(a.out)[:12])

def cmd_fuse(a):
    import pandas as pd
    from gogame.constants import RANKS
    from gogame.models import decide_ev
    from gogame.submission import write_rank_submission
    dse = np.load(a.dse); bg = np.load(E29T / "test_bags.npz"); qids = bg["qids"].tolist(); nn_ = bg["n"]; assert dse.shape == (800, 10)
    sp = np.load(a.set); assert sp.shape == (5, 3, 800, 10); pset = sp.mean((0, 1)); pset /= pset.sum(1, keepdims=True)
    w = a.w; p = w * dse + (1 - w) * pset; p /= p.sum(1, keepdims=True); pred = decide_ev(p)
    f = Path(a.out); write_rank_submission(qids, pred, f)
    out = {"file": str(f), "sha256": sha(f), "w_dse": w, "n_classes_predicted": int(len(np.unique(pred))), "pred_share": np.bincount(pred, minlength=K).tolist()}
    for name, ref in (a.ref or []):
        rdf = pd.read_csv(ref); assert list(rdf["question_id"]) == qids; r = np.array([RANKS.index(x) for x in rdf["pred_rank"]])
        out[f"vs_{name}"] = {"file": ref, "sha256": sha(ref), "identical_rows": int((pred == r).sum()), "agree": float((pred == r).mean()), "per_N_agree": {int(n): float((pred[nn_ == n] == r[nn_ == n]).mean()) for n in (5, 10, 15, 20)},
                             "mean_pred_minus_ref": float((pred - r).mean()), "file_bytes_identical": bool(Path(f).read_bytes() == Path(ref).read_bytes())}
    import check_submission
    out["check_submission_rc"] = check_submission.main([str(f), "--task", "rank", "--data-dir", str(DATA)]); Path(a.report).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    assert out["check_submission_rc"] == 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp_ = ap.add_subparsers(dest="cmd", required=True)
    v = sp_.add_parser("valcheck"); v.add_argument("--models", required=True); v.add_argument("--aux", required=True); v.add_argument("--x", required=True); v.add_argument("--F", type=int, required=True); v.add_argument("--out", required=True)
    i = sp_.add_parser("infer"); i.add_argument("--models", required=True); i.add_argument("--x", required=True); i.add_argument("--F", type=int, required=True); i.add_argument("--out", required=True)
    f = sp_.add_parser("fuse"); f.add_argument("--dse", required=True); f.add_argument("--set", required=True); f.add_argument("--w", type=float, required=True); f.add_argument("--out", required=True); f.add_argument("--report", required=True)
    f.add_argument("--ref", action="append", nargs=2, metavar=("NAME", "CSV"))
    a = ap.parse_args(); {"valcheck": cmd_valcheck, "infer": cmd_infer, "fuse": cmd_fuse}[a.cmd](a)
