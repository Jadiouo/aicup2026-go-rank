"""E63 (copy of e59_predict.py; X235 = X139 | PCA_k(tk, E55) | PCA_k(lk, E63 late window)).  Original: E55 v7 test side, A1T leg: fold-k models take X187_k = [X139 | PCA_k(trunk)] (the PCA of THEIR fold, as in training), so valcheck/infer build the per-fold input on the fly.
  valcheck  (python3+torch, CPU)  each fold's models re-predict their own validation questions -> must reproduce saved OOF probabilities (max diff <=1e-3, decode share >=.999)
  infer     (python3+torch, CPU)  test X139 (.scratch/e35/test/test_X120.npy) + G_test projected with PCA_k -> [5,3,800,10]
  fuse      -> scripts/oneoff/e48_predict.py fuse (same arithmetic as e37 fuse; --w .5 = v6 recipe)"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; S = ROOT / ".scratch/e55"; S63 = ROOT / ".scratch/e63"; sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from scripts.oneoff import e48_predict as E48
from scripts.experiments.e37.predict_test import _torch_env, E29T
def project(G, pca, chunk=100_000):
    out = np.zeros((len(G), 48), np.float32); mu, sd, C = pca["mu"], pca["sd"], pca["comps"]
    for a in range(0, len(G), chunk):
        z = (np.asarray(G[a:a + chunk], np.float32) - mu) / sd; bad = ~np.isfinite(z).all(1); z[bad] = 0; out[a:a + chunk] = z @ C.T
    return out
def cmd_valcheck(a):
    torch, sm, p29 = _torch_env(); d = S63 / "out/A1T2"; aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); X = np.load(ROOT / ".scratch/e36/bundle/X120_full.npy", mmap_mode="r"); G = np.load(S / "G_full.npy", mmap_mode="r"); LF = np.load(S63 / "L_full.npy", mmap_mode="r"); res = {}
    for k in range(5):
        rows = aux[f"val_rows_f{k}"]; qid = aux[f"val_qid_f{k}"]; uniq = np.unique(rows[rows >= 0]); pca = dict(np.load(S / f"pca/fold{k}.npz"))
        Xs = np.hstack([np.asarray(X[uniq], np.float32), project(G[uniq], pca), project(LF[uniq], dict(np.load(S63 / f"pca/fold{k}.npz")))]); remap = np.full(rows.shape, -1, np.int64); mk = rows >= 0; remap[mk] = np.searchsorted(uniq, rows[mk])
        ms, norm, W = E48._models(torch, sm, d, k, 235); Pp = p29._predict(torch, ms, norm, Xs, remap)
        for s in range(3):
            z = np.load(d / f"setprobs_ds_W{W}_fold{k}_s{s}.npz", allow_pickle=False); assert (z["qid"] == qid).all(); df = np.abs(Pp[s] - z["p"].astype(np.float64))
            res[f"ds_f{k}_s{s}"] = {"max_abs_diff": float(df.max()), "decode_equal_share": float((np.argmax(Pp[s] @ p29._S(), 1) == np.argmax(z["p"].astype(np.float64) @ p29._S(), 1)).mean())}
        print(k, {kk: v for kk, v in res.items() if kk.startswith(f"ds_f{k}_")}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1)); worst = max(v["max_abs_diff"] for v in res.values()); wd = min(v["decode_equal_share"] for v in res.values()); print("WORST max_abs_diff", worst, "WORST decode_equal_share", wd)
    if worst > 1e-3 or wd < 0.999: sys.exit("valcheck FAILED")
def cmd_infer(a):
    torch, sm, p29 = _torch_env(); d = S63 / "out/A1T2"; X = np.load(ROOT / ".scratch/e35/test/test_X120.npy"); GT = np.load(S / "G_test.npy"); LT = np.load(S63 / "L_test.npy"); assert X.shape == (10000, 139) and GT.shape == (10000, 768)
    rows = np.load(E29T / "test_bags.npz")["rows"].astype(np.int64); allp = []
    for k in range(5):
        X187 = np.hstack([X, project(GT, dict(np.load(S / f"pca/fold{k}.npz"))), project(LT, dict(np.load(S63 / f"pca/fold{k}.npz")))]).astype(np.float32); ms, norm, W = E48._models(torch, sm, d, k, 235); allp.append(p29._predict(torch, ms, norm, X187, rows))
    allp = np.stack(allp); np.save(a.out, allp); print("set", allp.shape, "row-sum dev", float(np.abs(allp.sum(-1) - 1).max()))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp_ = ap.add_subparsers(dest="cmd", required=True)
    v = sp_.add_parser("valcheck"); v.add_argument("--out", required=True); i = sp_.add_parser("infer"); i.add_argument("--out", required=True)
    f = sp_.add_parser("fuse"); f.add_argument("--dse", required=True); f.add_argument("--set", required=True); f.add_argument("--w", type=float, default=0.5); f.add_argument("--out", required=True); f.add_argument("--report", required=True); f.add_argument("--ref", action="append", nargs=2, metavar=("NAME", "CSV"))
    a = ap.parse_args()
    if a.cmd == "fuse": E48.cmd_fuse(a)
    else: {"valcheck": cmd_valcheck, "infer": cmd_infer}[a.cmd](a)
