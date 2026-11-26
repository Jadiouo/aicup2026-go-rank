"""E79 test side for A1P (only used if the PLAN section 3 criterion passes).
  valcheck : each fold's 3 models re-predict their own validation bags (DataP(k)) -> must reproduce the saved OOF probabilities (max diff <= 1e-3, decode share >= .999)
  infer    : test X139 (.scratch/e35/test/test_X120.npy; per-fold norm lo/hi/mu/sd) + test tokens (tok_test*.npy) -> [5,3,800,10]
  python3 scripts/oneoff/e79_predict.py valcheck --out J | infer --out NPY      (GPU via gpujob, or CPU with CUDA_VISIBLE_DEVICES="")"""
import argparse, json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts" / "oneoff")]
import numpy as np, torch
import e79_a1p as M
sm = M.sm
def load_model(k, s, W, bank, F=139):
    z = torch.load(M.OUT / "models" / f"a1p_W{W}_fold{k}_s{s}.pt", map_location="cpu"); st = np.load(M.OUT / f"tokstats_fold{k}.npz")
    m = M.A1P(F, W, sm.HP["drop"], z["p_in"], z["tdrop"], bank, st["mu"], st["sd"]).to(M.DEV); m.load_state_dict(z["state"], strict=False); m.eval(); return m
def cmd_valcheck(a):
    bank = M.TokenBank(); aux = np.load(M.HOME / "data/aux.npz", allow_pickle=False); bank.gcol = torch.from_numpy(aux["game_color"].astype(np.int64)).to(M.DEV); res = {}
    for k in range(5):
        W = json.loads((M.A1 / f"select_ds_fold{k}.json").read_text())["chosen_W"]; D = M.DataP(k)
        for s in range(3):
            m = load_model(k, s, W, bank); lp = sm.predict_logp(m, D, D.val["rows"]); p = np.exp(lp).astype(np.float64); p /= p.sum(1, keepdims=True)
            z = np.load(M.OUT / f"setprobs_ds_W{W}_fold{k}_s{s}.npz", allow_pickle=False); q = z["p"].astype(np.float64); assert (z["qid"] == D.val["qid"]).all()
            res[f"f{k}_s{s}"] = {"max_abs_diff": float(np.abs(p - q).max()), "decode_equal_share": float((sm.ev_decode(p) == sm.ev_decode(q)).mean())}; print(k, s, res[f"f{k}_s{s}"], flush=True)
        del D
    Path(a.out).write_text(json.dumps(res, indent=1)); worst = max(v["max_abs_diff"] for v in res.values()); wd = min(v["decode_equal_share"] for v in res.values()); print("WORST", worst, wd)
    if worst > 1e-3 or wd < 0.999: sys.exit("valcheck FAILED")
def cmd_infer(a):
    bank = M.TokenBank("tok_test"); bank.gcol = torch.from_numpy(np.load(M.TOK / "tok_test_gcol.npy").astype(np.int64)).to(M.DEV); assert (bank.gcol.cpu().numpy() >= 0).all()
    X = np.load(ROOT / ".scratch/e35/test/test_X120.npy"); assert X.shape == (10000, 139); rows = torch.from_numpy(np.load(ROOT / ".scratch/e29/test/test_bags.npz")["rows"].astype(np.int64)).to(M.DEV); allp = []
    for k in range(5):
        nz = np.load(M.OUT / f"norm_fold{k}.npz"); x = (np.clip(X.astype(np.float64), nz["lo"], nz["hi"]) - nz["mu"]) / nz["sd"]; Xg = torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float32)).to(M.DEV)
        W = json.loads((M.A1 / f"select_ds_fold{k}.json").read_text())["chosen_W"]; ps = []
        for s in range(3):
            m = load_model(k, s, W, bank); out = []
            with torch.no_grad():
                for i in range(0, len(rows), 256):
                    r = rows[i:i + 256]; rc = r.clamp(min=0); out.append(m(torch.cat([Xg[rc], rc.float()[..., None]], -1), r >= 0).float().cpu())
            p = torch.cat(out).exp().numpy().astype(np.float64); p /= p.sum(1, keepdims=True); ps.append(p)
        allp.append(np.stack(ps))
    allp = np.stack(allp); np.save(a.out, allp); print("set", allp.shape, "row-sum dev", float(np.abs(allp.sum(-1) - 1).max()))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp_ = ap.add_subparsers(dest="cmd", required=True); sp_.add_parser("valcheck").add_argument("--out", required=True); sp_.add_parser("infer").add_argument("--out", required=True); a = ap.parse_args()
    {"valcheck": cmd_valcheck, "infer": cmd_infer}[a.cmd](a)
