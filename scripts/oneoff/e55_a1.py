"""E55 (b): A1full set model (DeepSets, CE, E48 `train_var`, E29 setmodel unchanged) with 48 per-fold PCA trunk columns appended to X139 -> 187 columns.  Per fold the PCA (fit on player_fold != k games only,
e55_pca.py) projects G_full; X187 is written to .scratch/e55/home_a1/data/X.npy (atomic replace) and Data(k) is built; width per fold = v6 A1full chosen_W (not re-selected); seeds 0-2.
Outputs .scratch/e55/out/A1T/{models,select_ds_fold*.json,norm_fold*.npz,setprobs_ds_W*_fold*_s*.npz} (E48 format).   --npc 0 gives the E55 control path (X139 only, must equal E48 C0 up to seed/platform noise; not needed for the criterion).
  gpujob systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 python3 scripts/oneoff/e55_a1.py [--folds 0 1 2 3 4] [--seeds 0 1 2]"""
import argparse, json, os, sys, gc, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; S = ROOT / ".scratch/e55"; HOME = S / "home_a1"
os.environ["E29_HOME"] = str(HOME); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch
(HOME / "data").mkdir(parents=True, exist_ok=True)
auxl = HOME / "data/aux.npz"
if not auxl.exists(): auxl.symlink_to(ROOT / ".scratch/e29/bundle/aux.npz")
from scripts.experiments.e29 import setmodel as sm
from scripts.oneoff import e48_train as T
A1 = ROOT / ".scratch/e37/runs/v6/out/A1full"; OUT = S / "out/A1T"
def project(G, pca, chunk=100_000):
    out = np.zeros((len(G), 48), np.float32); mu, sd, C = pca["mu"], pca["sd"], pca["comps"]
    for a in range(0, len(G), chunk):
        z = (np.asarray(G[a:a + chunk], np.float32) - mu) / sd; bad = ~np.isfinite(z).all(1); z[bad] = 0; out[a:a + chunk] = z @ C.T
    return out
def build_x(k):
    X = np.load(ROOT / ".scratch/e36/bundle/X120_full.npy", mmap_mode="r"); G = np.load(S / "G_full.npy", mmap_mode="r"); pca = dict(np.load(S / f"pca/fold{k}.npz"))
    P = project(G, pca); tmp = HOME / "data/X.tmp.npy"; o = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=np.float32, shape=(len(X), 139 + 48))
    for a in range(0, len(X), 100_000): o[a:a + 100_000, :139] = X[a:a + 100_000]; o[a:a + 100_000, 139:] = P[a:a + 100_000]
    o.flush(); del o; os.replace(tmp, HOME / "data/X.npy"); return P
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); a = ap.parse_args()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for k in a.folds:
        sel = json.loads((A1 / f"select_ds_fold{k}.json").read_text()); W = sel["chosen_W"]
        if all((OUT / f"setprobs_ds_W{W}_fold{k}_s{s}.npz").exists() for s in a.seeds): continue
        t0 = time.perf_counter(); build_x(k); D = sm.Data(k); assert D.F == 187, D.F
        (OUT / f"select_ds_fold{k}.json").write_text(json.dumps({**sel, "note": "chosen_W copied from v6 A1full (not re-selected); X187 = X139 + per-fold PCA-48 trunk"}, indent=1)); np.savez(OUT / f"norm_fold{k}.npz", **D.norm)
        for s in a.seeds:
            f = OUT / f"setprobs_ds_W{W}_fold{k}_s{s}.npz"
            if f.exists(): continue
            model, info = T.train_var(D, W, "ce", s)
            torch.save({"state": {kk: v.detach().cpu() for kk, v in model.state_dict().items()}, "kind": "ds", "W": W, "seed": s, "fold": k, "F": D.F, "info": {x: info[x] for x in ("best_step", "steps_run", "status", "es")}}, OUT / "models" / f"ds_W{W}_fold{k}_s{s}.pt")
            lp = sm.predict_logp(model, D, D.val["rows"]); assert np.isfinite(lp).all(); p = np.exp(lp).astype(np.float64); p /= p.sum(1, keepdims=True)
            np.savez(str(f) + ".tmp.npz", qid=D.val["qid"], p=p.astype(np.float32), info=json.dumps(info)); Path(str(f) + ".tmp.npz").rename(f)
            print("A1T fold", k, "seed", s, "W", W, "best_step", info["best_step"], "es_nll", round(info["es"]["nll"], 4), info["status"], info["seconds"], "s", flush=True)
        del D; gc.collect(); torch.cuda.empty_cache(); print("fold", k, "done", round(time.perf_counter() - t0), "s", flush=True)
    print("A1T_DONE")
