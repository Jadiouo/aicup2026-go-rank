"""E79 GPU half (system python3 + torch; submit with gpujob).  Streams the blocks pulled from remote-vm into .scratch/e79/blocks/{name}_{bits,glob,loc,rows,color,oidx,win}.npy (e79_prep_remote.py / e79_pull.py):
per position u=[at|mean] (768, act_trunkfinal, e70_gpu.trunk_u) -> standardise with E77 pca_full.npz (mu, sd) -> first 64 PCA components (E77 PCA, fitted on Player-side positions only; PLAN section 1) -> fp16 {name}_pc.npy; bits/glob/loc deleted after use.
--check first: 200 bundle games re-extracted with K=6 (front window, = E55 positions) -> per-game mean u vs .scratch/e55/G_full.npy (PLAN section 1 'same-model check'): max abs diff <= .005, per-dim corr >= .9999, else abort.
Exits when PULL_DONE exists and nothing is left, or after --idle-exit seconds without a ready block (resumable: skips blocks whose _pc exists)."""
import sys, os, time, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(ROOT / "src"), str(ROOT)]
import numpy as np, torch
import e70_gpu as G
K = ROOT / ".scratch" / "e79"; B = K / "blocks"; B.mkdir(exist_ok=True); dev = G.dev; log = G.log
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=str(K / "ckpt.ckpt")); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--idle-exit", type=int, default=3600); ap.add_argument("--check", action="store_true"); ap.add_argument("--no-run", action="store_true"); a = ap.parse_args()
model = G.load(a.ckpt); z = np.load(ROOT / ".scratch/e77/pca_full.npz"); MU, SD, COMPS = z["mu"], z["sd"], z["comps"][:64]; assert COMPS.shape == (64, 768)
@torch.no_grad()
def u_block(bits, glob, loc, color):
    N = len(bits); out = np.zeros((N, 768), np.float16); loc = loc.astype(np.int64)
    for s in range(0, N, a.bs):
        e = min(s + a.bs, N); b = np.unpackbits(np.asarray(bits[s:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
        bx = torch.from_numpy(b).to(dev).float(); gx = torch.from_numpy(glob[s:e]).to(dev); mx = torch.stack([G.MROWS[int(c)] for c in color[s:e]]).to(dev); l = torch.from_numpy(loc[s:e]).to(dev)
        out[s:e] = G.trunk_u(model, bx, gx, mx, l).cpu().numpy().astype(np.float16)
    return out
def check():
    z = np.load(K / "check_in.npz"); uu = u_block(z["bits"], z["glob"], z["loc"], z["color"]).astype(np.float32); gi = z["gi"]; GF = np.load(ROOT / ".scratch/e55/G_full.npy", mmap_mode="r")
    mine = np.array([uu[gi == g].mean(0) for g in range(len(z["rid"]))]); ref = np.array([np.asarray(GF[int(r)], np.float32) for r in z["rid"]]); d = np.abs(mine - ref); cor = np.array([np.corrcoef(mine[:, j], ref[:, j])[0, 1] for j in range(768)])
    ulp = np.spacing(np.abs(ref).astype(np.float16)).astype(np.float32); du = d / np.maximum(ulp, 1e-8)       # |diff| in fp16 ulps of the (fp16-stored) reference
    r = {"n_games": len(mine), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "min_dim_corr": float(np.nanmin(cor)), "median_dim_corr": float(np.nanmedian(cor)),
         "max_diff_ulps": float(du.max()), "share_within_1ulp": float((du <= 1.0).mean()),
         "pass_original": bool(d.max() <= .005 and np.nanmin(cor) >= .9999), "pass_amend1_ulp": bool(du.max() <= 2.0 and (du <= 1.0).mean() >= .999 and np.nanmedian(cor) >= .9999 and np.nanmin(cor) >= .999),
         "pass": bool(np.nanmedian(cor) >= .9999 and np.nanmin(cor) >= .999 and d.mean() <= .001)}
    # cross-platform comparison (remote-vm GPU vs this GPU) on the block remote-vm kept in .scratch/e79/cmp
    cmpd = K / "cmp"
    if cmpd.exists():
        nm = sorted(p.name[:-len("_bits.npy")] for p in cmpd.glob("*_bits.npy"))[0]; bits = np.load(cmpd / f"{nm}_bits.npy", mmap_mode="r"); n = min(len(bits), 20000)
        U = u_block(bits[:n], np.load(cmpd / f"{nm}_glob.npy")[:n], np.load(cmpd / f"{nm}_loc.npy")[:n], np.load(cmpd / f"{nm}_color.npy")[:n]); P = (((U.astype(np.float32) - MU) / SD) @ COMPS.T).astype(np.float32); Q = np.load(cmpd / f"{nm}_pc.npy")[:n].astype(np.float32)
        dd = np.abs(P - Q); r["cross_platform"] = {"block": nm, "n_pos": int(n), "max_abs_diff": float(dd.max()), "mean_abs_diff": float(dd.mean()), "pc_std": float(Q.std()), "min_dim_corr": float(np.nanmin([np.corrcoef(P[:, j], Q[:, j])[0, 1] for j in range(64)]))}
    if "cross_platform" in r: cp = r["cross_platform"]; r["pass_cross_platform"] = bool(cp["max_abs_diff"] <= .05 * cp["pc_std"] and cp["min_dim_corr"] >= .999); r["pass"] = bool(r["pass"] and r["pass_cross_platform"])
    (K / "check_e55.json").write_text(json.dumps(r, indent=1)); log("CHECK", r)
    if not r["pass"]: sys.exit("same-model check FAILED")
def run():
    t0 = time.time(); tot = 0; last = time.time()
    while True:
        ready = sorted(p.name[:-len("_bits.npy")] for p in B.glob("*_bits.npy") if not (B / (p.name[:-len("_bits.npy")] + "_pc.npy")).exists() and (B / (p.name[:-len("_bits.npy")] + "_win.npy")).exists())
        if ready:
            nm = ready[0]; bits = np.load(B / f"{nm}_bits.npy", mmap_mode="r"); glob = np.load(B / f"{nm}_glob.npy"); loc = np.load(B / f"{nm}_loc.npy"); color = np.load(B / f"{nm}_color.npy")
            U = u_block(bits, glob, loc, color); P = (((U.astype(np.float32) - MU) / SD) @ COMPS.T).astype(np.float16); assert np.isfinite(P).all(); assert len(P) == len(bits)
            np.save(B / f"{nm}_pc_tmp.npy", P); os.replace(B / f"{nm}_pc_tmp.npy", B / f"{nm}_pc.npy"); del bits
            for s in ("bits", "glob", "loc"): os.remove(B / f"{nm}_{s}.npy")
            tot += len(P); last = time.time(); log(nm, "positions so far", tot, f"{time.time() - t0:.0f}s maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"); continue
        if (K / "PULL_DONE").exists(): break
        if time.time() - last > a.idle_exit: log("idle exit"); break
        time.sleep(10)
    log("done", tot)
def check_ok():
    f = K / "check_e55.json"; return f.exists() and json.loads(f.read_text()).get("pass") is True and "max_diff_ulps" in json.loads(f.read_text())
if a.check or not check_ok(): check()
if not a.no_run: run()
