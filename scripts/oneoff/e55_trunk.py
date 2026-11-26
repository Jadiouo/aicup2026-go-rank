"""E55 step 2 GPU half: KataGo b18c384nbt-humanv0 trunk (act_trunkfinal) per-GAME vectors = mean over the game's positions of [trunk at the played move (384) | trunk board mean (384)] -> fp16 (n_games, 768).
Reads .scratch/e55/{blk|tblk}{i}_{bits,glob,loc,rows,color}.npy written by e55_prep.py, writes {prefix}{i}_gv.npy / _gid.npy (gid = bundle row, or test_bags row for tblk) and deletes the block's bits.
System python3 + torch, submit with gpujob.  Processes ready blocks as they appear; exits when PREP_DONE and nothing is left, or after --idle-exit seconds without a ready block (so the GPU slot is not held while the CPU side is slow;
the scheduler resubmits).  Own copy of the E52 trunk2 logic (the E52 files are not touched)."""
import os, sys, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, torch
K = ROOT / ".scratch" / "e55"; ks = str(ROOT / ".scratch" / "e52" / "katago_src")
mr = np.load(ROOT / ".scratch" / "e52" / "kata" / "meta_rows.npy"); mrows = {0: torch.tensor(mr[0]), 1: torch.tensor(mr[1])}
def load_katago_trunk(ckpt, katago_src, device):
    sys.path.insert(0, str(Path(katago_src) / "python"))
    from katago.train.load_model import load_model
    model, _, _ = load_model(ckpt, use_swa=False, device=device, pos_len=19)
    class StopFwd(Exception):
        def __init__(self, x): self.x = x
    def hook(m, i, o): raise StopFwd(o)
    model.act_trunkfinal.register_forward_hook(hook)
    def trunk(bx, gx, mx):
        try: model(bx, gx, input_meta=mx)
        except StopFwd as e: return e.x
        raise RuntimeError("trunk hook did not fire")
    return model, trunk
DEV = "cuda"
model, trunk = load_katago_trunk(str(ROOT / ".scratch" / "e52" / "b18c384nbt-humanv0.ckpt"), ks, DEV)
def run_block(name):
    bits = np.load(K / f"{name}_bits.npy", mmap_mode="r"); glob = np.load(K / f"{name}_glob.npy"); loc = np.load(K / f"{name}_loc.npy"); rows = np.load(K / f"{name}_rows.npy"); color = np.load(K / f"{name}_color.npy")
    N = len(bits); vec = np.zeros((N, 768), np.float16); BS = 256
    with torch.no_grad():
        for a in range(0, N, BS):
            e = min(a + BS, N)
            b = np.unpackbits(np.asarray(bits[a:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
            bx = torch.from_numpy(b).to(DEV).float(); gx = torch.from_numpy(glob[a:e]).to(DEV); mx = torch.stack([mrows[int(c)] for c in color[a:e]]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16): f = trunk(bx, gx, mx).float()
            bm = f.mean((2, 3)); l = torch.from_numpy(loc[a:e].astype(np.int64)).to(DEV); pas = l < 0
            v = f.flatten(2).gather(2, l.clamp(min=0)[:, None, None].expand(-1, 384, 1))[:, :, 0]; v[pas] = bm[pas]
            vec[a:e] = torch.cat([v, bm], 1).cpu().numpy()
    gid, inv = np.unique(rows[:, 0], return_inverse=True); cnt = np.bincount(inv).astype(np.float32)
    gv = np.zeros((len(gid), 768), np.float32); np.add.at(gv, inv, vec.astype(np.float32)); gv /= cnt[:, None]
    assert np.isfinite(gv).all()
    np.save(K / f"{name}_gid.npy", gid); np.save(K / f"{name}_gv.tmp.npy", gv.astype(np.float16)); (K / f"{name}_gv.tmp.npy").rename(K / f"{name}_gv.npy")
    os.remove(K / f"{name}_bits.npy")
    return N
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--idle-exit", type=float, default=480); a = ap.parse_args()
    t0 = time.perf_counter(); tot = 0; idle = time.perf_counter()
    while True:
        ready = sorted(p.name[:-len("_bits.npy")] for p in K.glob("*blk*_bits.npy"))
        if ready:
            for nm in ready: tot += run_block(nm); print(f"{nm} done, positions {tot}, {time.perf_counter() - t0:.0f}s", flush=True)
            idle = time.perf_counter(); continue
        if (K / "PREP_DONE").exists(): break
        if time.perf_counter() - idle > a.idle_exit: print("idle exit", flush=True); break
        time.sleep(10)
    print("trunk exit", f"{time.perf_counter() - t0:.0f}s positions {tot}", flush=True)
