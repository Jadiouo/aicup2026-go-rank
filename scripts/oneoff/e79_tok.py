"""E79: assemble per-game token arrays from .scratch/e79/blocks/{name}_{pc,rows,oidx,win}.npy.  Bundle blocks b* -> tok_full.npy (712,945,24,64) fp16, tok_oidx.npy (int8 bucket own-move idx//5, 0..23), tok_mask.npy (bool); test t0 -> tok_test*.npy.
Asserts 100% game coverage (every bundle game / every test game has >= 1 position).  usage: e79_tok.py [--delete-blocks]"""
import sys, glob, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; K = ROOT / ".scratch/e79"; B = K / "blocks"
ap = argparse.ArgumentParser(); ap.add_argument("--delete-blocks", action="store_true"); a = ap.parse_args()
def load(names): 
    return [(n, np.load(B / f"{n}_pc.npy"), np.load(B / f"{n}_rows.npy"), np.load(B / f"{n}_oidx.npy")) for n in names]
def assemble(names, n_games, prefix):
    X = np.lib.format.open_memmap(str(K / f"{prefix}.tmp.npy"), mode="w+", dtype=np.float16, shape=(n_games, 24, 64)); X[:] = 0
    od = np.zeros((n_games, 24), np.int8); mk = np.zeros((n_games, 24), bool); gc = np.full(n_games, -1, np.int8); tot = 0
    for n in names:
        pc = np.load(B / f"{n}_pc.npy"); rows = np.load(B / f"{n}_rows.npy"); oi = np.load(B / f"{n}_oidx.npy"); gc[rows[:, 0]] = np.load(B / f"{n}_color.npy"); assert len(pc) == len(rows) == len(oi) and rows[:, 1].max() < 24
        g, j = rows[:, 0], rows[:, 1]; assert not mk[g, j].any(); X[g, j] = pc; od[g, j] = np.minimum(oi // 5, 23).astype(np.int8); mk[g, j] = True; tot += len(pc)
    X.flush(); del X; (K / f"{prefix}.tmp.npy").rename(K / f"{prefix}.npy"); np.save(K / f"{prefix}_oidx.npy", od); np.save(K / f"{prefix}_mask.npy", mk); np.save(K / f"{prefix}_gcol.npy", gc)
    cov = int((mk.sum(1) > 0).sum()); print(prefix, "games", n_games, "with tokens", cov, "positions", tot, "dist", np.bincount(mk.sum(1)).tolist(), flush=True); assert cov == n_games, f"coverage {cov}/{n_games}"
    return {"games": n_games, "covered": cov, "positions": tot}
names = sorted((p.name[:-len("_pc.npy")] for p in B.glob("b*_pc.npy")), key=lambda s: int(s[1:])); info = {}
info["bundle"] = assemble(names, 712_945, "tok_full"); info["test"] = assemble(["t0"], 10_000, "tok_test"); (K / "tok_info.json").write_text(json.dumps(info))
if a.delete_blocks:
    for p in list(B.glob("*_pc.npy")) + list(B.glob("*_rows.npy")) + list(B.glob("*_oidx.npy")) + list(B.glob("*_win.npy")) + list(B.glob("*_color.npy")): p.unlink()
