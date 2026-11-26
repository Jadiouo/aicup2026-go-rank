"""E79 local CPU prep worker (<= 2 workers): claims item pickles from the TAIL of remote-vm:~/gogame_e79/items (atomic `mv` into items_claimed on remote-vm; the remote-vm prep takes from the head), pulls them, and writes the same
blk files as e79_prep_remote.py into .scratch/e79/blocks/{name}_* (bits last).  Maths = e77_prep_full.py work24 (K=12 two windows).  Stops when no pickle is left (or ITEMS_DONE and none left).  Backpressure: <= --maxpend local blocks of bits.
usage: e79_prep_local.py [--workers 2] [--maxpend 12] [--stop-file F]"""
import os, sys, time, argparse, pickle, subprocess
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "oneoff")]
import numpy as np
from multiprocessing import Pool
from scripts.experiments.e13 import encode as E, lib
import e52_kata_prep as P1
from gogame.sgf import parse_sgf
KS_, CFG_ = P1.KS, P1.CFG; _enc = [None]
def late_idx(n_own, k):
    idx = lib.own_move_idx_eval(n_own, k=k, lo=60, hi=120)
    if len(idx) == 0: idx = lib.own_move_idx_eval(n_own, k=k, lo=max(2, n_own - 58), hi=n_own)
    return idx
def work24(chunk):
    if _enc[0] is None: _enc[0] = E.KataEncoder(KS_, CFG_)
    bl, gl, pl, rows, oi, wi = [], [], [], [], [], []
    for uid, color, sgf, rid in chunk:
        pg = parse_sgf(sgf)
        if E._skip_reason(pg): continue
        moves = pg.moves; n_own = int((moves[:, 0] == color).sum()); j = 0
        for w, idx in ((0, lib.own_move_idx_eval(n_own, k=12)), (1, late_idx(n_own, 12))):
            if len(idx) == 0: continue
            b, g, plies = _enc[0].encode_game(moves, int(color), idx)
            loc = np.array([(-1 if moves[p][1] < 0 else int(moves[p][1]) * 19 + int(moves[p][2])) for p in plies], np.int16)
            bl.append(np.packbits(b.reshape(len(b), -1), axis=1)); gl.append(g); pl.append(loc); oi.append(np.asarray(idx, np.int16)); wi.append(np.full(len(idx), w, np.int8))
            rows += [(rid, j + t) for t in range(len(plies))]; j += len(plies)
    return (np.concatenate(bl), np.concatenate(gl), np.concatenate(pl), np.array(rows), np.concatenate(oi), np.concatenate(wi)) if bl else None
def ssh(cmd): return subprocess.run(["ssh", os.environ.get("REMOTE_HOST", "remote-host"), cmd], capture_output=True, text=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--maxpend", type=int, default=12); ap.add_argument("--stop-file", default=""); ap.add_argument("--items-dir", default=""); a = ap.parse_args()
    OUT = ROOT / ".scratch/e79/blocks"; OUT.mkdir(parents=True, exist_ok=True); TMP = ROOT / ".scratch/e79/items_local"; TMP.mkdir(exist_ok=True); t0 = time.time()
    with Pool(a.workers) as pool:
        while True:
            if a.stop_file and Path(a.stop_file).exists(): print("stop file", flush=True); break
            while len(list(OUT.glob("*_bits.npy"))) >= a.maxpend: time.sleep(20)
            r = ssh("cd ~/gogame_e79/items && mkdir -p ../items_claimed && ls *.pkl 2>/dev/null | sed 's/.pkl//' | sort -V | tail -3; test -f ITEMS_DONE && echo ITEMSDONE"); toks = r.stdout.split()
            names = [x for x in toks if x != "ITEMSDONE"]; 
            if not names:
                if "ITEMSDONE" in toks: break
                time.sleep(30); continue
            nm = names[-1]
            m = ssh(f"cd ~/gogame_e79 && mv items/{nm}.pkl items_claimed/{nm}.pkl")
            if m.returncode != 0: continue
            subprocess.run(["rsync", "-a", "--remove-source-files", f"{os.environ.get('REMOTE_HOST', 'remote-host')}:~/gogame_e79/items_claimed/{nm}.pkl", str(TMP) + "/"], check=True)
            items = pickle.load(open(TMP / f"{nm}.pkl", "rb")); colmap = {it[3]: it[1] for it in items}
            chunks = [items[i:i + 100] for i in range(0, len(items), 100)]; res = [x for x in pool.map(work24, chunks, chunksize=1) if x is not None]; rw = np.concatenate([x[3] for x in res])
            np.save(OUT / f"{nm}_glob.npy", np.concatenate([x[1] for x in res])); np.save(OUT / f"{nm}_loc.npy", np.concatenate([x[2] for x in res])); np.save(OUT / f"{nm}_rows.npy", rw)
            np.save(OUT / f"{nm}_oidx.npy", np.concatenate([x[4] for x in res])); np.save(OUT / f"{nm}_win.npy", np.concatenate([x[5] for x in res])); np.save(OUT / f"{nm}_color.npy", np.array([colmap[int(x)] for x in rw[:, 0]], np.int8))
            np.save(OUT / f"{nm}_bits_tmp.npy", np.concatenate([x[0] for x in res])); os.replace(OUT / f"{nm}_bits_tmp.npy", OUT / f"{nm}_bits.npy"); (TMP / f"{nm}.pkl").unlink()
            print(time.strftime("%H:%M:%S"), nm, "games", len(items), "pos", len(rw), f"{time.time() - t0:.0f}s", flush=True)
    (ROOT / ".scratch/e79/LOCAL_DONE").write_text("ok"); print("local prep done", flush=True)
