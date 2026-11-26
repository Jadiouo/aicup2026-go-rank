"""E52 step 5 prep: KataGo input planes for a PILOT subset of the sim rows (800 train players x 12 games + 400 eval players x 15 games), 12 evenly spaced own-move positions per game.
Reuses E13's KataEncoder (scripts/experiments/e13/encode.py).  CPU, multiprocessing.  Output .scratch/e52/kata/pilot_{bits,glob,pos}.npy / rows.parquet.
H players never appear (sim meta has no H)."""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import json, numpy as np, pandas as pd
from multiprocessing import Pool
from scripts.experiments.e13 import encode as E, lib
from gogame.sgf import parse_sgf
OUT = ROOT / ".scratch" / "e52" / "kata"; OUT.mkdir(parents=True, exist_ok=True)
KS = str(ROOT / ".scratch" / "e52" / "katago_src"); CFG = json.load(open(ROOT / ".scratch" / "e52" / "model_config.json"))
K = 12
_enc = None
def work(chunk):
    global _enc
    if _enc is None: _enc = E.KataEncoder(KS, CFG)
    bl, gl, pl, rows = [], [], [], []
    for uid, color, sgf, rid in chunk:
        pg = parse_sgf(sgf)
        if E._skip_reason(pg): continue
        moves = pg.moves; n_own = int((moves[:, 0] == color).sum()); idx = lib.own_move_idx_eval(n_own, k=K)
        if len(idx) == 0: continue
        b, g, plies = _enc.encode_game(moves, int(color), idx)
        loc = np.array([(-1 if moves[p][1] < 0 else int(moves[p][1]) * 19 + int(moves[p][2])) for p in plies], np.int16)
        bl.append(np.packbits(b.reshape(len(b), -1), axis=1)); gl.append(g); pl.append(loc)
        rows += [(rid, j) for j in range(len(plies))]
    return (np.concatenate(bl), np.concatenate(gl), np.concatenate(pl), np.array(rows)) if bl else None
if __name__ == "__main__":
    t0 = time.perf_counter()
    m = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet"); rng = np.random.default_rng(52)
    pick = []
    for grp, npl, ng in (("train", 800, 12), ("eval", 400, 15)):
        mm = m[m.grp == grp]; cnt = mm.groupby("player_id").size(); el = cnt[cnt >= 28].index.to_numpy()
        for p in rng.choice(el, npl, replace=False):
            r = mm.index[mm.player_id == p].to_numpy(); pick += list(rng.permutation(r)[:ng])
    sel = m.loc[np.sort(pick)].copy(); sel["row"] = sel.index.to_numpy(); print("rows", len(sel), sel.groupby("grp").player_id.nunique().to_dict(), flush=True)
    sg = E.read_sgf_for_uids(lib.TRAIN_CSV_DIR, set(sel.uid)); print("sgf read", len(sg), f"{time.perf_counter() - t0:.0f}s", flush=True)
    sgm = dict(zip(sg.uid, sg.sgf)); items = [(r.uid, int(r.color), sgm[r.uid], int(r.row)) for r in sel.itertuples()]
    chunks = [items[i:i + 100] for i in range(0, len(items), 100)]
    with Pool(6) as pool: res = pool.map(work, chunks, chunksize=1)
    res = [r for r in res if r is not None]
    np.save(OUT / "pilot_bits.npy", np.concatenate([r[0] for r in res])); np.save(OUT / "pilot_glob.npy", np.concatenate([r[1] for r in res]))
    np.save(OUT / "pilot_loc.npy", np.concatenate([r[2] for r in res])); rw = np.concatenate([r[3] for r in res]); np.save(OUT / "pilot_rows.npy", rw)
    sel.to_parquet(OUT / "pilot_sel.parquet"); print("done positions", len(rw), f"{time.perf_counter() - t0:.0f}s", flush=True)
