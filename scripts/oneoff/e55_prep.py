"""E55 step 2 prep: KataGo input planes for ALL non-H bundle games (aux uid order, 712,945) and the 10,000 Rank test games (test_bags uid order), K=6 own-move positions per game
(evenly spaced in own moves [2,60), same as E52 prep2), in blocks = one 10k-row CSV chunk filtered to the bundle -> .scratch/e55/{blk|tblk}{i}_{bits,glob,loc,rows,color}.npy.
CPU only (4 workers, nice).  Backpressure: waits while >= MAXP blocks are waiting for the GPU (e55_trunk.py deletes bits after it writes gv).  Resumable (skips blocks whose gv exists).
Reuses E13's KataEncoder read-only.  No H: the bundle has none (asserted against the split).  Usage: e55_prep.py [--workers 4] [--maxp 24]"""
import os, sys, time, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import json, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from scripts.experiments.e13 import encode as E, lib
from gogame.sgf import parse_sgf
OUT = ROOT / ".scratch" / "e55"; OUT.mkdir(parents=True, exist_ok=True)
KS = str(ROOT / ".scratch" / "e52" / "katago_src"); CFG = json.load(open(ROOT / ".scratch" / "e52" / "model_config.json"))
K = 6; _enc = None
def work(chunk):
    global _enc
    if _enc is None: _enc = E.KataEncoder(KS, CFG)
    bl, gl, pl, rows = [], [], [], []
    for uid, color, sgf, rid in chunk:
        try: pg = parse_sgf(sgf)
        except Exception: continue
        if E._skip_reason(pg): continue
        moves = pg.moves; n_own = int((moves[:, 0] == color).sum()); idx = lib.own_move_idx_eval(n_own, k=K)
        if len(idx) == 0: continue
        b, g, plies = _enc.encode_game(moves, int(color), idx)
        loc = np.array([(-1 if moves[p][1] < 0 else int(moves[p][1]) * 19 + int(moves[p][2])) for p in plies], np.int16)
        bl.append(np.packbits(b.reshape(len(b), -1), axis=1)); gl.append(g); pl.append(loc)
        rows += [(rid, j) for j in range(len(plies))]
    return (np.concatenate(bl), np.concatenate(gl), np.concatenate(pl), np.array(rows)) if bl else None
def save_block(pool, prefix, bi, items, color_by_row):
    chunks = [items[i:i + 100] for i in range(0, len(items), 100)]
    res = [r for r in pool.map(work, chunks, chunksize=1) if r is not None]
    rw = np.concatenate([r[3] for r in res])
    np.save(OUT / f"{prefix}{bi}_glob.npy", np.concatenate([r[1] for r in res])); np.save(OUT / f"{prefix}{bi}_loc.npy", np.concatenate([r[2] for r in res]))
    np.save(OUT / f"{prefix}{bi}_rows.npy", rw); np.save(OUT / f"{prefix}{bi}_color.npy", color_by_row[rw[:, 0]].astype(np.int8))
    np.save(OUT / f"{prefix}{bi}_bits.tmp.npy", np.concatenate([r[0] for r in res])); (OUT / f"{prefix}{bi}_bits.tmp.npy").rename(OUT / f"{prefix}{bi}_bits.npy")   # bits last = ready flag
    return len(items), len(set(rw[:, 0].tolist()))
def pending(): return len(list(OUT.glob("*blk*_bits.npy")))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--maxp", type=int, default=24); ap.add_argument("--only-test", action="store_true"); a = ap.parse_args()
    t0 = time.perf_counter()
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); uid = aux["uid"].astype(str); gcol = aux["game_color"].astype(np.int8)
    row_of = pd.Series(np.arange(len(uid), dtype=np.int64), index=uid)
    split = pd.read_parquet(ROOT / "data/splits/split.parquet")
    H = set(split.loc[split["is_holdout"], "player_id"]) if "is_holdout" in split else set()
    meta = pd.read_parquet(ROOT / ".scratch/e37/dse_home/data/cache/games_meta.parquet"); assert set(meta["uid"]) == set(uid) and not (set(meta["player_id"]) & H)
    print("bundle games", len(uid), "H players in bundle", len(set(meta["player_id"]) & H), flush=True)
    with Pool(a.workers, maxtasksperchild=200) as pool:
        # ---- test games (tblk0)
        if not (OUT / "tblk0_gv.npy").exists() and not (OUT / "tblk0_bits.npy").exists():
            bg = np.load(ROOT / ".scratch/e29/test/test_bags.npz", allow_pickle=True); tu = [str(u) for u in bg["uids"]]; assert len(tu) == 10000
            tab = {}
            for f in ("public", "private"):
                d = pd.read_csv(ROOT / f"data/contest_1104/rank_prediction_test_{f}.csv")
                for _, r in d.iterrows():
                    for i in range(1, int(r.num_games) + 1): tab[f"{r.question_id}#{i}"] = (r[f"sgf_{i}"], 0 if r[f"color_{i}"] == "B" else 1)
            items = [(u, tab[u][1], tab[u][0], i) for i, u in enumerate(tu)]; tcol = np.array([it[1] for it in items], np.int8)
            print("test block", save_block(pool, "tblk", 0, items, tcol), f"{time.perf_counter() - t0:.0f}s", flush=True)
        if a.only_test: sys.exit(0)
        # ---- training games: stream CSV chunks, prefetch next chunk in a thread
        files = sorted((ROOT / "data" / "training").glob("train_*.csv"))
        def chunks():
            for f in files:
                for c, ch in enumerate(pd.read_csv(f, usecols=["game_id", "rank", "sgf_content"], chunksize=10_000)):
                    ch["uid"] = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str); yield f.name, c, ch[ch["uid"].isin(row_of.index)]
        it = chunks(); bi = 0; tot = 0; seen = set(); ex = ThreadPoolExecutor(1); nxt = ex.submit(lambda: next(it, None))
        while True:
            cur = nxt.result()
            if cur is None: break
            nxt = ex.submit(lambda: next(it, None)); fn, c, ch = cur
            if len(ch) == 0: continue
            if (OUT / f"blk{bi}_gv.npy").exists(): bi += 1; seen |= set(ch["uid"]); continue
            while pending() >= a.maxp: time.sleep(20)
            rid = row_of.reindex(ch["uid"]).to_numpy(); items = [(u, int(gcol[r]), s, int(r)) for u, s, r in zip(ch["uid"], ch["sgf_content"], rid)]; seen |= set(ch["uid"])
            n, ng = save_block(pool, "blk", bi, items, gcol); tot += n
            print(f"block {bi} {fn}#{c}: games {n} encoded {ng} total {tot} {time.perf_counter() - t0:.0f}s pending {pending()}", flush=True); bi += 1
        assert len(seen) == len(uid), (len(seen), len(uid)); (OUT / "PREP_DONE").write_text(str(bi))
    print("prep done", bi, "blocks", f"{time.perf_counter() - t0:.0f}s", flush=True)
