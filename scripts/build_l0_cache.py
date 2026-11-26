"""Compute L0 features for all training games (multi-core) and store them as a GameTable cache (SDD 6.1).

  .venv/bin/python scripts/build_l0_cache.py [--jobs 14] [--chunk 20000]

Per-rank partial results go to data/cache/l0_parts/ (resumable); the merged table is data/cache/l0/
(dense.parquet, sparse.npz, meta.json - gametable.save_table format) plus data/cache/games_meta.parquet.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gogame.constants import RANKS  # noqa: E402
from gogame.features_l0 import DENSE_NAMES, compute_l0_many  # noqa: E402
from gogame.gametable import GameTable, save_table  # noqa: E402
from gogame.io import GameRecord, read_train_csv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "training"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--ranks", nargs="*", default=list(RANKS))
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    parts = cache / "l0_parts"
    parts.mkdir(parents=True, exist_ok=True)

    for r in a.ranks:
        done = parts / f"{r}.done"
        if done.exists():
            print(f"[{r}] cached", flush=True)
            continue
        t0 = time.perf_counter()
        df = read_train_csv(Path(a.data_dir) / f"train_{r}.csv")
        dense, sp = [], []
        for s in range(0, len(df), a.chunk):
            sub = df.iloc[s:s + a.chunk]
            recs = [GameRecord(u, p, int(k), int(c), g, "train") for u, p, k, c, g in
                    zip(sub["uid"], sub["player_id"], sub["rank_id"], sub["color_id"], sub["sgf_content"])]
            b = compute_l0_many(recs, n_jobs=a.jobs)
            dense.append(b.dense)
            sp.append(b.sparse)
        meta = df[["uid", "player_id", "game_id", "rank_id", "color_id"]].rename(columns={"color_id": "color"})
        meta.to_parquet(parts / f"{r}_meta.parquet", index=False)
        np.save(parts / f"{r}_dense.npy", np.vstack(dense))
        sps.save_npz(parts / f"{r}_sparse.npz", sps.vstack(sp).tocsr())
        done.write_text("ok")
        print(f"[{r}] {len(df)} games in {time.perf_counter() - t0:.0f}s", flush=True)
        del df, dense, sp

    if set(a.ranks) != set(RANKS):
        return 0
    metas = [pd.read_parquet(parts / f"{r}_meta.parquet") for r in RANKS]
    meta = pd.concat(metas, ignore_index=True)
    dense = np.vstack([np.load(parts / f"{r}_dense.npy") for r in RANKS])
    sparse = sps.vstack([sps.load_npz(parts / f"{r}_sparse.npz") for r in RANKS]).tocsr()
    assert len(meta) == len(dense) == sparse.shape[0] and meta["uid"].is_unique
    meta["rank_id"] = meta["rank_id"].astype("int8")
    meta["color"] = meta["color"].astype("int8")
    meta.to_parquet(cache / "games_meta.parquet", index=False)
    t = GameTable("l0", meta["uid"].to_numpy(), dense, tuple("l0_" + n for n in DENSE_NAMES), sparse, {},
                  {"route": "l0", "model_id": "l0", "fold": None, "config_hash": "", "external_resources": []})
    save_table(t, cache / "l0")
    print(f"merged: {len(meta)} games, dense {dense.shape}, sparse nnz {sparse.nnz}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
