"""E79 step 1 (CPU, local): dump the item lists for the remote (remote-vm) prep.  One pickle per 10k-row CSV chunk filtered to the bundle (e63_prep order), plus the 10,000 Rank test games.
items = [(uid, color, sgf, rid)]; rid = bundle row (aux uid order) or test_bags row.  Each pickle is rsynced to remote-vm:~/gogame_e79/items/ and removed locally (--remove-source-files).
Names: b{i}.pkl (bundle chunks, i sequential), t0.pkl (test).  No H (bundle has none; asserted).  usage: e79_items.py [--no-sync]"""
import sys, os, pickle, subprocess, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np, pandas as pd
OUT = ROOT / ".scratch/e79/items"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--no-sync", action="store_true"); a = ap.parse_args()
def sync(f):
    if a.no_sync: return
    subprocess.run(["rsync", "-a", "--remove-source-files", str(f), os.environ.get("REMOTE_HOST", "remote-host") + ":~/gogame_e79/items/"], check=True)
aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); uid = aux["uid"].astype(str); gcol = aux["game_color"].astype(np.int8)
row_of = pd.Series(np.arange(len(uid), dtype=np.int64), index=uid)
split = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(split.loc[split["is_holdout"], "player_id"])
meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet"); assert set(meta["uid"]) >= set(uid)
assert not (set(meta.loc[meta["uid"].isin(set(uid)), "player_id"]) & H), "H player in bundle"
bg = np.load(ROOT / ".scratch/e29/test/test_bags.npz", allow_pickle=True); tu = [str(u) for u in bg["uids"]]; assert len(tu) == 10000
tab = {}
for f in ("public", "private"):
    d = pd.read_csv(ROOT / f"data/contest_1104/rank_prediction_test_{f}.csv")
    for _, r in d.iterrows():
        for i in range(1, int(r.num_games) + 1): tab[f"{r.question_id}#{i}"] = (r[f"sgf_{i}"], 0 if r[f"color_{i}"] == "B" else 1)
items = [(u, tab[u][1], tab[u][0], i) for i, u in enumerate(tu)]; p = OUT / "t0.pkl"; pickle.dump(items, open(p, "wb")); print("test items", len(items), flush=True); sync(p)
seen = 0; bi = 0
for f in sorted((ROOT / "data" / "training").glob("train_*.csv")):
    for ch in pd.read_csv(f, usecols=["game_id", "rank", "sgf_content"], chunksize=10_000):
        ch["uid"] = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str); ch = ch[ch["uid"].isin(row_of.index)]
        if len(ch) == 0: continue
        rid = row_of.reindex(ch["uid"]).to_numpy(); its = [(u, int(gcol[r]), s, int(r)) for u, s, r in zip(ch["uid"], ch["sgf_content"], rid)]
        p = OUT / f"b{bi}.pkl"; pickle.dump(its, open(p, "wb")); seen += len(its); print(f"b{bi} {f.name} n={len(its)} total {seen}", flush=True); sync(p); bi += 1
assert seen == len(uid), (seen, len(uid)); (OUT / "ITEMS_DONE").write_text(str(bi))
subprocess.run(["ssh", os.environ.get("REMOTE_HOST", "remote-host"), f"echo {bi} > ~/gogame_e79/items/ITEMS_DONE"], check=True) if not a.no_sync else None
print("done blocks", bi, flush=True)
