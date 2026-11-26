"""Build the full-scan shard manifest (run locally, CPU only).

Outputs (data/scan/):
  manifest.parquet  one row per game: uid, player_id, color, rank_id, tier, grp_idx, order, shard_id, pilot_done
  shards.parquet    one row per shard: shard_id, tier, n_games, n_pilot, content_hash, uids_hash
  sgf/<shard_id>.parquet  uid, color, sgf   (what a runner needs; self-contained, zstd)

Priority: tier 1 = up to 15 games per (player, color) [pilot games count first, they are already done],
tier 2 = everything else. Inside a tier games are ordered breadth-first (k-th game of every group before
the (k+1)-th), ties broken by a seeded hash, so any prefix of the order is balanced over players.
Pilot games (data/pilot/pilot_games.parquet, 16,000) form shard P0000 and are marked done.
Deterministic: same inputs -> same shard ids and hashes.
"""
import hashlib, os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEED = "gogame-scan-v1"
SHARD_TARGET = int(os.environ.get("SHARD_TARGET", 4000))
T1_PER_GROUP = 15
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def content_hash(uids, colors, sgfs):
    h = hashlib.sha256()
    for u, c, s in zip(uids, colors, sgfs):
        h.update(f"{u}\t{int(c)}\t{s}\n".encode())
    return h.hexdigest()


def main():
    out = os.path.join(ROOT, "data/scan"); os.makedirs(out + "/sgf", exist_ok=True)
    frames = []
    for i, r in enumerate(RANKS):
        d = pd.read_csv(os.path.join(ROOT, f"data/training/train_{r}.csv"), dtype=str)
        d["uid"] = r + "_" + d.game_id
        d["rank_id"] = np.int8(i)
        d["color"] = (d.color == "W").astype(np.int8)
        frames.append(d[["uid", "player_id", "color", "rank_id", "sgf_content"]])
        print(r, len(d), flush=True)
    g = pd.concat(frames, ignore_index=True)
    assert g.uid.is_unique and len(g) == 1_000_000
    pilot = set(pd.read_parquet(os.path.join(ROOT, "data/pilot/pilot_games.parquet")).uid)
    g["pilot_done"] = g.uid.isin(pilot)
    assert g.pilot_done.sum() == 16000
    g["h"] = [int.from_bytes(hashlib.sha256((SEED + u).encode()).digest()[:8], "big") for u in g.uid]
    # within (player,color): pilot games first, then by hash
    g = g.sort_values(["player_id", "color", "pilot_done", "h"], ascending=[True, True, False, True], kind="stable")
    g["grp_idx"] = g.groupby(["player_id", "color"], sort=False).cumcount().astype(np.int16)
    g["tier"] = np.where(g.grp_idx < T1_PER_GROUP, 1, 2).astype(np.int8)
    g = g.sort_values(["pilot_done", "tier", "grp_idx", "h"], ascending=[False, True, True, True], kind="stable").reset_index(drop=True)
    g["order"] = np.arange(len(g), dtype=np.int32)
    g["shard_id"] = ""
    g.loc[g.pilot_done, "shard_id"] = "P0000"
    shard_rows = []
    cnt = 0
    todo = g[~g.pilot_done]
    for tier in (1, 2):
        idx = todo.index[todo.tier == tier].to_numpy()
        n_sh = int(np.ceil(len(idx) / SHARD_TARGET))
        for part in np.array_split(idx, n_sh):
            cnt += 1
            sid = f"S{cnt:04d}"
            g.loc[part, "shard_id"] = sid
    for sid, d in g.groupby("shard_id", sort=True):
        d = d.sort_values("order")
        row = dict(shard_id=sid, tier=int(d.tier.min()), n_games=len(d), n_pilot=int(d.pilot_done.sum()),
                   content_hash=content_hash(d.uid, d.color, d.sgf_content),
                   uids_hash=hashlib.sha256("\n".join(d.uid).encode()).hexdigest())
        shard_rows.append(row)
        if sid != "P0000":
            sub = d[["uid", "color", "sgf_content"]].rename(columns={"sgf_content": "sgf"})
            sub.to_parquet(f"{out}/sgf/{sid}.parquet", index=False, compression="zstd", compression_level=9)
    man = g.drop(columns=["sgf_content", "h"])
    man.to_parquet(out + "/manifest.parquet", index=False, compression="zstd")
    sh = pd.DataFrame(shard_rows)
    sh.to_parquet(out + "/shards.parquet", index=False)
    print(sh.groupby("tier").agg(n_shards=("shard_id", "count"), games=("n_games", "sum")))
    print("pilot shard:", sh[sh.shard_id == "P0000"].n_games.tolist(), "max/min shard size",
          sh[sh.shard_id != "P0000"].n_games.max(), sh[sh.shard_id != "P0000"].n_games.min())
    t1 = man[(man.tier == 1)]
    print("tier1 games", len(t1), "of which pilot", int(t1.pilot_done.sum()), "groups", t1.groupby(["player_id", "color"]).ngroups)


if __name__ == "__main__":
    main()
