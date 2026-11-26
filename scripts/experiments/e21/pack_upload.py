"""E21 extract step 1 (local): pack non-H, HD<2 games into upload shards (uid, color, sgf).
H players' games are filtered out here and asserted absent.  Run:
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 nice -n 10 .venv/bin/python scripts/experiments/e21/pack_upload.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / ".scratch/e21/upload"
SHARD = 10000
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]

sp = pd.read_parquet(ROOT / "data/splits/split.parquet")
Hset = set(sp.player_id[sp.is_holdout])
g = pd.read_parquet(ROOT / ".scratch/e19/games_hd.parquet")
g["uid"] = g["rank"] + "_" + g["game_id"]
h_uids = set(g.uid[g.player_id.isin(Hset)])
hdt = pd.read_parquet(ROOT / ".scratch/e20/hd_uid.parquet")
nonh = set(hdt.uid)
assert not (nonh & h_uids), "E20 non-H uid table intersects H games"
assert len(nonh) + len(h_uids) == len(g) == 1_000_000
hd = dict(zip(hdt.uid, hdt.hd.astype(int)))
keep = {u for u, v in hd.items() if v < 2}
assert not (keep & h_uids)
print("nonH", len(nonh), "keep(HD<2)", len(keep), "H games", len(h_uids), flush=True)

OUT.mkdir(parents=True, exist_ok=True)
buf, sid, total, seen = [], 0, 0, set()
def flush(final=False):
    global buf, sid
    while len(buf) >= SHARD or (final and buf):
        part, buf = buf[:SHARD], buf[SHARD:]
        d = pd.DataFrame(part, columns=["uid", "color", "sgf"])
        assert not (set(d.uid) & h_uids)
        d.to_parquet(OUT / f"shard_{sid:04d}.parquet", index=False, compression="zstd")
        sid += 1
for r in RANKS:
    for ch in pd.read_csv(ROOT / f"data/training_v1.1/training/train_{r}.csv", dtype=str, keep_default_na=False, chunksize=5000):
        ch["uid"] = ch["rank"] + "_" + ch["game_id"]
        ch = ch[ch.uid.isin(keep)]
        assert not (set(ch.uid) & h_uids)
        seen.update(ch.uid)
        buf += list(zip(ch.uid, (ch.color != "B").astype(int), ch.sgf_content))
        flush()
flush(True)
assert seen == keep, (len(seen), len(keep))
assert not (seen & h_uids)
print("shards", sid, "games", len(seen), "H intersection 0", flush=True)
(OUT / "DONE").write_text(f"{sid} {len(seen)}\n")
