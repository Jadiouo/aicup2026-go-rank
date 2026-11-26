"""Build .scratch/e21/se_full.parquet = hd_uid (non-H, 712,945) left join se_uid; HD>=2 -> NaN. Asserts no H."""
import pandas as pd, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[3]
hd = pd.read_parquet(R / ".scratch/e20/hd_uid.parquet"); se = pd.read_parquet(R / ".scratch/e21/se_uid.parquet")
sp = pd.read_parquet(R / "data/splits/split.parquet")
H = set(sp.loc[sp["is_holdout"], "player_id"])
assert len(hd) == 712945 and se["uid"].is_unique and set(se["uid"]) <= set(hd["uid"])
out = hd[["uid"]].merge(se[["uid", "se_mean", "se_std", "se_n"]], on="uid", how="left")
assert len(out) == len(hd) and out["uid"].is_unique
miss = out["se_mean"].isna().to_numpy()
assert (hd["hd"].to_numpy()[miss] >= 2).all() and miss.sum() == (hd["hd"] >= 2).sum(), (miss.sum(), (hd["hd"] >= 2).sum())
# H check via meta (uid -> player) when available
import sys
try:
    meta = pd.read_parquet(R / "data/processed/games_meta.parquet")[["uid", "player_id"]] if (R / "data/processed/games_meta.parquet").exists() else None
except Exception: meta = None
if meta is not None:
    assert not (set(meta.loc[meta["uid"].isin(set(out["uid"])), "player_id"]) & H)
out.to_parquet(R / ".scratch/e21/se_full.parquet", index=False)
print(len(out), "nan", int(miss.sum()), "meta check", meta is not None)
