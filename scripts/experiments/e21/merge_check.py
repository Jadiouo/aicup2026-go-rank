"""E21 extract step 3 (local): merge school shards -> .scratch/e21/se_uid.parquet and run the completion checks.
  rsync -a $REMOTE_HOST:~/gogame_e21/out/ .scratch/e21/out/ ; .venv/bin/python scripts/experiments/e21/merge_check.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[3]
files = sorted((ROOT / ".scratch/e21/out").glob("se_shard_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
hdt = pd.read_parquet(ROOT / ".scratch/e20/hd_uid.parquet")
exp = set(hdt.uid[hdt.hd.astype(int) < 2])
sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(sp.player_id[sp.is_holdout])
g = pd.read_parquet(ROOT / ".scratch/e19/games_hd.parquet"); g["uid"] = g["rank"] + "_" + g["game_id"]
hu = set(g.uid[g.player_id.isin(H)])
print("shards", len(files), "rows", len(df), "expected", len(exp), "dup", int(df.uid.duplicated().sum()))
print("missing", len(exp - set(df.uid)), "extra", len(set(df.uid) - exp), "H overlap", len(set(df.uid) & hu))
print("se_mean NaN", int(df.se_mean.isna().sum()), "se_n<32", int((df.se_n < 32).sum()), "se_n==0", int((df.se_n == 0).sum()))
print(df[["se_mean", "se_std", "se_n"]].describe().round(3).to_string())
ref = pd.read_csv(ROOT / ".scratch/e21/sanity.csv"); ref["uid"] = ref["rank"] + "_" + ref.game_id
s = ref.merge(df, on="uid")
print("sanity-sample rho vs rank (n=%d): %.3f" % (len(s), spearmanr(s.se_mean, s.rank_id)[0]))
r = df.assign(rk=df.uid.str.split("_").str[0]).groupby("rk").se_mean.mean()
print("mean se_mean by rank:", r.round(2).to_dict())
if len(files) == 69 and len(df) == len(exp) and not df.se_mean.isna().any() and not (set(df.uid) & hu):
    df.to_parquet(ROOT / ".scratch/e21/se_uid.parquet", index=False); print("WROTE .scratch/e21/se_uid.parquet")
else:
    df.to_parquet(ROOT / ".scratch/e21/se_uid_partial.parquet", index=False); print("incomplete: wrote se_uid_partial.parquet")
