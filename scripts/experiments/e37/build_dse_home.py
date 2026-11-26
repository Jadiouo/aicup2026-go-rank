"""E37 Dse arm: build the static, H-free "home" tree that dse_w120.py reads (local, light, run once; a host keeps it between runs).
  PYTHONPATH=src:scripts systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 nice -n 10 .venv/bin/python -m scripts.experiments.e37.build_dse_home
Output .scratch/e37/dse_home/ (mirrors the repo-relative paths the E10/E21 code expects, restricted to the 712,945 non-H bundle games / non-H players):
  data/cache/l0/{dense.parquet,sparse.npz,meta.json}   L0 rows of the bundle games only (same row order as the original table)
  data/cache/games_meta.parquet, data/splits/split.parquet   non-H games / players only (split has no holdout row, so H = empty set on the host)
  data/se_full.parquet                                  E21 SE table (already non-H only)
  ref/                                                  E10 questions_q2 + oof_A (M04 leg), E10 meta_D / meta_A, E21 oof_M03SE / oof_Dse / meta  (the saved v4 Dse; reproduction reference)
  manifest.json                                         sha256 of every file + H check.  HSL columns are NOT here: they come from the X file (W60 or W120) of the run.
Reads H only to filter it out; never reads or writes h_eval_ledger.jsonl."""
from __future__ import annotations
import json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, scipy.sparse as sps

from scripts.experiments.e37.common import *   # noqa: F403
from gogame.gametable import load_table

HOME = S37 / "dse_home"
E10_OOF = ROOT / "experiments/E10_tier2_retrain/oof"
E21_RES = ROOT / "experiments/E21_strength_estimator/results"
E29B = ROOT / ".scratch/e29/bundle"


def main():
    t0 = time.time()
    if HOME.exists():
        shutil.rmtree(HOME)
    for d in ("data/cache/l0", "data/splits", "ref"):
        (HOME / d).mkdir(parents=True)
    aux = np.load(E29B / "aux.npz", allow_pickle=False)
    uid = aux["uid"].astype(str); names = aux["names"].astype(str).tolist()
    split = pd.read_parquet(ROOT / "data/splits/split.parquet")
    H = set(split.loc[split["is_holdout"], "player_id"])
    nonh = split[~split["is_holdout"]].reset_index(drop=True)
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    meta_nh = meta[meta["player_id"].isin(set(nonh["player_id"]))].reset_index(drop=True)
    assert not (set(meta_nh["player_id"]) & H) and set(meta_nh["uid"]) == set(uid) and len(meta_nh) == len(uid), "non-H meta != bundle uid set"
    hsl_cols = [c for c in pq.read_schema(ROOT / "data/cache/e10/hsl_games.parquet").names if c != "uid"]
    assert hsl_cols == names[:104], "bundle HSL column names/order != hsl_games.parquet"
    # L0 (rows of the bundle games only, original order)
    tab = load_table(ROOT / "data/cache/l0")
    keep = np.isin(np.asarray(tab.uids).astype(str), uid)
    assert int(keep.sum()) == len(uid)
    pq.write_table(pa.table({"uid": np.asarray(tab.uids)[keep].astype(str), **{n: tab.dense[keep, i] for i, n in enumerate(tab.names)}}), HOME / "data/cache/l0/dense.parquet", compression="zstd")
    sps.save_npz(HOME / "data/cache/l0/sparse.npz", tab.sparse[np.flatnonzero(keep)].tocsr())
    shutil.copy2(ROOT / "data/cache/l0/meta.json", HOME / "data/cache/l0/meta.json")
    del tab
    meta_nh.to_parquet(HOME / "data/cache/games_meta.parquet", index=False)
    nonh.to_parquet(HOME / "data/splits/split.parquet", index=False)
    se = pd.read_parquet(ROOT / ".scratch/e21/se_full.parquet")
    assert set(se["uid"]) == set(uid) and se["uid"].is_unique
    se.to_parquet(HOME / "data/se_full.parquet", index=False)
    for src, dst in [(E10_OOF / "questions_q2.parquet", "questions_q2.parquet")] + \
            [(E10_OOF / f"{p}_fold{k}.{e}", f"{p}_fold{k}.{e}") for k in range(5) for p, e in (("oof_A", "parquet"), ("meta_D", "json"), ("meta_A", "json"))] + \
            [(E21_RES / f"{p}_fold{k}.{e}", f"{p}_fold{k}.{e}" if p != "meta" else f"meta_E21_fold{k}.json") for k in range(5) for p, e in (("oof_M03SE", "parquet"), ("oof_Dse", "parquet"), ("meta", "json"))]:
        shutil.copy2(src, HOME / "ref" / dst)
    # H check on what was written
    q2 = pd.read_parquet(HOME / "ref/questions_q2.parquet")
    spl = pd.read_parquet(HOME / "data/splits/split.parquet"); gm = pd.read_parquet(HOME / "data/cache/games_meta.parquet", columns=["uid", "player_id"])
    l0u = pq.read_table(HOME / "data/cache/l0/dense.parquet", columns=["uid"]).column("uid").to_pylist()
    chk = {"n_H_removed": len(H), "split_holdout_rows": int(spl["is_holdout"].sum()), "split_players_inter_H": len(set(spl["player_id"]) & H), "meta_players_inter_H": len(set(gm["player_id"]) & H),
           "q2_players_inter_H": len(set(q2["player_id"]) & H), "l0_uids_not_in_bundle": len(set(l0u) - set(uid)), "se_uids_not_in_bundle": len(set(se["uid"]) - set(uid)),
           "n_games": int(len(uid)), "n_players": int(len(spl))}
    assert chk["split_holdout_rows"] == chk["split_players_inter_H"] == chk["meta_players_inter_H"] == chk["q2_players_inter_H"] == chk["l0_uids_not_in_bundle"] == chk["se_uids_not_in_bundle"] == 0, f"H LEAK {chk}"
    files = {str(p.relative_to(HOME)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(HOME.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    man = {"built": time.strftime("%F %T"), "h_check": chk, "hsl_cols": len(hsl_cols), "files": files, "seconds": round(time.time() - t0), "total_mb": round(sum(f["bytes"] for f in files.values()) / 1e6)}
    (HOME / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
