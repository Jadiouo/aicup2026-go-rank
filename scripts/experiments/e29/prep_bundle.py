"""E29 data bundle (local, light, run once).  Non-H games only; per-game feature matrix + fold roles + ES / val bags as row indices.
Output: .scratch/e29/bundle/{X.npy, aux.npz, manifest.json}.  Plan: experiments/E29_set_model/PLAN.md.
Run:  .venv/bin/python -m scripts.experiments.e29.prep_bundle
"""
from __future__ import annotations

import hashlib, json, sys, time
from pathlib import Path

import numpy as np, pandas as pd

from scripts.experiments.e10 import lib
from gogame.questions import Bag, build_player_index
import run_e01 as e01

ROOT = lib.ROOT
OUT = ROOT / ".scratch" / "e29" / "bundle"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def bags_to_rows(bags, row_of, pid_of):
    n = len(bags)
    rows = np.full((n, 20), -1, np.int32)
    for i, b in enumerate(bags):
        rows[i, : b.n] = [row_of[u] for u in b.uids]
    return {"rows": rows, "n": np.array([b.n for b in bags], np.int16), "rank": np.array([b.rank_id for b in bags], np.int8),
            "player": np.array([pid_of[b.player_id] for b in bags], np.int32), "color": np.array([b.color for b in bags], np.int8),
            "qid": np.array([b.qid for b in bags])}


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.read_parquet(ROOT / "data/splits/split.parquet")
    H = set(split.loc[split["is_holdout"], "player_id"])
    nonh = split[~split["is_holdout"]].reset_index(drop=True)
    nonh_pl = set(nonh["player_id"])
    assert len(H) and not (H & nonh_pl)
    meta_all = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    meta = meta_all[meta_all["player_id"].isin(nonh_pl)].sort_values("uid").reset_index(drop=True)
    assert not (set(meta["player_id"]) & H)
    se = pd.read_parquet(ROOT / ".scratch/e21/se_full.parquet")
    assert set(se["uid"]) == set(meta["uid"]) and se["uid"].is_unique, "SE table uid set != non-H games"
    uids = meta["uid"].to_numpy()
    n = len(uids)
    print("non-H games", n, "players", len(nonh_pl), "H players", len(H), flush=True)
    # features
    hsl = pd.read_parquet(ROOT / "data/cache/e10/hsl_games.parquet")
    hcols = [c for c in hsl.columns if c != "uid"]
    hsl = hsl.set_index("uid").reindex(uids)
    assert not hsl.isna().any().any()
    l0 = pd.read_parquet(ROOT / "data/cache/l0/dense.parquet")
    lcols = [c for c in l0.columns if c != "uid"]
    l0 = l0.set_index("uid").reindex(uids)
    assert not l0.isna().any().any()
    se = se.set_index("uid").reindex(uids)
    X = np.empty((n, len(hcols) + len(lcols) + 4), np.float32)
    X[:, : len(hcols)] = hsl[hcols].to_numpy(np.float32); del hsl
    X[:, len(hcols): len(hcols) + len(lcols)] = l0[lcols].to_numpy(np.float32); del l0
    j = len(hcols) + len(lcols)
    X[:, j] = se["se_mean"].to_numpy(np.float32); X[:, j + 1] = se["se_std"].to_numpy(np.float32)
    X[:, j + 2] = np.isnan(X[:, j]).astype(np.float32)          # SE missing flag
    X[:, j + 3] = meta["color"].astype(int).to_numpy(np.float32) if meta["color"].dtype != object else (meta["color"] == "W").astype(np.float32).to_numpy()
    names = hcols + lcols + ["se_mean", "se_std", "se_missing", "is_white"]
    print("X", X.shape, "se nan share", float(X[:, j + 2].mean()), flush=True)
    # players
    pl_ids = sorted(nonh_pl)
    pid_of = {p: i for i, p in enumerate(pl_ids)}
    row_of = {u: i for i, u in enumerate(uids)}
    pidx = meta["player_id"].map(pid_of).to_numpy(np.int32)
    prank = nonh.set_index("player_id").loc[pl_ids, "rank_id"].to_numpy(np.int8)
    pfold = nonh.set_index("player_id").loc[pl_ids, "fold"].to_numpy(np.int8)
    assert (meta.groupby("player_id")["rank_id"].nunique() == 1).all()
    ptotal = np.bincount(pidx, minlength=len(pl_ids)).astype(np.int32)       # total games (both colours)
    # roles per fold (E21 sets) + ES / val bags
    index = build_player_index(meta.assign(color_id=(meta["color"].astype(int) if meta["color"].dtype != object else (meta["color"] == "W").astype(int))))
    q2 = e01.frame_to_bags(pd.read_parquet(lib.OUT / "oof" / "questions_q2.parquet"))
    assert not ({b.player_id for b in q2} & H)
    aux = {"uid": uids.astype(str), "game_player": pidx, "game_color": X[:, j + 3].astype(np.int8), "game_rank": meta["rank_id"].to_numpy(np.int8),
           "player_ids": np.array(pl_ids), "player_rank": prank, "player_fold": pfold, "player_total": ptotal, "names": np.array(names)}
    rep = {}
    for k in range(5):
        fs = lib.fold_sets(nonh, k)
        lib.assert_fold_disjoint(fs, H)
        role = np.zeros(len(pl_ids), np.int8)               # 1 fit, 2 es, 3 val, 0 none
        for r, key in ((1, "fit"), (2, "es"), (3, "val")):
            for p in fs[key]:
                role[pid_of[p]] = r
        assert (role > 0).all() and (role == 3).sum() == len(fs["val"])
        aux[f"role_f{k}"] = role
        es = lib.make_es_bags(index, fs["es"], k, False)
        val = [b for b in q2 if b.player_id in fs["val"]]
        assert {b.player_id for b in es} <= fs["es"] and {b.player_id for b in val} <= fs["val"]
        leak = lib.assert_no_game_leak(meta_all, [u for b in val for u in b.uids], fs["train_all"])
        for nm, bags in (("es", es), ("val", val)):
            for kk, v in bags_to_rows(bags, row_of, pid_of).items():
                aux[f"{nm}_{kk}_f{k}"] = v
        rep[k] = {"n_fit": int((role == 1).sum()), "n_es": int((role == 2).sum()), "n_val": int((role == 3).sum()), "n_es_bags": len(es),
                  "n_val_bags": len(val), "game_leak_overlap": leak["n_overlap"]}
        print("fold", k, rep[k], flush=True)
    np.save(OUT / "X.npy", X)
    np.savez(OUT / "aux.npz", **aux)
    man = {"n_games": int(n), "n_players": len(pl_ids), "n_features": int(X.shape[1]), "names": names, "folds": rep,
           "n_H_players_in_bundle": 0, "sha256": {"X.npy": sha(OUT / "X.npy"), "aux.npz": sha(OUT / "aux.npz")},
           "seconds": round(time.perf_counter() - t0)}
    # H check, done on the written artefacts
    a = np.load(OUT / "aux.npz")
    assert not (set(a["player_ids"].tolist()) & H), "H player in bundle"
    gm = meta_all.set_index("uid").loc[a["uid"], "player_id"]
    assert not (set(gm) & H), "H game in bundle"
    man["h_check"] = {"n_H": len(H), "bundle_players_inter_H": 0, "bundle_games_owned_by_H": 0, "uid_set_equals_se_table": True}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print("done", man["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
