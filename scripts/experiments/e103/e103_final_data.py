"""E103 交件版訓練資料（E98 ft4 的訓練盤 574,553 盤 ＋ K 線全部 fold 4 玩家的盤）。不改 e98_data.py（本資料夾那份是逐位複本）；抽取直接用 e98_data.cmd_extract。
子命令
  select   （本機，.venv python）選出「新增盤」＝K 線計畫裡 fold 4 玩家的全部盤。斷言：與 E98 的訓練盤（select_train.parquet）不相交、聯集＝K 線 712,945 盤、無 H 玩家、
           訓練玩家集合＝原 5,898 位 ∪ fold 4 的 1,469 位。輸出 select_f4_extra.parquet（gid 從 0 起，只在本檔內有意義）、old_train_players.txt、select_info.json。
  extract  （學校）＝ e98_data.py extract（同一個取局面規則 pick_positions：己方手、ply>=2、多於 12 個以 crc32(uid) 子種子 permutation 取前 12 再排序）。
  merge    （學校）舊 bundle（~/gogame_e98/bundle_train）原樣放前段、新增盤接在後面 -> bundle_train_final；gid 接續（新盤 gid ＝ 574,553 ＋ 序）。
  verify   （學校）核對：總盤數、每盤 12 局面、舊段與舊 bundle 逐位相同、玩家集合斷言、各段位盤數；--spot N 抽 N 盤新增盤與 shards 直接重取逐位比對。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e98_data as E  # noqa: E402  (本資料夾的逐位複本)

BITS_W, N_RANK = E.BITS_W, E.N_RANK
EXPECTED_ALL = 712_945
POS = 12


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------- select (local)
def cmd_select(a) -> None:
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    aux = np.load(root / ".scratch/e29/bundle/aux.npz", allow_pickle=True)
    order = np.load(root / ".scratch/e89/k/plan/order.npy")
    plan = json.load(open(root / ".scratch/e89/k/plan/plan.json"))
    gps, nsh = plan["games_per_shard"], plan["n_shards"]
    sp = pd.read_parquet(root / "data/splits/split.parquet")
    H = set(sp.loc[sp["is_holdout"], "player_id"])
    spi = sp.set_index("player_id")
    uid_all, pids, gp = aux["uid"].astype(str), aux["player_ids"].astype(str), aux["game_player"]
    assert len(set(uid_all)) == len(uid_all) == EXPECTED_ALL, len(uid_all)
    shard_of = np.full(len(uid_all), -1, np.int32)
    for k in range(nsh):
        shard_of[order[k * gps:(k + 1) * gps]] = k
    assert (shard_of >= 0).all()
    df = pd.DataFrame({"row": np.arange(len(uid_all), dtype=np.int32), "uid": uid_all, "player_id": pids[gp], "color": aux["game_color"].astype(np.int8), "shard": shard_of})
    df["rank_id"] = spi.loc[df["player_id"], "rank_id"].to_numpy().astype(np.int8)
    df["fold"] = spi.loc[df["player_id"], "fold"].to_numpy().astype(np.int8)
    assert not spi.loc[df["player_id"], "is_holdout"].to_numpy().any(), "H player in the K bundle"
    assert not (set(df["player_id"]) & H)
    assert (df["rank_id"].to_numpy() == aux["game_rank"]).all()
    assert (df["fold"].to_numpy() == aux["player_fold"][gp]).all()
    old = pd.read_parquet(Path(a.old_select))                       # E98 的訓練盤（fold 0-3）
    assert len(old) == 574_553 and old["player_id"].nunique() == 5898 and (old["fold"] != 4).all()
    ex = df[df["fold"] == 4].copy()
    assert not (set(ex["uid"]) & set(old["uid"])), "extra games overlap the E98 train games"
    assert len(ex) + len(old) == EXPECTED_ALL, (len(ex), len(old))
    assert set(ex["uid"]) | set(old["uid"]) == set(uid_all), "union != all K games"
    f4_file = [x for x in Path(a.fold4).read_text().split()]
    f4 = sorted(set(ex["player_id"]))
    assert f4 == sorted(f4_file) and len(f4) == 1469, (len(f4), len(f4_file))
    old_players = sorted(set(old["player_id"]))
    assert not (set(old_players) & set(f4))
    all_train_players = set(old_players) | set(f4)
    assert all_train_players == set(df["player_id"]) and len(all_train_players) == 5898 + 1469
    ex = ex.sort_values(["shard", "row"]).reset_index(drop=True)
    ex["gid"] = np.arange(len(ex), dtype=np.int32)
    cols = ["gid", "row", "uid", "player_id", "rank_id", "fold", "color", "shard"]
    ex[cols].to_parquet(out / "select_f4_extra.parquet", index=False)
    (out / "old_train_players.txt").write_text("\n".join(old_players) + "\n")
    info = {"n_extra_games": int(len(ex)), "n_extra_players": int(ex.player_id.nunique()), "extra_games_by_rank": np.bincount(ex["rank_id"], minlength=N_RANK).tolist(),
            "extra_players_by_rank": [int(ex.loc[ex.rank_id == r, "player_id"].nunique()) for r in range(N_RANK)],
            "old_games": int(len(old)), "old_players": len(old_players), "total_games": int(len(old) + len(ex)), "total_players": len(all_train_players),
            "n_shards_with_extra": int(ex["shard"].nunique()), "H_players_in_K": 0, "built": time.strftime("%F %T")}
    (out / "select_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# --------------------------------------------------------------------------------------------- merge (school)
def cmd_merge(a) -> None:
    old, parts_dir, out = Path(a.old_bundle), Path(a.parts), Path(a.bundle)
    sel = pd.read_parquet(a.select)
    sids = sorted(int(s) for s in sel["shard"].unique())
    parts = parts_dir / "parts"
    for s in sids:
        assert (parts / f"{s:05d}" / "OK").exists(), f"missing part {s}"
    sizes = {s: np.load(parts / f"{s:05d}" / "gid.npy", mmap_mode="r").shape[0] for s in sids}
    Nn = sum(sizes.values())
    om = np.load(old / "meta.npz", allow_pickle=False)
    G_old = len(om["g_uid"])
    gold = np.load(old / "gid.npy", mmap_mode="r")
    N_old = gold.shape[0]
    N = N_old + Nn
    need_gb = (Nn * (BITS_W + 38 + 4 + 4) + N_old * (BITS_W + 38 + 4 + 4)) / 1e9
    if E.free_gb(out.parent) < need_gb + 1.5:
        raise SystemExit(f"disk free {E.free_gb(out.parent):.1f}GB < need {need_gb + 1.5:.1f}GB")
    # 新盤的 gid：sel 的 gid（0..）在每片內已升冪、跨片依 (shard,row) 升冪；全部有局面才算數
    gid_new = np.concatenate([np.load(parts / f"{s:05d}" / "gid.npy") for s in sids])
    assert (np.diff(gid_new) >= 0).all(), "gids not sorted across shards"
    cnt = np.bincount(gid_new, minlength=len(sel))
    assert len(cnt) == len(sel) and (cnt > 0).all(), f"{int((cnt == 0).sum())} extra games have no position"
    sel = sel.sort_values("gid").reset_index(drop=True)
    assert (sel["gid"].to_numpy() == np.arange(len(sel))).all()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    arr = {"bits": (np.uint8, (N, BITS_W)), "glob": (np.float16, (N, 19)), "loc": (np.int16, (N,)), "ply": (np.int16, (N,)), "gid": (np.int32, (N,))}
    mm = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=sh) for k, (dt, sh) in arr.items()}
    CH = 200_000
    for k in ("bits", "glob", "loc", "ply", "gid"):                 # 舊段原樣
        x = np.load(old / f"{k}.npy", mmap_mode="r")
        assert x.dtype == mm[k].dtype and x.shape[0] == N_old and x.shape[1:] == mm[k].shape[1:], (k, x.dtype, x.shape)
        for c in range(0, N_old, CH):
            e_ = min(c + CH, N_old)
            mm[k][c:e_] = x[c:e_]
        log("copied old", k)
    o = N_old
    for s in sids:
        n = sizes[s]
        for k in ("bits", "glob", "loc", "ply"):
            x = np.load(parts / f"{s:05d}" / f"{k}.npy", mmap_mode="r")
            assert x.dtype == mm[k].dtype and x.shape[0] == n and x.shape[1:] == mm[k].shape[1:], (k, x.dtype, x.shape)
            mm[k][o:o + n] = x
        mm["gid"][o:o + n] = np.load(parts / f"{s:05d}" / "gid.npy").astype(np.int64) + G_old
        o += n
    assert o == N
    for m in mm.values():
        m.flush()
    meta = {k: om[k] for k in om.files}
    new_meta = dict(g_uid=sel["uid"].to_numpy().astype(str), g_player=sel["player_id"].to_numpy().astype(str), g_rank=sel["rank_id"].to_numpy().astype(np.int8),
                    g_fold=sel["fold"].to_numpy().astype(np.int8), g_color=sel["color"].to_numpy().astype(np.int8), g_shard=sel["shard"].to_numpy().astype(np.int16),
                    g_row=sel["row"].to_numpy().astype(np.int32))
    assert set(meta) == set(new_meta), (set(meta), set(new_meta))
    np.savez(out / "meta.npz", **{k: np.concatenate([meta[k], new_meta[k]]) for k in meta})
    info = {"role": "train_final", "complete": True, "n_games_old": int(G_old), "n_games_new": int(len(sel)), "n_games": int(G_old + len(sel)), "n_positions_old": int(N_old),
            "n_positions_new": int(Nn), "n_positions": int(N), "built": time.strftime("%F %T")}
    (out / "MERGED.json").write_text(json.dumps(info, indent=1))
    log("merge done", json.dumps(info))


# --------------------------------------------------------------------------------------------- verify (school)
def _eq_chunked(a_, b_, CH=300_000):
    assert a_.shape[0] <= b_.shape[0]
    for c in range(0, a_.shape[0], CH):
        e_ = min(c + CH, a_.shape[0])                                  # b_ 比 a_ 長：只比 a_ 的長度（舊段）
        if not np.array_equal(np.asarray(a_[c:e_]), np.asarray(b_[c:e_])):
            return False
    return True


def cmd_verify(a) -> None:
    new, old = Path(a.bundle), Path(a.old_bundle)
    d, od = E.load_bundle(new), E.load_bundle(old)
    f4 = set(Path(a.fold4).read_text().split())
    old_players = set(Path(a.old_players).read_text().split())
    P, fold, rank = d["g_player"], d["g_fold"], d["g_rank"]
    G, G_old = len(P), len(od["g_player"])
    gid = np.asarray(d["gid"])
    N, N_old = len(gid), len(np.asarray(od["gid"]))
    sel_dir = Path(a.sel_dir)
    n_new = int(json.loads((sel_dir / "select_info.json").read_text())["n_extra_games"])
    print("games total", G, "old", G_old, "new", G - G_old, "positions", N, "old", N_old, "new", N - N_old)
    assert G == 574_553 + n_new and G_old == 574_553, (G, n_new)
    assert (np.diff(gid) >= 0).all() and gid.min() == 0 and gid.max() == G - 1
    cnt = np.bincount(gid, minlength=G)
    assert (cnt == POS).all(), f"games without exactly {POS} positions: {int((cnt != POS).sum())} (min {cnt.min()}, max {cnt.max()})"
    assert N == G * POS and d["ply"].shape[0] == N == d["bits"].shape[0] == d["glob"].shape[0] == d["loc"].shape[0]
    assert int(np.asarray(d["ply"]).min()) >= E.PLY_MIN
    # 玩家集合：訓練玩家 ＝ 原 5,898 位 ∪ fold 4 玩家
    assert set(P.tolist()) == old_players | f4 and len(old_players) == 5898 and len(f4) == 1469, (len(set(P.tolist())), len(old_players), len(f4))
    assert not (old_players & f4)
    assert (fold[:G_old] != 4).all() and (fold[G_old:] == 4).all() and set(P[G_old:].tolist()) == f4 and not (set(P[:G_old].tolist()) & f4)
    assert len(set(d["g_uid"].tolist())) == G, "duplicate uid"
    print("player-set assertion OK: train players", len(set(P.tolist())), "= old", len(old_players), "+ fold4", len(f4))
    # 舊段逐位相同
    for k in ("bits", "glob", "loc", "ply", "gid"):
        assert _eq_chunked(od[k], d[k]), f"old segment differs: {k}"
    for k in od:
        if k in ("bits", "glob", "loc", "ply", "gid"):
            continue
        assert np.array_equal(od[k], d[k][:G_old]), f"old meta differs: {k}"
    print("old segment bit-identical to", old, "(bits/glob/loc/ply/gid + meta)")
    print("new games by rank:", np.bincount(rank[G_old:], minlength=N_RANK).tolist(), " players by rank:", [len(set(P[G_old:][rank[G_old:] == r].tolist())) for r in range(N_RANK)])
    print("all games by rank:", np.bincount(rank, minlength=N_RANK).tolist())
    if a.spot:
        import numpy as _np
        shards = Path(a.shards)
        sel = pd.read_parquet(sel_dir / "select_f4_extra.parquet")
        rng = np.random.default_rng(20261120)
        pick = rng.choice(len(sel), a.spot, replace=False)
        bad = 0
        for i in pick:
            r = sel.iloc[int(i)]
            sid, uid = int(r["shard"]), r["uid"]
            gnew = G_old + int(r["gid"])
            assert d["g_uid"][gnew] == uid
            dd = shards / f"{sid:05d}"
            pos = pd.read_parquet(dd / "pos.parquet", columns=["uid", "ply", "loc"])
            bits = _np.load(dd / "bits.npy", mmap_mode="r")
            glob = _np.load(dd / "glob.npy", mmap_mode="r")
            rows_all, _ = E.pick_positions(pos, {uid: 0}, POS)
            rows = rows_all[0]
            s0, s1 = int(np.searchsorted(gid, gnew)), int(np.searchsorted(gid, gnew, side="right"))
            ok = (s1 - s0 == len(rows) == POS and np.array_equal(np.asarray(d["bits"][s0:s1]), bits[rows]) and np.array_equal(np.asarray(d["glob"][s0:s1]), glob[rows])
                  and np.array_equal(np.asarray(d["loc"][s0:s1]), pos["loc"].to_numpy()[rows].astype(np.int16)) and np.array_equal(np.asarray(d["ply"][s0:s1]), pos["ply"].to_numpy()[rows].astype(np.int16))
                  and int(d["g_color"][gnew]) == int(r["color"]) and int(d["g_rank"][gnew]) == int(r["rank_id"]))
            bad += (not ok)
        print(f"spot check: {a.spot} extra games re-extracted directly from shards, bit-identical: {a.spot - bad}/{a.spot}")
        assert bad == 0
    print("VERIFY OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    s = sp_.add_parser("select"); s.add_argument("--root", default=str(Path(__file__).resolve().parents[3])); s.add_argument("--out", required=True)
    s.add_argument("--old-select", required=True); s.add_argument("--fold4", required=True)
    m = sp_.add_parser("merge"); m.add_argument("--select", required=True); m.add_argument("--parts", required=True, help="dir that contains parts/"); m.add_argument("--old-bundle", required=True); m.add_argument("--bundle", required=True)
    v = sp_.add_parser("verify"); v.add_argument("--bundle", required=True); v.add_argument("--old-bundle", required=True); v.add_argument("--fold4", required=True); v.add_argument("--old-players", required=True)
    v.add_argument("--sel-dir", required=True, help="folder with select_info.json and select_f4_extra.parquet")
    v.add_argument("--shards", default=""); v.add_argument("--spot", type=int, default=0)
    a = ap.parse_args()
    {"select": cmd_select, "merge": cmd_merge, "verify": cmd_verify}[a.cmd](a)
