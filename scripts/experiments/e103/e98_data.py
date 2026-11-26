"""E98 資料：全量 K 線局面 -> 訓練 bundle（排除 fold 4 全部玩家）與驗證 bundle（fold 4 驗證題 4,568 題的 41,589 盤）。
抽取規則與 E94-C（scripts/experiments/e94/e94_c_data.py）完全相同：每盤隨機取己方手局面（ply>=2，種子 0，以 uid 的 crc32 為子種子），所以同一盤的 12 個局面逐位相同。

子命令
  select   （本機，.venv python）選盤：全部非 fold 4 的盤＝訓練；驗證題 fold 4 的 41,589 盤＝驗證（含驗證題結構）。產生 select_train.parquet、select_val.parquet、
           val_questions.npz、fold4_players.txt、select_info.json（含目前已完成片的涵蓋情形）。不讀 H 的任何東西（K 線計畫本來就沒有 H 玩家，這裡再斷言一次）。
  extract  （學校，~/gogame_e89/.venv/bin/python，nice -n 19，只讀 shards）每片抽成 parts/<sid>/*.npy；可續做（已有的片跳過）；沒完成（無 .done）的片跳過並回報。
  merge    （學校，numpy＋pandas）把 parts 串成一個 bundle 目錄（bits／glob／loc／ply／gid 的 .npy，可 mmap；meta.npz）。預設要求全部片都在，--partial 才允許缺片。
  verify   （學校或本機）核對 bundle：訓練 bundle 無 fold 4 玩家、與驗證玩家不相交、每盤至少一個局面、各段位盤數。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
PLY_MIN = 2
BITS_W = 993
N_RANK = 10
MIN_FREE_GB = float(os.environ.get("E98_MIN_FREE_GB", "6"))


def crc(s: str) -> int:
    return zlib.crc32(s.encode())


def free_gb(p) -> float:
    return shutil.disk_usage(p).free / 1e9


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------- select (local)
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
    assert len(set(uid_all)) == len(uid_all)
    shard_of = np.full(len(uid_all), -1, np.int32)
    for k in range(nsh):
        shard_of[order[k * gps:(k + 1) * gps]] = k
    assert (shard_of >= 0).all()
    df = pd.DataFrame({"row": np.arange(len(uid_all), dtype=np.int32), "uid": uid_all, "player_id": pids[gp], "color": aux["game_color"].astype(np.int8), "shard": shard_of})
    df["rank_id"] = spi.loc[df["player_id"], "rank_id"].to_numpy().astype(np.int8)
    df["fold"] = spi.loc[df["player_id"], "fold"].to_numpy().astype(np.int8)
    assert not spi.loc[df["player_id"], "is_holdout"].to_numpy().any(), "H player in the K bundle"
    assert (df["rank_id"].to_numpy() == aux["game_rank"]).all()
    assert (df["fold"].to_numpy() == aux["player_fold"][gp]).all(), "split fold != bundle player_fold"
    # ---- validation questions (fold 4) and the games they use
    q = dict(rows=aux["val_rows_f4"].astype(np.int64), ns=aux["val_n_f4"].astype(np.int64), y=aux["val_rank_f4"].astype(np.int64),
             player=pids[aux["val_player_f4"]], qid=aux["val_qid_f4"].astype(str), color=aux["val_color_f4"].astype(np.int64))
    rows = q["rows"]
    assert rows.shape == (4568, 20) and ((rows >= 0).sum(1) == q["ns"]).all()
    val_rows = np.unique(rows[rows >= 0])
    assert len(val_rows) == 41589, len(val_rows)
    r0 = np.load(root / ".scratch/e90/stage0r/r0_games.npz")
    assert (np.sort(r0["rid"]) == val_rows).all(), "val games differ from r0_games.npz (the E92/E95 export set)"
    val = df.iloc[val_rows].copy()
    assert (val["fold"] == 4).all() and (val["row"].to_numpy() == val_rows).all()
    # every game of one question belongs to that question's player and has the question's rank
    pl_of_row = df["player_id"].to_numpy()
    for i in range(len(rows)):
        r = rows[i][rows[i] >= 0]
        assert (pl_of_row[r] == q["player"][i]).all() and (df["rank_id"].to_numpy()[r] == q["y"][i]).all()
    # ---- train = all non-fold-4 games (every K shard)
    tr = df[df["fold"] != 4].copy()
    assert not (set(tr["player_id"]) & set(df.loc[df["fold"] == 4, "player_id"])), "player in both train and fold 4"
    assert not (set(tr["player_id"]) & set(val["player_id"])) and not (set(tr["player_id"]) & H)
    f4_players = sorted(set(df.loc[df["fold"] == 4, "player_id"]))
    tr = tr.sort_values(["shard", "row"]).reset_index(drop=True)
    tr["gid"] = np.arange(len(tr), dtype=np.int32)
    val = val.sort_values(["shard", "row"]).reset_index(drop=True)
    val["gid"] = np.arange(len(val), dtype=np.int32)
    cols = ["gid", "row", "uid", "player_id", "rank_id", "fold", "color", "shard"]
    tr[cols].to_parquet(out / "select_train.parquet", index=False)
    val[cols].to_parquet(out / "select_val.parquet", index=False)
    np.savez(out / "val_questions.npz", **q)
    (out / "fold4_players.txt").write_text("\n".join(f4_players) + "\n")
    # ---- coverage of the validation games by the shards that are done now
    info = {"n_shards": nsh, "train_games": int(len(tr)), "train_players": int(tr.player_id.nunique()), "val_games": int(len(val)), "val_players": int(val.player_id.nunique()),
            "val_questions": int(len(rows)), "fold4_players_all": len(f4_players), "train_games_by_rank": np.bincount(tr["rank_id"], minlength=N_RANK).tolist(),
            "train_players_by_rank": [int(tr.loc[tr.rank_id == r, "player_id"].nunique()) for r in range(N_RANK)],
            "train_games_per_shard_min_max": [int(tr.groupby("shard").size().min()), int(tr.groupby("shard").size().max())],
            "val_games_per_shard_min_max": [int(val.groupby("shard").size().min()), int(val.groupby("shard").size().max())],
            "seed": SEED, "built": time.strftime("%F %T")}
    if a.done_list:
        done = sorted(int(x) for x in Path(a.done_list).read_text().split())
        ok_shard = np.zeros(nsh, bool)
        ok_shard[done] = True
        vmiss = val[~ok_shard[val["shard"].to_numpy()]]
        miss_rows = set(vmiss["row"].tolist())
        qmiss = np.array([any(int(r) in miss_rows for r in rows[i][rows[i] >= 0]) for i in range(len(rows))])
        info["coverage_asof"] = {"n_done_shards": len(done), "val_games_missing": int(len(vmiss)), "val_games_missing_by_rank": np.bincount(vmiss["rank_id"], minlength=N_RANK).tolist(),
                                 "val_games_missing_by_hd_or_color": {"color0": int((vmiss["color"] == 0).sum()), "color1": int((vmiss["color"] == 1).sum())},
                                 "val_games_missing_shards": sorted(set(int(s) for s in vmiss["shard"])), "questions_with_missing_game": int(qmiss.sum()),
                                 "questions_missing_by_N": {str(n): int(((q["ns"] == n) & qmiss).sum()) for n in (5, 10, 15, 20)},
                                 "questions_missing_by_rank": np.bincount(q["y"][qmiss], minlength=N_RANK).tolist(),
                                 "train_games_in_missing_shards": int((~ok_shard[tr["shard"].to_numpy()]).sum())}
    (out / "select_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(json.dumps(info, ensure_ascii=False, indent=1))


# ----------------------------------------------------------------------------------------------- extract (school)
def pick_positions(pos: pd.DataFrame, want: dict, ppg: int):
    """E94-C 的取法：ply>=2 的己方局面；多於 ppg 個就取 permutation(len)[:ppg]（子種子 crc32(uid)）再排序。-> rows(list of arrays), gids"""
    pos = pos.copy()
    pos["row"] = np.arange(len(pos))
    pp = pos[pos["uid"].isin(want) & (pos["ply"] >= PLY_MIN)]
    rows_all, gid_l = [], []
    for uid, gg in pp.groupby("uid", sort=False):
        r = gg["row"].to_numpy()
        if len(r) > ppg:
            r = r[np.sort(np.random.default_rng([SEED, 300, crc(uid)]).permutation(len(r))[:ppg])]
        rows_all.append(r)
        gid_l.append(np.full(len(r), want[uid], np.int32))
    return rows_all, gid_l


def cmd_extract(a) -> None:
    shards, out = Path(a.shards), Path(a.out)
    sel = pd.read_parquet(a.select)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    sids = sorted(int(s) for s in sel["shard"].unique())
    if a.only:
        sids = [s for s in sids if s in set(int(x) for x in a.only.split(","))]
    if a.max_shards:
        sids = sids[:a.max_shards]
    t0, n_done, pending, missing = time.time(), 0, [], {}
    for sid in sids:
        dst = parts / f"{sid:05d}"
        if (dst / "OK").exists():
            continue
        if not (shards / f"{sid:05d}.done").exists():
            pending.append(sid)
            continue
        if free_gb(out) < MIN_FREE_GB:
            raise SystemExit(f"disk free {free_gb(out):.1f}GB < {MIN_FREE_GB}GB: stop")
        g = sel[sel["shard"] == sid]
        d = shards / f"{sid:05d}"
        pos = pd.read_parquet(d / "pos.parquet", columns=["uid", "ply", "loc"])
        bits = np.load(d / "bits.npy", mmap_mode="r")
        glob = np.load(d / "glob.npy", mmap_mode="r")
        assert len(bits) == len(pos) == len(glob) and bits.shape[1] == BITS_W
        want = dict(zip(g["uid"], g["gid"]))
        rows_all, gid_l = pick_positions(pos, want, a.pos)
        got = set(int(x[0]) for x in gid_l)
        miss = sorted(set(g["uid"][~g["gid"].isin(got)]))
        if miss:
            missing[sid] = miss
        rows = np.concatenate(rows_all)
        gid = np.concatenate(gid_l)
        o = np.argsort(gid, kind="stable")
        rows, gid = rows[o], gid[o]
        tmp = parts / f".tmp_{sid:05d}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        np.save(tmp / "bits.npy", np.ascontiguousarray(bits[rows]))
        np.save(tmp / "glob.npy", np.ascontiguousarray(glob[rows]))
        np.save(tmp / "loc.npy", pos["loc"].to_numpy()[rows].astype(np.int16))
        np.save(tmp / "ply.npy", pos["ply"].to_numpy()[rows].astype(np.int16))
        np.save(tmp / "gid.npy", gid)
        (tmp / "info.json").write_text(json.dumps({"sid": sid, "n_games_selected": int(len(g)), "n_games_with_positions": len(got), "n_rows": int(len(rows)), "missing_uids": miss}))
        (tmp / "OK").write_text("ok")
        os.replace(tmp, dst) if not dst.exists() else (shutil.rmtree(dst), os.replace(tmp, dst))
        n_done += 1
        log(f"shard {sid} games {len(g)} with-positions {len(got)} rows {len(rows)} ({n_done} done this call, {time.time() - t0:.0f}s)")
    left = [s for s in sids if not (parts / f"{s:05d}" / "OK").exists()]
    log(f"extract finished: this call {n_done} shards; pending (shard not .done yet) {len(pending)}; not extracted {len(left)}; games without any eligible position: {sum(len(v) for v in missing.values())}")
    if missing:
        log("  games without positions (first 5):", {k: v[:5] for k, v in list(missing.items())[:5]})


# ----------------------------------------------------------------------------------------------- merge (school)
def cmd_merge(a) -> None:
    out, bdir = Path(a.out), Path(a.bundle)
    sel = pd.read_parquet(a.select)
    sids = sorted(int(s) for s in sel["shard"].unique())
    parts = out / "parts"
    have = [s for s in sids if (parts / f"{s:05d}" / "OK").exists()]
    if len(have) != len(sids) and not a.partial:
        raise SystemExit(f"merge needs all {len(sids)} shards; have {len(have)} (use --partial for a test bundle)")
    sel_h = sel[sel["shard"].isin(have)]
    sizes = {s: np.load(parts / f"{s:05d}" / "gid.npy", mmap_mode="r").shape[0] for s in have}
    N = sum(sizes.values())
    need_gb = N * (BITS_W + 38 + 4 + 4) / 1e9
    if free_gb(out) < need_gb + 3:
        raise SystemExit(f"disk free {free_gb(out):.1f}GB < need {need_gb + 3:.1f}GB")
    shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True)
    gid_all = np.concatenate([np.load(parts / f"{s:05d}" / "gid.npy") for s in have])
    assert (np.diff(gid_all) >= 0).all(), "gids not sorted across shards (gid must follow (shard,row) order)"
    sel_h = sel_h.sort_values("gid").reset_index(drop=True)
    cnt = np.bincount(gid_all, minlength=int(sel["gid"].max()) + 1)[sel_h["gid"].to_numpy()]
    keep = cnt > 0
    dropped = sel_h[~keep]
    sel_k = sel_h[keep].reset_index(drop=True)
    remap = np.full(int(sel["gid"].max()) + 1, -1, np.int64)
    remap[sel_k["gid"].to_numpy()] = np.arange(len(sel_k))
    arr = {"bits": (np.uint8, (N, BITS_W)), "glob": (np.float16, (N, 19)), "loc": (np.int16, (N,)), "ply": (np.int16, (N,)), "gid": (np.int32, (N,))}
    mm = {k: np.lib.format.open_memmap(bdir / f"{k}.npy", mode="w+", dtype=dt, shape=sh) for k, (dt, sh) in arr.items()}
    o = 0
    for s in have:
        n = sizes[s]
        for k in ("bits", "glob", "loc", "ply"):
            x = np.load(parts / f"{s:05d}" / f"{k}.npy", mmap_mode="r")
            assert x.dtype == mm[k].dtype and x.shape[1:] == mm[k].shape[1:], (k, x.dtype, x.shape)
            mm[k][o:o + n] = x
        g = np.load(parts / f"{s:05d}" / "gid.npy")
        mm["gid"][o:o + n] = remap[g].astype(np.int32)
        o += n
        log(f"merged shard {s} rows {o}/{N}")
    assert o == N and (mm["gid"][:] >= 0).all()
    for m in mm.values():
        m.flush()
    meta = dict(g_uid=sel_k["uid"].to_numpy().astype(str), g_player=sel_k["player_id"].to_numpy().astype(str), g_rank=sel_k["rank_id"].to_numpy().astype(np.int8),
                g_fold=sel_k["fold"].to_numpy().astype(np.int8), g_color=sel_k["color"].to_numpy().astype(np.int8), g_shard=sel_k["shard"].to_numpy().astype(np.int16),
                g_row=sel_k["row"].to_numpy().astype(np.int32))
    np.savez(bdir / "meta.npz", **meta)
    info = {"role": a.role, "complete": len(have) == len(sids), "n_shards": len(have), "n_games": int(len(sel_k)), "n_positions": int(N), "n_games_dropped_no_position": int(len(dropped)),
            "dropped_uids": dropped["uid"].tolist()[:50], "bytes_bits": int(N * BITS_W), "built": time.strftime("%F %T")}
    (bdir / "MERGED.json").write_text(json.dumps(info, indent=1))
    log("merge done", json.dumps(info))
    if a.role == "val" and len(dropped):
        raise SystemExit(f"validation bundle has {len(dropped)} games without positions - coverage broken, do not use")
    if a.delete_parts:
        shutil.rmtree(parts)
        log("parts deleted")


# ----------------------------------------------------------------------------------------------- verify
def load_bundle(bdir: Path, mmap=True):
    bdir = Path(bdir)
    m = np.load(bdir / "meta.npz", allow_pickle=False)
    d = {k: m[k] for k in m.files}
    for k in ("bits", "glob", "loc", "ply", "gid"):
        d[k] = np.load(bdir / f"{k}.npy", mmap_mode="r" if mmap else None)
    return d


def cmd_verify(a) -> None:
    d = load_bundle(a.bundle)
    f4 = set(Path(a.fold4).read_text().split())
    P, fold, rank, gid = d["g_player"], d["g_fold"], d["g_rank"], np.asarray(d["gid"])
    G = len(P)
    assert (np.diff(gid) >= 0).all() and gid.min() == 0 and gid.max() == G - 1
    cnt = np.bincount(gid, minlength=G)
    assert (cnt > 0).all(), "game without positions"
    assert d["bits"].shape[0] == len(gid) == d["glob"].shape[0] == d["loc"].shape[0] == d["ply"].shape[0]
    assert d["ply"].min() >= PLY_MIN
    if a.role == "train":
        assert not (set(P.tolist()) & f4), "FOLD-4 PLAYER IN TRAIN BUNDLE"
        assert (fold != 4).all() and set(fold.tolist()) <= {0, 1, 2, 3}
        if a.val_select:
            vp = set(pd.read_parquet(a.val_select)["player_id"])
            assert not (set(P.tolist()) & vp), "val player in train"
    else:
        assert set(P.tolist()) <= f4 and (fold == 4).all()
    print(a.role, "games", G, "players", len(set(P.tolist())), "positions", len(gid), "per game min/mean/max", cnt.min(), round(float(cnt.mean()), 2), cnt.max(), "ply min", int(d["ply"].min()))
    print("games by rank:", np.bincount(rank, minlength=N_RANK).tolist(), " players by rank:", [len(set(P[rank == r].tolist())) for r in range(N_RANK)])
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    s = sp_.add_parser("select"); s.add_argument("--root", default=str(Path(__file__).resolve().parents[3])); s.add_argument("--out", required=True); s.add_argument("--done-list", default="")
    e = sp_.add_parser("extract"); e.add_argument("--shards", required=True); e.add_argument("--select", required=True); e.add_argument("--out", required=True)
    e.add_argument("--pos", type=int, default=12); e.add_argument("--only", default=""); e.add_argument("--max-shards", type=int, default=0)
    m = sp_.add_parser("merge"); m.add_argument("--select", required=True); m.add_argument("--out", required=True); m.add_argument("--bundle", required=True)
    m.add_argument("--role", choices=["train", "val"], required=True); m.add_argument("--partial", action="store_true"); m.add_argument("--delete-parts", action="store_true")
    v = sp_.add_parser("verify"); v.add_argument("--bundle", required=True); v.add_argument("--role", choices=["train", "val"], required=True); v.add_argument("--fold4", required=True); v.add_argument("--val-select", default="")
    a = ap.parse_args()
    {"select": cmd_select, "extract": cmd_extract, "merge": cmd_merge, "verify": cmd_verify}[a.cmd](a)
