"""E103 C1：Rank 測試盤的 KataGo 輸入編碼（與 K 線／E98 逐位相同的路徑），組成 E98 格式的 bundle，並做逐位核對。

編碼器＝`encode_k.py` 的 KataEncoder／parse_sgf／skip_reason／move_loc（直接 import 受審版本，不複製；學校用 ~/gogame_e89/code/scripts/experiments/e89/encode_k.py，md5 與本機相同）。
取局面規則＝`e98_data.pick_positions`（本資料夾的 e98_data.py 是原檔的逐位複本）：己方手、ply >= 2、多於 12 個時以 np.random.default_rng([0, 300, crc32(uid)]).permutation 取前 12 再排序。
與 K 線 shards 的差別只有一處（刻意）：K 線先把每盤「全部」己方局面編碼再由 e98 抽取；這裡先算出要取的局面（只用 ply 與 uid），只編碼那 12 個。
  局面的特徵只由「到該手為止的盤面歷史」決定，所以結果應相同——這一點由 `check` 子命令與 shards 逐位核對（含全部局面的完整重編）。
測試盤 uid ＝ test_bags.npz 的 `<question_id>#<k>`（例 pub_q_0001#3，k 從 1 起，依題目 CSV 的欄位順序）；確定性，crc32 的子種子由這個字串得出。

子命令
  prep-test   （本機 .venv）讀 data/contest_1104/rank_prediction_test_{public,private}.csv -> games_test.parquet、test_questions.npz（與 test_bags／test_uids／t_games_uids 逐項核對）
  pick-check  （本機 .venv）從訓練 CSV 挑核對用的盤（黑白、定先、跨片、訓練與驗證都有）-> check_games.parquet
  build       （學校 e89 venv，nice 19，4 workers）games parquet -> E98 格式 bundle 目錄（bits/glob/loc/ply/gid .npy、meta.npz、MERGED.json、own_idx.npy）
  check       （學校）用 build 的路徑重編 check_games，與 K 線 shards／bundle_train／bundle_val12 逐位比對
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE
sys.path.insert(0, str(HERE))
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

import e98_data as D98                  # noqa: E402  (verbatim copy of scripts/experiments/e98/e98_data.py)

PPG = 12
BITS_W, N_GLOB, N_CH = 993, 19, 22
RID0 = 30_000_000                       # Rank test games: rid = 30M + position in test_bags uids order (= t_games.npz convention)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_encode_k():
    school = Path.home() / "gogame_e89/code/scripts/experiments/e89/encode_k.py"
    p = Path(os.environ.get("E103_ENCODE_K") or (school if school.exists() else ROOT / "scripts/experiments/e89/encode_k.py"))
    spec = importlib.util.spec_from_file_location("encode_k", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["encode_k"] = m
    spec.loader.exec_module(m)
    return m, p


# ----------------------------------------------------------------------------------------------- prep-test (local)
def cmd_prep_test(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    EK, _ = load_encode_k()
    bags = np.load(ROOT / ".scratch/e29/test/test_bags.npz", allow_pickle=True)
    ru = bags["uids"].astype(str)
    tu = np.load(ROOT / ".scratch/e91/test_uids.npz")
    t_uids = np.load(ROOT / ".scratch/e90/data/t_games_uids.npz")["uids"][14200:24200]
    assert len(ru) == 10000 and (t_uids == ru).all() and (tu["uids"] == ru).all()
    assert (tu["rid"] == RID0 + np.arange(10000)).all()
    rows_ = []
    for w, sp in (("public", 0), ("private", 1)):
        q = pd.read_csv(ROOT / f"data/contest_1104/rank_prediction_test_{w}.csv", dtype=str)
        for r in q.itertuples(index=False):
            d = dict(zip(q.columns, r))
            for k in range(1, int(d["num_games"]) + 1):
                rows_.append(dict(uid=f"{d['question_id']}#{k}", qid=d["question_id"], split=sp, k=k, color=0 if d[f"color_{k}"] == "B" else 1, sgf=d[f"sgf_{k}"], n=int(d["num_games"])))
                assert d[f"color_{k}"] in ("B", "W")
            assert all(pd.isna(d[f"sgf_{k}"]) for k in range(int(d["num_games"]) + 1, 21))
    g = pd.DataFrame(rows_)
    assert len(g) == 10000 and (g["uid"].to_numpy() == ru).all(), "uid order != test_bags uids"
    g["rid"] = (RID0 + np.arange(10000)).astype(np.int64)
    g["hd"] = np.array([EK.hd_of(s) for s in g["sgf"]], np.int8)
    assert (g["hd"].to_numpy() == tu["hd"]).all(), "hd differs from test_uids"
    assert (g["color"].to_numpy() == tu["eff"]).all(), "colour differs from test_uids eff (own colour)"
    assert (g["split"].to_numpy() == tu["split"]).all()
    g.to_parquet(out / "games_test.parquet", index=False, compression="zstd")
    qids = bags["qids"].astype(str)
    rows = np.full((800, 20), -1, np.int64)
    ns = np.zeros(800, np.int64)
    qcol = np.zeros(800, np.int64)
    qsplit = np.zeros(800, np.int64)
    pos_of = {u: i for i, u in enumerate(ru)}
    for i, qid in enumerate(qids):
        gg = g[g["qid"] == qid]
        assert len(gg) == int(bags["n"][i]) and (gg["n"] == len(gg)).all() and gg["color"].nunique() == 1
        assert int(gg["color"].iloc[0]) == int(bags["color"][i])
        rows[i, :len(gg)] = [RID0 + pos_of[u] for u in gg["uid"]]
        ns[i], qcol[i], qsplit[i] = len(gg), int(gg["color"].iloc[0]), int(gg["split"].iloc[0])
        # the bag rows of test_bags (positions) must be the same games
        r = bags["rows"][i]
        assert (RID0 + r[r >= 0] == rows[i][rows[i] >= 0]).all()
    np.savez(out / "test_questions.npz", rows=rows, ns=ns, y=np.full(800, -1, np.int64), player=qids.astype("<U16"), qid=qids, color=qcol, split=qsplit)
    info = dict(n_games=10000, n_questions=800, public=int((qsplit == 0).sum()), private=int((qsplit == 1).sum()), N_counts={int(n): int((ns == n).sum()) for n in (5, 10, 15, 20)},
                hd_counts=np.bincount(g["hd"], minlength=3).tolist(), color_counts=np.bincount(g["color"], minlength=2).tolist(), uid_rule="test_bags.npz uids = '<question_id>#<k>' (k from 1), order = public CSV rows then private CSV rows",
                rid_rule="30,000,000 + position in that order (same as t_games.npz / test_uids.npz rid)")
    (out / "prep_test_info.json").write_text(json.dumps(info, indent=1))
    log("prep-test OK", json.dumps(info))


# ----------------------------------------------------------------------------------------------- pick-check (local)
def cmd_pick_check(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    EK, _ = load_encode_k()
    rng = np.random.default_rng(103)
    st = pd.read_parquet(ROOT / ".scratch/e98/sel/select_train.parquet")
    sv = pd.read_parquet(ROOT / ".scratch/e98/sel/select_val.parquet")
    cand = pd.concat([st.iloc[rng.permutation(len(st))[:a.n_train]].assign(role="train"), sv.iloc[rng.permutation(len(sv))[:a.n_val]].assign(role="val")])
    want = dict(zip(cand["uid"], range(len(cand))))
    got = {}
    for f in sorted((ROOT / "data/training_v1.1/training").glob("train_*.csv")):
        for ch in pd.read_csv(f, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], chunksize=5000, dtype=str):
            u = ch["rank"] + "_" + ch["game_id"]
            m = u.isin(want)
            for uu, col, s, pl in zip(u[m], ch["color"][m], ch["sgf_content"][m], ch["player_id"][m]):
                got[uu] = (col, s, pl)
        log("read", f.name, len(got))
    assert set(got) == set(want), (len(got), len(want))
    cand["color_csv"] = [0 if got[u][0] == "B" else 1 for u in cand["uid"]]
    assert (cand["color_csv"].to_numpy() == cand["color"].to_numpy()).all(), "CSV colour != select colour"
    cand["sgf"] = [got[u][1] for u in cand["uid"]]
    cand["hd"] = [EK.hd_of(s) for s in cand["sgf"]]
    log("candidates by role/hd/colour:", cand.groupby(["role", "hd", "color"]).size().to_dict())
    pick, used = [], set()

    def take(df, n, label):
        for _, r in df.sample(frac=1, random_state=103).iterrows():
            if n == 0:
                break
            if (r["role"], r["shard"]) in used:
                continue
            used.add((r["role"], r["shard"]))
            pick.append(dict(r, why=label))
            n -= 1
        assert n == 0, f"not enough candidates for {label}"
    tr, va = cand[cand.role == "train"], cand[cand.role == "val"]
    take(tr[(tr.hd == 0) & (tr.color == 0)], 8, "train hd0 black")
    take(tr[(tr.hd == 0) & (tr.color == 1)], 6, "train hd0 white")
    take(tr[(tr.hd == 1) & (tr.color == 0)], 3, "train hd1 black")
    take(tr[(tr.hd == 1) & (tr.color == 1)], 3, "train hd1 white")
    take(tr[tr.hd >= 2], 2, "train hd>=2 (extra, test has none)")
    take(va[(va.hd == 0) & (va.color == 0)], 3, "val hd0 black")
    take(va[(va.hd == 0) & (va.color == 1)], 3, "val hd0 white")
    take(va[(va.hd == 1)], 3, "val hd1")
    p = pd.DataFrame(pick)[["uid", "role", "color", "shard", "hd", "rank_id", "fold", "gid", "row", "sgf", "why"]]
    p.to_parquet(out / "check_games.parquet", index=False)
    log("check games", len(p), "roles", p.role.value_counts().to_dict(), "colours", p.color.value_counts().to_dict(), "hd", p.hd.value_counts().to_dict(), "distinct shards", p[["role", "shard"]].drop_duplicates().shape[0])


# ----------------------------------------------------------------------------------------------- build (school)
_W = {}


def _enc():
    if "enc" not in _W:
        _W["enc"] = _W["EK"].KataEncoder(_W["katago_src"], json.load(open(_W["model_config"])))
    return _W["enc"]


def own_positions(EK, sgf: str, color: int):
    """(moves, own_plies[list], why).  Same skip rule and own-move definition as encode_k.encode_shard."""
    pg = EK.parse_sgf(sgf)
    why = EK.skip_reason(pg)
    if why is not None:
        return None, [], why
    own_plies = [i for i, m in enumerate(pg.moves.tolist()) if m[0] == color]
    if not own_plies:
        return None, [], "no_own_move"
    return pg.moves, own_plies, None


def process_game(args):
    uid, color, sgf, ppg = args
    EK = _W["EK"]
    moves, own_plies, why = own_positions(EK, sgf, int(color))
    if why is not None:
        return dict(uid=uid, status=why)
    n_own = len(own_plies)
    pos = pd.DataFrame({"uid": [uid] * n_own, "ply": np.array(own_plies, np.int32), "own_idx": np.arange(n_own, dtype=np.int16)})
    rows_all, gid_l = D98.pick_positions(pos, {uid: 0}, ppg)          # the very function e98_data.cmd_extract uses
    if not rows_all:
        return dict(uid=uid, status="no_eligible_position")
    rows = np.concatenate(rows_all)
    own_idx = pos["own_idx"].to_numpy()[rows].astype(np.int64)
    try:
        b, g, plies = _enc().encode_game(moves, int(color), own_idx)
    except Exception as e:                                             # recorded, never silent
        return dict(uid=uid, status=f"encode_error:{type(e).__name__}:{e}")
    assert b.shape == (len(rows), N_CH, 19, 19) and g.shape == (len(rows), N_GLOB)
    assert (plies == pos["ply"].to_numpy()[rows]).all()
    return dict(uid=uid, status="ok", bits=np.packbits(b.reshape(len(rows), -1), axis=1), glob=g.astype(np.float16), loc=EK.move_loc(moves, plies), ply=plies.astype(np.int16),
                own_idx=own_idx.astype(np.int16), n_own=n_own, n_eligible=int((pos["ply"] >= D98.PLY_MIN).sum()))


def run_games(games: pd.DataFrame, a) -> list:
    import multiprocessing as mp
    EK, p = load_encode_k()
    _W.update(EK=EK, katago_src=a.katago_src, model_config=a.model_config)
    log("encode_k from", p)
    tasks = [(u, c, s, PPG) for u, c, s in zip(games["uid"], games["color"], games["sgf"])]
    t0 = time.time()
    res = []
    if a.workers <= 1:
        for i, t in enumerate(tasks):
            res.append(process_game(t))
            if i % 100 == 0:
                log(f"{i}/{len(tasks)} {time.time() - t0:.0f}s")
    else:
        with mp.get_context("fork").Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap(process_game, tasks, chunksize=8)):
                res.append(r)
                if i % 500 == 0:
                    log(f"{i}/{len(tasks)} {time.time() - t0:.0f}s")
    log(f"encoded {len(res)} games in {time.time() - t0:.0f}s")
    return res


def write_bundle(games: pd.DataFrame, res: list, bdir: Path, role_meta: dict) -> dict:
    """E98 bundle format (e98_data.cmd_merge): bits uint8 (N,993), glob float16 (N,19), loc int16, ply int16, gid int32 (nondecreasing), meta.npz (g_uid, g_player, g_rank, g_fold, g_color, g_shard, g_row)."""
    bad = [(r["uid"], r["status"]) for r in res if r["status"] != "ok"]
    assert [r["uid"] for r in res] == games["uid"].tolist()
    ok = [i for i, r in enumerate(res) if r["status"] == "ok"]
    gid = np.concatenate([np.full(len(res[i]["ply"]), j, np.int32) for j, i in enumerate(ok)])
    bdir.mkdir(parents=True, exist_ok=True)
    np.save(bdir / "bits.npy", np.concatenate([res[i]["bits"] for i in ok]).astype(np.uint8))
    np.save(bdir / "glob.npy", np.concatenate([res[i]["glob"] for i in ok]).astype(np.float16))
    np.save(bdir / "loc.npy", np.concatenate([res[i]["loc"] for i in ok]).astype(np.int16))
    np.save(bdir / "ply.npy", np.concatenate([res[i]["ply"] for i in ok]).astype(np.int16))
    np.save(bdir / "gid.npy", gid)
    np.save(bdir / "own_idx.npy", np.concatenate([res[i]["own_idx"] for i in ok]).astype(np.int16))
    gk = games.iloc[ok].reset_index(drop=True)
    meta = dict(g_uid=gk["uid"].to_numpy().astype(str), g_player=role_meta["player"](gk), g_rank=role_meta["rank"](gk), g_fold=role_meta["fold"](gk), g_color=gk["color"].to_numpy().astype(np.int8),
                g_shard=role_meta["shard"](gk), g_row=role_meta["row"](gk))
    np.savez(bdir / "meta.npz", **meta)
    cnt = np.bincount(gid, minlength=len(ok))
    info = dict(complete=len(bad) == 0, n_games=len(ok), n_games_input=len(games), n_positions=int(len(gid)), n_games_dropped_no_position=len(bad), dropped=bad[:50], positions_per_game_min_mean_max=[int(cnt.min()), float(cnt.mean()), int(cnt.max())],
                n_games_fewer_than_12=int((cnt < PPG).sum()), built=time.strftime("%F %T"), seed=D98.SEED, ply_min=D98.PLY_MIN)
    (bdir / "MERGED.json").write_text(json.dumps(info, indent=1))
    return info


def cmd_build(a) -> None:
    games = pd.read_parquet(a.games)
    if a.limit:
        games = games.iloc[:a.limit].reset_index(drop=True)
    res = run_games(games, a)
    rm = dict(player=lambda g: g["qid"].to_numpy().astype(str), rank=lambda g: np.full(len(g), -1, np.int8), fold=lambda g: np.full(len(g), 4, np.int8),
              shard=lambda g: np.full(len(g), -1, np.int16), row=lambda g: g["rid"].to_numpy().astype(np.int32))
    info = write_bundle(games, res, Path(a.bundle), rm)
    log("bundle", json.dumps(info))
    if not info["complete"]:
        raise SystemExit(f"{info['n_games_dropped_no_position']} games without positions: {info['dropped'][:5]}")


# ----------------------------------------------------------------------------------------------- check (school)

def cmd_check(a) -> None:
    games = pd.read_parquet(a.games)
    res = run_games(games, a)
    EK = _W["EK"]
    shards, kgames = Path(a.shards), Path(a.kgames)
    bt, bv = D98.load_bundle(a.bundle_train), D98.load_bundle(a.bundle_val)
    gid_t, gid_v = np.asarray(bt["gid"]), np.asarray(bv["gid"])
    ut = {u: i for i, u in enumerate(bt["g_uid"].tolist())}
    uv = {u: i for i, u in enumerate(bv["g_uid"].tolist())}
    rep, n_games_ok, n_pos12, n_posall = [], 0, 0, 0
    all_ok = True
    for (r, gr) in zip(res, games.itertuples(index=False)):
        uid, sid = gr.uid, int(gr.shard)
        d = shards / f"{sid:05d}"
        rec = dict(uid=uid, role=gr.role, color=int(gr.color), hd=int(gr.hd), shard=sid, why=gr.why)
        pos = pd.read_parquet(d / "pos.parquet")
        pos["row"] = np.arange(len(pos))
        sub = pos[pos["uid"] == uid]
        bits_s = np.load(d / "bits.npy", mmap_mode="r")
        glob_s = np.load(d / "glob.npy", mmap_mode="r")
        # (0) input SGF / colour identical to what the K line was given
        kg = pd.read_parquet(kgames / f"{sid:05d}.parquet", columns=["uid", "color", "sgf"])
        kg = kg[kg["uid"] == uid]
        rec["sgf_equals_kline_input"] = bool(len(kg) == 1 and kg["sgf"].iloc[0] == gr.sgf and int(kg["color"].iloc[0]) == int(gr.color))
        assert r["status"] == "ok", (uid, r["status"])
        # (a) the 12 positions chosen by the new path == the K-line + e98_data.cmd_extract choice (pick_positions on the shard's pos.parquet)
        rows_all, _ = D98.pick_positions(pos[["uid", "ply", "loc"]], {uid: 0}, PPG)
        rows_e98 = np.concatenate(rows_all)
        rec["n_positions"] = int(len(rows_e98))
        rec["same_selection_as_e98_pick_on_shard_pos"] = bool(np.array_equal(pos["ply"].to_numpy()[rows_e98], r["ply"]) and np.array_equal(pos["own_idx"].to_numpy()[rows_e98], r["own_idx"]))
        # (b) bits / glob / loc / ply of those positions vs the shard rows (bitwise)
        rs = sub["row"].to_numpy()[r["own_idx"].astype(np.int64)]
        assert (pos["own_idx"].to_numpy()[rs] == r["own_idx"]).all() and (pos["ply"].to_numpy()[rs] == r["ply"]).all()
        rec["bits_eq_shard"] = bool(np.array_equal(r["bits"], bits_s[rs]))
        rec["glob_eq_shard"] = bool(np.array_equal(r["glob"].view(np.uint16), np.asarray(glob_s[rs]).view(np.uint16)))
        rec["loc_eq_shard"] = bool(np.array_equal(r["loc"], pos["loc"].to_numpy()[rs].astype(np.int16)))
        rec["ply_eq_shard"] = bool(np.array_equal(r["ply"], pos["ply"].to_numpy()[rs].astype(np.int16)))
        # (c) vs the E98 bundle (train or val12) that the model was trained / validated on
        ub, gidx, B = (ut, gid_t, bt) if gr.role == "train" else (uv, gid_v, bv)
        j = ub[uid]
        s0, s1 = np.searchsorted(gidx, j), np.searchsorted(gidx, j, side="right")
        rec["bundle_rows"] = int(s1 - s0)
        rec["bits_eq_bundle"] = bool(np.array_equal(r["bits"], np.asarray(B["bits"][s0:s1])))
        rec["glob_eq_bundle"] = bool(np.array_equal(r["glob"].view(np.uint16), np.asarray(B["glob"][s0:s1]).view(np.uint16)))
        rec["loc_eq_bundle"] = bool(np.array_equal(r["loc"], np.asarray(B["loc"][s0:s1])))
        rec["ply_eq_bundle"] = bool(np.array_equal(r["ply"], np.asarray(B["ply"][s0:s1])))
        # (d) full re-encode of EVERY own position of the game from the raw SGF with the K-line routine, vs the whole shard block
        moves, own_plies, why = own_positions(EK, gr.sgf, int(gr.color))
        n_own = len(own_plies)
        b, g, plies = _enc().encode_game(moves, int(gr.color), np.arange(n_own))
        pk = np.packbits(b.reshape(n_own, -1), axis=1)
        rec["n_own_positions"] = n_own
        rec["shard_has_same_count"] = bool(len(sub) == n_own)
        rec["all_bits_eq_shard"] = bool(len(sub) == n_own and np.array_equal(pk, bits_s[sub["row"].to_numpy()]))
        rec["all_glob_eq_shard"] = bool(len(sub) == n_own and np.array_equal(g.astype(np.float16).view(np.uint16), np.asarray(glob_s[sub["row"].to_numpy()]).view(np.uint16)))
        rec["all_loc_ply_eq_shard"] = bool(len(sub) == n_own and np.array_equal(plies.astype(np.int16), sub["ply"].to_numpy()) and np.array_equal(EK.move_loc(moves, plies), sub["loc"].to_numpy()))
        ok = all(v for k, v in rec.items() if isinstance(v, bool)) and rec["bundle_rows"] == rec["n_positions"]
        rec["ALL_OK"] = bool(ok)
        all_ok &= ok
        n_games_ok += int(ok)
        n_pos12 += rec["n_positions"]
        n_posall += n_own
        rep.append(rec)
        log(("OK  " if ok else "FAIL"), uid, gr.role, "color", rec["color"], "hd", rec["hd"], "shard", sid, "12-sel", rec["n_positions"], "all", n_own)
    summ = dict(n_games=len(rep), n_games_all_ok=n_games_ok, n_selected_positions_compared=n_pos12, n_all_positions_compared=n_posall, all_ok=bool(all_ok),
                by_role=games["role"].value_counts().to_dict(), by_color=games["color"].value_counts().to_dict(), by_hd=games["hd"].value_counts().to_dict(), n_distinct_shards=int(games[["role", "shard"]].drop_duplicates().shape[0]))
    Path(a.out).write_text(json.dumps(dict(summary=summ, games=rep), indent=1, default=str))
    log("CHECK SUMMARY", json.dumps(summ))
    if not all_ok:
        raise SystemExit("CHECK FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep-test"); p.add_argument("--out", required=True)
    p = sp.add_parser("pick-check"); p.add_argument("--out", required=True); p.add_argument("--n-train", type=int, default=1500); p.add_argument("--n-val", type=int, default=300)

    def common(p):
        p.add_argument("--katago-src", default=str(Path.home() / "gogame_e89/katago_src")); p.add_argument("--model-config", default=str(Path.home() / "gogame_e89/model_config.json")); p.add_argument("--workers", type=int, default=4)
    p = sp.add_parser("build"); p.add_argument("--games", required=True); p.add_argument("--bundle", required=True); p.add_argument("--limit", type=int, default=0); common(p)
    p = sp.add_parser("check"); p.add_argument("--games", required=True); p.add_argument("--out", required=True); common(p)
    p.add_argument("--shards", default=str(Path.home() / "gogame_e89/k/shards")); p.add_argument("--kgames", default=str(Path.home() / "gogame_e89/k/games"))
    p.add_argument("--bundle-train", default=str(Path.home() / "gogame_e98/bundle_train")); p.add_argument("--bundle-val", default=str(Path.home() / "gogame_e98/bundle_val12"))
    a = ap.parse_args()
    {"prep-test": cmd_prep_test, "pick-check": cmd_pick_check, "build": cmd_build, "check": cmd_check}[a.cmd](a)
