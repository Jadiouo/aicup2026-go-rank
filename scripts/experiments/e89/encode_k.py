"""E89 K line: KataGo input features for EVERY own-move position of the 712,945 non-H bundle games (pure CPU, resumable shards).

Subcommands
  plan   (local only; reads bundle aux.npz + training CSV; H check)  -> <out>/games/<sid>.parquet, plan.json, order.npy
  run    (school; 4 workers)                                          -> <root>/shards/<sid>/..., <sid>.done, status.json
  encode-games (local, for the 50-game check): encode given uids with this file's encoder -> npz
  show   one-line peek into a finished shard

Encoder = verbatim copy of scripts/experiments/e13/encode.py KataEncoder (KataGo python Features, 22 spatial planes + 19 global,
profile rank_1d, Japanese rules).  Skip rule = E13/E79 `_skip_reason` (setup stones / bad size / non-alternating / bad coord /
first move not black), recorded per reason.  Games whose KataGo replay raises are skipped and recorded as encode_error:<type>.
Plain CPU Python; no network, no GPU, no external binaries.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[3]          # repo root locally; ~/gogame_e89/code on the school machine
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from gogame.sgf import parse_sgf                    # noqa: E402

PROFILE = "rank_1d"
SEED = 89
GAMES_PER_SHARD = 2000
N_CH, N_GLOB, BITS_W = 22, 19, 993
_RANK_INV = {**{f"{i}d": 10 - i for i in range(1, 10)}, **{f"{i}k": 9 + i for i in range(1, 21)}}


# ----------------------------------------------------------------------------------------------- encoder (copy of E13 KataEncoder)
class KataEncoder:
    def __init__(self, katago_src: str, model_config: dict, profile: str = PROFILE):
        sys.path.insert(0, str(Path(katago_src) / "python"))
        from katago.game.board import Board
        from katago.game.features import Features
        from katago.game.gamestate import GameState
        from katago.game.sgfmetadata import SGFMetadata
        self.Board, self.GameState, self.SGFMetadata = Board, GameState, SGFMetadata
        self.feats = Features(model_config, 19)
        self.meta = self._profile_meta(profile)
        self.meta_rows = {c: np.asarray(self.meta.get_metadata_row(pla, 361), np.float32)
                          for c, pla in ((0, Board.BLACK), (1, Board.WHITE))}

    def _profile_meta(self, name: str):
        pre = name.startswith("preaz_")
        s = name.split("_", 1)[1]
        parts = s.split("_") if "_" in s else [s, s]
        return self.SGFMetadata(inverseBRank=_RANK_INV[parts[0]], inverseWRank=_RANK_INV[parts[1]], bIsHuman=True, wIsHuman=True,
                                gameRatednessIsUnknown=True, tcIsUnknown=False, tcIsByoYomi=True, mainTimeSeconds=1200,
                                periodTimeSeconds=30, byoYomiPeriods=5,
                                gameDate=datetime.date(2016, 9, 1) if pre else datetime.date(2020, 3, 1),
                                source=self.SGFMetadata.SOURCE_KGS)

    def encode_game(self, moves: np.ndarray, color: int, own_idx: np.ndarray, tick=None):
        B = self.Board
        want = set(int(j) for j in own_idx)
        own_plies = [i for i, m in enumerate(moves.tolist()) if m[0] == color]
        target = {own_plies[j]: j for j in want}
        gs = self.GameState(19, self.GameState.RULES_JAPANESE)
        out_b, out_g, plies = [], [], []
        for ply, (c, r, cc) in enumerate(moves.tolist()):
            if ply in target:
                b, g = gs.get_input_features(self.feats)
                bb = np.asarray(b[0])
                if not np.isin(bb, (0, 1)).all():
                    raise ValueError("non-binary spatial plane: bit packing is invalid for this KataGo version")
                out_b.append(bb.astype(np.uint8))
                out_g.append(np.asarray(g[0], np.float32))
                plies.append(ply)
            loc = B.PASS_LOC if r < 0 else gs.board.loc(int(cc), int(r))         # x = column, y = row
            gs.play(B.BLACK if c == 0 else B.WHITE, loc)
            if ply >= max(target):
                break
        return (np.stack(out_b), np.stack(out_g), np.array(plies, np.int32)) if out_b else \
            (np.zeros((0, 1, 19, 19), np.uint8), np.zeros((0, 1), np.float32), np.zeros(0, np.int32))


_HD_RE = re.compile(r"HD\[(\d+)\]")


def hd_of(sgf: str) -> int:
    """Handicap tag of the v1.1 SGF root (0 if absent).  HD>=2 games carry NO setup stones and B/W reversed (sgf.py normalisation note);
    E13/E79 encoded them as plain games (no stones) and so does this script -- they are only TAGGED here (games.parquet `hd`)."""
    m = _HD_RE.search(sgf[:200])
    return int(m.group(1)) if m else 0


def skip_reason(pg) -> str | None:
    f = pg.flags
    for name in ("has_setup", "bad_size", "nonalt", "bad_coord", "first_not_black"):
        if getattr(f, name):
            return name
    return None


def move_loc(moves: np.ndarray, plies: np.ndarray) -> np.ndarray:
    """Played point of each position's own move: row*19+col, -1 for a pass (same as E79 prep)."""
    return np.array([(-1 if moves[p][1] < 0 else int(moves[p][1]) * 19 + int(moves[p][2])) for p in plies], np.int16)


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- plan (local)
def cmd_plan(a) -> None:
    out = Path(a.out)
    (out / "games").mkdir(parents=True, exist_ok=True)
    aux = np.load(a.aux, allow_pickle=True)
    uid, gp, gc, pids = aux["uid"], aux["game_player"], aux["game_color"], aux["player_ids"]
    n = len(uid)
    sp = pd.read_parquet(a.split)
    H = set(sp.loc[sp["is_holdout"], "player_id"])
    bundle_players = set(pids[np.unique(gp)])
    assert not (bundle_players & H), f"H players in bundle: {sorted(bundle_players & H)[:3]}"          # assert (1): bundle has no H player
    # order: round r = r-th game of every player (per-player shuffle, seed 89), players in index order inside a round
    rnd = np.empty(n, np.int32)
    for p in np.unique(gp):
        rows = np.flatnonzero(gp == p)
        perm = np.random.default_rng([SEED, int(p)]).permutation(len(rows))
        rnd[rows[perm]] = np.arange(len(rows))
    order = np.lexsort((np.arange(n), gp, rnd)).astype(np.int32)       # sort by round, then player, then row
    n_sh = (n + GAMES_PER_SHARD - 1) // GAMES_PER_SHARD
    # SGF text
    want = {u: i for i, u in enumerate(uid)}
    sgf = [None] * n
    csv_pid = [None] * n
    md5_new = {}
    for f in sorted(Path(a.csv_dir).glob("train_*.csv")):
        for ch in pd.read_csv(f, usecols=["game_id", "rank", "player_id", "color", "sgf_content"], chunksize=40_000):
            u = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str)
            for uu, pl, col, s in zip(u.tolist(), ch["player_id"].tolist(), ch["color"].tolist(), ch["sgf_content"].tolist()):
                i = want.get(uu)
                if i is None:
                    continue
                assert sgf[i] is None, f"duplicate uid {uu}"
                assert pl == pids[gp[i]], f"{uu}: csv player != bundle player"
                assert (0 if col == "B" else 1) == int(gc[i]), f"{uu}: colour mismatch"
                sgf[i], csv_pid[i] = s, pl
                md5_new[i] = hashlib.md5(s.encode()).digest()
        print("read", f.name, flush=True)
    missing = [uid[i] for i in range(n) if sgf[i] is None]
    assert not missing, f"{len(missing)} uids without SGF: {missing[:3]}"
    assert not (set(csv_pid) & H), "H player in the CSV rows of the bundle uids"                         # assert (2): by CSV player_id
    # compare with the older data/training copy (hash only)
    n_diff = n_old_missing = 0
    if a.old_csv_dir:
        for f in sorted(Path(a.old_csv_dir).glob("train_*.csv")):
            for ch in pd.read_csv(f, usecols=["game_id", "rank", "sgf_content"], chunksize=40_000):
                u = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str)
                for uu, s in zip(u.tolist(), ch["sgf_content"].tolist()):
                    i = want.get(uu)
                    if i is not None:
                        n_diff += hashlib.md5(s.encode()).digest() != md5_new[i]
    # skip flags and own-move counts
    reasons, n_pos_total, n_pos_sh = {}, 0, np.zeros(n_sh, np.int64)
    skip = [None] * n
    n_own = np.zeros(n, np.int32)
    for i in range(n):
        pg = parse_sgf(sgf[i])
        r = skip_reason(pg)
        if r is None:
            n_own[i] = int((pg.moves[:, 0] == int(gc[i])).sum())
            if n_own[i] == 0:
                r = "no_own_move"
        skip[i] = r
        if r is not None:
            reasons[r] = reasons.get(r, 0) + 1
    pos_of = np.where(np.array([s is None for s in skip]), n_own, 0)
    for k in range(n_sh):
        sel = order[k * GAMES_PER_SHARD:(k + 1) * GAMES_PER_SHARD]
        n_pos_sh[k] = int(pos_of[sel].sum())
    n_pos_total = int(n_pos_sh.sum())
    shard_hash = {}
    for k in range(n_sh):
        sel = order[k * GAMES_PER_SHARD:(k + 1) * GAMES_PER_SHARD]
        df = pd.DataFrame({"uid": uid[sel].astype(str), "row": sel.astype(np.int32), "player_id": [str(csv_pid[i]) for i in sel],
                           "color": gc[sel].astype(np.int8), "round": rnd[sel], "sgf": [sgf[i] for i in sel]})
        f = out / "games" / f"{k:05d}.parquet"
        df.to_parquet(f, index=False, compression="zstd", compression_level=9)
    plan = {"n_games": int(n), "n_shards": int(n_sh), "games_per_shard": GAMES_PER_SHARD, "seed": SEED, "n_players": int(len(bundle_players)),
            "skip_reasons_plan": reasons, "n_skipped_plan": int(sum(reasons.values())), "n_positions_plan": n_pos_total,
            "positions_per_shard_min_max": [int(n_pos_sh.min()), int(n_pos_sh.max())], "max_round": int(rnd.max()),
            "csv_dir": str(a.csv_dir), "n_sgf_differs_from_old_data_training": int(n_diff), "H_players_in_split": len(H),
            "h_check": "bundle players and CSV player_id both disjoint from split.is_holdout (assert)",
            "est_bytes_per_position": BITS_W + N_GLOB * 2 + 14, "built": time.strftime("%F %T")}
    plan["est_disk_GB"] = round(n_pos_total * plan["est_bytes_per_position"] / 1e9, 1)
    write_json_atomic(out / "plan.json", plan)
    np.save(out / "order.npy", order)
    print(json.dumps(plan, ensure_ascii=False, indent=1))


# ----------------------------------------------------------------------------------------------- shard encode (worker)
_ENC = [None]
_HB = [None]


def _get_enc(cfg: dict):
    if _ENC[0] is None:
        _ENC[0] = KataEncoder(cfg["katago_src"], json.load(open(cfg["model_config"])))
    return _ENC[0]


def _beat(root: Path) -> None:
    hb = root / "hb"
    hb.mkdir(exist_ok=True)
    (hb / f"w{os.getpid()}").write_text(str(time.time()))


def encode_shard(root: Path, cfg: dict, sid: int) -> dict:
    t0 = time.time()
    shards = root / "shards"
    final, tmp, done = shards / f"{sid:05d}", shards / f".tmp_{sid:05d}", shards / f"{sid:05d}.done"
    for p in (tmp, final):
        if p.exists():
            shutil.rmtree(p)
    if done.exists():
        done.unlink()
    gpath = root / "games" / f"{sid:05d}.parquet"
    games = pd.read_parquet(gpath)
    enc = _get_enc(cfg)
    ppid0 = os.getppid()
    parsed, n_cap = [], 0
    for r in games.itertuples(index=False):
        pg = parse_sgf(r.sgf)
        why = skip_reason(pg)
        n_own = int((pg.moves[:, 0] == r.color).sum()) if why is None else 0
        if why is None and n_own == 0:
            why = "no_own_move"
        parsed.append((pg.moves if why is None else None, n_own, why))
        n_cap += n_own
    bits = np.empty((n_cap, BITS_W), np.uint8)
    glob = np.empty((n_cap, N_GLOB), np.float16)
    gi_l, own_l, ply_l, loc_l, status = [], [], [], [], []
    k = 0
    for gi, (r, (moves, n_own, why)) in enumerate(zip(games.itertuples(index=False), parsed)):
        if gi % 20 == 0:
            if os.getppid() != ppid0:
                raise SystemExit("parent died")
            _beat(root)
        n_done = 0
        if why is None:
            try:
                idx = np.arange(n_own, dtype=np.int64)
                b, g, plies = enc.encode_game(moves, int(r.color), idx)
                assert b.shape == (n_own, N_CH, 19, 19) and g.shape == (n_own, N_GLOB) and len(plies) == n_own, (b.shape, g.shape)
                pk = np.packbits(b.reshape(n_own, -1), axis=1)
                assert pk.shape[1] == BITS_W
                bits[k:k + n_own] = pk
                glob[k:k + n_own] = g.astype(np.float16)
                gi_l.append(np.full(n_own, gi, np.int32))
                own_l.append(np.arange(n_own, dtype=np.int16))
                ply_l.append(plies.astype(np.int16))
                loc_l.append(move_loc(moves, plies))
                k += n_own
                n_done = n_own
            except Exception as e:                                      # recorded, never silent
                why = f"encode_error:{type(e).__name__}"
        status.append((why or "ok", n_done))
    bits, glob = bits[:k], glob[:k]
    gidx = np.concatenate(gi_l) if gi_l else np.zeros(0, np.int32)
    pos = pd.DataFrame({"uid": games["uid"].to_numpy()[gidx], "row": games["row"].to_numpy()[gidx],
                        "own_idx": np.concatenate(own_l) if own_l else np.zeros(0, np.int16),
                        "ply": np.concatenate(ply_l) if ply_l else np.zeros(0, np.int16),
                        "loc": np.concatenate(loc_l) if loc_l else np.zeros(0, np.int16),
                        "color": games["color"].to_numpy()[gidx].astype(np.int8)})
    gm = games.drop(columns=["sgf"]).copy()
    gm["status"] = [s for s, _ in status]
    gm["n_pos"] = [n for _, n in status]
    gm["hd"] = np.array([hd_of(x) for x in games["sgf"]], np.int8)
    reasons = {}
    for s, _ in status:
        if s != "ok":
            reasons[s] = reasons.get(s, 0) + 1
    tmp.mkdir(parents=True)
    meta = {"sid": sid, "n_games": int(len(games)), "n_games_encoded": int(sum(s == "ok" for s, _ in status)),
            "n_skipped": int(sum(reasons.values())), "skip_reasons": reasons,
            "n_hd_ge2_games": int((gm["hd"] >= 2).sum()), "n_hd_ge2_positions": int(gm.loc[gm["hd"] >= 2, "n_pos"].sum()), "n_pos": int(k), "n_channels": N_CH, "n_global": N_GLOB,
            "bits_width": BITS_W, "games_sha256": sha256_path(gpath), "profile": PROFILE, "secs": round(time.time() - t0, 1),
            "pid": os.getpid(), "finished": time.strftime("%F %T")}
    np.save(tmp / "bits.npy", bits)
    np.save(tmp / "glob.npy", glob)
    gm.to_parquet(tmp / "games.parquet", index=False)
    pos.to_parquet(tmp / "pos.parquet", index=False, compression="zstd")
    (tmp / "meta.json").write_text(json.dumps(meta))
    os.rename(tmp, final)
    write_json_atomic(done, meta)                                       # `.done` last = shard complete
    return meta


def _worker(args):
    root, cfg, sid = args
    return encode_shard(Path(root), cfg, sid)


# ----------------------------------------------------------------------------------------------- driver (school)
def scan_done(root: Path, n_sh: int) -> dict:
    tot = {"done_shards": 0, "done_games": 0, "done_positions": 0, "skipped_games": 0, "hd_ge2_games": 0, "hd_ge2_positions": 0, "skip_reasons": {}, "last_shard_ts": 0.0}
    for f in sorted((root / "shards").glob("*.done")):
        sid = int(f.stem)
        if not (root / "shards" / f"{sid:05d}").is_dir():
            f.unlink()                                                  # `.done` without data: redo
            continue
        m = json.loads(f.read_text())
        tot["done_shards"] += 1
        tot["done_games"] += m["n_games"]
        tot["done_positions"] += m["n_pos"]
        tot["skipped_games"] += m["n_skipped"]
        tot["hd_ge2_games"] += m.get("n_hd_ge2_games", 0)
        tot["hd_ge2_positions"] += m.get("n_hd_ge2_positions", 0)
        for k, v in m["skip_reasons"].items():
            tot["skip_reasons"][k] = tot["skip_reasons"].get(k, 0) + v
        tot["last_shard_ts"] = max(tot["last_shard_ts"], f.stat().st_mtime)
    return tot


def done_set(root: Path) -> set:
    return {int(f.stem) for f in (root / "shards").glob("*.done") if (root / "shards" / f.stem).is_dir()}


def write_status(root: Path, plan: dict, state: str, tot: dict, t_start: float, extra: dict | None = None) -> None:
    st = {"state": state, "n_shards": plan["n_shards"], "n_games_total": plan["n_games"], **tot,
          "ts": time.strftime("%F %T"), "ts_epoch": time.time(), "started_epoch": t_start, "driver_pid": os.getpid(), "workers": int(os.environ.get("E89K_WORKERS", "4")),
          "positions_plan": plan.get("n_positions_plan")}
    st.update(extra or {})
    write_json_atomic(root / "status.json", st)


def cmd_run(a) -> None:
    import multiprocessing as mp
    root = Path(a.root)
    (root / "shards").mkdir(exist_ok=True)
    (root / "hb").mkdir(exist_ok=True)
    plan = json.load(open(root / "plan.json"))
    cfg = {"katago_src": a.katago_src, "model_config": a.model_config}
    os.environ["E89K_WORKERS"] = str(a.workers)
    (root / "driver.pid").write_text(str(os.getpid()))
    t_start = time.time()
    for p in (root / "shards").glob(".tmp_*"):
        shutil.rmtree(p)
    done = done_set(root)
    todo = [s for s in range(plan["n_shards"]) if s not in done]
    tot = scan_done(root, plan["n_shards"])
    print(f"{time.strftime('%F %T')} driver start pid={os.getpid()} done={len(done)} todo={len(todo)} workers={a.workers}", flush=True)
    write_status(root, plan, "RUNNING" if todo else "DONE", tot, t_start)
    if todo:
        with mp.get_context("fork").Pool(a.workers) as pool:
            for m in pool.imap_unordered(_worker, [(str(root), cfg, s) for s in todo], chunksize=1):
                tot = scan_done(root, plan["n_shards"])
                write_status(root, plan, "RUNNING", tot, t_start)
                print(f"{time.strftime('%F %T')} shard {m['sid']:05d} done n_pos={m['n_pos']} skipped={m['n_skipped']} {m['secs']}s "
                      f"-> {tot['done_shards']}/{plan['n_shards']}", flush=True)
    tot = scan_done(root, plan["n_shards"])
    state = "DONE" if tot["done_shards"] == plan["n_shards"] else "INCOMPLETE"
    write_status(root, plan, state, tot, t_start)
    print(f"{time.strftime('%F %T')} driver end state={state}", flush=True)
    sys.exit(0 if state == "DONE" else 3)


# ----------------------------------------------------------------------------------------------- helpers: local check, peek
def cmd_encode_games(a) -> None:
    """Encode all own-move positions of the given uids (read from a games parquet) -> npz; `--impl e13` uses the E13 class."""
    games = pd.concat([pd.read_parquet(f) for f in a.games])
    games = games[games["uid"].isin(a.uids.split(","))]
    if a.impl == "e13":
        from scripts.experiments.e13 import encode as E
        enc = E.KataEncoder(a.katago_src, json.load(open(a.model_config)))
    else:
        enc = KataEncoder(a.katago_src, json.load(open(a.model_config)))
    out = {}
    for r in games.itertuples(index=False):
        pg = parse_sgf(r.sgf)
        n_own = int((pg.moves[:, 0] == r.color).sum())
        b, g, plies = enc.encode_game(pg.moves, int(r.color), np.arange(n_own))
        out[r.uid + "|bits"] = np.packbits(b.reshape(len(b), -1), axis=1)
        out[r.uid + "|glob"] = g.astype(np.float16)
        out[r.uid + "|ply"] = plies.astype(np.int16)
        out[r.uid + "|loc"] = move_loc(pg.moves, plies)
    np.savez(a.out, **out)


def cmd_show(a) -> None:
    d = Path(a.root) / "shards" / f"{int(a.sid):05d}"
    for f in ("bits.npy", "glob.npy"):
        x = np.load(d / f, mmap_mode="r")
        print(f, x.dtype, x.shape)
    pos = pd.read_parquet(d / "pos.parquet")
    print(pos.dtypes.to_dict(), len(pos))
    print(pos.head(3).to_string())
    print(json.dumps(json.loads((d / "meta.json").read_text())))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--aux", default=str(ROOT / ".scratch/e29/bundle/aux.npz"))
    p.add_argument("--split", default=str(ROOT / "data/splits/split.parquet"))
    p.add_argument("--csv-dir", default=str(ROOT / "data/training_v1.1/training"))
    p.add_argument("--old-csv-dir", default=str(ROOT / "data/training"))
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_plan)
    p = sp.add_parser("run")
    p.add_argument("--root", required=True)
    p.add_argument("--katago-src", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(f=cmd_run)
    p = sp.add_parser("encode-games")
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--uids", required=True)
    p.add_argument("--katago-src", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--impl", choices=["e13", "k"], default="k")
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_encode_games)
    p = sp.add_parser("show")
    p.add_argument("--root", required=True)
    p.add_argument("--sid", required=True)
    p.set_defaults(f=cmd_show)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
