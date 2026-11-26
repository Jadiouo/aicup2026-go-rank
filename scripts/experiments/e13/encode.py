"""Pre-encode SGF positions into KataGo inputs (shards).  Resumable and multi-process.

Stages (CLI in __main__):  prepare  -> games/<sid>.parquet with SGF text of the chosen games (H never included)
                           encode   -> shards/<sid>/...   (Pool of workers, finished shards are skipped)
The KataGo python package is imported lazily and only inside the worker (`--katago-src`); tests use FakeEncoder.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gogame.constants import SEED
from gogame.seeds import derive_seed
from gogame.sgf import parse_sgf

from . import lib, shards

PASS_LOC_NAME = "PASS"


# ------------------------------------------------------------------------------------------------ SGF text for the chosen games
def read_sgf_for_uids(csv_dir: Path, uids: set, chunksize: int = 50_000) -> pd.DataFrame:
    """uid -> sgf from data/training/train_<rank>.csv (uid = '<rank>_<game_id>').  Rows of other players are dropped at once."""
    parts = []
    for f in sorted(Path(csv_dir).glob("train_*.csv")):
        for ch in pd.read_csv(f, usecols=["game_id", "rank", "sgf_content"], chunksize=chunksize):
            ch["uid"] = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str)
            ch = ch[ch["uid"].isin(uids)]
            if len(ch):
                parts.append(ch[["uid", "sgf_content"]].rename(columns={"sgf_content": "sgf"}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["uid", "sgf"])
    missing = uids - set(out["uid"])
    if missing:
        raise RuntimeError(f"{len(missing)} uids have no SGF, e.g. {sorted(missing)[:3]}")
    return out.drop_duplicates("uid")


def prepare_split(root: Path, split: str, games: pd.DataFrame, allowed: set, forbidden: set, csv_dir: Path = lib.TRAIN_CSV_DIR,
                  sgf_loader=read_sgf_for_uids) -> dict:
    """Write games/<sid>.parquet.  `games` has uid, player_id, rank_id, color.  Leak checks run before any SGF is read."""
    lib.assert_games_clean(games, allowed, forbidden, split)
    games = games.sort_values("uid").reset_index(drop=True)
    sg = sgf_loader(csv_dir, set(games["uid"]))
    g = games.merge(sg, on="uid", how="left", validate="1:1")
    assert g["sgf"].notna().all()
    d = Path(root) / split / "games"
    d.mkdir(parents=True, exist_ok=True)
    n_sh = (len(g) + lib.GAMES_PER_SHARD - 1) // lib.GAMES_PER_SHARD
    for sid in range(n_sh):
        g.iloc[sid * lib.GAMES_PER_SHARD:(sid + 1) * lib.GAMES_PER_SHARD].to_parquet(d / f"{sid:05d}.parquet", index=False)
    info = {"split": split, "n_games": int(len(g)), "n_shards": n_sh, "n_players": int(g["player_id"].nunique())}
    lib.write_json(Path(root) / split / "prepare.json", info)
    return info


# ------------------------------------------------------------------------------------------------ encoders
class FakeEncoder:
    """Deterministic stand-in with the same interface as KataEncoder (tests, CPU dry runs).  Planes are a few binary
    functions of the (row, col, colour) board so that models can learn from them; NOT KataGo features."""
    n_channels, n_global = 4, 3

    def __init__(self, *a, **k):
        self.meta_rows = {0: np.zeros(8, np.float32), 1: np.ones(8, np.float32)}

    def encode_game(self, moves: np.ndarray, color: int, own_idx: np.ndarray):
        own_plies = [i for i, m in enumerate(moves.tolist()) if m[0] == color]
        out_b, out_g, plies = [], [], []
        for j in own_idx:
            ply = own_plies[int(j)]
            b = np.zeros((self.n_channels, 19, 19), np.uint8)
            for k, (c, r, cc) in enumerate(moves[:ply].tolist()):
                if r >= 0:
                    b[0 if c == color else 1, r, cc] = 1
            b[2, :, :] = 1
            b[3, ply % 19, :] = 1
            out_b.append(b)
            out_g.append(np.array([ply / 200.0, color, 1.0], np.float32))
            plies.append(ply)
        return np.stack(out_b) if out_b else np.zeros((0, self.n_channels, 19, 19), np.uint8), \
            np.stack(out_g) if out_g else np.zeros((0, self.n_global), np.float32), np.array(plies, np.int32)


_RANK_INV = {**{f"{i}d": 10 - i for i in range(1, 10)}, **{f"{i}k": 9 + i for i in range(1, 21)}}


class KataEncoder:
    """KataGo `Features` + `SGFMetadata` (python package of the KataGo repo, v1.18.2 verified in the smoke).
    Mirrors experiments/E13_humansl_ft/smoke/scripts/compare.py: same GameState/Features calls, metadata of
    makeBasicRankProfile (byo-yomi 1200s/30s x5, KGS, 2020-03-01, ratedness unknown), Japanese rules, default komi."""

    def __init__(self, katago_src: str, model_config: dict, profile: str = lib.PROFILE):
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

    def encode_game(self, moves: np.ndarray, color: int, own_idx: np.ndarray):
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
            loc = B.PASS_LOC if r < 0 else gs.board.loc(int(cc), int(r))         # x = column, y = row (same as compare.py)
            gs.play(B.BLACK if c == 0 else B.WHITE, loc)
            if ply >= max(target):
                break
        return (np.stack(out_b), np.stack(out_g), np.array(plies, np.int32)) if out_b else \
            (np.zeros((0, 1, 19, 19), np.uint8), np.zeros((0, 1), np.float32), np.zeros(0, np.int32))


def make_encoder(kind: str, **kw):
    if kind == "fake":
        return FakeEncoder()
    if kind == "katago":
        return KataEncoder(kw["katago_src"], kw["model_config"], kw.get("profile", lib.PROFILE))
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ one shard
def _skip_reason(pg) -> str | None:
    f = pg.flags
    if f.has_setup or f.bad_size or f.nonalt or f.bad_coord or f.first_not_black:
        return "flags"
    return None


def encode_shard(root: Path, split: str, sid: int, encoder, mode: str, seed_tag: str = "train") -> dict:
    """mode 'train': random 16 own-move positions per game (named seed per uid);  'eval': ALL own-move positions in
    [POS_LO, POS_HI) (<= 58 per game); inference picks 16 evenly spaced or all of them (lib.select_positions)."""
    gpath = Path(root) / split / "games" / f"{sid:05d}.parquet"
    games_sha = shards.sha256_path(gpath)
    games = pd.read_parquet(gpath)
    bits_l, glob_l, rows, skipped, n_ch = [], [], [], 0, None
    for r in games.itertuples(index=False):
        pg = parse_sgf(r.sgf)
        if _skip_reason(pg):
            skipped += 1
            continue
        moves = pg.moves
        n_own = int((moves[:, 0] == r.color).sum())
        if mode == "train":
            idx = lib.own_move_idx_train(n_own, np.random.default_rng(derive_seed(SEED, "e13", "plies", seed_tag, r.uid)))
        else:
            idx = lib.own_move_idx_all(n_own)
        if len(idx) == 0:
            skipped += 1
            continue
        b, g, plies = encoder.encode_game(moves, int(r.color), idx)
        n_ch = int(b.shape[1]) if n_ch is None else n_ch
        assert b.shape[1] == n_ch
        bits_l.append(np.packbits(b.reshape(len(b), -1), axis=1))
        glob_l.append(g)
        for j, ply in zip(idx.tolist(), plies.tolist()):
            rows.append((r.uid, r.player_id, int(r.color), int(r.rank_id), int(j), int(ply)))
    pos = pd.DataFrame(rows, columns=["uid", "player_id", "color", "label", "own_idx", "ply"])
    if not len(pos):
        raise RuntimeError(f"{split}/{sid}: no positions")
    shards.write_shard(root, split, sid, pos, np.concatenate(bits_l), np.concatenate(glob_l), n_channels=n_ch, games_sha256=games_sha)
    return {"sid": sid, "n_pos": len(pos), "n_games": int(games.shape[0]), "skipped": skipped}


def limit_threads() -> None:
    """One BLAS/OpenMP thread per encode process: with N workers on N cores, intra-op threads only oversubscribe the CPU."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def default_workers() -> int:
    """Encode worker count: env E13_ENCODE_WORKERS, else 4 (PLAN 10: school 6, remote-vm nproc-2 but at most 4)."""
    return max(1, int(os.environ.get("E13_ENCODE_WORKERS", "4")))


def _worker(args):
    root, split, sids, kind, kw, mode = args
    limit_threads()                                  # before the encoder (and thereby torch/numpy BLAS) is imported
    enc = make_encoder(kind, **kw)
    if kind == "katago" and "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)      # the KataGo package imports torch inside make_encoder
    return [encode_shard(root, split, s, enc, mode) for s in sids]


def check_shards_current(root: Path, split: str, sids: list[int]) -> None:
    """A finished shard whose games list changed since it was encoded is refused (never silently re-used or overwritten)."""
    for s in sids:
        m = shards.read_meta(root, split, s)
        if m["games_sha256"] != shards.sha256_path(Path(root) / split / "games" / f"{s:05d}.parquet"):
            raise RuntimeError(f"{split}/{s:05d}: games list changed since the shard was encoded (sha256 mismatch); "
                               f"delete {shards.shard_dir(root, split, s)} (or the whole shards dir) and run encode again")


def encode_split(root: Path, split: str, mode: str, kind: str, workers: int | None = None, kw: dict | None = None) -> dict:
    """Encode all unfinished shards.  Resumable: finished shards are skipped, interrupted ones are redone."""
    root = Path(root)
    workers = default_workers() if workers is None else workers
    info = __import__("json").loads((root / split / "prepare.json").read_text())
    done = shards.done_shards(root, split)
    check_shards_current(root, split, done)
    todo = [s for s in range(info["n_shards"]) if s not in set(done)]
    kw = kw or {}
    res = []
    limit_threads()
    if workers <= 1 or len(todo) <= 1:
        res = _worker((root, split, todo, kind, kw, mode))
    else:
        import multiprocessing as mp
        chunks = [todo[i::workers] for i in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(_worker, [(root, split, c, kind, kw, mode) for c in chunks if c]):
                res += r
    return {"split": split, "encoded_now": len(res), "n_pos_now": int(sum(r["n_pos"] for r in res)),
            "skipped_games": int(sum(r["skipped"] for r in res)), "n_shards_total": info["n_shards"]}
