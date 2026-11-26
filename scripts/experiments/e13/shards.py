"""Position shards on disk.  numpy / pandas only.

Layout:  <root>/<split>/games/<sid>.parquet      (prepare: uid, player_id, rank_id, color, sgf)
         <root>/<split>/shards/<sid>/{bits.npy, glob.npy, color.npy, label.npy, pos.parquet}   (encode)
A shard directory is written as `.tmp_<sid>` and renamed when complete, so a killed worker never leaves a half shard and
re-running skips every finished shard (resume).  bits = packbits of the binary spatial planes, flattened (C*19*19).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_path(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def shard_dir(root: Path, split: str, sid: int) -> Path:
    return Path(root) / split / "shards" / f"{sid:05d}"


def done_shards(root: Path, split: str) -> list[int]:
    d = Path(root) / split / "shards"
    if not d.exists():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit() and (p / "pos.parquet").exists())


def read_meta(root: Path, split: str, sid: int) -> dict:
    f = shard_dir(root, split, sid) / "meta.json"
    if not f.exists():
        raise RuntimeError(f"{split}/{sid:05d}: no meta.json (shard written by an older format); delete the shard directory and run encode again")
    return json.loads(f.read_text())


def write_shard(root: Path, split: str, sid: int, pos: pd.DataFrame, bits: np.ndarray, glob: np.ndarray, *, n_channels: int,
                games_sha256: str) -> None:
    final = shard_dir(root, split, sid)
    tmp = final.with_name(f".tmp_{sid:05d}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n = len(pos)
    assert bits.shape[0] == glob.shape[0] == n and bits.dtype == np.uint8
    assert bits.shape[1] == (n_channels * 19 * 19 + 7) // 8, f"bits width {bits.shape[1]} does not match {n_channels} channels"
    (tmp / "meta.json").write_text(json.dumps({"n_channels": int(n_channels), "n_global": int(glob.shape[1]), "bits_width": int(bits.shape[1]),
                                               "n_pos": int(n), "games_sha256": games_sha256}))
    np.save(tmp / "bits.npy", bits)
    np.save(tmp / "glob.npy", glob.astype(np.float32))
    np.save(tmp / "color.npy", pos["color"].to_numpy(np.uint8))
    np.save(tmp / "label.npy", pos["label"].to_numpy(np.int8))
    pos.to_parquet(tmp / "pos.parquet", index=False)          # written last: its presence marks completeness
    if final.exists():
        shutil.rmtree(final)
    os.rename(tmp, final)


def unpack_bits(bits: np.ndarray, n_channels: int, size: int = 19) -> np.ndarray:
    """(n, ceil(C*size*size/8)) uint8 -> (n, C, size, size) uint8 0/1."""
    flat = np.unpackbits(bits, axis=1, count=n_channels * size * size)
    return flat.reshape(len(bits), n_channels, size, size)


class ShardSet:
    """Read-only view over the finished shards of one split (memory-mapped)."""

    def __init__(self, root: Path, split: str, n_channels: int, expected_shards: int | None = None, verify_games: bool = True):
        self.root, self.split, self.n_channels = Path(root), split, n_channels
        sids = done_shards(root, split)
        if expected_shards is not None and len(sids) != expected_shards:
            raise RuntimeError(f"{split}: {len(sids)} of {expected_shards} shards finished; run encode again")
        self.sids = sids
        self._bits, self._glob, self._color, self._label, lens = [], [], [], [], []
        for s in sids:
            d = shard_dir(root, split, s)
            self._check_meta(s, n_channels, verify_games)
            self._bits.append(np.load(d / "bits.npy", mmap_mode="r"))
            self._glob.append(np.load(d / "glob.npy", mmap_mode="r"))
            self._color.append(np.load(d / "color.npy"))
            self._label.append(np.load(d / "label.npy"))
            lens.append(len(self._label[-1]))
        self.offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.n = int(self.offsets[-1])

    def _check_meta(self, sid: int, n_channels: int, verify_games: bool) -> None:
        m = read_meta(self.root, self.split, sid)
        if m["n_channels"] != n_channels:
            raise RuntimeError(f"{self.split}/{sid:05d}: shard has {m['n_channels']} channels but {n_channels} were requested "
                               f"(wrong --n-channels or a shard from another encoder)")
        gf = Path(self.root) / self.split / "games" / f"{sid:05d}.parquet"
        if verify_games and gf.exists() and sha256_path(gf) != m["games_sha256"]:
            raise RuntimeError(f"{self.split}/{sid:05d}: games list changed after this shard was encoded (sha256 mismatch); "
                               f"refusing to use it.  Delete {shard_dir(self.root, self.split, sid)} and run encode again")

    def shard_positions(self, i: int, columns=None) -> pd.DataFrame:
        return pd.read_parquet(shard_dir(self.root, self.split, self.sids[i]) / "pos.parquet", columns=columns)

    def uids(self) -> np.ndarray:
        return np.concatenate([self.shard_positions(i, ["uid"])["uid"].to_numpy() for i in range(len(self.sids))]) if self.sids else np.zeros(0, object)

    def labels(self) -> np.ndarray:
        return np.concatenate(self._label) if self._label else np.zeros(0, np.int8)

    def positions(self) -> pd.DataFrame:
        """pos.parquet of all shards in order (uid, own_idx, ply, color, label); row i = global index i."""
        return pd.concat([pd.read_parquet(shard_dir(self.root, self.split, s) / "pos.parquet") for s in self.sids], ignore_index=True)

    def get(self, idx: np.ndarray):
        """Gather global indices (any order) -> bits (n,B) uint8, glob (n,G) f32, color (n,) u8, label (n,) i8."""
        idx = np.asarray(idx, np.int64)
        sh = np.searchsorted(self.offsets, idx, side="right") - 1
        off = idx - self.offsets[sh]
        B, G = self._bits[0].shape[1], self._glob[0].shape[1]
        bits, glob = np.empty((len(idx), B), np.uint8), np.empty((len(idx), G), np.float32)
        color, label = np.empty(len(idx), np.uint8), np.empty(len(idx), np.int8)
        for s in np.unique(sh):
            m = sh == s
            o = off[m]
            bits[m], glob[m] = self._bits[s][o], self._glob[s][o]
            color[m], label[m] = self._color[s][o], self._label[s][o]
        return bits, glob, color, label
