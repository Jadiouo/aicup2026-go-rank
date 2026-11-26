"""L0 hand-crafted features (SDD 5.7 / 9).  Pure functions, no randomness.

Only `board.iter_own_positions` (+ `replay_summary` for trailing enemy captures) and `symmetry` are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sps

from .board import iter_own_positions, replay_summary
from .constants import BIGRAM_BUCKETS, BOARD, OPEN_OWN_MOVES, OPEN_PLIES
from .io import GameRecord
from .sgf import ParsedGame, normalize_sgf, parse_sgf
from .symmetry import SHAPE_INDEX, SHAPE_VOCAB, _MAPS, canonical_open_sym

DENSE_NAMES: tuple[str, ...] = (
    "n_plies", "n_own", "n_own_pass", "cap_own", "cap_lost",
    "line_1", "line_2", "line_3", "line_4", "line_c",
    "lineo_1", "lineo_2", "lineo_3", "lineo_4", "lineo_c",
    "iso_frac", "contact_opp_frac",
    "dopp_1", "dopp_2", "dopp_3", "dopp_4_6", "dopp_7p", "dopp_na", "dopp_mean",
    "dself_1", "dself_2", "dself_3", "dself_4_6", "dself_7p", "dself_na", "dself_mean",
)
SPARSE_SIZE = 10125
N_OPEN_POS = BOARD * BOARD                       # 361
OFF_BIGRAM = N_OPEN_POS                          # 361
OFF_SHAPE = OFF_BIGRAM + BIGRAM_BUCKETS          # 1385
LINEO_MOVES = 30
_N = BOARD
_MULT = 2654435761


@dataclass(frozen=True)
class L0Schema:
    dense_names: tuple[str, ...]
    sparse_size: int
    groups: dict[str, tuple[int, int]]


@dataclass
class L0Game:
    dense: np.ndarray        # float32[31]
    sp_idx: np.ndarray       # int32[K], strictly increasing
    sp_val: np.ndarray       # float32[K]


@dataclass
class L0Batch:
    dense: np.ndarray        # float32[G,31]
    sparse: sps.csr_matrix   # float32[G,10125]
    uids: list[str]


def l0_schema() -> L0Schema:
    return L0Schema(DENSE_NAMES, SPARSE_SIZE,
                    {"open_pos": (0, OFF_BIGRAM), "open_bigram": (OFF_BIGRAM, OFF_SHAPE),
                     "shape": (OFF_SHAPE, SPARSE_SIZE)})


# ---------------------------------------------------------------- lookup tables ------------------------------------
def _build_tables():
    codes = np.arange(65536, dtype=np.int64)
    digits = np.stack([(codes >> (2 * (7 - i))) & 3 for i in range(8)], axis=1)
    best = np.full(65536, 1 << 30, dtype=np.int64)
    for k in range(4):
        for mirror in (False, True):
            idx = [((-i if mirror else i) + 2 * k) % 8 for i in range(8)]
            img = digits[:, idx]
            c = np.zeros(65536, dtype=np.int64)
            for j in range(8):
                c = c * 4 + img[:, j]
            best = np.minimum(best, c)
    shape_idx = np.searchsorted(SHAPE_VOCAB, best).astype(np.int32)
    has_own_or_opp = ((digits == 1) | (digits == 2)).any(axis=1)
    has_opp = (digits == 2).any(axis=1)
    return shape_idx, ~has_own_or_opp, has_opp


_SHAPE_OF_CODE, _ISO_OF_CODE, _CONTACT_OF_CODE = _build_tables()
_WEIGHTS = np.array([4 ** (7 - i) for i in range(8)], dtype=np.int64)

# padded 21x21 buffer; ring point offsets (N,NE,E,SE,S,SW,W,NW) as (dr,dc)
_RING_OFF = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
_RING_IDX = np.zeros((_N * _N, 8), dtype=np.int64)
for _r in range(_N):
    for _c in range(_N):
        for _i, (_dr, _dc) in enumerate(_RING_OFF):
            _RING_IDX[_r * _N + _c, _i] = (_r + 1 + _dr) * (_N + 2) + (_c + 1 + _dc)


def _line_of(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.minimum(np.minimum(r, c), np.minimum(18 - r, 18 - c)) + 1


def _line_fracs(lines: np.ndarray) -> list[float]:
    n = lines.size
    if n == 0:
        return [0.0] * 5
    return [float((lines == k).sum()) / n for k in (1, 2, 3, 4)] + [float((lines >= 5).sum()) / n]


def _dist_fracs(d: np.ndarray, n_np: int) -> list[float]:
    """[b1,b2,b3,b4_6,b7p, na, mean] from the defined distances d; na denominator is n_np."""
    n_def = d.size
    if n_np == 0:
        return [0.0] * 6 + [float("nan")]
    if n_def == 0:
        return [0.0] * 5 + [(n_np - n_def) / n_np, float("nan")]
    b = [float((d <= 1).sum()), float((d == 2).sum()), float((d == 3).sum()),
         float(((d >= 4) & (d <= 6)).sum()), float((d >= 7).sum())]
    return [x / n_def for x in b] + [(n_np - n_def) / n_np, float(d.mean())]


def compute_l0(game: ParsedGame, annot_color: int) -> L0Game:
    moves = game.moves
    T = len(moves)
    positions = list(iter_own_positions(game, annot_color))
    n_own = len(positions)

    cap_own = 0
    lost = 0
    buf = np.zeros((_N + 2, _N + 2), dtype=np.int8)
    buf[0, :] = buf[-1, :] = 3
    buf[:, 0] = buf[:, -1] = 3
    flat = buf.reshape(-1)
    own_rc: list[tuple[int, int]] = []
    own_ply: list[int] = []
    d_opp: list[int] = []
    d_self: list[int] = []
    codes: list[int] = []
    for p in positions:
        cap_own += p.captured
        lost += p.lost_since_prev_own
        r, c = p.move
        if r < 0:
            continue
        own_rc.append((r, c))
        own_ply.append(p.ply)
        if p.prev_opp is not None:
            d_opp.append(max(abs(r - p.prev_opp[0]), abs(c - p.prev_opp[1])))
        if p.prev_own is not None:
            d_self.append(max(abs(r - p.prev_own[0]), abs(c - p.prev_own[1])))
        buf[1:_N + 1, 1:_N + 1] = p.grid_own
        codes.append(int(flat[_RING_IDX[r * _N + c]] @ _WEIGHTS))
    n_np = len(own_rc)
    n_pass = n_own - n_np
    # enemy captures made after the annotated side's last move are invisible to the iterator
    if T and int(moves[-1][0]) != annot_color:
        last_own_ply = positions[-1].ply if positions else -1
        caps = replay_summary(game).captures
        lost += int(caps[last_own_ply + 1:].sum())

    dense = np.empty(31, dtype=np.float32)
    dense[0:5] = (T, n_own, n_pass, cap_own, lost)
    if n_np:
        arr = np.array(own_rc, dtype=np.int64)
        lines = _line_of(arr[:, 0], arr[:, 1])
    else:
        lines = np.zeros(0, dtype=np.int64)
    dense[5:10] = _line_fracs(lines)
    dense[10:15] = _line_fracs(lines[:LINEO_MOVES])
    if n_np:
        cd = np.array(codes, dtype=np.int64)
        dense[15] = float(_ISO_OF_CODE[cd].sum()) / n_np
        dense[16] = float(_CONTACT_OF_CODE[cd].sum()) / n_np
    else:
        cd = np.zeros(0, dtype=np.int64)
        dense[15] = dense[16] = 0.0
    dense[17:24] = _dist_fracs(np.array(d_opp, dtype=np.int64), n_np)
    dense[24:31] = _dist_fracs(np.array(d_self, dtype=np.int64), n_np)

    # ---- sparse
    entries: dict[int, float] = {}
    if n_np:
        plies = [(int(r), int(c)) for _, r, c in moves[:OPEN_PLIES].tolist()]
        tau = canonical_open_sym(plies)
        mp = _MAPS[tau]

        def pos(r: int, c: int) -> int:
            return int(mp[r * _N + c])

        # open_pos: first 12 non-pass own moves
        for r, c in own_rc[:OPEN_OWN_MOVES]:
            entries[pos(r, c)] = 1.0
        # open_bigram: ply t in 1..OPEN_PLIES-1 own non-pass, ply t-1 non-pass
        own_set = set(own_ply)
        ml = moves.tolist()
        for t in range(1, min(T, OPEN_PLIES)):
            if t in own_set and ml[t - 1][1] >= 0:
                key = pos(ml[t - 1][1], ml[t - 1][2]) * 361 + pos(ml[t][1], ml[t][2])
                b = OFF_BIGRAM + (((key * _MULT) & 0xFFFFFFFF) >> 22)
                entries[b] = entries.get(b, 0.0) + 1.0
        # shape
        cnt = np.bincount(_SHAPE_OF_CODE[cd], minlength=SHAPE_VOCAB.size)
        for i in np.nonzero(cnt)[0].tolist():
            entries[OFF_SHAPE + i] = float(cnt[i]) / n_np
    idx = np.array(sorted(entries), dtype=np.int32)
    val = np.array([entries[i] for i in idx.tolist()], dtype=np.float32)
    return L0Game(dense, idx, val)


# ---------------------------------------------------------------- batches ------------------------------------------
def _work(args: tuple[str, int]) -> L0Game:
    sgf, color = args
    ng = normalize_sgf(sgf, color)      # current format: ng.parsed == parse_sgf(sgf), eff_color == color; HD>=2: stones + swap
    return compute_l0(ng.parsed, ng.eff_color)


def compute_l0_many(records: Iterable[GameRecord], n_jobs: int = 1) -> L0Batch:
    recs = list(records)
    args = [(r.sgf, r.color) for r in recs]
    if n_jobs > 1 and len(args) > 1:
        with Pool(n_jobs) as pool:
            games = pool.map(_work, args, chunksize=max(1, min(256, len(args) // (4 * n_jobs) or 1)))
    else:
        games = [_work(a) for a in args]
    G = len(games)
    dense = np.stack([g.dense for g in games]) if G else np.zeros((0, 31), dtype=np.float32)
    indptr = np.zeros(G + 1, dtype=np.int64)
    if G:
        indptr[1:] = np.cumsum([g.sp_idx.size for g in games])
        indices = np.concatenate([g.sp_idx for g in games]).astype(np.int32)
        data = np.concatenate([g.sp_val for g in games]).astype(np.float32)
    else:
        indices = np.zeros(0, dtype=np.int32)
        data = np.zeros(0, dtype=np.float32)
    sp = sps.csr_matrix((data, indices, indptr), shape=(G, SPARSE_SIZE))
    return L0Batch(dense, sp, [r.uid for r in recs])
