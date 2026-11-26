"""8 board symmetries, 3x3 ring canonicalisation and opening canonicalisation (SDD 5.5)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .constants import BOARD, OPEN_PLIES

M = BOARD - 1
PASS_CODE = BOARD * BOARD  # 361

SYMS = (
    lambda r, c: (r, c),
    lambda r, c: (r, M - c),
    lambda r, c: (M - r, c),
    lambda r, c: (M - r, M - c),
    lambda r, c: (c, r),
    lambda r, c: (c, M - r),
    lambda r, c: (M - c, r),
    lambda r, c: (M - c, M - r),
)


def apply_sym(k: int, row: int, col: int) -> tuple[int, int]:
    return SYMS[k](row, col)


def _build_index_maps() -> np.ndarray:
    """maps[k][r*19+c] = flat index of the image of (r,c)."""
    maps = np.zeros((8, BOARD * BOARD), dtype=np.int64)
    for k in range(8):
        for r in range(BOARD):
            for c in range(BOARD):
                rr, cc = SYMS[k](r, c)
                maps[k, r * BOARD + c] = rr * BOARD + cc
    return maps


_MAPS = _build_index_maps()


def apply_sym_grid(k: int, grid: np.ndarray) -> np.ndarray:
    """out[apply_sym(k,r,c)] = grid[r,c]."""
    out = np.empty_like(grid)
    out.reshape(-1)[_MAPS[k]] = grid.reshape(-1)
    return out


def canonical_open_sym(plies: Sequence[tuple[int, int]]) -> int:
    """Smallest-k symmetry giving the lexicographically smallest sequence over the first OPEN_PLIES plies.

    Each ply is (row, col); a pass is (-1, -1) (or any negative row) and is encoded as 361.  Colour is ignored.
    """
    pts = [PASS_CODE if r < 0 else r * BOARD + c for r, c in list(plies)[:OPEN_PLIES]]
    arr = np.array(pts, dtype=np.int64)
    best_k, best = 0, None
    for k in range(8):
        img = np.where(arr == PASS_CODE, PASS_CODE, _MAPS[k][np.minimum(arr, PASS_CODE - 1)]).tolist()
        if best is None or img < best:
            best, best_k = img, k
    return best_k


def ring_code(ring: Sequence[int]) -> int:
    """Ring order N,NE,E,SE,S,SW,W,NW (clockwise from north); values 0 empty/1 own/2 enemy/3 off-board.
    code = sum ring[i] * 4**(7-i)."""
    code = 0
    for v in ring:
        code = code * 4 + int(v)
    return code


def _ring_images(ring: Sequence[int]):
    for k in range(4):
        yield [ring[(i + 2 * k) % 8] for i in range(8)]
        yield [ring[(-i + 2 * k) % 8] for i in range(8)]


def canon_ring(ring: Sequence[int]) -> int:
    return min(ring_code(r) for r in _ring_images(ring))


def _build_vocab() -> np.ndarray:
    codes = np.arange(65536, dtype=np.int64)
    digits = np.stack([(codes >> (2 * (7 - i))) & 3 for i in range(8)], axis=1)   # [65536,8]
    best = np.full(65536, 1 << 30, dtype=np.int64)
    for k in range(4):
        for mirror in (False, True):
            idx = [((-i if mirror else i) + 2 * k) % 8 for i in range(8)]
            img = digits[:, idx]
            c = np.zeros(65536, dtype=np.int64)
            for j in range(8):
                c = c * 4 + img[:, j]
            best = np.minimum(best, c)
    return np.unique(best).astype(np.int32)


SHAPE_VOCAB: np.ndarray = _build_vocab()
SHAPE_INDEX: dict[int, int] = {int(c): i for i, c in enumerate(SHAPE_VOCAB)}
