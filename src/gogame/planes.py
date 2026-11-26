"""Shared board encoder (SDD 5.6).  Layers (viewpoint = the annotated side that is about to move):
0 own stones, 1 enemy stones, 2 empty, 3 previous enemy move, 4 previous own (non-pass) move, 5 ko point,
6 liberties of own groups = clip(liberties,1,4) on own stones (uint8 1..4; to_float() rescales by 1/4),
7 colour flag (all 1 if the annotated side is black, else all 0).  Optional layer 8: one-hot of the actual move.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .board import OwnPosition, iter_own_positions, liberty_map
from .constants import BOARD, PASS_INDEX
from .sgf import ParsedGame
from .symmetry import apply_sym, apply_sym_grid

N_PLANES: int = 8


@dataclass(frozen=True)
class PlaneSampling:
    mode: Literal["all", "first_k", "even", "random"]
    k: int = 64
    seed: int = 0


@dataclass
class PlaneBatch:
    planes: np.ndarray       # uint8[M,9,19,19]: 8 position layers + move one-hot (pass: all zero)
    move_index: np.ndarray   # int16[M]: row*19+col, pass = 361 (padding rows also hold 361; use mask)
    own_idx: np.ndarray      # int16[M]: own-move index of the row (padding: -1)
    mask: np.ndarray         # bool[M]: False for padding rows


def encode_position(pos: OwnPosition, include_move: bool = False) -> np.ndarray:
    n = N_PLANES + (1 if include_move else 0)
    a = np.zeros((n, BOARD, BOARD), dtype=np.uint8)
    g = pos.grid_own
    a[0] = g == 1
    a[1] = g == 2
    a[2] = g == 0
    if pos.prev_opp is not None:
        a[3][pos.prev_opp] = 1
    if pos.prev_own is not None:
        a[4][pos.prev_own] = 1
    if pos.ko_point is not None:
        a[5][pos.ko_point] = 1
    a[6] = np.where(g == 1, np.clip(liberty_map(g, 1), 1, 4), 0)
    if pos.annot_color == 0:
        a[7] = 1
    if include_move and pos.move[0] >= 0:
        a[8][pos.move] = 1
    return a


def to_float(planes: np.ndarray) -> np.ndarray:
    """uint8 planes ([...,C,19,19]) -> float16, with the liberty layer (index 6) scaled by 1/4."""
    out = planes.astype(np.float16)
    out[..., 6, :, :] *= np.float16(0.25)
    return out


def _select(m: int, sample: PlaneSampling) -> np.ndarray:
    if sample.mode == "all" or m == 0:
        return np.arange(m)
    k = sample.k
    if m <= k:
        return np.arange(m)
    if sample.mode == "first_k":
        return np.arange(k)
    if sample.mode == "even":
        return np.round(np.linspace(0, m - 1, k)).astype(np.int64)
    if sample.mode == "random":
        return np.sort(np.random.default_rng(sample.seed).choice(m, size=k, replace=False))
    raise ValueError(f"unknown sampling mode {sample.mode!r}")


def sym_augment(batch: PlaneBatch, k: int) -> PlaneBatch:
    """Apply symmetry k to every plane and move index of a batch (same k for the whole game)."""
    planes = np.empty_like(batch.planes)
    for i in range(planes.shape[0]):
        for layer in range(planes.shape[1]):
            planes[i, layer] = apply_sym_grid(k, batch.planes[i, layer])
    mv = batch.move_index.copy()
    for i, m in enumerate(mv.tolist()):
        if m != PASS_INDEX and batch.mask[i]:
            r, c = apply_sym(k, *divmod(m, BOARD))
            mv[i] = r * BOARD + c
    return PlaneBatch(planes, mv, batch.own_idx.copy(), batch.mask.copy())


def encode_game(game: ParsedGame, annot_color: int, sample: PlaneSampling, sym: int | None = None) -> PlaneBatch:
    positions = list(iter_own_positions(game, annot_color))
    idx = _select(len(positions), sample)
    rows = len(idx) if sample.mode == "all" else sample.k
    planes = np.zeros((rows, N_PLANES + 1, BOARD, BOARD), dtype=np.uint8)
    move_index = np.full(rows, PASS_INDEX, dtype=np.int16)
    own_idx = np.full(rows, -1, dtype=np.int16)
    mask = np.zeros(rows, dtype=bool)
    for j, i in enumerate(idx.tolist()):
        p = positions[i]
        planes[j] = encode_position(p, include_move=True)
        move_index[j] = PASS_INDEX if p.move[0] < 0 else p.move[0] * BOARD + p.move[1]
        own_idx[j] = p.own_idx
        mask[j] = True
    batch = PlaneBatch(planes, move_index, own_idx, mask)
    return sym_augment(batch, sym) if sym else batch
