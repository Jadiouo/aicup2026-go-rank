"""Board replay with captures (SDD 5.4). Shared core of every feature family.

Rules: after placing a stone, adjacent enemy groups without liberties are captured; then the own group is
checked: no liberties => suicide (recorded as illegal, group removed like sgfmill does).  Simple ko: capturing
exactly one stone with a single-stone group left with one liberty sets `ko_point` for the next move only.
Occupied points are overwritten and counted.  Nothing here raises on illegal data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .constants import BOARD
from .sgf import ParsedGame

N = BOARD
_NBRS: list[tuple[int, ...]] = []
for _r in range(N):
    for _c in range(N):
        _n = []
        if _r > 0:
            _n.append((_r - 1) * N + _c)
        if _c < N - 1:
            _n.append(_r * N + _c + 1)
        if _r < N - 1:
            _n.append((_r + 1) * N + _c)
        if _c > 0:
            _n.append(_r * N + _c - 1)
        _NBRS.append(tuple(_n))

_SWAP = np.array([0, 2, 1], dtype=np.int8)


@dataclass
class PlayResult:
    captured: int            # enemy stones removed by this move
    suicide: bool
    self_captured: int       # own stones removed because of suicide (not counted in `captured`)
    occupied: bool
    ko_point: tuple[int, int] | None   # ko point created BY this move (valid for the next move only)
    ko_violation: bool       # this move was played on the ko point forbidden by the previous move


class Board:
    """19x19 board. `grid` (int8[19,19], 0 empty / 1 black / 2 white, absolute colors) is a snapshot property."""

    __slots__ = ("_cells", "ko_point")

    def __init__(self) -> None:
        self._cells: list[int] = [0] * (N * N)
        self.ko_point: tuple[int, int] | None = None   # forbidden point for the NEXT move

    @property
    def grid(self) -> np.ndarray:
        return np.array(self._cells, dtype=np.int8).reshape(N, N)

    def copy(self) -> "Board":
        b = Board()
        b._cells = list(self._cells)
        b.ko_point = self.ko_point
        return b

    def set_stone(self, color: int, row: int, col: int) -> None:
        """Place a setup stone (AB/AW) without any capture logic."""
        self._cells[row * N + col] = color + 1

    def _group(self, p: int) -> tuple[list[int], bool]:
        """Flood fill the group containing p. Returns (stones, has_liberty)."""
        cells = self._cells
        color = cells[p]
        stack = [p]
        seen = {p}
        libs = False
        while stack:
            q = stack.pop()
            for n in _NBRS[q]:
                v = cells[n]
                if v == 0:
                    libs = True
                elif v == color and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return list(seen), libs

    def liberties(self, row: int, col: int) -> int:
        """Number of distinct liberties of the group at (row,col); 0 if the point is empty."""
        p = row * N + col
        if self._cells[p] == 0:
            return 0
        stones, _ = self._group(p)
        libs = set()
        for q in stones:
            for n in _NBRS[q]:
                if self._cells[n] == 0:
                    libs.add(n)
        return len(libs)

    def play(self, color: int, row: int, col: int) -> PlayResult:
        """color 0=B/1=W; pass is row=-1."""
        prev_ko = self.ko_point
        self.ko_point = None
        if row < 0 or col < 0:
            return PlayResult(0, False, 0, False, None, False)
        cells = self._cells
        p = row * N + col
        me = color + 1
        opp = 3 - me
        occupied = cells[p] != 0
        ko_violation = prev_ko == (row, col)
        cells[p] = me
        captured = 0
        cap_point = -1
        checked: set[int] = set()
        for n in _NBRS[p]:
            if cells[n] == opp and n not in checked:
                stones, libs = self._group(n)
                if libs:
                    checked.update(stones)
                else:
                    for q in stones:
                        cells[q] = 0
                    captured += len(stones)
                    cap_point = stones[0]
        own, own_libs = self._group(p)
        suicide = False
        self_captured = 0
        ko_point = None
        if not own_libs:
            suicide = True
            for q in own:
                cells[q] = 0
            self_captured = len(own)
        elif captured == 1 and len(own) == 1:
            nlibs = sum(1 for n in _NBRS[p] if cells[n] == 0)
            if nlibs == 1:
                ko_point = divmod(cap_point, N)
        self.ko_point = ko_point
        return PlayResult(captured, suicide, self_captured, occupied, ko_point, ko_violation)


def _initial_board(game: ParsedGame) -> Board:
    b = Board()
    for color, r, c in game.setup.tolist():
        b.set_stone(color, r, c)
    return b


def replay(game: ParsedGame) -> Iterator[tuple[int, Board, PlayResult]]:
    """Yields (ply, board_after_move, result). The same mutable Board object is reused; copy() to keep it."""
    b = _initial_board(game)
    for ply, (color, r, c) in enumerate(game.moves.tolist()):
        res = b.play(color, r, c)
        yield ply, b, res


@dataclass(frozen=True)
class OwnPosition:
    ply: int
    own_idx: int
    grid_own: np.ndarray                 # int8[19,19] board BEFORE the move: 0 empty, 1 own, 2 enemy
    move: tuple[int, int]
    prev_opp: tuple[int, int] | None
    prev_own: tuple[int, int] | None
    ko_point: tuple[int, int] | None
    captured: int
    lost_since_prev_own: int
    annot_color: int = 0                 # extension: color (0=B/1=W) of the annotated side


def iter_own_positions(game: ParsedGame, annot_color: int) -> Iterator[OwnPosition]:
    """One OwnPosition per move of the annotated side (pass included). Color swap, never a geometric flip."""
    b = _initial_board(game)
    moves = game.moves.tolist()
    own_idx = 0
    prev_own: tuple[int, int] | None = None
    last_move: tuple[int, int, int] | None = None
    lost = 0
    for ply, (color, r, c) in enumerate(moves):
        if color == annot_color:
            g = b.grid
            if annot_color == 1:
                g = _SWAP[g]
            prev_opp = None
            if last_move is not None and last_move[0] != annot_color and last_move[1] >= 0:
                prev_opp = (last_move[1], last_move[2])
            ko_before = b.ko_point
            res = b.play(color, r, c)
            yield OwnPosition(ply, own_idx, g, (r, c), prev_opp, prev_own, ko_before, res.captured, lost, annot_color)
            own_idx += 1
            lost = 0
            if r >= 0:
                prev_own = (r, c)
        else:
            res = b.play(color, r, c)
            lost += res.captured
        last_move = (color, r, c)


@dataclass
class ReplaySummary:
    captures: np.ndarray        # int16[T] enemy stones captured per ply
    self_captures: np.ndarray   # int16[T] own stones removed by suicide per ply
    grid: np.ndarray            # final int8[19,19] absolute colors
    counts: dict[str, int]      # suicide / occupied / ko_violation

    @property
    def illegal(self) -> int:
        """Illegal moves proper: suicide + occupied (ko violations are reported separately)."""
        return self.counts["suicide"] + self.counts["occupied"]


def replay_summary(game: ParsedGame) -> ReplaySummary:
    T = len(game.moves)
    caps = np.zeros(T, dtype=np.int16)
    selfc = np.zeros(T, dtype=np.int16)
    counts = {"suicide": 0, "occupied": 0, "ko_violation": 0}
    final = _initial_board(game)
    for ply, b, res in replay(game):
        caps[ply] = res.captured
        selfc[ply] = res.self_captured
        counts["suicide"] += res.suicide
        counts["occupied"] += res.occupied
        counts["ko_violation"] += res.ko_violation
        final = b
    return ReplaySummary(caps, selfc, final.grid, counts)


def liberty_map(grid: np.ndarray, value: int) -> np.ndarray:
    """int16[19,19]: for every stone equal to `value`, the liberty count of its group (0 elsewhere)."""
    flat = grid.reshape(-1).tolist()
    out = np.zeros(N * N, dtype=np.int16)
    seen = [False] * (N * N)
    for p0 in range(N * N):
        if flat[p0] != value or seen[p0]:
            continue
        stack, group, libs = [p0], [p0], set()
        seen[p0] = True
        while stack:
            q = stack.pop()
            for n in _NBRS[q]:
                v = flat[n]
                if v == 0:
                    libs.add(n)
                elif v == value and not seen[n]:
                    seen[n] = True
                    stack.append(n)
                    group.append(n)
        for q in group:
            out[q] = len(libs)
    return out.reshape(N, N)
