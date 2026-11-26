"""E89 S line: fast 19x19 board replay (numba) that emits own-side positions in a compact form.

Rules mirror src/gogame/board.py exactly (capture enemy groups without liberties, then suicide check, simple ko =
capture of exactly one stone by a single-stone group left with one liberty; occupied points are overwritten).
Move encoding (int16): pos + 362 * colour_bit, pos = row*19+col, pass = 361, colour_bit 0 = black, 1 = white.

encode_game() returns, for each selected own-move ordinal (ordinal counts passes, like board.py own_idx):
  grid[K,361]   uint8  0 empty / 1 own / 2 opponent (own = side that is about to move), BEFORE the move
  lib[K,361]    uint8  liberties of the group on that point, clipped to 4 (0 on empty points)
  prev_opp/prev_own/ko/move  int16 point index or -1
  ply/own_idx   int16
build_planes_np() turns that into the 18 PLAN planes (reference implementation; the GPU twin lives in train_s.py).
"""
from __future__ import annotations

import numpy as np
from numba import njit

N = 19
NN = 361
PASS = 361

HANDICAP_STONES = {
    2: ["pd", "dp"], 3: ["pd", "dp", "dd"], 4: ["pd", "dp", "dd", "pp"], 5: ["pd", "dp", "dd", "pp", "jj"],
    6: ["pd", "dp", "dd", "pp", "dj", "pj"], 7: ["pd", "dp", "dd", "pp", "dj", "pj", "jj"],
    8: ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp"], 9: ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp", "jj"],
}   # identical to scripts/katago/kgio.py HANDICAP_STONES (asserted in verify_engine.py)


def _hd_points(hd: int) -> np.ndarray:
    return np.array([(ord(s[1]) - 97) * N + (ord(s[0]) - 97) for s in HANDICAP_STONES.get(hd, [])], dtype=np.int16)


HD_PTS = [_hd_points(h) for h in range(10)]


def _nbr_table() -> np.ndarray:
    t = np.full((NN, 4), -1, np.int16)
    for r in range(N):
        for c in range(N):
            n = []
            if r > 0: n.append((r - 1) * N + c)
            if c < N - 1: n.append(r * N + c + 1)
            if r < N - 1: n.append((r + 1) * N + c)
            if c > 0: n.append(r * N + c - 1)
            t[r * N + c, :len(n)] = n
    return t


NBR = _nbr_table()


@njit(cache=True)
def _flood(cells, p, nbr, stack, mark, stamp, grp):
    """Flood the group at p. Returns (n_stones, has_liberty); stones are written to grp[:n]."""
    color = cells[p]
    sp = 0
    stack[0] = p
    sp = 1
    mark[p] = stamp
    n = 0
    libs = False
    while sp > 0:
        sp -= 1
        q = stack[sp]
        grp[n] = q
        n += 1
        for k in range(4):
            m = nbr[q, k]
            if m < 0:
                break
            v = cells[m]
            if v == 0:
                libs = True
            elif v == color and mark[m] != stamp:
                mark[m] = stamp
                stack[sp] = m
                sp += 1
    return n, libs


@njit(cache=True)
def _play(cells, color, p, nbr, stack, mark, stamp0, grp):
    """Place `color` (1 black / 2 white) at p. Returns (new_stamp, ko_point)."""
    stamp = stamp0
    me = color
    opp = 3 - me
    cells[p] = me
    captured = 0
    cap_point = -1
    for k in range(4):
        m = nbr[p, k]
        if m < 0:
            break
        if cells[m] == opp:
            stamp += 1
            n, libs = _flood(cells, m, nbr, stack, mark, stamp, grp)
            if not libs:
                for i in range(n):
                    cells[grp[i]] = 0
                captured += n
                cap_point = grp[0]
    stamp += 1
    n, libs = _flood(cells, p, nbr, stack, mark, stamp, grp)
    ko = -1
    if not libs:
        for i in range(n):
            cells[grp[i]] = 0
    elif captured == 1 and n == 1:
        nl = 0
        for k in range(4):
            m = nbr[p, k]
            if m < 0:
                break
            if cells[m] == 0:
                nl += 1
        if nl == 1:
            ko = cap_point
    return stamp, ko


@njit(cache=True)
def _libmap(cells, nbr, stack, mark, lmark, stamp, grp, out):
    """out[p] = liberties (clipped to 4) of the group on p, 0 for empty points.  Returns the new stamp."""
    for p in range(NN):
        out[p] = 0
    for p0 in range(NN):
        if cells[p0] == 0 or out[p0] != 0:
            continue
        stamp += 1
        color = cells[p0]
        sp = 1
        stack[0] = p0
        mark[p0] = stamp
        n = 0
        nl = 0
        while sp > 0:
            sp -= 1
            q = stack[sp]
            grp[n] = q
            n += 1
            for k in range(4):
                m = nbr[q, k]
                if m < 0:
                    break
                v = cells[m]
                if v == 0:
                    if lmark[m] != stamp:
                        lmark[m] = stamp
                        nl += 1
                elif v == color and mark[m] != stamp:
                    mark[m] = stamp
                    stack[sp] = m
                    sp += 1
        c = nl if nl < 4 else 4
        for i in range(n):
            out[grp[i]] = c
    return stamp


@njit(cache=True)
def encode_game(mv, eff_color, hd_pts, sel, nbr):
    """mv int16[T] encoded moves; eff_color 0/1 own side; hd_pts int16[S] handicap stones (black) ; sel int32[K] sorted own-move
    ordinals to emit.  Returns the arrays described in the module docstring."""
    T = mv.shape[0]
    K = sel.shape[0]
    cells = np.zeros(NN, np.int8)
    for i in range(hd_pts.shape[0]):
        cells[hd_pts[i]] = 1
    stack = np.empty(NN, np.int16)
    grp = np.empty(NN, np.int16)
    mark = np.zeros(NN, np.int64)
    lmark = np.zeros(NN, np.int64)
    stamp = 0
    grid = np.zeros((K, NN), np.uint8)
    lib = np.zeros((K, NN), np.uint8)
    prev_opp = np.full(K, -1, np.int16)
    prev_own = np.full(K, -1, np.int16)
    ko_out = np.full(K, -1, np.int16)
    move = np.full(K, -1, np.int16)
    ply_out = np.zeros(K, np.int16)
    oi_out = np.zeros(K, np.int16)
    libtmp = np.zeros(NN, np.uint8)
    own_val = 1 + eff_color
    ko = -1
    own_ord = 0
    last_col = -1
    last_pos = -1
    last_own = -1
    k = 0
    for t in range(T):
        e = mv[t]
        col = 1 if e >= 362 else 0
        pos = e - 362 * col
        if col == eff_color:
            if k < K and sel[k] == own_ord:
                for p in range(NN):
                    v = cells[p]
                    grid[k, p] = 0 if v == 0 else (1 if v == own_val else 2)
                stamp = _libmap(cells, nbr, stack, mark, lmark, stamp, grp, libtmp)
                for p in range(NN):
                    lib[k, p] = libtmp[p]
                if last_col >= 0 and last_col != eff_color and last_pos != PASS:
                    prev_opp[k] = last_pos
                prev_own[k] = last_own
                ko_out[k] = ko
                move[k] = -1 if pos == PASS else pos
                ply_out[k] = t
                oi_out[k] = own_ord
                k += 1
            own_ord += 1
        if pos == PASS:
            ko = -1
        else:
            stamp, ko = _play(cells, 1 + col, pos, nbr, stack, mark, stamp, grp)
        if col == eff_color and pos != PASS:
            last_own = pos
        last_col = col
        last_pos = pos
        if k >= K:
            break
    return grid, lib, prev_opp, prev_own, ko_out, move, ply_out, oi_out


@njit(cache=True)
def replay_states(mv, hd_pts, nbr):
    """Debug/verification: board after every ply (absolute colours 0/1 black/2 white) and captures per ply."""
    T = mv.shape[0]
    cells = np.zeros(NN, np.int8)
    for i in range(hd_pts.shape[0]):
        cells[hd_pts[i]] = 1
    stack = np.empty(NN, np.int16)
    grp = np.empty(NN, np.int16)
    mark = np.zeros(NN, np.int64)
    stamp = 0
    out = np.zeros((T, NN), np.int8)
    kos = np.full(T, -1, np.int16)
    for t in range(T):
        e = mv[t]
        col = 1 if e >= 362 else 0
        pos = e - 362 * col
        ko = -1
        if pos != PASS:
            stamp, ko = _play(cells, 1 + col, pos, nbr, stack, mark, stamp, grp)
        out[t] = cells
        kos[t] = ko
    return out, kos


def valid_ordinals(mv: np.ndarray, eff_color: int) -> np.ndarray:
    """Own-move ordinals (passes counted, like board.py own_idx) of the NON-pass own moves."""
    col = (mv >= 362).astype(np.int8)
    pos = mv - 362 * col.astype(np.int16)
    own = col == eff_color
    ordv = np.cumsum(own) - 1
    return ordv[own & (pos != PASS)].astype(np.int32)


def count_own(mv: np.ndarray, eff_color: int) -> int:
    return int(((mv >= 362).astype(np.int8) == eff_color).sum())


def encode(mv: np.ndarray, eff_color: int, hd: int, sel: np.ndarray):
    return encode_game(np.ascontiguousarray(mv, dtype=np.int16), int(eff_color), HD_PTS[int(hd)], np.ascontiguousarray(sel, dtype=np.int32), NBR)


def build_planes_np(grid, lib, prev_opp, prev_own, ko, move, ply, own_idx, eff_color) -> np.ndarray:
    """Reference plane builder -> float32 [K,18,19,19].
    0 own, 1 opp, 2 empty, 3-6 own libs ==1/==2/==3/>=4, 7-10 opp libs, 11 prev opp move, 12 prev own move, 13 ko point,
    14 actual move, 15 colour flag (black 1), 16 ply/200, 17 min(own_idx/120, 1)."""
    K = grid.shape[0]
    X = np.zeros((K, 18, NN), np.float32)
    g = grid
    X[:, 0] = g == 1
    X[:, 1] = g == 2
    X[:, 2] = g == 0
    for i in range(4):
        X[:, 3 + i] = (g == 1) & (lib == i + 1)
        X[:, 7 + i] = (g == 2) & (lib == i + 1)
    for ch, a in ((11, prev_opp), (12, prev_own), (13, ko), (14, move)):
        rows = np.flatnonzero(a >= 0)
        X[rows, ch, a[rows]] = 1
    X[:, 15] = 1.0 if eff_color == 0 else 0.0
    X[:, 16] = (ply.astype(np.float32) / 200.0)[:, None]
    X[:, 17] = np.minimum(own_idx.astype(np.float32) / 120.0, 1.0)[:, None]
    return X.reshape(K, 18, N, N)
