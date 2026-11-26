"""SGF -> Academia Sinica Strength Estimator (SE) input planes [N,18,19,19].

Every layer is checked against the original sources (read via GitHub raw, nothing cloned or compiled):

  GO = https://github.com/rlglab/minizero/blob/main/minizero/environment/go/go.cpp
  SL = https://github.com/rlglab/minizero/blob/main/minizero/utils/sgf_loader.cpp
  RO = https://github.com/rlglab/minizero/blob/main/minizero/utils/rotation.h
  BE = https://github.com/rlglab/minizero/blob/main/minizero/environment/base/base_env.h
  GW = https://github.com/rlglab/strength-estimator/blob/iclr2025/strength/misc/game_wrapper.cpp
  MH = https://github.com/rlglab/strength-estimator/blob/iclr2025/strength/st_mode_handler.cpp
  (the strength-estimator default branch is `iclr2025`, not `main`; line numbers are those of 2026-11-02)

Facts established from the sources
----------------------------------
* Position index (BE L235-241, `EnvironmentLoader::getFeatures(pos)`): replays actions [0, pos) on an empty board and
  returns env.getFeatures().  GW L58-61 `calculateFeatures(env_loader,pos,rot)` just forwards to it.  MH L176-182 scores
  positions pos = 0..T-1 of EVERY ply: pos = number of moves already played, i.e. the board BEFORE move `pos` is played
  and `turn_` is the side about to play it.  => "pre" position of ply k = position index k.  "post" = index k+1.
* Planes (GO L291-319 `GoEnv::getFeatures`), 18 channels, row-major pos = y*19 + x:
    channel c<16: `last_n_turn = history.size()-1 - c/2`; if <0 -> all zero; else the stone bitboard of
      `player = (c%2==0 ? turn_ : opponent of turn_)` taken from `stone_bitboard_history_[last_n_turn]` (L303-309).
      So c=0/1 = side-to-move / opponent stones in the CURRENT board, c=2/3 = the board one move earlier, ... c=14/15 =
      seven moves earlier.  "own/opponent" is always relative to turn_ (not to who moved in that historic board).
    channel 16 = 1 everywhere if turn_ == Player1 (black)   (L311-312)
    channel 17 = 1 everywhere if turn_ == Player2 (white)   (L313-314)
  The history has one entry per action, pushed AFTER the action incl. captures (GO L145 for pass, L179-180 for a stone;
  captured blocks are removed from stone_bitboard_ in removeBlockFromBoard, L442).  The empty start board is NOT in
  the history (reset(), L127, clears it), so with k moves played, planes of age >= k are zero.
  A pass pushes the unchanged board (L145) and flips the turn like any move.
* Colour: reset() sets turn_ = Player1 (L105) and the first act is black in SGF; Player1 == black (channel 16 is
  documented "black turn" in the L293-297 comment).  After an action `turn_ = action.nextPlayer()` (L140) = the opposite
  of the mover, also for passes.
* Orientation (SL L113-119 `sgfStringToActionID`): x = letter0 - 'A';  y = (19-1) - (letter1 - 'A');  id = y*19 + x.
  The SGF second letter is the row counted from the TOP, so the tensor row index is 18 - sgf_row (vertical flip
  relative to our own `coord_to_rc`), and the tensor column is the SGF first letter.  The tensor is [C, y, x] with
  pos = y*19+x reshaped to (19,19).
* Rotation (RO L52-81 `getPositionByRotating`, GO L301 uses `getRotatePosition(pos, reversed_rotation[rot])`): output
  feature at pos n equals the unrotated feature at pos reversed_rotation[rot]-applied(n); equivalently an original
  point o moves to getPositionByRotating(rot, o).  Training used a random one of the 8 rotations per sample
  (strength-estimator st_data_loader.cpp L58, mode handler MH L181), so all 8 are equally valid inputs.
* Strength score: cfg `bt_use_weight=false` (strength-estimator cfg/se_go.cfg), so the aggregation in MH L184-190 is
  just the `score` head output (no weight head) averaged per game.
* Handicap: games with HD >= 2 have swapped colours and no recorded stones in our data (v1.1), so they are excluded
  by the caller; this module refuses a game with SGF setup stones.
* Not modelled (no effect on the planes): komi (cfg only), ko rule (legality only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gogame.board import Board  # noqa: E402
from gogame.sgf import ParsedGame, parse_sgf  # noqa: E402

N = 19
HIST = 8          # last 8 positions
NCH = 18
BLACK, WHITE = 0, 1


def _rotation_index(rotation: int) -> np.ndarray:
    """perm[n] = original position o with getPositionByRotating(rotation, o) == n (RO L52-81)."""
    center = (N - 1) / 2.0
    perm = np.zeros(N * N, dtype=np.int64)
    for o in range(N * N):
        x = o % N - center
        y = o // N - center
        rx, ry = {
            0: (x, y), 1: (y, -x), 2: (-x, -y), 3: (-y, x),
            4: (x, -y), 5: (-y, -x), 6: (-x, y), 7: (y, x),
        }[rotation]
        n = int(round((ry + center) * N + (rx + center)))
        perm[n] = o
    return perm


_PERMS = [_rotation_index(r) for r in range(8)]


def board_history(game: ParsedGame) -> tuple[np.ndarray, np.ndarray]:
    """Replay with captures.  Returns (hist int8[T,19,19], movers int8[T]).

    hist[i] is the board AFTER move i in MiniZero orientation (row index = 18 - sgf_row), 0 empty / 1 black / 2 white.
    """
    if len(game.setup):
        raise ValueError("games with setup stones (HD>=2) are not supported")
    b = Board()
    T = len(game.moves)
    hist = np.zeros((T, N, N), dtype=np.int8)
    movers = np.zeros(T, dtype=np.int8)
    for i, (color, r, c) in enumerate(game.moves.tolist()):
        b.play(color, r, c)
        hist[i] = b.grid[::-1]          # flip rows: tensor y = 18 - sgf row (SL L117)
        movers[i] = color
    return hist, movers


def encode_positions(hist: np.ndarray, movers: np.ndarray, positions, rotation: int = 0) -> np.ndarray:
    """Planes for the given position indices p (p = number of moves already played, 0..T).  float32 [n,18,19,19]."""
    positions = list(positions)
    out = np.zeros((len(positions), NCH, N, N), dtype=np.float32)
    perm = _PERMS[rotation]
    for j, p in enumerate(positions):
        turn = BLACK if p == 0 else 1 - int(movers[p - 1])       # GO L140 nextPlayer, L105 initial Player1
        own, opp = turn + 1, 2 - turn
        for c in range(16):
            age = c // 2
            idx = p - 1 - age                                   # GO L303
            if idx < 0:
                continue
            who = own if c % 2 == 0 else opp                    # GO L308
            out[j, c] = (hist[idx] == who)
        out[j, 16] = 1.0 if turn == BLACK else 0.0              # GO L311-312
        out[j, 17] = 1.0 if turn == WHITE else 0.0              # GO L313-314
    if rotation:
        flat = out.reshape(len(positions), NCH, N * N)[:, :, perm]
        out = np.ascontiguousarray(flat.reshape(len(positions), NCH, N, N))
    return out


def own_move_plies(game: ParsedGame, color: int, n_max: int | None = None, include_pass: bool = False) -> np.ndarray:
    """Plies played by `color`; if n_max, an evenly spread subset of that size."""
    mv = game.moves
    sel = np.where((mv[:, 0] == color) & (include_pass | (mv[:, 1] >= 0)))[0]
    if n_max is not None and len(sel) > n_max:
        sel = sel[np.linspace(0, len(sel) - 1, n_max).round().astype(int)]
    return sel


def encode_game(sgf: str, color: int, when: str = "pre", n_max: int | None = 30, rotation: int = 0) -> np.ndarray:
    """Planes for ~n_max moves of `color` (0=B,1=W).  when='pre': the board before the move (side to move == color);
    when='post': the board after the move (side to move == opponent)."""
    if when not in ("pre", "post"):
        raise ValueError(when)
    g = parse_sgf(sgf)
    hist, movers = board_history(g)
    plies = own_move_plies(g, color, n_max)
    pos = plies if when == "pre" else plies + 1
    return encode_positions(hist, movers, pos, rotation)
