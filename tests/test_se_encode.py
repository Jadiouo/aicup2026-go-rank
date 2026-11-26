"""E21: SGF -> SE planes.  Hand-computed expectations (see se_encode.py docstring for source references)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "e21"))
import se_encode as se  # noqa: E402
from gogame.sgf import parse_sgf  # noqa: E402


def planes(sgf, positions, rotation=0):
    h, m = se.board_history(parse_sgf(sgf))
    return se.encode_positions(h, m, positions, rotation)


def pts(plane):
    return sorted(map(tuple, np.argwhere(plane > 0).tolist()))


def test_empty_start_black_to_move():
    x = planes("(;B[pd])", [0])
    assert x.shape == (1, 18, 19, 19) and x.dtype == np.float32
    assert x[0, :16].sum() == 0 and x[0, 16].min() == 1 and x[0, 17].max() == 0


def test_two_moves_orientation_and_history():
    # pd = col 15,row 3 from top -> tensor y = 18-3 = 15, x = 15 ; dp = col 3,row 15 -> y = 3, x = 3
    x = planes("(;B[pd];W[dp])", [2])[0]
    assert pts(x[0]) == [(15, 15)]      # side to move (black) stones now
    assert pts(x[1]) == [(3, 3)]        # opponent (white) stones now
    assert pts(x[2]) == [(15, 15)]      # one move earlier: own(black)=pd
    assert x[3].sum() == 0              # one move earlier: white had none
    assert x[4:16].sum() == 0
    assert x[16].min() == 1 and x[17].max() == 0


def test_white_to_move_flags_and_colour_relative_planes():
    x = planes("(;B[pd];W[dp];B[dd])", [3])[0]
    assert x[17].min() == 1 and x[16].max() == 0
    assert pts(x[0]) == [(3, 3)]                       # white (to move) = dp
    assert pts(x[1]) == [(15, 3), (15, 15)]            # black dd, pd
    assert pts(x[2]) == [(3, 3)]                       # one move earlier, white
    assert pts(x[3]) == [(15, 15)]                     # one move earlier, black had only pd
    assert pts(x[4]) == [] and pts(x[5]) == [(15, 15)]  # two moves earlier: white none yet, black pd


def test_capture_removes_stone_in_current_and_keeps_it_in_history():
    # white stone in corner aa (tensor y=18,x=0) captured by black ba (y=18,x=1) + ab (y=17,x=0)
    x = planes("(;B[ba];W[aa];B[ab])", [3])[0]
    assert pts(x[0]) == []                                   # white (to move) has nothing left
    assert pts(x[1]) == [(17, 0), (18, 1)]                   # black ab, ba
    assert pts(x[2]) == [(18, 0)]                            # one move earlier: white aa still there
    assert pts(x[3]) == [(18, 1)]
    assert x[4].sum() == 0 and pts(x[5]) == [(18, 1)]


def test_multi_stone_capture_and_suicide_free_replay():
    # white two stones aa,ba captured by black ca, ab, bb
    sgf = "(;B[ab];W[aa];B[bb];W[ba];B[ca])"
    x = planes(sgf, [5])[0]
    assert x[1].sum() == 3 and x[0].sum() == 0           # white (to move) lost both stones
    assert planes(sgf, [4])[0][1].sum() == 2              # before ca: white (opponent of the black mover) has aa,ba
    assert planes(sgf, [4])[0][0].sum() == 2              # black (to move after W) has ab,bb


def test_pass_repeats_board_and_flips_turn():
    x = planes("(;B[pd];W[];B[dd])", [2])[0]       # after B, W-pass: black to move
    assert x[16].min() == 1
    assert pts(x[0]) == [(15, 15)] and x[1].sum() == 0
    assert pts(x[2]) == [(15, 15)]                  # one move earlier = board after B (pass keeps the board)
    assert x[3].sum() == 0 and x[4:].sum() == x[16].sum()   # nothing older; only the turn flag remains


def test_history_depth_is_8_and_older_dropped():
    moves = ";".join(("B" if i % 2 == 0 else "W") + "[" + chr(97 + i) + chr(97 + i) + "]" for i in range(12))
    x = planes("(;" + moves + ")", [12])[0]
    # position 12: black to move; plane 14/15 = 7 moves earlier = board after move index 4 (5 stones: 0..4)
    assert x[14].sum() + x[15].sum() == 5
    assert x[0].sum() + x[1].sum() == 12


@pytest.mark.parametrize("rot", range(8))
def test_rotations_follow_minizero(rot):
    # independent restatement of RO L52-81 + GO L301 (reversed_rotation): out[n] = in[getPositionByRotating(rev[rot], n)]
    rev = [0, 3, 2, 1, 4, 5, 6, 7]

    def f(r, o):
        c = 9.0
        x, y = o % 19 - c, o // 19 - c
        rx, ry = [(x, y), (y, -x), (-x, -y), (-y, x), (x, -y), (-y, -x), (-x, y), (y, x)][r]
        return int(round((ry + c) * 19 + (rx + c)))

    base = planes("(;B[dc];W[pq])", [2])[0]     # asymmetric: dc -> (y=16,x=3), pq -> (y=2,x=15)
    got = planes("(;B[dc];W[pq])", [2], rotation=rot)[0]
    exp = np.zeros_like(base).reshape(18, 361)
    flat = base.reshape(18, 361)
    for n in range(361):
        exp[:, n] = flat[:, f(rev[rot], n)]
    assert np.array_equal(got, exp.reshape(18, 19, 19))
    assert got[0].sum() == 1 and got[1].sum() == 1          # stone counts kept
    assert got[16].min() == 1 and got[17].max() == 0         # turn flags untouched
    # the black stone must sit where the forward map says
    assert pts(got[0]) == [divmod(f(rot, 16 * 19 + 3), 19)]


def test_eight_rotations_all_distinct_and_composable():
    outs = [planes("(;B[dc];W[pq])", [2], rotation=r)[0][0] for r in range(8)]
    keys = {o.tobytes() for o in outs}
    assert len(keys) == 8
    # rotation 2 = point reflection through the centre
    assert np.array_equal(outs[2], outs[0][::-1, ::-1])
    assert np.array_equal(outs[4], outs[0][::-1, :])        # kHorizontalRotation: y -> -y
    assert np.array_equal(outs[6], outs[0][:, ::-1])        # x -> -x


def test_encode_game_pre_post_and_selection():
    sgf = "(;B[pd];W[dp];B[dd];W[pp])"
    pre = se.encode_game(sgf, 0, "pre", n_max=None)        # black plies 0,2
    post = se.encode_game(sgf, 0, "post", n_max=None)
    assert pre.shape[0] == post.shape[0] == 2
    assert pre[0, 16].min() == 1 and post[0, 17].min() == 1       # pre: black to move, post: white to move
    assert pre[0, :16].sum() == 0 and post[0, 1].sum() == 1
    wpre = se.encode_game(sgf, 1, "pre", n_max=None)
    assert wpre[0, 17].min() == 1
    assert se.encode_game(sgf, 0, "pre", n_max=1).shape[0] == 1


def test_setup_stones_rejected():
    with pytest.raises(ValueError):
        se.board_history(parse_sgf("(;SZ[19]AB[dd][pp];W[dp])"))
