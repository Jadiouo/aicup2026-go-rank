import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from fixtures.handmade import HANDMADE
from gogame.board import iter_own_positions
from gogame.planes import N_PLANES, PlaneSampling, encode_game, encode_position, sym_augment, to_float
from gogame.sgf import parse_sgf
from gogame.symmetry import apply_sym, apply_sym_grid
from helpers import sgf

GOLDEN_CSV = Path(__file__).parent / "fixtures" / "golden" / "games_50.csv"
ALL = PlaneSampling("all")


def game(name):
    return parse_sgf(HANDMADE.get(name, name))


def positions(name, color):
    return list(iter_own_positions(game(name), color))


def long_game():
    return parse_sgf(pd.read_csv(GOLDEN_CSV, dtype=str, keep_default_na=False)["sgf_content"].iloc[0])


def test_planes_shape_dtype():
    p = positions("mini_B", 0)[0]
    a = encode_position(p)
    assert N_PLANES == 8 and a.shape == (8, 19, 19) and a.dtype == np.uint8
    assert encode_position(p, include_move=True).shape == (9, 19, 19)
    b = encode_game(game("mini_B"), 0, ALL)
    assert b.planes.shape == (3, 9, 19, 19) and b.planes.dtype == np.uint8
    assert b.move_index.shape == (3,) and b.own_idx.shape == (3,) and b.mask.shape == (3,)
    assert b.move_index.dtype == np.int16 and b.mask.dtype == np.bool_


def test_own_opp_empty_partition():
    for color in (0, 1):
        for p in list(iter_own_positions(long_game(), color))[::7]:
            a = encode_position(p)
            assert np.all(a[0].astype(int) + a[1] + a[2] == 1)
            assert set(np.unique(a[0])) <= {0, 1} and set(np.unique(a[2])) <= {0, 1}


def test_white_annotated_equals_color_swapped_black():
    b = encode_game(game("mini_B"), 0, ALL)
    w = encode_game(game("white_first"), 1, ALL)
    assert b.planes.shape == w.planes.shape
    # layers 0..6 (own, enemy, empty, prev_opp, prev_own, ko, liberties) are identical element by element;
    # layer 7 is the colour flag (1 for black, 0 for white) and layer 8 the move one-hot (identical too).
    assert np.array_equal(b.planes[:, :7], w.planes[:, :7])
    assert np.array_equal(b.planes[:, 8], w.planes[:, 8])
    assert b.planes[:, 7].min() == 1 and w.planes[:, 7].max() == 0
    assert np.array_equal(b.move_index, w.move_index)


def test_last_move_planes():
    a = [encode_position(p) for p in positions("mini_B", 0)]
    assert a[0][3].sum() == 0 and a[0][4].sum() == 0              # first move: no previous moves
    assert a[1][3].sum() == 1 and a[1][3][15, 3] == 1             # prev_opp = dp
    assert a[1][4].sum() == 1 and a[1][4][3, 15] == 1             # prev_own = pd
    assert a[2][3][3, 3] == 1 and a[2][4][15, 15] == 1
    w = [encode_position(p) for p in positions("mini_B", 1)]
    assert w[0][3][3, 15] == 1 and w[0][4].sum() == 0             # white's first move: prev_opp = black's first move


def test_ko_plane():
    ps = positions("ko_recapture", 1)
    a = encode_position(ps[-1])
    assert a[5][9, 2] == 1 and a[5].sum() == 1
    assert all(encode_position(p)[5].sum() == 0 for p in ps[:-1])


def test_liberty_plane_clip():
    # one liberty: black corner stone aa with white on ba, position before black plays ss
    p = positions(sgf("B:aa W:ba B:ss"), 0)[-1]
    a = encode_position(p)
    assert a[6][0, 0] == 1 and to_float(a)[6][0, 0] == 0.25
    # 6-liberty two-stone group in the centre clips to 4 -> 1.0
    p = positions(sgf("B:jj W:aa B:kj W:ab B:ss"), 0)[-1]
    a = encode_position(p)
    assert a[6][9, 9] == 4 and a[6][9, 10] == 4 and to_float(a)[6][9, 9] == 1.0
    assert a[6][0, 0] == 0                                         # enemy stones carry no liberty value
    assert a[6][a[0] == 0].sum() == 0                              # only own stones
    assert to_float(encode_position(p)).dtype == np.float16


def test_color_flag_plane():
    assert np.all(encode_position(positions("mini_B", 0)[1])[7] == 1)
    assert np.all(encode_position(positions("mini_B", 1)[1])[7] == 0)


def test_move_onehot_matches_index():
    for name, color in (("mini_B", 0), ("pass_variants", 1), ("edge_coords", 0), ("pass_variants", 0)):
        b = encode_game(game(name), color, ALL)
        for i in range(len(b.move_index)):
            layer = b.planes[i, 8]
            if b.move_index[i] == 361:
                assert layer.sum() == 0
            else:
                assert layer.sum() == 1 and layer.reshape(-1)[b.move_index[i]] == 1
    b = encode_game(game("mini_B"), 0, ALL)
    assert b.move_index.tolist() == [3 * 19 + 15, 15 * 19 + 15, 9 * 19 + 10]
    assert encode_game(game("pass_variants"), 1, ALL).move_index.tolist() == [361, 361]
    assert encode_game(game("edge_coords"), 0, ALL).move_index.tolist() == [18 * 19 + 18]


def test_sampling_even_deterministic():
    g = long_game()
    n_own = len(list(iter_own_positions(g, 0)))
    a = encode_game(g, 0, PlaneSampling("even", k=10))
    b = encode_game(g, 0, PlaneSampling("even", k=10))
    assert np.array_equal(a.planes, b.planes) and np.array_equal(a.own_idx, b.own_idx)
    assert a.own_idx.tolist() == sorted(set(a.own_idx.tolist())) and len(a.own_idx) == 10
    assert a.own_idx[0] == 0 and a.own_idx[-1] == n_own - 1       # equally spaced covering the whole game
    r1 = encode_game(g, 0, PlaneSampling("random", k=10, seed=1))
    r2 = encode_game(g, 0, PlaneSampling("random", k=10, seed=1))
    r3 = encode_game(g, 0, PlaneSampling("random", k=10, seed=2))
    assert np.array_equal(r1.own_idx, r2.own_idx) and not np.array_equal(r1.own_idx, r3.own_idx)
    f = encode_game(g, 0, PlaneSampling("first_k", k=10))
    assert f.own_idx.tolist() == list(range(10))
    assert encode_game(g, 0, ALL).planes.shape[0] == n_own


def test_sampling_pads_with_mask():
    b = encode_game(game("mini_B"), 0, PlaneSampling("first_k", k=5))
    assert b.planes.shape[0] == 5
    assert b.mask.tolist() == [True, True, True, False, False]
    assert b.planes[3:].sum() == 0 and b.own_idx[:3].tolist() == [0, 1, 2]
    e = encode_game(game("mini_B"), 0, PlaneSampling("even", k=5))
    assert e.mask.sum() == 3 and e.own_idx[:3].tolist() == [0, 1, 2]
    r = encode_game(game("mini_B"), 0, PlaneSampling("random", k=5, seed=3))
    assert r.mask.tolist() == [True, True, True, False, False]


def test_sym_augmentation_consistent_across_game():
    g = long_game()
    base = encode_game(g, 0, PlaneSampling("even", k=8))
    for k in (1, 4, 6):
        aug = encode_game(g, 0, PlaneSampling("even", k=8), sym=k)
        assert np.array_equal(aug.planes, sym_augment(base, k).planes)
        for i in range(8):
            for layer in range(9):
                assert np.array_equal(aug.planes[i, layer], apply_sym_grid(k, base.planes[i, layer]))
            r, c = divmod(int(base.move_index[i]), 19)
            assert int(aug.move_index[i]) == apply_sym(k, r, c)[0] * 19 + apply_sym(k, r, c)[1]
            assert aug.planes[i, 8].reshape(-1)[aug.move_index[i]] == 1
