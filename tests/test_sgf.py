import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fixtures.handmade import HANDMADE
from gogame.errors import DataFormatError
from gogame.sgf import coord_to_rc, parse_sgf, rc_to_coord
from helpers import sgf

GOLDEN_CSV = Path(__file__).parent / "fixtures" / "golden" / "games_50.csv"


def moves_of(name_or_text):
    text = HANDMADE.get(name_or_text, name_or_text)
    return [tuple(int(x) for x in m) for m in parse_sgf(text).moves]


def test_coord_not_transposed_pd():
    assert moves_of(sgf("B:pd")) == [(0, 3, 15)]
    assert coord_to_rc("pd") == (3, 15)


def test_coord_asymmetric_aj():
    assert moves_of(sgf("B:pd W:aj"))[1] == (1, 9, 0)
    assert moves_of("transpose_probe") == [(0, 3, 15), (1, 9, 0)]
    assert coord_to_rc("aj") == (9, 0)


def test_coord_roundtrip():
    letters = "abcdefghijklmnopqrs"
    n = 0
    for a in letters:
        for b in letters:
            s = a + b
            r, c = coord_to_rc(s)
            assert 0 <= r < 19 and 0 <= c < 19
            assert rc_to_coord(r, c) == s
            n += 1
    assert n == 361


def test_pass_empty_brackets():
    assert moves_of("(;SZ[19];B[pd];W[])")[1] == (1, -1, -1)
    assert coord_to_rc("") == (-1, -1)


def test_pass_tt():
    assert moves_of("(;SZ[19];B[pd];W[tt])")[1] == (1, -1, -1)
    assert coord_to_rc("tt") == (-1, -1)


def test_pass_variants():
    g = parse_sgf(HANDMADE["pass_variants"])
    assert len(g.moves) == 5
    assert tuple(g.moves[1]) == (1, -1, -1) and tuple(g.moves[3]) == (1, -1, -1)
    assert g.flags.n_pass == 2


def test_edge_ss_not_pass():
    g = parse_sgf(HANDMADE["edge_coords"])
    assert tuple(g.moves[0]) == (0, 18, 18)
    assert tuple(g.moves[1]) == (1, -1, -1)
    assert g.flags.bad_coord == 0


def test_bad_coord_flagged():
    g = parse_sgf(HANDMADE["bad_coord"])
    assert g.flags.bad_coord == 1
    assert len(g.moves) == 1 and tuple(g.moves[0]) == (0, -1, -1)
    with pytest.raises(DataFormatError):
        coord_to_rc("ta")


def test_move_count_matches_regex_count():
    texts = list(HANDMADE.values()) + list(pd.read_csv(GOLDEN_CSV, dtype=str)["sgf_content"])
    assert len(texts) == 16 + 50
    for t in texts:
        assert len(parse_sgf(t).moves) == t.count(";B[") + t.count(";W[")


def test_ignore_AB_AW_as_moves():
    g = parse_sgf("(;SZ[19]AB[dd][pp]AW[dp];B[aa])")
    assert len(g.moves) == 1


def test_setup_stones_parsed():
    g = parse_sgf(HANDMADE["setup_stones"])
    assert g.flags.has_setup
    assert g.setup.shape == (2, 3)
    assert sorted(map(tuple, g.setup.tolist())) == [(0, 3, 3), (0, 15, 15)]
    assert len(g.moves) == 1
    g2 = parse_sgf("(;SZ[19]AW[dp];B[aa])")
    assert g2.setup.tolist() == [[1, 15, 3]]
    assert parse_sgf(HANDMADE["mini_B"]).setup.shape == (0, 3)


def test_flags_nonalt_first_not_black():
    f = parse_sgf(HANDMADE["nonalt"]).flags
    assert f.nonalt and not f.first_not_black
    f = parse_sgf(HANDMADE["white_first"]).flags
    assert f.first_not_black and not f.nonalt
    f = parse_sgf(HANDMADE["mini_B"]).flags
    assert not (f.nonalt or f.first_not_black or f.bad_size or f.has_setup) and f.bad_coord == 0 and f.n_pass == 0


def test_empty_game_no_exception():
    g = parse_sgf(HANDMADE["empty_game"])
    assert g.moves.shape == (0, 3) and g.setup.shape == (0, 3)
    assert parse_sgf("").moves.shape == (0, 3)


def test_bad_size_flag_keeps_data():
    g = parse_sgf(HANDMADE["bad_size"])
    assert g.flags.bad_size
    assert len(g.moves) == 1


def test_whitespace_newlines_tolerated():
    t = "(;SZ[19]\n;B[pd]\n ;W[dp]\r\n; B[pp])"
    assert moves_of(t) == [(0, 3, 15), (1, 15, 3), (0, 15, 15)]


def test_moves_dtype():
    g = parse_sgf(HANDMADE["mini_B"])
    assert g.moves.dtype == np.int16 and g.moves.shape == (5, 3)


@pytest.mark.parametrize("bad", [None, 3, b"(;B[aa])"])
def test_non_str_raises(bad):
    with pytest.raises(DataFormatError):
        parse_sgf(bad)


@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_flags_all_zero(training_dir):
    rng = np.random.default_rng(0)
    total = {"bad_size": 0, "has_setup": 0, "n_pass": 0, "nonalt": 0, "first_not_black": 0, "bad_coord": 0}
    n = 0
    for rank in ("D", "3D"):
        df = pd.read_csv(training_dir / f"train_{rank}.csv", dtype=str, keep_default_na=False, usecols=["sgf_content"])
        for t in df["sgf_content"].iloc[rng.choice(len(df), 2500, replace=False)]:
            g = parse_sgf(t)
            assert len(g.moves) == t.count(";B[") + t.count(";W[")
            for k in total:
                total[k] += int(getattr(g.flags, k))
            n += 1
    assert n == 5000
    assert all(v == 0 for v in total.values()), total
