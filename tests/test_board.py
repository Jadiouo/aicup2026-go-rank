import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fixtures.handmade import HANDMADE
from gogame.board import Board, OwnPosition, iter_own_positions, replay, replay_summary
from gogame.sgf import parse_sgf
from helpers import sgf

sys.path.insert(0, str(Path(__file__).parent))
from tools.golden_from_sgfmill import replay_with_sgfmill  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "golden"


def summ(name_or_text):
    return replay_summary(parse_sgf(HANDMADE.get(name_or_text, name_or_text)))


def count(grid):
    return int((grid == 1).sum()), int((grid == 2).sum())


def results(name_or_text):
    g = parse_sgf(HANDMADE.get(name_or_text, name_or_text))
    return [(ply, b.copy(), r) for ply, b, r in replay(g)]


def test_single_capture():
    s = summ("capture_single")
    assert s.captures.tolist() == [0, 0, 0, 0, 0, 0, 1]
    assert s.grid[9, 9] == 0
    assert count(s.grid) == (4, 2)


def test_corner_capture():
    s = summ("capture_corner")
    assert s.captures.tolist() == [0, 0, 1]
    assert s.grid[0, 0] == 0
    assert count(s.grid) == (2, 0)


def test_multi_stone_capture():
    s = summ("capture_multi")
    assert s.captures.tolist() == [0] * 14 + [3]
    for c in (9, 10, 11):
        assert s.grid[9, c] == 0
    assert count(s.grid) == (8, 4)


def test_capture_does_not_remove_own_group():
    # In `ko` the capturing stone dj has zero liberties before the capture and exactly one afterwards.
    s = summ("ko")
    assert s.grid[9, 3] == 1          # dj stays
    assert s.grid[9, 2] == 0          # cj captured
    assert count(s.grid) == (5, 3)    # black: ci ck bj sa dj ; white: di dk ej (cj captured)


def test_ko_point_set():
    r = results("ko")
    assert [x[2].captured for x in r][-1] == 1
    assert r[8][2].ko_point == (9, 2)
    assert all(x[2].ko_point is None for x in r[:8])


def test_ko_point_cleared_after_other_move():
    r = results(HANDMADE["ko"][:-1] + ";W[ss])")
    assert r[8][2].ko_point == (9, 2)
    assert r[9][2].ko_point is None and not r[9][2].ko_violation


def test_ko_point_cleared_by_pass():
    r = results(HANDMADE["ko"][:-1] + ";W[])")
    assert r[9][2].ko_point is None


def test_ko_violation_counted():
    r = results("ko_recapture")
    res = r[9][2]
    assert res.captured == 1 and res.ko_violation is True
    assert res.ko_point == (9, 3)
    final = r[9][1].grid
    assert count(final) == (4, 4)       # replayed and captured anyway
    assert summ("ko_recapture").counts["ko_violation"] == 1


def test_no_ko_when_capturer_has_liberties():
    r = results("capture_single")
    assert r[-1][2].captured == 1 and r[-1][2].ko_point is None


def test_suicide_recorded_illegal_and_removed():
    r = results("suicide")
    res = r[4][2]
    assert res.suicide is True and res.self_captured == 1 and res.captured == 0
    s = summ("suicide")
    assert s.grid[0, 0] == 0
    assert count(s.grid) == (2, 2)
    assert s.counts["suicide"] == 1 and s.counts["occupied"] == 0
    assert s.self_captures.tolist() == [0, 0, 0, 0, 1]


MULTI_SUICIDE = sgf("B:aa W:ca B:ba W:bb B:sa W:ac B:ab")


@pytest.mark.parametrize("text", [HANDMADE["suicide"], MULTI_SUICIDE])
def test_suicide_matches_sgfmill(text):
    ref = replay_with_sgfmill(text)
    s = replay_summary(parse_sgf(text))
    assert s.captures.tolist() == ref["captures_per_ply"]
    assert s.self_captures.tolist() == ref["self_captures_per_ply"]
    assert count(s.grid) == (ref["final_black"], ref["final_white"])
    assert hashlib.sha256(s.grid.astype(np.int8).tobytes()).hexdigest() == ref["final_grid_hash"]


def test_multi_stone_suicide_values():
    s = replay_summary(parse_sgf(MULTI_SUICIDE))
    assert s.self_captures.tolist()[-1] == 3
    assert s.counts["suicide"] == 1
    assert count(s.grid) == (1, 3)


def test_illegal_counts_reported():
    text = HANDMADE["ko_recapture"][:-1] + "".join(
        [";B[sb];W[ba];B[ss];W[ab];B[aa];B[pd];W[pd]"]) + ")"
    s = replay_summary(parse_sgf(text))
    assert s.counts == {"suicide": 1, "occupied": 1, "ko_violation": 1}
    assert s.illegal == 2   # suicide + occupied (ko violations are reported separately)


def test_occupied_flagged():
    r = results(sgf("B:aa W:aa"))
    assert r[0][2].occupied is False
    assert r[1][2].occupied is True


def test_pass_does_not_change_grid():
    g = parse_sgf(HANDMADE["pass_variants"])
    prev = None
    for ply, board, res in replay(g):
        if g.moves[ply][1] < 0:
            assert np.array_equal(board.grid, prev)
            assert res.captured == 0 and not res.suicide and not res.occupied
        prev = board.grid.copy()


def test_row_col_not_transposed_in_grid():
    s = summ(sgf("B:pd"))
    assert s.grid[3, 15] == 1 and s.grid[15, 3] == 0


def test_board_copy_independent():
    b = Board()
    b.play(0, 3, 3)
    c = b.copy()
    c.play(1, 4, 4)
    assert b.grid[4, 4] == 0 and c.grid[4, 4] == 2 and c.grid[3, 3] == 1


def test_setup_stones_on_initial_board():
    g = parse_sgf(HANDMADE["setup_stones"])
    s = replay_summary(g)
    assert count(s.grid) == (2, 1)
    pos = list(iter_own_positions(g, 1))
    assert int((pos[0].grid_own == 2).sum()) == 2 and int((pos[0].grid_own == 1).sum()) == 0


# ---- own-perspective positions -------------------------------------------------------

def own(name, color):
    return list(iter_own_positions(parse_sgf(HANDMADE.get(name, name)), color))


def test_grid_own_black_perspective():
    p = own("mini_B", 0)[2]       # third black move kj, before the move
    assert p.grid_own[3, 15] == 1 and p.grid_own[15, 15] == 1     # pd, pp are own
    assert p.grid_own[15, 3] == 2 and p.grid_own[3, 3] == 2       # dp, dd are enemy
    assert p.grid_own[9, 10] == 0                                  # board BEFORE the move
    assert p.move == (9, 10) and p.ply == 4 and p.own_idx == 2


def test_grid_own_white_is_color_swap_not_flip():
    p = own("mini_B", 1)[1]       # second white move dd (ply 3)
    assert p.grid_own[15, 3] == 1                    # dp is white -> own
    assert p.grid_own[3, 15] == 2 and p.grid_own[15, 15] == 2     # black stones -> enemy, same place
    assert p.grid_own[3, 3] == 0                                   # dd not yet played
    assert int((p.grid_own == 1).sum()) == 1 and int((p.grid_own == 2).sum()) == 2
    assert p.ply == 3 and p.own_idx == 1


def test_iter_own_positions_counts():
    assert len(own("mini_B", 0)) == 3
    assert len(own("mini_B", 1)) == 2
    assert [p.ply for p in own("mini_B", 0)] == [0, 2, 4]


def test_prev_opp_none_for_first_move():
    assert own("mini_B", 0)[0].prev_opp is None
    assert own("mini_B", 1)[0].prev_opp == (3, 15)        # black's first move
    assert own("mini_B", 1)[1].prev_opp == (15, 15)


def test_prev_opp_none_after_pass():
    assert [p.prev_opp for p in own("pass_variants", 0)] == [None, None, None]   # W passed twice
    assert [p.prev_opp for p in own("pass_variants", 1)] == [(3, 15), (3, 3)]


def test_prev_own_skips_pass():
    ps = own(sgf("B:pd W:dp B:dd W: B:pp W:cc"), 1)
    assert [p.prev_own for p in ps] == [None, (15, 3), (15, 3)]
    assert [p.move for p in ps] == [(15, 3), (-1, -1), (2, 2)]
    assert [p.own_idx for p in ps] == [0, 1, 2]


def test_captured_and_lost_since_prev_own():
    ps = own(sgf("B:ji W:jj B:ij W:aa B:kj W:ab B:jk W:ss"), 1)
    assert [p.lost_since_prev_own for p in ps] == [0, 0, 0, 1]
    assert [p.captured for p in ps] == [0, 0, 0, 0]
    assert own("capture_single", 0)[-1].captured == 1


def test_own_position_ko_point():
    ps = own("ko_recapture", 1)
    assert ps[-1].ply == 9 and ps[-1].ko_point == (9, 2)
    assert all(p.ko_point is None for p in ps[:-1])


def test_own_position_is_frozen_dataclass():
    p = own("mini_B", 0)[0]
    assert isinstance(p, OwnPosition)
    with pytest.raises(Exception):
        p.ply = 3  # type: ignore[misc]


# ---- golden ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden():
    games = pd.read_csv(GOLDEN / "games_50.csv", dtype=str, keep_default_na=False)
    ref = json.loads((GOLDEN / "golden_board.json").read_text())
    return games, ref


def test_stone_count_conservation_golden(golden):
    games, _ = golden
    for t in games["sgf_content"]:
        g = parse_sgf(t)
        s = replay_summary(g)
        n_moves = int((g.moves[:, 1] >= 0).sum())
        b, w = count(s.grid)
        assert s.counts["occupied"] == 0
        assert b + w == n_moves - int(s.captures.sum()) - int(s.self_captures.sum())


def test_golden_matches_sgfmill(golden):
    games, ref = golden
    assert len(games) == 50 and len(ref) == 50
    n_caps = 0
    for row in games.itertuples():
        uid = f"{row.rank}_{row.game_id}"
        exp = ref[uid]
        s = replay_summary(parse_sgf(row.sgf_content))
        assert len(s.captures) == exp["n_plies"]
        assert s.captures.tolist() == exp["captures_per_ply"], uid
        assert s.self_captures.tolist() == exp["self_captures_per_ply"], uid
        assert count(s.grid) == (exp["final_black"], exp["final_white"]), uid
        assert hashlib.sha256(s.grid.astype(np.int8).tobytes()).hexdigest() == exp["final_grid_hash"], uid
        n_caps += int(s.captures.sum())
    assert n_caps > 100    # the golden sample really exercises captures


@pytest.mark.real_data
def test_real_sample_illegal_rare(training_dir):
    from gogame.constants import RANKS
    tot = {"suicide": 0, "occupied": 0, "ko_violation": 0}
    n = 0
    for r in RANKS:
        df = pd.read_csv(training_dir / f"train_{r}.csv", dtype=str, keep_default_na=False, nrows=100)
        for t in df["sgf_content"]:
            for k, v in replay_summary(parse_sgf(t)).counts.items():
                tot[k] += v
            n += 1
    print("real-sample illegal counts over", n, "games:", tot)
    assert n == 1000
    assert tot["suicide"] + tot["occupied"] <= 1, tot
