import math

import numpy as np
import pytest

from gogame.metrics import (
    balanced_weights, by_group, player_score, player_scores, rank_score, rank_score_matrix, rank_scores,
)

E1 = math.exp(-1)


def test_rank_score_exact_diff0():
    assert rank_scores(np.array([0, 5, 9]), np.array([0, 5, 9])).tolist() == [1, 1, 1]


def test_rank_score_diff1():
    # pred=A(3), truth=1D(4)
    assert rank_scores(np.array([3]), np.array([4]))[0] == pytest.approx(0.3678794412, abs=1e-10)


def test_rank_score_diff2_and_more_zero():
    assert rank_scores(np.array([0, 0, 4]), np.array([2, 9, 7])).tolist() == [0, 0, 0]


def test_rank_score_symmetric():
    for a in range(10):
        for b in range(10):
            assert rank_score(np.array([a]), np.array([b])) == rank_score(np.array([b]), np.array([a]))


def test_rank_score_mean():
    assert rank_score(np.array([0, 1, 2]), np.array([0, 0, 0])) == pytest.approx((1 + E1 + 0) / 3)


@pytest.mark.parametrize("bad", [10, -1])
def test_rank_score_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        rank_scores(np.array([bad]), np.array([0]))
    with pytest.raises(ValueError):
        rank_scores(np.array([0]), np.array([bad]))


def test_rank_score_empty_raises():
    with pytest.raises(ValueError):
        rank_score(np.array([], dtype=int), np.array([], dtype=int))


def test_player_score_positions():
    ids = np.array(["a", "b", "c", "d", "e"])
    expected = [1, 0.367879, 0.135335, 0.049787, 0.018316]
    for r in range(5):
        got = player_scores(ids[None, :], np.array([ids[r]]))[0]
        assert got == pytest.approx(expected[r], abs=1e-6)
        assert got == pytest.approx(math.exp(-r), abs=1e-12)


def test_player_score_absent_zero():
    ids = np.array([["a", "b", "c", "d", "e"]])
    assert player_scores(ids, np.array(["z"]))[0] == 0


def test_player_score_dup_ids_raises():
    with pytest.raises(ValueError):
        player_scores(np.array([["a", "b", "a", "d", "e"]]), np.array(["a"]))


def test_player_score_mean_over_questions():
    top5 = np.array([list("abcde")] * 4)
    truth = np.array(["a", "b", "z", "e"])
    assert player_score(top5, truth) == pytest.approx((1 + E1 + 0 + math.exp(-4)) / 4)


def test_ev_score_matrix():
    s = rank_score_matrix()
    assert s.shape == (10, 10)
    for j in range(10):
        for k in range(10):
            want = 1.0 if j == k else (E1 if abs(j - k) == 1 else 0.0)
            assert s[j, k] == pytest.approx(want, abs=1e-15)


def test_balanced_weights():
    w = balanced_weights(np.array([0, 0, 1, 2]))
    assert w == pytest.approx([2 / 3, 2 / 3, 4 / 3, 4 / 3])
    assert w.mean() == pytest.approx(1.0)


def test_by_group():
    df = by_group(np.array([1.0, 0.0, 0.5, 0.5]), np.array(["a", "a", "b", "b"]))
    d = df.set_index(df.columns[0])
    assert d.loc["a", "mean"] == 0.5 and d.loc["a", "n"] == 2
    assert d.loc["b", "mean"] == 0.5
