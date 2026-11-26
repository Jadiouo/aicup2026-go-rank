import numpy as np
import pytest

from gogame.io import GameRecord
from gogame.overlap import (build_prefix_index, build_train_index, bag_centroid_shift, decide_overlap, exact_overlap,
                            game_key, library_centroid_check, prefix_key, prefix_overlap)
from helpers import random_game, sgf


def rec(s, color=0, uid="t#1"):
    return GameRecord(uid=uid, player_id="", rank_id=None, color=color, sgf=s, source="test_pub")


def test_game_key_ignores_header_whitespace():
    a = "(;SZ[19];B[pd];W[dp];B[pp])"
    b = "(;GM[1]SZ[19]PB[x]\n;B[pd] ;W[dp]\n;B[pp];)"
    assert game_key(a) == game_key(b)
    assert game_key(a) != game_key("(;SZ[19];B[pd];W[dp];B[dd])")
    assert game_key("(;SZ[19];B[pd];W[tt])") == game_key("(;SZ[19];B[pd];W[])")       # pass spelled two ways
    assert len(game_key(a)) == 40


def test_exact_overlap_counts_and_color_split():
    rng = np.random.default_rng(0)
    train = [random_game(rng, 0.5, 80) for _ in range(20)]
    tcol = [i % 2 for i in range(20)]                       # train colours
    idx, colors = build_train_index(train, tcol)
    fresh = [random_game(rng, 0.5, 80) for _ in range(7)]
    test = [rec(train[0], tcol[0]),                          # same game, same colour
            rec(train[2], tcol[2]),                          # same game, same colour
            rec(train[1], 1 - tcol[1]),                      # same game, other colour -> the opponent side
            ] + [rec(s, 0) for s in fresh]
    st = exact_overlap(test, idx, colors)
    assert st.n_test == 10 and st.n_exact == 3 and st.n_same_color == 2 and st.n_other_side == 1
    assert st.exact_frac == pytest.approx(0.3) and st.same_color_frac == pytest.approx(0.2)


def test_exact_overlap_disjoint_all_zero():
    rng = np.random.default_rng(1)
    idx, colors = build_train_index([random_game(rng, 0.5, 70) for _ in range(10)], [0] * 10)
    st = exact_overlap([rec(random_game(rng, 0.5, 70)) for _ in range(10)], idx, colors)
    assert (st.n_exact, st.n_same_color, st.n_other_side) == (0, 0, 0)
    assert exact_overlap([], idx, colors).same_color_frac == 0.0


def _seq(n, last=None):
    pts = [f"{chr(97 + (i * 7) % 19)}{chr(97 + (i * 3) % 19)}" for i in range(n)]
    return pts


def _game(pts):
    return "(;SZ[19]" + "".join(f";{'BW'[i % 2]}[{p}]" for i, p in enumerate(pts)) + ")"


def test_prefix_overlap_boundary_29_vs_30():
    base = _seq(60)
    tr = _game(base)
    differ30 = base[:29] + ["ss"] + base[30:]      # first 29 plies equal, 30th differs
    same30 = base[:30] + ["ss"] + base[31:]        # first 30 plies equal
    pi = build_prefix_index([tr], ks=(6, 10, 16, 24, 30))
    st = prefix_overlap([rec(_game(differ30))], pi, k=30)
    assert st.n_match[30] == 0 and st.n_match[24] == 1
    st = prefix_overlap([rec(_game(same30))], pi, k=30)
    assert st.n_match[30] == 1 and st.frac[30] == 1.0


def test_prefix_overlap_reports_reference_ks():
    base = _seq(60)
    tr = _game(base)
    pi = build_prefix_index([tr], ks=(6, 10, 16, 24, 30))
    # shares exactly 12 plies
    other = _game(base[:12] + ["ss", "sr"] + base[14:])
    st = prefix_overlap([rec(other)], pi, k=30)
    assert set(st.frac) == {6, 10, 16, 24, 30}
    assert st.frac[6] == 1.0 and st.frac[10] == 1.0 and st.frac[16] == 0.0 and st.frac[24] == 0.0 and st.frac[30] == 0.0


def test_prefix_short_game_cannot_match():
    base = _seq(60)
    pi = build_prefix_index([_game(base)], ks=(30,))
    assert prefix_key(_game(base[:29]), 30) is None
    assert prefix_overlap([rec(_game(base[:29]))], pi, k=30).n_match[30] == 0


def test_prefix_requires_k_in_index():
    pi = build_prefix_index([_game(_seq(40))], ks=(6,))
    with pytest.raises(ValueError):
        prefix_overlap([rec(_game(_seq(40)))], pi, k=30)


# ------------------------------------------------------------------ centroid shift
def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_bag_centroid_shift_p99_fraction():
    rng = np.random.default_rng(0)
    D = 32
    train = unit(rng.normal(size=(300, D)))
    held = unit(rng.normal(size=(3000, D)))                         # unseen players
    dup = unit(train[rng.integers(0, 300, 3000)] + 0.05 * rng.normal(size=(3000, D)))   # near copies
    indep = unit(rng.normal(size=(3000, D)))
    s_dup = bag_centroid_shift(dup, train, held)
    s_ind = bag_centroid_shift(indep, train, held)
    assert s_dup.p99_frac > 0.95 and s_dup.ks_pvalue < 0.01
    assert 0.0 <= s_ind.p99_frac < 0.03 and s_ind.ks_pvalue > 0.05
    assert s_dup.n_test == 3000 and set(s_dup.quantiles) >= {0.5, 0.99}
    assert s_dup.holdout_p99 == pytest.approx(np.quantile(s_dup.holdout_max_sim, 0.99))


def test_library_centroid_check():
    rng = np.random.default_rng(1)
    D = 32
    train = unit(rng.normal(size=(400, D)))
    held = unit(rng.normal(size=(2000, D)))
    cand_mixed = np.vstack([unit(train[:10] + 0.05 * rng.normal(size=(10, D))), unit(rng.normal(size=(10, D)))])
    st = library_centroid_check(cand_mixed, train, held)
    assert st.p99_frac == pytest.approx(0.5, abs=0.1) and st.n_test == 20
    big = library_centroid_check(unit(rng.normal(size=(4000, D))), train, held)
    assert big.p99_frac < 0.03


def test_zero_vectors_do_not_crash_or_overlap():
    rng = np.random.default_rng(2)
    train = unit(rng.normal(size=(20, 8)))
    st = bag_centroid_shift(np.zeros((5, 8)), train, unit(rng.normal(size=(50, 8))))
    assert np.isfinite(st.p99_frac) and st.p99_frac == 0.0


# ------------------------------------------------------------------ decision rule
BASE = dict(exact_same_color_frac=0.0, prefix30_frac=0.0, nn_p99_frac=0.01, library_p99_frac=0.01)


@pytest.mark.parametrize("key,lo,hi", [("exact_same_color_frac", 0.009, 0.01), ("prefix30_frac", 0.009, 0.01),
                                       ("nn_p99_frac", 0.049, 0.05), ("library_p99_frac", 0.049, 0.05)])
def test_overlap_decision_or_rule(key, lo, hi):
    assert decide_overlap({**BASE, key: lo}).overlap is False
    d = decide_overlap({**BASE, key: hi})
    assert d.overlap is True and d.label == "待人工確認" and d.triggered[key] is True
    assert sum(d.triggered.values()) == 1                      # only this criterion fired


def test_overlap_decision_none_and_cfg():
    d = decide_overlap(BASE)
    assert d.overlap is False and d.label == "否" and not any(d.triggered.values())
    assert decide_overlap({**BASE, "prefix30_frac": 0.02}, {"prefix30_frac_thr": 0.05}).overlap is False
    assert decide_overlap({**BASE, "prefix30_frac": 0.02}, {"prefix30_frac_thr": 0.02}).overlap is True
    d = decide_overlap({"exact_same_color_frac": 0.0, "prefix30_frac": 0.0})            # (c),(d) not evaluated
    assert d.overlap is False and set(d.not_evaluated) == {"nn_p99_frac", "library_p99_frac"}
    assert d.evidence["prefix30_frac"] == 0.0


def test_overlap_same_player_different_games_detected():
    """Test players are training players but no game is shared: (a)(b) are 0, only the centroid evidence fires."""
    rng = np.random.default_rng(3)
    D = 32
    train_games = [random_game(rng, 0.5, 90) for _ in range(30)]
    idx, colors = build_train_index(train_games, [0] * 30)
    pi = build_prefix_index(train_games, ks=(30,))
    test = [rec(random_game(rng, 0.5, 90)) for _ in range(30)]
    ex, pf = exact_overlap(test, idx, colors), prefix_overlap(test, pi, k=30)
    train_c = unit(rng.normal(size=(200, D)))
    held = unit(rng.normal(size=(2000, D)))
    bags = unit(train_c[rng.integers(0, 200, 500)] + 0.05 * rng.normal(size=(500, D)))
    sh = bag_centroid_shift(bags, train_c, held)
    d = decide_overlap({"exact_same_color_frac": ex.same_color_frac, "prefix30_frac": pf.frac[30],
                        "nn_p99_frac": sh.p99_frac})
    assert ex.same_color_frac == 0 and pf.frac[30] == 0
    assert d.overlap and d.triggered["nn_p99_frac"] and not d.triggered["exact_same_color_frac"]
