"""games_from_moves must survive games with <=20 own moves (no 'late' rows); found in the 2026-10-30 school smoke run."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "katago"))
import scan_runner as sr  # noqa: E402

PROF = ["rank_12k", "rank_9k", "rank_6k", "rank_3k", "rank_1k", "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d"]


def _mv(uid, n, seed):
    rng = np.random.default_rng(seed)
    d = {"uid": uid, "own_move_idx": np.arange(n), "ply": np.arange(n) * 2}
    for i in range(len(PROF)):
        d[f"logp_{i}"] = rng.normal(-2, 1, n)
        d[f"rank_{i}"] = rng.integers(1, 20, n)
        d[f"ent_{i}"] = rng.random(n)
    return pd.DataFrame(d)


def test_short_game_does_not_crash_and_has_no_nan():
    long_, short = _mv("a", 60, 1), _mv("b", 15, 2)
    gm = sr.games_from_moves(pd.concat([long_, short], ignore_index=True), PROF)
    assert list(gm.uid) == ["a", "b"] and not gm.drop(columns="uid").isna().any().any()
    b = gm[gm.uid == "b"].iloc[0]
    for p in PROF:                                   # no late rows -> late falls back to the whole-game mean
        assert np.isclose(b[f"mllate_{p}"], b[f"ml_{p}"], atol=1e-5)
        assert np.isclose(b[f"mlearly_{p}"], b[f"ml_{p}"], atol=1e-5)
    assert np.isclose(b["ml_exp_idx_late"], b["ml_exp_idx"], atol=1e-4)


def test_normal_game_unchanged_by_short_neighbour():
    long_, short = _mv("a", 60, 1), _mv("b", 15, 2)
    alone = sr.games_from_moves(long_, PROF).iloc[0]
    both = sr.games_from_moves(pd.concat([long_, short], ignore_index=True), PROF)
    pd.testing.assert_series_equal(both[both.uid == "a"].iloc[0], alone, check_names=False)
