import numpy as np
import pandas as pd
import pytest

from gogame.errors import GogameError
from gogame.seeds import derive_seed
from gogame.split import (PlayerLeakError, assert_no_player_leak, make_folds, make_player_holdout,
                          make_split_table, read_split, write_split)
from helpers import make_corpus
from helpers_batch2 import heavy_players_table, players_table


def _players(ppr=12, heavy=False):
    c = make_corpus(1, players_per_rank=ppr, games_per_player=4, heavy=heavy, n_heavy=ppr // 2)
    return players_table(c)


def test_each_player_one_fold():
    p = _players()
    f = make_folds(p, 5)
    assert len(f) == len(p) and f["player_id"].is_unique
    assert set(f["player_id"]) == set(p["player_id"])
    assert f["fold"].between(0, 4).all() and f["fold"].dtype == np.int8


def test_rank_stratified_balanced():
    for ppr in (12, 13, 17, 23):
        f = make_folds(_players(ppr), 5)
        cnt = f.groupby(["rank_id", "fold"]).size().unstack(fill_value=0)
        assert (cnt.max(axis=1) - cnt.min(axis=1)).max() <= 1


def test_row_order_invariant():
    p = _players()
    a = make_folds(p, 5)
    b = make_folds(p.sample(frac=1.0, random_state=3).reset_index(drop=True), 5)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_seed_reproducible_and_sensitive():
    p = _players()
    a, b = make_folds(p, 5, seed=1), make_folds(p, 5, seed=1)
    c = make_folds(p, 5, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["fold"].equals(c["fold"])


def test_default_seed_is_named_split_seed():
    p = _players()
    pd.testing.assert_frame_equal(make_folds(p, 5), make_folds(p, 5, seed=derive_seed(2026, "split")))


def test_assert_no_player_leak():
    assert_no_player_leak({"a", "b"}, {"c"})
    with pytest.raises(PlayerLeakError) as e:
        assert_no_player_leak({"a", "b"}, {"b", "c"})
    assert isinstance(e.value, GogameError)


def test_small_rank_group_warns_and_does_not_crash():
    p = pd.DataFrame({"player_id": ["a", "b", "c"], "rank_id": np.int8([0, 0, 0])})
    with pytest.warns(UserWarning):
        f = make_folds(p, 5)
    assert f["fold"].nunique() == 3


def test_duplicate_player_rows_rejected():
    p = pd.DataFrame({"player_id": ["a", "a"], "rank_id": np.int8([0, 1])})
    with pytest.raises(GogameError):
        make_folds(p, 5)


def test_player_holdout_only_heavy_players():
    pt = heavy_players_table()
    h = make_player_holdout(pt, min_black=120, n_holdout=20)
    heavy = set(pt.loc[pt["n_black"] >= 120, "player_id"])
    assert h and h <= heavy
    # a threshold nobody meets -> nobody in H
    assert make_player_holdout(pt, min_black=10_000, n_holdout=20) == set()


def test_player_holdout_size_stratified_deterministic():
    pt = heavy_players_table()
    h = make_player_holdout(pt, min_black=120, n_holdout=20)
    assert len(h) == 20
    per_rank = pt[pt["player_id"].isin(h)].groupby("rank_id").size()
    assert (per_rank == 2).all() and len(per_rank) == 10          # half of 4 heavy per rank
    assert h == make_player_holdout(pt.sample(frac=1, random_state=0), 120, 20)
    assert h != make_player_holdout(pt, 120, 20, seed=5)
    assert make_player_holdout(pt, 120, 20, seed=5) == make_player_holdout(pt, 120, 20, seed=5)


def test_player_holdout_default_seed_name():
    pt = heavy_players_table()
    assert make_player_holdout(pt, 120, 20) == make_player_holdout(pt, 120, 20, seed=derive_seed(2026, "player_holdout"))


def test_rank_folds_exclude_holdout_and_table():
    pt = heavy_players_table()
    tab = make_split_table(pt, n_folds=5, min_black=120, n_holdout=20)
    h = set(tab.loc[tab["is_holdout"], "player_id"])
    assert len(h) == 20
    assert ((tab["fold"] == -1) == tab["is_holdout"]).all()
    assert set(tab.loc[~tab["is_holdout"], "fold"]) <= set(range(5))
    assert len(tab) == len(pt) and tab["player_id"].is_unique
    assert list(tab.columns) == ["player_id", "rank_id", "fold", "is_holdout", "n_black"]
    # stratified balance among non-H
    cnt = tab[~tab["is_holdout"]].groupby(["rank_id", "fold"]).size().unstack(fill_value=0)
    assert (cnt.max(axis=1) - cnt.min(axis=1)).max() <= 1


@pytest.mark.parametrize("k", [3, 5])
def test_folds_k3_and_k5_valid(k):
    tab = make_split_table(heavy_players_table(), n_folds=k, min_black=120, n_holdout=20)
    nh = tab[~tab["is_holdout"]]
    assert nh["fold"].nunique() == k
    cnt = nh.groupby(["rank_id", "fold"]).size().unstack(fill_value=0)
    assert (cnt.max(axis=1) - cnt.min(axis=1)).max() <= 1


def test_any_two_folds_and_holdout_disjoint():
    tab = make_split_table(heavy_players_table(), n_folds=5, min_black=120, n_holdout=20)
    sets = [set(tab.loc[tab["fold"] == k, "player_id"]) for k in range(5)] + [set(tab.loc[tab["is_holdout"], "player_id"])]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j])
            assert_no_player_leak(sets[i], sets[j])


def test_split_parquet_roundtrip(tmp_path):
    tab = make_split_table(heavy_players_table(), n_folds=5, min_black=120, n_holdout=20)
    write_split(tab, tmp_path / "split.parquet")
    back = read_split(tmp_path / "split.parquet")
    pd.testing.assert_frame_equal(tab, back)
    import pyarrow.parquet as pq
    assert pq.read_schema(tmp_path / "split.parquet").metadata[b"schema_version"] == b"1"
    # inconsistent file is refused
    bad = tab.copy()
    bad.loc[bad.index[0], "is_holdout"] = ~bad.loc[bad.index[0], "is_holdout"]
    write_split(bad, tmp_path / "bad.parquet")
    with pytest.raises(GogameError):
        read_split(tmp_path / "bad.parquet")
