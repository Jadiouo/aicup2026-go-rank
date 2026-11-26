import numpy as np
import pandas as pd
import pytest

from gogame.errors import ConfigError, DataFormatError
from gogame.questions import (Bag, bag_class_weights, build_player_index, build_player_pool_sim, build_val_questions,
                              n_class_report, sample_train_bags, to_questions_frame)
from gogame.split import PlayerLeakError, assert_no_player_leak, make_folds


def synth_meta(spec):
    """spec: {player: (rank_id, n_black, n_white)} -> meta frame (uid, player_id, rank_id, color_id)."""
    rows = []
    for p, (rid, nb, nw) in spec.items():
        for c, n in ((0, nb), (1, nw)):
            for i in range(n):
                rows.append((f"{rid}_{p}_{'BW'[c]}{i:03d}", p, rid, c))
    return pd.DataFrame(rows, columns=["uid", "player_id", "rank_id", "color_id"])


def rich_spec(per_rank=6, n_ranks=10, nb=40, nw=40):
    return {f"p{r}_{i}": (r, nb, nw) for r in range(n_ranks) for i in range(per_rank)}


@pytest.fixture(scope="module")
def world():
    meta = synth_meta(rich_spec())
    idx = build_player_index(meta)
    pl = pd.DataFrame({"player_id": sorted(set(meta["player_id"])), "rank_id": [int(p[1]) for p in sorted(set(meta["player_id"]))]})
    folds = make_folds(pl, 5)
    return meta, idx, folds


def fold_players(folds, k):
    return set(folds.loc[folds["fold"] == k, "player_id"])


def test_val_questions_only_from_fold_players(world):
    _, idx, folds = world
    for k in range(5):
        qs = build_val_questions(idx, fold_players(folds, k))
        assert qs and {q.player_id for q in qs} <= fold_players(folds, k)


def test_train_bags_exclude_val_players_and_games(world):
    meta, idx, folds = world
    k = 2
    val = fold_players(folds, k)
    train = set(folds["player_id"]) - val
    tb = sample_train_bags(idx, train, bags_per_player_color=2)
    vb = build_val_questions(idx, val)
    assert not ({b.player_id for b in tb} & {b.player_id for b in vb})
    assert not ({u for b in tb for u in b.uids} & {u for b in vb for u in b.uids})
    assert_no_player_leak({b.player_id for b in tb}, val)


def test_no_player_in_both_train_and_val_for_every_fold(world):
    _, idx, folds = world
    allp = set(folds["player_id"])
    for k in range(5):
        val = fold_players(folds, k)
        tb = sample_train_bags(idx, allp - val, bags_per_player_color=1)
        vb = build_val_questions(idx, val)
        assert not ({b.player_id for b in tb} & {b.player_id for b in vb})


def test_sample_train_bags_rejects_nothing_outside_given_players(world):
    _, idx, folds = world
    some = set(sorted(folds["player_id"])[:7])
    tb = sample_train_bags(idx, some, bags_per_player_color=2)
    assert {b.player_id for b in tb} <= some


def test_bag_same_color_no_dup_and_valid(world):
    meta, idx, _ = world
    color = dict(zip(meta["uid"], meta["color_id"]))
    pl = dict(zip(meta["uid"], meta["player_id"]))
    for b in sample_train_bags(idx, set(meta["player_id"]), bags_per_player_color=2) + build_val_questions(idx, set(meta["player_id"])):
        assert len(set(b.uids)) == len(b.uids) == b.n
        assert {color[u] for u in b.uids} == {b.color}
        assert {pl[u] for u in b.uids} == {b.player_id}


def test_val_n_values(world):
    _, idx, _ = world
    qs = build_val_questions(idx, set(idx.rank))
    assert {q.n for q in qs} == {5, 10, 15, 20}


def test_at_most_one_question_per_player_per_n(world):
    _, idx, _ = world
    qs = build_val_questions(idx, set(idx.rank))
    df = pd.DataFrame({"p": [q.player_id for q in qs], "n": [q.n for q in qs]})
    assert df.groupby(["p", "n"]).size().max() == 1
    assert len(df) == 4 * len(idx.rank)           # every rich player gets all four Ns
    assert len({q.qid for q in qs}) == len(qs)


def test_val_color_choice_requires_enough_games():
    idx = build_player_index(synth_meta({"x": (3, 3, 12)}))
    qs = {q.n: q for q in build_val_questions(idx, {"x"})}
    assert qs[10].color == 1 and qs[5].color == 1
    assert 15 not in qs and 20 not in qs
    assert set(qs) == {5, 10}


def test_val_both_colors_possible_picks_either_deterministically():
    idx = build_player_index(synth_meta({"x": (3, 30, 30), "y": (3, 30, 30), "z": (3, 30, 30)}))
    a = build_val_questions(idx, {"x", "y", "z"})
    b = build_val_questions(idx, {"z", "y", "x"})
    assert [(q.qid, q.color, q.uids) for q in sorted(a, key=lambda q: q.qid)] == \
           [(q.qid, q.color, q.uids) for q in sorted(b, key=lambda q: q.qid)]
    assert {q.color for q in a} == {0, 1}  # some of each in 12 draws (seeded; sanity)


def test_train_n_covers_5_to_20(world):
    _, idx, _ = world
    tb = sample_train_bags(idx, set(idx.rank), bags_per_player_color=8)
    assert {b.n for b in tb} == set(range(5, 21))
    assert all(5 <= b.n <= 20 for b in tb)


def test_train_bag_n_first_no_truncation():
    spec = {"short": (0, 7, 7), "long": (1, 25, 25), "tiny": (2, 4, 4)}
    idx = build_player_index(synth_meta(spec))
    seq = [10, 20, 5, 7, 12, 25 - 5, 6] * 20
    tb = sample_train_bags(idx, set(spec), bags_per_player_color=3, n_sequence=iter(seq))
    assert all(b.n == len(b.uids) for b in tb)
    assert all(b.player_id != "tiny" for b in tb)                       # <5 games per colour: never a bag
    for b in tb:
        if b.n > 7:
            assert b.player_id == "long"
        if b.n > 25:
            raise AssertionError
    assert any(b.player_id == "short" for b in tb) and any(b.n >= 10 for b in tb)
    # N exactly as injected, in order (dedup redraws only the games, not N)
    assert [b.n for b in tb] == seq[:len(tb)]


def test_train_bag_no_candidates_for_n_redraws_then_errors():
    idx = build_player_index(synth_meta({"a": (0, 6, 0)}))
    # every drawn N is larger than any group -> redraw 100 times, then give up
    with pytest.raises(DataFormatError):
        sample_train_bags(idx, {"a"}, bags_per_player_color=1, n_sequence=iter([10] * 1000))
    assert sample_train_bags(idx, {"a"}, bags_per_player_color=1, n_range=(10, 12)) == []


def test_bag_total_count(world):
    meta, idx, _ = world
    tb = sample_train_bags(idx, set(meta["player_id"]), bags_per_player_color=3)
    groups = meta.groupby(["player_id", "color_id"]).size()
    assert len(tb) == 3 * int((groups >= 5).sum())
    assert len({frozenset(b.uids) for b in tb}) == len(tb)               # de-duplicated
    assert len({b.qid for b in tb}) == len(tb)


def test_questions_reproducible_with_seed(world):
    _, idx, _ = world
    pl = set(sorted(idx.rank)[:12])
    a = sample_train_bags(idx, pl, 2, seed=1)
    assert a == sample_train_bags(idx, pl, 2, seed=1)
    assert a != sample_train_bags(idx, pl, 2, seed=2)
    v = build_val_questions(idx, pl, seed=1)
    assert v == build_val_questions(idx, pl, seed=1)
    assert [q.uids for q in v] != [q.uids for q in build_val_questions(idx, pl, seed=2)]


def test_train_n_class_report():
    # low ranks have few games -> N=20 bags are mostly high rank
    spec = {f"lo{i}": (0, 8, 8) for i in range(10)}
    spec.update({f"hi{i}": (9, 40, 40) for i in range(10)})
    idx = build_player_index(synth_meta(spec))
    tb = sample_train_bags(idx, set(spec), bags_per_player_color=40)
    rep = n_class_report(tb)
    cnt = rep[[f"c_{r}" for r in range(10)]]
    frac = rep[[f"f_{r}" for r in range(10)]]
    assert (cnt.sum(axis=1).values == rep["total"].values).all()
    by_n = pd.Series([b.n for b in tb]).value_counts()
    assert all(rep.set_index("n").loc[n, "total"] == by_n[n] for n in by_n.index)
    assert np.allclose(frac.sum(axis=1), 1.0)
    r = rep.set_index("n")
    assert r.loc[20, "f_0"] < r.loc[5, "f_0"]
    assert r.loc[20, "f_0"] == 0.0


def test_n_class_report_flag():
    from gogame.questions import flag_n_class_shift
    bags = [Bag("q", "p", 0, 0, 5, ()), Bag("q", "p", 0, 0, 5, ()), Bag("q", "p", 1, 0, 20, ()), Bag("q", "p", 1, 0, 20, ())]
    flags = flag_n_class_shift(n_class_report(bags), thr=0.05)
    assert flags[5] and flags[20]


def test_sample_weights_balance_classes():
    bags = []
    for rid, players in ((3, 20), (4, 5)):
        for p in range(players):
            for j in range(4):
                bags.append(Bag(f"q{rid}{p}{j}", f"p{rid}_{p}", rid, 0, 5, ()))
    for mode in ("inv_players", "balanced"):
        w = bag_class_weights(bags, mode=mode)
        assert w.dtype == np.float64 and abs(w.mean() - 1) < 1e-12
        tot = {rid: sum(wi for wi, b in zip(w, bags) if b.rank_id == rid) for rid in (3, 4)}
        assert tot[3] == pytest.approx(tot[4])
    w2 = bag_class_weights(bags, "inv_players", players_per_rank={3: 20, 4: 5})
    assert w2.mean() == pytest.approx(1)


def test_questions_frame_format():
    bags = [Bag("a", "p", 3, 1, 5, tuple(f"u{i}" for i in range(5))), Bag("b", "p2", 9, 0, 20, tuple(f"v{i}" for i in range(20)))]
    df = to_questions_frame(bags)
    assert list(df.columns) == ["qid", "player_id", "rank_id", "color", "n"] + [f"uid_{i}" for i in range(1, 21)]
    assert df["rank_id"].dtype == np.int8 and df["color"].dtype == np.int8 and df["n"].dtype == np.int8
    assert df.loc[0, "uid_5"] == "u4" and df.loc[0, "uid_6"] == "" and df.loc[1, "uid_20"] == "v19"
    assert len(to_questions_frame([])) == 0


# ---------------------------------------------------------------- pool simulation
def pool_world(n_h=12, nb=60):
    spec = {f"h{i}": (i % 10, nb, 5) for i in range(n_h)}
    spec["outsider"] = (0, nb, 5)
    return build_player_index(synth_meta(spec)), sorted(f"h{i}" for i in range(n_h))


def test_pool_sim_queries_disjoint_from_library():
    idx, H = pool_world()
    sims = build_player_pool_sim(idx, H, pool_size=8, n_draws=3, lib_games=30, holdout_set=set(H))
    assert len(sims) == 3
    for s in sims:
        for p in s.players:
            assert p in H
            lib = s.library[p]
            assert len(lib) == len(set(lib)) == 30 and all(u.endswith(tuple(f"B{i:03d}" for i in range(60))) for u in lib)
        for q in s.queries:
            assert q.color == 0 and q.player_id in s.library
            assert not (set(q.uids) & set(s.library[q.player_id]))
            assert len(set(q.uids)) == q.n


def test_pool_sim_pool_size_and_draws():
    idx, H = pool_world()
    sims = build_player_pool_sim(idx, H, pool_size=8, n_draws=4, lib_games=30)
    assert len(sims) == 4 and all(len(s.players) == 8 == len(set(s.players)) for s in sims)
    for s in sims:
        per = pd.DataFrame({"p": [q.player_id for q in s.queries], "n": [q.n for q in s.queries]}).groupby(["p", "n"]).size()
        assert per.max() == 1 and len(per) == 8 * 4
    with pytest.raises(ConfigError):
        build_player_pool_sim(idx, H, pool_size=13, n_draws=1, lib_games=30)


def test_pool_sim_rejects_non_holdout_player():
    idx, H = pool_world()
    with pytest.raises(PlayerLeakError):
        build_player_pool_sim(idx, H + ["outsider"], pool_size=8, n_draws=1, lib_games=30, holdout_set=set(H))


def test_pool_sim_requires_enough_games():
    spec = {f"h{i}": (0, 60, 0) for i in range(6)}
    spec["thin"] = (0, 34, 0)                       # < lib_games + 5
    idx = build_player_index(synth_meta(spec))
    H = sorted(spec)
    sims = build_player_pool_sim(idx, H, pool_size=6, n_draws=2, lib_games=30)
    assert all("thin" not in s.players for s in sims)
    with pytest.raises(ConfigError):
        build_player_pool_sim(idx, H, pool_size=7, n_draws=1, lib_games=30)     # only 6 eligible


def test_pool_sim_reproducible():
    idx, H = pool_world()
    a = build_player_pool_sim(idx, H, 8, 2, 30, seed=3)
    assert a == build_player_pool_sim(idx, H, 8, 2, 30, seed=3)
    assert a != build_player_pool_sim(idx, H, 8, 2, 30, seed=4)
