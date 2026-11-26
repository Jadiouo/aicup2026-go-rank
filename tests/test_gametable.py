"""tdd 4.9 (gametable part)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from gogame.aggregate import aggregate_bag, make_spec
from gogame.errors import ConfigError, DataFormatError
from gogame.gametable import (GameTable, L0Provider, assert_crossfit, build_oof_table, load_table, merge_tables,
                              require_oof_table, save_table)
from gogame.io import GameRecord
from helpers import make_corpus


def table(route="l0", prefix="l0_", uids=("a", "b", "c"), names=("x", "y_mean"), seed=0, sparse=True, vecs=None, meta=None):
    rng = np.random.default_rng(seed)
    dense = rng.random((len(uids), len(names))).astype(np.float32)
    dense[0, -1] = np.nan
    sp = sps.csr_matrix(rng.random((len(uids), 7)).astype(np.float32) * (rng.random((len(uids), 7)) < .4)) if sparse else None
    return GameTable(route=route, uids=np.array(uids), dense=dense, names=tuple(prefix + n for n in names),
                     sparse=sp, vecs=vecs or {}, meta=meta or {"route": route, "model_id": "t", "fold": None,
                                                              "config_hash": "h", "external_resources": []})


def test_gametable_roundtrip(tmp_path):
    emb = np.random.default_rng(1).random((3, 4)).astype(np.float32)
    t = table(vecs={"emb": emb})
    save_table(t, tmp_path / "t")
    u = load_table(tmp_path / "t")
    assert u.route == t.route and u.names == t.names and u.meta == t.meta
    np.testing.assert_array_equal(u.uids, t.uids)
    np.testing.assert_array_equal(u.dense, t.dense)          # NaN positions equal too
    assert u.dense.dtype == np.float32
    assert (u.sparse != t.sparse).nnz == 0
    np.testing.assert_array_equal(u.vecs["emb"], emb)


def test_roundtrip_without_sparse(tmp_path):
    t = table(route="A", prefix="ka_", names=("p", "q_mean"), sparse=False)
    save_table(t, tmp_path / "t")
    assert load_table(tmp_path / "t").sparse is None


def test_merge_tables_align_by_uid():
    a = table(uids=("a", "b", "c"), names=("x", "y_mean"))
    b = table(route="A", prefix="ka_", uids=("c", "a", "b"), names=("p", "q_mean"), sparse=False, seed=5)
    m = merge_tables([a, b])
    assert list(m.uids) == ["a", "b", "c"]
    assert m.names == a.names + b.names
    np.testing.assert_array_equal(m.dense[:, :2], a.dense)
    np.testing.assert_array_equal(m.dense[:, 2:], b.dense[[1, 2, 0]])
    assert m.sparse is not None
    bad = table(route="A", prefix="ka_", uids=("a", "b", "zzz"), names=("p", "q_mean"), sparse=False)
    with pytest.raises(DataFormatError):
        merge_tables([a, bad])


def test_gametable_name_prefix():
    with pytest.raises(DataFormatError):
        table(route="A", prefix="l0_")
    t = table(route="A", prefix="ka_", sparse=False)
    assert all(n.startswith("ka_") for n in t.names)
    with pytest.raises(DataFormatError):
        GameTable(route="l0", uids=np.array(["a"]), dense=np.zeros((1, 2), np.float32), names=("l0_x", "l0_x"),
                  sparse=None, vecs={}, meta={})


def test_gametable_nan_only_in_allowed_columns():
    d = np.zeros((2, 2), np.float32)
    d[0, 0] = np.nan
    with pytest.raises(DataFormatError):
        GameTable(route="l0", uids=np.array(["a", "b"]), dense=d, names=("l0_x", "l0_y_mean"), sparse=None, vecs={}, meta={})


def test_aggregate_l0_only_regression():
    a = table(uids=tuple(f"g{i}" for i in range(6)))
    spec_p = make_spec(a.names, np.array([1, 4], np.int32))
    spec_u = make_spec(("x", "y_mean"), np.array([1, 4], np.int32))
    v1 = aggregate_bag([a], None, spec_p, 1)
    v2 = aggregate_bag(a.dense, a.sparse, spec_u, 1)
    np.testing.assert_array_equal(v1, v2)


def test_vecs_aggregation_mean_then_l2():
    from gogame.aggregate import aggregate_vecs
    emb = np.random.default_rng(0).standard_normal((7, 5)).astype(np.float32)
    v = aggregate_vecs(emb)
    assert v.shape == (5,) and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(v, emb.mean(0) / np.linalg.norm(emb.mean(0)), rtol=1e-5)


def split_frame():
    rows = [(f"p{i}", i % 3, False) for i in range(12)] + [(f"h{i}", -1, True) for i in range(3)]
    return pd.DataFrame(rows, columns=["player_id", "fold", "is_holdout"])


def test_assert_crossfit_detects_leak():
    sp = split_frame()
    f2 = set(sp[sp.fold == 2].player_id)
    ok = table(meta={"route": "l0", "fold": 2, "train_players": sorted(set(sp[(sp.fold != 2) & ~sp.is_holdout].player_id))})
    assert_crossfit(ok, sp)
    leak = table(meta={"route": "l0", "fold": 2, "train_players": sorted(f2 | {"p0"})})
    with pytest.raises(DataFormatError):
        assert_crossfit(leak, sp)
    leak_h = table(meta={"route": "l0", "fold": 0, "train_players": ["p1", "h0"]})
    with pytest.raises(DataFormatError):
        assert_crossfit(leak_h, sp)
    with pytest.raises(DataFormatError):
        assert_crossfit(table(meta={"route": "l0", "fold": 0}), sp)


def _recs(split):
    out = []
    for i, pid in enumerate(split.player_id):
        for g in range(2):
            out.append(GameRecord(uid=f"{pid}#{g}", player_id=pid, rank_id=0, color=0, sgf="(;SZ[19])", source="train"))
    return out


def test_crossfit_net_train_players_disjoint_from_fold():
    sp = split_frame()
    seen = {}

    def train(players, j):
        seen[j] = set(players)
        return j

    build_oof_table(_recs(sp), sp, K=3, train_net=train, apply_net=lambda net, recs: np.full((len(recs), 1), float(net), np.float32),
                    route="B", names=("st_v",))
    H = set(sp[sp.is_holdout].player_id)
    assert sorted(seen) == [0, 1, 2]
    for j, tp in seen.items():
        fj = set(sp[sp.fold == j].player_id)
        assert tp and not (tp & fj) and not (tp & H)


def test_oof_table_rows_from_matching_net():
    sp = split_frame()
    recs = _recs(sp)
    t = build_oof_table(recs, sp, K=3, train_net=lambda players, j: float(j + 1),
                        apply_net=lambda net, rs: np.full((len(rs), 1), net, np.float32), route="B", names=("st_v",))
    fold_of = dict(zip(sp.player_id, sp.fold))
    rf = t.meta["row_fold"]
    for i, r in enumerate(recs):
        j = fold_of[r.player_id]
        if j >= 0:
            assert rf[i] == j and t.dense[i, 0] == j + 1
        else:
            assert rf[i] == -1 and t.dense[i, 0] == pytest.approx((1 + 2 + 3) / 3)
    assert t.meta["fold"] == "oof" and len(t.meta["train_players_per_fold"]) == 3
    assert_crossfit(t, sp)


def test_lgbm_oof_uses_only_oof_features():
    sp = split_frame()
    recs = _recs(sp)
    oof = build_oof_table(recs, sp, K=3, train_net=lambda p, j: j, apply_net=lambda n, r: np.zeros((len(r), 1), np.float32),
                          route="B", names=("st_v",))
    require_oof_table(oof)
    full = table(route="B", prefix="st_", names=("v_mean",), sparse=False, meta={"route": "B", "fold": "avg"})
    with pytest.raises(DataFormatError):
        require_oof_table(full)
    require_oof_table(table())      # L0 / A have no learned parameters


def test_emb_vecs_rejected_as_rank_input():
    with pytest.raises(ConfigError):
        make_spec(("a",), np.array([], np.int32), vec_keys=("emb",))


def test_l0_provider_compute():
    df = make_corpus(2, n_ranks=1, players_per_rank=2, games_per_player=2)
    recs = [GameRecord(uid=g, player_id=p, rank_id=0, color=0, sgf=s, source="train")
            for p, g, s in zip(df.player_id, df.game_id, df.sgf_content)]
    t = L0Provider().compute(recs, None)
    assert t.route == "l0" and len(t.names) == 31 and all(n.startswith("l0_") for n in t.names)
    assert t.dense.shape == (4, 31) and t.sparse.shape == (4, 10125) and list(t.uids) == [r.uid for r in recs]
