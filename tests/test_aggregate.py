"""tdd 4.9 (aggregate part)."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sps

from gogame.aggregate import (BagFeatureSpec, L0Store, aggregate_bag, aggregate_many, fit_sparse_selector,
                              make_spec)
from gogame.errors import ConfigError, DataFormatError

NAMES = ("a", "b")


def spec(cols=(), names=NAMES):
    return make_spec(names, np.array(cols, dtype=np.int32))


def bag_of(dense, sp=None, color=0, cols=()):
    if sp is None:
        sp = sps.csr_matrix((len(dense), 10), dtype=np.float32)
    return aggregate_bag(np.asarray(dense, dtype=np.float32), sp, spec(cols), color)


def idx(s, name):
    return list(s.names).index(name)


def test_stats_hand_computed():
    d = np.column_stack([[1, 3, 5, 7, 9], [10] * 5]).astype(np.float32)
    s = spec()
    v = aggregate_bag(d, sps.csr_matrix((5, 10), dtype=np.float32), s, 0)
    g = lambda n: float(v[idx(s, n)])
    assert g("a__mean") == 5 and g("a__std") == pytest.approx(np.sqrt(8), rel=1e-6)
    assert g("a__q25") == 3 and g("a__q50") == 5 and g("a__q75") == 7
    assert g("b__mean") == 10 and g("b__std") == 0
    assert v.dtype == np.float32


def test_output_layout_and_names():
    s = spec(cols=[2, 5])
    assert s.stats == ("mean", "std", "q25", "q50", "q75")
    assert len(s.names) == 2 * 5 + 2 + 2
    assert s.names[-2:] == ("bag_n", "is_white")
    v = aggregate_bag(np.ones((7, 2), np.float32), sps.csr_matrix((7, 10), dtype=np.float32), s, 1)
    assert v.shape == (len(s.names),)


def test_any_n_5_to_20_same_length():
    rng = np.random.default_rng(0)
    s = spec(cols=[1, 3])
    lens = {aggregate_bag(rng.random((n, 2)).astype(np.float32), sps.csr_matrix(rng.random((n, 10)).astype(np.float32)), s, 0).size
            for n in range(5, 21)}
    assert lens == {len(s.names)}


def test_mean_not_sum_scale_invariant():
    rng = np.random.default_rng(1)
    d = rng.random((5, 2)).astype(np.float32)
    sp = sps.csr_matrix(rng.random((5, 10)).astype(np.float32))
    s = spec(cols=[0, 4])
    v1 = aggregate_bag(d, sp, s, 0)
    v2 = aggregate_bag(np.vstack([d, d]), sps.vstack([sp, sp]).tocsr(), s, 0)
    n = len(s.names)
    np.testing.assert_allclose(v1[:n - 2], v2[:n - 2], rtol=1e-5, atol=1e-6)
    assert v1[-2] == 5 and v2[-2] == 10


def test_order_invariant():
    rng = np.random.default_rng(2)
    d = rng.random((9, 2)).astype(np.float32)
    sp = sps.csr_matrix(rng.random((9, 10)).astype(np.float32))
    s = spec(cols=[0, 4, 9])
    p = rng.permutation(9)
    np.testing.assert_array_equal(aggregate_bag(d, sp, s, 0), aggregate_bag(d[p], sp[p], s, 0))


def test_nan_aware_stats():
    d = np.array([[1, np.nan], [3, np.nan], [np.nan, np.nan], [5, np.nan], [7, np.nan]], dtype=np.float32)
    s = spec()
    v = aggregate_bag(d, sps.csr_matrix((5, 10), dtype=np.float32), s, 0)
    assert v[idx(s, "a__mean")] == 4
    assert np.isnan(v[idx(s, "b__mean")]) and np.isnan(v[idx(s, "b__std")]) and np.isnan(v[idx(s, "b__q50")])
    assert not np.isinf(v).any()


def test_sparse_mean_only_selected_cols():
    sp = sps.csr_matrix(np.array([[0, 1, 0, 2], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0]], np.float32))
    s = make_spec(NAMES, np.array([1, 3], np.int32))
    v = aggregate_bag(np.zeros((5, 2), np.float32), sp, s, 0)
    n_dense = 10
    assert v[n_dense] == pytest.approx(0.8) and v[n_dense + 1] == pytest.approx(1.2)
    assert len(s.names) == n_dense + 2 + 2


def test_selector_min_df_label_free_train_only():
    # 200 rows; col0 non-zero in exactly 1 row (df=0.005), col1 in 0 rows, col2 in 2 rows.
    m = np.zeros((200, 4), np.float32)
    m[0, 0] = 1
    m[0, 2] = m[1, 2] = 1
    cols = fit_sparse_selector(sps.csr_matrix(m), min_df=0.005)
    assert cols.tolist() == [0, 2] and cols.dtype == np.int32
    assert fit_sparse_selector(sps.csr_matrix(m), min_df=0.0075).tolist() == [2]


def test_bag_n_and_color_columns():
    s = spec()
    v = aggregate_bag(np.ones((6, 2), np.float32), sps.csr_matrix((6, 10), dtype=np.float32), s, 1)
    assert v[-2] == 6 and v[-1] == 1
    v0 = aggregate_bag(np.ones((6, 2), np.float32), sps.csr_matrix((6, 10), dtype=np.float32), s, 0)
    assert v0[-1] == 0


def test_no_inf_in_output():
    s = spec(cols=[0])
    v = aggregate_bag(np.zeros((1, 2), np.float32), sps.csr_matrix((1, 10), dtype=np.float32), s, 0)
    assert not np.isinf(v).any() and not np.isnan(v).any()
    assert v[idx(s, "a__std")] == 0


def test_make_spec_rejects_emb_vecs():
    with pytest.raises(ConfigError):
        make_spec(NAMES, np.array([], np.int32), vec_keys=("emb",))


def _store(G=30, seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.random((G, 2)).astype(np.float32)
    sp = sps.csr_matrix((rng.random((G, 10)) * (rng.random((G, 10)) < 0.3)).astype(np.float32))
    return L0Store(uids=[f"u{i}" for i in range(G)], dense=dense, sparse=sp, names=NAMES), dense, sp


def test_aggregate_many_matches_aggregate_bag():
    store, dense, sp = _store()
    s = make_spec(NAMES, np.array([1, 3, 7], np.int32))
    bags = [SimpleNamespace(uids=tuple(f"u{i}" for i in ids), color=c)
            for ids, c in (([0, 5, 7, 9, 11], 0), ([1, 2, 3, 4, 5, 6, 7], 1))]
    X = aggregate_many(bags, store, s)
    assert X.shape == (2, len(s.names)) and X.dtype == np.float32
    for b, row in zip(bags, X):
        rows = [int(u[1:]) for u in b.uids]
        np.testing.assert_array_equal(row, aggregate_bag(dense[rows], sp[rows], s, b.color))


def test_aggregate_many_unknown_uid_raises():
    store, _, _ = _store()
    s = make_spec(NAMES, np.array([], np.int32))
    with pytest.raises(DataFormatError):
        aggregate_many([SimpleNamespace(uids=("nope",) * 5, color=0)], store, s)


def test_aggregate_many_name_mismatch_raises():
    store, _, _ = _store()
    s = make_spec(("x", "y"), np.array([], np.int32))
    with pytest.raises(DataFormatError):
        aggregate_many([SimpleNamespace(uids=("u0",) * 5, color=0)], store, s)
