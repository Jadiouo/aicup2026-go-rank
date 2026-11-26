"""tdd 4.12: models (LightGBM wrapper, decisions, prior correction, OOF, memory baseline, final model)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from gogame import models
from gogame.aggregate import L0Store, aggregate_many, make_spec
from gogame.errors import DataFormatError
from gogame.metrics import rank_score_matrix, rank_scores
from gogame.models import (MemoryBaseline, OofFold, RankModel, apply_prior_correction, decide_argmax, decide_ev,
                           final_num_rounds, fit_lgbm, oof_scores, run_oof, train_final, predict_bags,
                           constant_prediction)

E1 = float(np.exp(-1))
SMALL = {"num_boost_round": 30, "early_stopping_rounds": 10, "num_leaves": 4, "min_data_in_leaf": 5}


def onehot(i, n=10):
    p = np.zeros(n)
    p[i] = 1
    return p


# ----------------------------------------------------------------- decisions -------------------------------------
def test_decide_argmax():
    p = np.array([[.1, .5, .4] + [0] * 7, [0, 0, 0, .2, .8] + [0] * 5])
    assert decide_argmax(p).tolist() == [1, 4] and decide_argmax(p).dtype == np.int64


def test_decide_ev_differs_from_argmax():
    p = np.zeros(10)
    p[1], p[2], p[3], p[5] = .30, .18, .26, .26
    assert decide_argmax(p[None])[0] == 1
    assert decide_ev(p[None])[0] == 2
    ev = p @ rank_score_matrix()
    assert ev[2] == pytest.approx(.18 + E1 * (.30 + .26), abs=1e-9) and ev[1] == pytest.approx(.30 + E1 * .18)


def test_decide_ev_tie_picks_lowest():
    q = np.zeros((1, 10))
    q[0, 3] = q[0, 7] = .5
    assert decide_ev(q)[0] == 3


def test_decide_ev_equals_argmax_when_peaked():
    p = np.stack([onehot(i) for i in range(10)])
    assert decide_ev(p).tolist() == decide_argmax(p).tolist() == list(range(10))


# ----------------------------------------------------------------- prior correction ------------------------------
def test_prior_correction_formula():
    out = apply_prior_correction(np.array([[.5, .5]]), np.array([.8, .2]), np.array([.5, .5]))
    np.testing.assert_allclose(out, [[.2, .8]], atol=1e-12)


def test_prior_correction_uniform_default():
    p = np.array([[.1, .6, .3]])
    tp = np.array([.2, .5, .3])
    out = apply_prior_correction(p, tp)                       # target None -> uniform
    ref = p / tp
    np.testing.assert_allclose(out, ref / ref.sum(1, keepdims=True))
    np.testing.assert_allclose(apply_prior_correction(p, tp, tp), p)
    np.testing.assert_allclose(apply_prior_correction(p, np.full(3, 1 / 3)), p)


def test_prior_correction_zero_train_prior_class_stays_zero():
    out = apply_prior_correction(np.array([[.5, .5, 0.]]), np.array([.5, .5, 0.]))
    np.testing.assert_allclose(out, [[.5, .5, 0.]])


# ----------------------------------------------------------------- LightGBM --------------------------------------
def synth(n=400, seed=0, classes=(0, 1, 2, 3), signal=True):
    rng = np.random.default_rng(seed)
    y = rng.choice(classes, size=n)
    X = rng.standard_normal((n, 4))
    if signal:
        X[:, 0] += y
    return X.astype(np.float32), y.astype(np.int64)


@pytest.mark.lgbm
def test_predict_proba_shape_and_sum():
    Xtr, ytr = synth(500, 0, tuple(range(10)))
    Xva, yva = synth(200, 1, tuple(range(10)))
    m = fit_lgbm(Xtr, ytr, Xva, yva, SMALL, seed=1)
    assert isinstance(m, RankModel)
    for raw in (True, False):
        p = m.predict_proba(Xva, raw=raw)
        assert p.shape == (200, 10) and p.dtype == np.float32
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    assert m.best_iteration >= 1


@pytest.mark.lgbm
def test_missing_class_padded():
    Xtr, ytr = synth(400, 0, (0, 1, 2, 3, 4, 5, 6))
    Xva, yva = synth(150, 1, (0, 1, 2, 3, 4, 5, 6))
    m = fit_lgbm(Xtr, ytr, Xva, yva, SMALL, seed=1)
    for raw in (True, False):
        p = m.predict_proba(Xva, raw=raw)
        assert p.shape == (150, 10)
        assert (p[:, 7:] == 0).all()
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    assert m.train_prior[7:].sum() == 0


@pytest.mark.lgbm
def test_fit_deterministic():
    Xtr, ytr = synth(400, 0)
    Xva, yva = synth(150, 1)
    a = fit_lgbm(Xtr, ytr, Xva, yva, SMALL, seed=7).predict_proba(Xva, raw=True)
    b = fit_lgbm(Xtr, ytr, Xva, yva, SMALL, seed=7).predict_proba(Xva, raw=True)
    np.testing.assert_array_equal(a, b)


@pytest.mark.lgbm
def test_signal_learned():
    Xtr, ytr = synth(800, 0)
    Xva, yva = synth(300, 1)
    m = fit_lgbm(Xtr, ytr, Xva, yva, SMALL, seed=1)
    pred = decide_argmax(m.predict_proba(Xva))
    assert (pred == yva).mean() > 0.45          # chance is 0.25


@pytest.mark.lgbm
def test_sample_weight_changes_effective_prior():
    rng = np.random.default_rng(0)
    y = np.where(rng.random(500) < .8, 3, 4).astype(np.int64)
    X = rng.standard_normal((500, 3)).astype(np.float32)
    Xv, yv = X[:100], y[:100]
    m0 = fit_lgbm(X, y, Xv, yv, SMALL, seed=0)
    assert m0.train_prior[3] == pytest.approx((y == 3).mean())
    cnt = np.bincount(y, minlength=10).astype(float)
    w = 1.0 / cnt[y]
    w = w / w.mean()
    m1 = fit_lgbm(X, y, Xv, yv, SMALL, seed=0, w_tr=w, w_va=w[:100])
    assert m1.train_prior[3] == pytest.approx(0.5, abs=1e-9) and m1.train_prior[4] == pytest.approx(0.5, abs=1e-9)
    assert m0.train_prior.sum() == pytest.approx(1.0)


@pytest.mark.lgbm
def test_prior_correction_balances_predictions():
    rng = np.random.default_rng(3)
    n = 1500
    y = np.where(rng.random(n) < .8, 3, 4).astype(np.int64)     # A : 1D = 4 : 1, features carry no signal
    X = rng.standard_normal((n, 3)).astype(np.float32)
    Xt = rng.standard_normal((1000, 3)).astype(np.float32)
    m = fit_lgbm(X[:1000], y[:1000], X[1000:], y[1000:], dict(SMALL, num_boost_round=5), seed=0)
    raw = m.predict_proba(Xt, raw=True).mean(0)
    cor = m.predict_proba(Xt).mean(0)
    assert raw[3] == pytest.approx(0.8, abs=0.05)
    assert cor[3] == pytest.approx(0.5, abs=0.05) and cor[4] == pytest.approx(0.5, abs=0.05)
    # explicit target prior changes the output
    t = np.zeros(10)
    t[3], t[4] = .9, .1
    assert m.predict_proba(Xt, target_prior=t).mean(0)[3] > cor[3] + 0.2


# ----------------------------------------------------------------- bags / store helpers --------------------------
def make_world(n_players=12, games=8, seed=0):
    rng = np.random.default_rng(seed)
    uids, dense, rows = [], [], []
    bags = []
    for pi in range(n_players):
        rank = pi % 4
        mine = []
        for g in range(games):
            u = f"p{pi}_g{g}"
            uids.append(u)
            mine.append(u)
            dense.append([rank + rng.normal(0, .5), rng.normal()])
        for b in range(3):
            ids = list(rng.choice(games, size=5, replace=False))
            bags.append(SimpleNamespace(qid=f"q{pi}_{b}", player_id=f"p{pi}", rank_id=rank, color=0, n=5,
                                        uids=tuple(mine[i] for i in ids)))
    store = L0Store(uids=uids, dense=np.array(dense, np.float32), sparse=sps.csr_matrix((len(uids), 6), dtype=np.float32),
                    names=("a", "b"))
    return store, bags


class StubModel:
    best_iteration = 7
    train_prior = np.full(10, .1)

    def __init__(self, n):
        self.n = n
        self.spec = None

    def predict_proba(self, X, target_prior=None, raw=False):
        return np.full((len(X), 10), .1, dtype=np.float32)


def stub_trainer(log):
    def tr(Xtr, ytr, Xva, yva, params, seed, w_tr=None, w_va=None):
        log.append((len(Xtr), None if Xva is None else len(Xva)))
        return StubModel(len(Xtr))
    return tr


def folds_of(bags, K=3):
    players = sorted({b.player_id for b in bags}, key=lambda s: int(s[1:]))
    out = []
    for k in range(K):
        val = {p for i, p in enumerate(players) if i % K == k}
        rest = [p for p in players if p not in val]
        es = {rest[0]}
        out.append(OofFold(fold=k, train_bags=[b for b in bags if b.player_id in rest[1:]],
                           es_bags=[b for b in bags if b.player_id in es],
                           val_bags=[b for b in bags if b.player_id in val]))
    return out


def test_oof_no_player_leak_and_each_question_once():
    store, bags = make_world()
    log = []
    res = run_oof({"params": {}, "seed": 1}, store, folds_of(bags), trainer=stub_trainer(log))
    assert len(log) == 3
    for k, tp in res.train_players.items():
        assert not (tp & res.val_players[k])
    assert res.frame["qid"].is_unique and set(res.frame["qid"]) == {b.qid for b in bags}   # val sets partition players
    assert res.best_iterations == [7, 7, 7]
    assert {"qid", "player_id", "rank_id", "n", "color", "fold"} <= set(res.frame.columns)
    assert all(f"praw_{i}" in res.frame.columns and f"p_{i}" in res.frame.columns for i in range(10))


def test_oof_covers_all_val_questions():
    store, bags = make_world()
    folds = folds_of(bags)
    res = run_oof({}, store, folds, trainer=stub_trainer([]))
    assert len(res.frame) == sum(len(f.val_bags) for f in folds)


def test_oof_rejects_leaky_folds():
    store, bags = make_world()
    f = folds_of(bags)
    f[0].train_bags = f[0].train_bags + f[0].val_bags[:1]
    with pytest.raises(DataFormatError):
        run_oof({}, store, f, trainer=stub_trainer([]))


def test_oof_rejects_holdout_in_training():
    store, bags = make_world()
    with pytest.raises(DataFormatError):
        run_oof({"holdout": {"p0"}}, store, folds_of(bags), trainer=stub_trainer([]))


def test_sparse_selector_fit_on_train_folds_only(monkeypatch):
    store, bags = make_world()
    seen = []
    orig = models.fit_sparse_selector

    def spy(sp, min_df=0.005):
        seen.append(sp.shape[0])
        return orig(sp, min_df)
    monkeypatch.setattr(models, "fit_sparse_selector", spy)
    folds = folds_of(bags)
    run_oof({}, store, folds, trainer=stub_trainer([]))
    for f, n in zip(folds, seen):
        used = {u for b in f.train_bags for u in b.uids}
        assert n == len(used)
        assert not (used & {u for b in f.val_bags for u in b.uids})


def test_constant_baseline_e00():
    y = np.array([3, 3, 3, 4, 2, 3, 5])
    c = constant_prediction(y)
    assert c == 3
    assert rank_scores(np.full(4, c), np.array([3, 4, 2, 9])).mean() == pytest.approx((1 + E1 + E1 + 0) / 4)


def _frame(rank_ids, pred_class, shuffle=False):
    rows = []
    for r, c in zip(rank_ids, pred_class):
        p = np.full(10, 1e-3)
        p[c] = 1
        p /= p.sum()
        rows.append(p)
    P = np.array(rows)
    d = {"qid": [f"q{i}" for i in range(len(rank_ids))], "rank_id": list(rank_ids)}
    for i in range(10):
        d[f"praw_{i}"] = P[:, i]
        d[f"p_{i}"] = P[:, i]
    return pd.DataFrame(d)


def test_eval_reports_raw_and_balanced():
    rk = [0] * 6 + [1] * 2
    pr = [0] * 6 + [5] * 2                  # class 0 perfect, class 1 wrong
    s = oof_scores(_frame(rk, pr))
    assert set(s) == {"raw_praw", "raw_p", "balanced_praw", "balanced_p"}
    assert s["raw_p"] == pytest.approx(6 / 8) and s["balanced_p"] == pytest.approx(0.5)
    s2 = oof_scores(_frame(rk, pr), decision="ev")
    assert s2["raw_p"] == pytest.approx(6 / 8)


def test_eval_decision_acts_on_corrected_p():
    f = _frame([0, 0], [0, 0])
    f["p_0"], f["p_9"] = 0.0, 1.0           # corrected p disagrees with praw
    s = oof_scores(f)
    assert s["raw_praw"] == 1.0 and s["raw_p"] == 0.0


# ----------------------------------------------------------------- final model -----------------------------------
def test_final_model_rounds_median_of_oof():
    assert final_num_rounds([100, 140, 120]) == 120
    assert final_num_rounds([100, 141]) == 121          # 120.5 rounds half up
    assert final_num_rounds([100, 140, 120, 160]) == 130


def test_final_model_uses_all_nonholdout_players():
    store, bags = make_world()
    log = []
    got = {}

    def tr(Xtr, ytr, Xva, yva, params, seed, w_tr=None, w_va=None):
        got.update(params=params, Xva=Xva)
        return StubModel(len(Xtr))
    m = train_final(store, bags, holdout={"zz"}, best_iterations=[100, 140, 120], params={"learning_rate": .1}, seed=3,
                    trainer=tr)
    assert got["params"]["num_boost_round"] == 120 and got["Xva"] is None
    assert m.meta["train_players"] == sorted({b.player_id for b in bags})
    assert m.meta["rounds"] == 120
    with pytest.raises(DataFormatError):
        train_final(store, bags, holdout={"p3"}, best_iterations=[10], params={}, seed=3, trainer=tr)


@pytest.mark.lgbm
def test_final_model_applies_prior_correction():
    store, bags = make_world(n_players=16)
    m = train_final(store, bags, holdout=set(), best_iterations=[10], params={"num_leaves": 4, "min_data_in_leaf": 3}, seed=0)
    raw = predict_bags(m, bags, store, raw=True)
    uni = predict_bags(m, bags, store)
    t = np.zeros(10)
    t[0] = 1.0
    skew = predict_bags(m, bags, store, target_prior=t)
    ref = apply_prior_correction(raw, m.train_prior)
    np.testing.assert_allclose(uni, ref, atol=1e-6)
    assert not np.allclose(uni, skew)
    assert skew[:, 0].mean() > uni[:, 0].mean()
    np.testing.assert_allclose(uni.sum(1), 1, atol=1e-5)


def test_train_vs_test_path_same_features():
    from gogame.features_l0 import compute_l0_many
    from gogame.io import GameRecord, explode_question
    from helpers import make_contest, make_corpus
    corpus = make_corpus(5, n_ranks=3, players_per_rank=2, games_per_player=2)
    contest = make_contest(corpus, n_per_n=1, pool_size=4, lib_games=5)
    for prefix_split in ("pub",):
        df = contest["rank_public"]
        for _, row in df.iterrows():
            recs = explode_question(row, "pub")
            n = int(row["num_games"])
            assert len(recs) == n
            # training path: games live in the training store under their own uids
            train_recs = [GameRecord(uid=f"tr_{i}", player_id="x", rank_id=0, color=r.color, sgf=r.sgf, source="train")
                          for i, r in enumerate(recs)]
            st_train = L0Store.from_batch(compute_l0_many(train_recs))
            st_test = L0Store.from_batch(compute_l0_many(recs))
            spec = make_spec(st_train.names, np.arange(0, 10125, 7, dtype=np.int32))
            bag_tr = SimpleNamespace(uids=tuple(r.uid for r in train_recs), color=train_recs[0].color)
            bag_te = SimpleNamespace(uids=tuple(r.uid for r in recs), color=recs[0].color)
            a = aggregate_many([bag_tr], st_train, spec)
            b = aggregate_many([bag_te], st_test, spec)
            assert np.array_equal(a, b, equal_nan=True)
            assert a.shape[1] == len(spec.names)


# ----------------------------------------------------------------- memory baseline -------------------------------
def _mem_data():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 3, size=(6, 5))
    X, pid, rid = [], [], []
    for p in range(6):
        for _ in range(8):
            X.append(centers[p] + rng.normal(0, .3, 5))
            pid.append(f"p{p}")
            rid.append(p // 2)
    return np.array(X), np.array(pid), np.array(rid), centers


def test_memory_baseline_self_match():
    X, pid, rid, _ = _mem_data()
    mb = MemoryBaseline().fit(X, pid, rid)
    pred, sim = mb.predict(X, k=1)
    assert (pred == rid).all() and (sim > 0.9).all() and pred.dtype == np.int64


def test_memory_baseline_oof_lower_than_self():
    rng = np.random.default_rng(1)
    d = 5
    tr_centers = rng.normal(0, 3, size=(40, d))
    rank = np.arange(40) % 10
    # rank signal is weak: players differ mostly by individual offset
    X = np.vstack([tr_centers[p] + rng.normal(0, .3, (6, d)) for p in range(40)])
    pid = np.repeat([f"p{p}" for p in range(40)], 6)
    rid = np.repeat(rank, 6)
    mb = MemoryBaseline().fit(X, pid, rid)
    self_acc = (mb.predict(X)[0] == rid).mean()
    new_centers = rng.normal(0, 3, size=(40, d))
    Xn = np.vstack([new_centers[p] + rng.normal(0, .3, (6, d)) for p in range(40)])
    rn = np.repeat(rank, 6)
    oof_acc = (mb.predict(Xn)[0] == rn).mean()
    assert self_acc > 0.95 and oof_acc < self_acc - 0.3


def test_memory_baseline_k_vote():
    # three players: two of rank 1 close to the query, one of rank 5 closest
    X = np.array([[1, 0], [1, .05], [0.9, .6], [0.8, .7], [1.0, 0.9], [.9, 1.0]], float)
    pid = np.array(["a", "a", "b", "b", "c", "c"])
    rid = np.array([5, 5, 1, 1, 1, 1])
    mb = MemoryBaseline(standardize=False).fit(X, pid, rid)
    q = np.array([[1.0, .1]])
    assert mb.predict(q, k=1)[0][0] == 5
    assert mb.predict(q, k=3)[0][0] == 1
