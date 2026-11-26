"""End-to-end test of scripts/predict.py on a tiny synthetic corpus / contest (no real data needed)."""
import sys

import numpy as np
import pandas as pd
import pytest

from helpers import make_contest, make_corpus, write_contest


@pytest.fixture(scope="module")
def pr(project_root):
    sys.path.insert(0, str(project_root / "scripts"))
    import predict
    return predict


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(3, players_per_rank=3, games_per_player=40)


@pytest.fixture(scope="module")
def models(pr, corpus, tmp_path_factory):
    from gogame.constants import RANK_TO_ID
    from gogame.io import GameRecord
    recs = [GameRecord(f"{r}_{g}", p, RANK_TO_ID[r], 0 if c == "B" else 1, s, "train")
            for p, g, r, c, s in zip(corpus.player_id, corpus.game_id, corpus["rank"], corpus.color, corpus.sgf_content)]
    store = pr.l0_store(recs, 1)
    meta = pd.DataFrame({"uid": [r.uid for r in recs], "player_id": [r.player_id for r in recs],
                         "rank_id": [r.rank_id for r in recs], "color": [r.color for r in recs]})
    d = tmp_path_factory.mktemp("models")
    pr.L0LgbmRank.fit(store, meta, rounds=4, threads=1, bags_per_player_color=2, min_df=0.05).save(d)
    pr.L0CosinePlayer.fit(store, min_df=0.05).save(d)
    return d


@pytest.fixture(scope="module")
def contest_dir(corpus, tmp_path_factory):
    d = tmp_path_factory.mktemp("contest")
    write_contest(make_contest(corpus, n_per_n=2, pool_size=8, lib_games=6, with_prefix=False, seed=1), d)
    return d


def test_end_to_end_passes_format_check(pr, models, contest_dir, tmp_path):
    rm, pm = pr.RANK_MODELS["l0_lgbm"].load(models, 1), pr.PLAYER_MODELS["l0_cosine"].load(models, 1)
    res = pr.run_predict(contest_dir, tmp_path, rm, pm, check=True, log=lambda *_: None)
    assert res["ok"]
    rk = pd.read_csv(res["rank"], dtype=str)
    pl = pd.read_csv(res["player"], dtype=str)
    assert len(rk) == len(pl) == 16                       # 2 splits x 4 N x 2 questions
    assert list(rk.columns) == ["question_id", "pred_rank"]
    assert rk["question_id"].str.startswith(("pub_", "priv_")).all()      # prefix added to un-prefixed test ids
    assert set(rk["pred_rank"]) <= set(pr.RANKS)
    assert (pl[[f"top_{i}" for i in range(1, 6)]].nunique(axis=1) >= 1).all()
    for sp in ("pub", "priv"):
        rows = pl[pl["question_id"].str.startswith(sp + "_")]
        assert rows[[f"top_{i}" for i in range(1, 6)]].map(lambda x: x.startswith(sp + "_")).all().all()


def test_prefixed_ids_accepted(pr, models, corpus, tmp_path):
    d = tmp_path / "c"
    write_contest(make_contest(corpus, n_per_n=1, pool_size=6, lib_games=5, with_prefix=True, seed=2), d)
    rm, pm = pr.RANK_MODELS["l0_lgbm"].load(models, 1), pr.PLAYER_MODELS["l0_cosine"].load(models, 1)
    assert pr.run_predict(d, tmp_path / "o", rm, pm, log=lambda *_: None)["ok"]


def test_pluggable_models(pr, contest_dir, tmp_path):
    class Const(pr.RankModel):
        def predict(self, questions):
            return np.full(len(questions), 4)

    class First5(pr.PlayerModel):
        def rank(self, candidates, queries):
            return np.array([sorted(candidates)[:5]] * len(queries), dtype=object)

    res = pr.run_predict(contest_dir, tmp_path, Const(), First5(), log=lambda *_: None)
    assert res["ok"]
    assert set(pd.read_csv(res["rank"])["pred_rank"]) == {"1D"}


def test_check_catches_bad_output(pr, contest_dir, tmp_path):
    class Bad(pr.PlayerModel):
        def rank(self, candidates, queries):          # duplicates in the top-5
            return np.array([[sorted(candidates)[0]] * 5] * len(queries), dtype=object)

    class Const(pr.RankModel):
        def predict(self, questions):
            return np.zeros(len(questions), dtype=int)

    res = pr.run_predict(contest_dir, tmp_path, Const(), Bad(), log=lambda *_: None)
    assert not res["ok"]


def test_report_with_train_overlap(pr, contest_dir, corpus, tmp_path):
    tdir = tmp_path / "train"
    tdir.mkdir()
    corpus[corpus["rank"] == "D"].to_csv(tdir / "train_D.csv", index=False)   # one rank file only; others skipped
    sub = corpus[corpus["rank"] == "D"].iloc[:1]
    c = make_contest(corpus, n_per_n=1, pool_size=5, lib_games=5, with_prefix=False, seed=5)
    c["rank_public"].loc[0, "sgf_1"] = sub.sgf_content.iloc[0]   # plant one training game as test game
    c["rank_public"].loc[0, "color_1"] = sub.color.iloc[0]
    d = tmp_path / "c"
    write_contest(c, d)
    rep = pr.run_report(d, tdir, log=lambda *_: None)
    o = rep["rank_pub"]["train_overlap"]
    assert o["exact"] >= 1 and rep["rank_pub"]["n_dist"] == {5: 1, 10: 1, 15: 1, 20: 1}
    assert rep["rank_priv"]["train_overlap"]["exact"] == 0
    assert "exact_vs_candidates" in rep["player_pub"]


def test_too_small_pool_rejected(pr, models, contest_dir):
    from gogame.errors import DataFormatError
    pm = pr.PLAYER_MODELS["l0_cosine"].load(models, 1)
    with pytest.raises(DataFormatError):
        pm.rank({"a": []}, [])


# ---------------------------------------------------------------------------------- E04v2 (player_rerank)
def _fake_z(rng, n_players=12, lib=15, dim=40):
    mus = rng.normal(size=(n_players, dim))
    libs = [(mus[i] + rng.normal(size=(lib, dim)) * 2).astype(np.float32) for i in range(n_players)]
    qs = [(mus[i % n_players] + rng.normal(size=(n, dim)) * 2).astype(np.float32) for i, n in enumerate([5, 10, 15, 20] * 6)]
    return libs, qs, np.arange(dim, dtype=np.int32)


def _knn_ref(libs, qs, f, sel):
    from gogame import player_rerank as prr
    nq, k = f["top"].shape
    ref = np.empty((nq, k, 3))
    for i, q in enumerate(qs):
        for j, c in enumerate(f["top"][i]):
            s = prr.unit(q[:, sel].astype(np.float64)) @ prr.unit(libs[c][:, sel].astype(np.float64)).T
            ref[i, j] = [s.max(1).mean(), s.mean(), np.sort(s, axis=1)[:, -3:].mean()]
    return ref


def test_candidate_features_paths_and_shapes():
    from gogame import player_rerank as prr
    rng = np.random.default_rng(0)
    libs, qs, sel = _fake_z(rng)
    f = prr.candidate_features(libs, qs, sel, 8, 10)                       # equal library sizes (E05 path)
    assert f["X"].shape == (len(qs) * 8, 12 + 2 * len(sel)) and f["top"].shape == (len(qs), 8)
    X = f["X"].reshape(len(qs), 8, -1)
    assert np.allclose(X[:, :, 9:12], _knn_ref(libs, qs, f, sel), atol=1e-5)
    assert (X[:, :, 1] == np.arange(1, 9)).all()                             # rank column
    assert (f["cos"][np.arange(len(qs)), f["top"][:, 0]] == f["cos"].max(1)).all()
    libs_u = [l[: 10 + i] for i, l in enumerate(libs)]                      # unequal sizes -> per-candidate path
    fu = prr.candidate_features(libs_u, qs, sel, 8, 10)
    assert np.allclose(fu["X"].reshape(len(qs), 8, -1)[:, :, 9:12], _knn_ref(libs_u, qs, fu, sel), atol=1e-5)
    assert prr.candidate_features(libs[:5], qs, sel, 20, 10)["k"] == 5       # pool smaller than K


def test_top5_indices():
    from gogame import player_rerank as prr
    top = np.array([[7, 3, 1, 9, 2, 5, 0]])
    assert list(prr.top5_indices(top, None)[0]) == [7, 3, 1, 9, 2]
    sc = np.array([0.1, 0.9, 0.2, 0.8, 0.0, 0.3, 0.5])
    assert list(prr.top5_indices(top, sc)[0]) == [3, 9, 0, 5, 1]


@pytest.fixture(scope="module")
def e04_model(pr, corpus, models, tmp_path_factory):
    from gogame.io import GameRecord
    from gogame.constants import RANK_TO_ID
    recs = [GameRecord(f"{r}_{g}", p, RANK_TO_ID[r], 0 if c == "B" else 1, s, "train")
            for p, g, r, c, s in zip(corpus.player_id, corpus.game_id, corpus["rank"], corpus.color, corpus.sgf_content)]
    store = pr.l0_store(recs, 1)
    meta = pd.DataFrame({"uid": [r.uid for r in recs], "player_id": [r.player_id for r in recs],
                         "rank_id": [r.rank_id for r in recs], "color": [r.color for r in recs]})
    players = set(meta.player_id)
    m = pr.E04v2Player.fit(store, meta, players, rounds=3, threads=1, pool_size=8, n_draws=2, lib_games=10, min_df=0.05,
                           log=lambda *_: None)
    d = tmp_path_factory.mktemp("e04")
    m.save(d)
    return d


def test_e04v2_default_and_end_to_end(pr, models, e04_model, contest_dir, tmp_path):
    assert pr.DEFAULT_PLAYER_MODEL == "e04v2" and "l0_cosine" in pr.PLAYER_MODELS
    rm = pr.RANK_MODELS["l0_lgbm"].load(models, 1)
    pm = pr.PLAYER_MODELS["e04v2"].load(e04_model, 1)
    assert pm.meta["rounds"] == 3 and (e04_model / "player_e04v2.txt").exists()
    res = pr.run_predict(contest_dir, tmp_path, rm, pm, check=True, log=lambda *_: None)
    assert res["ok"]
    pl = pd.read_csv(res["player"], dtype=str)
    assert len(pl) == 16 and (pl[[f"top_{i}" for i in range(1, 6)]].nunique(axis=1) == 5).all()


def test_e04v2_train_players_only_and_pool_too_small(pr, e04_model):
    from gogame.errors import DataFormatError
    pm = pr.PLAYER_MODELS["e04v2"].load(e04_model, 1)
    with pytest.raises(DataFormatError):
        pm.rank({"a": []}, [])
