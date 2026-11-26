"""E05 pool-sim wiring on synthetic players: disjointness, subpool consistency, feature/label alignment."""
import numpy as np

from gogame.questions import build_player_index, build_player_pool_sim
from scripts.experiments.e05 import run


class FakeData:
    def __init__(self, n_players=30, games=130, dim=60, seed=0):
        import pandas as pd
        rng = np.random.default_rng(seed)
        rows, Z = [], []
        for p in range(n_players):
            mu = rng.normal(size=dim)
            for g in range(games):
                rows.append((f"p{p}_{g}", f"p{p}", 0, 0))
                Z.append(mu + rng.normal(size=dim) * 2)
        meta = pd.DataFrame(rows, columns=["uid", "player_id", "rank_id", "color"])
        self.Z = np.asarray(Z, np.float32)
        self.rowof = {u: i for i, u in enumerate(meta.uid)}
        self.sel = np.arange(dim, dtype=np.int32)
        self.index = build_player_index(meta.assign(color_id=meta["color"]))
        self.players = sorted(meta.player_id.unique())

    def rows(self, uids):
        return np.fromiter((self.rowof[u] for u in uids), dtype=np.int64, count=len(uids))


def test_sim_features_and_subpool(monkeypatch):
    monkeypatch.setattr(run, "N_DENSE", 10)
    d = FakeData()
    sims = build_player_pool_sim(d.index, d.players, 20, 2, 100, run.NS, seed=7)
    for s in sims:
        run.check_sim(s)
        sub = run.subsim(s, 8, "t")
        run.check_sim(sub)
        assert len(sub.players) == 8 and {q.player_id for q in sub.queries} <= set(sub.players)
        pf = run.pool_features(d, s)
        nq = len(s.queries)
        assert pf["X"].shape[0] == nq * run.K and pf["y"].shape == (nq * run.K,)
        # label marks exactly the true player among candidates, and recall matches
        assert pf["y"].reshape(nq, run.K).sum(1).max() <= 1
        # M01 top1 equals highest-cosine candidate (rank feature 1 column 1)
        assert (pf["X"].reshape(nq, run.K, -1)[:, 0, 1] == 1).all()
        df = run.score_pool(pf, None)
        assert df["m01"].mean() > 0.3   # separable synthetic players
        assert (df["m01"] <= 1).all()
