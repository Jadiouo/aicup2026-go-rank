import numpy as np
import pytest

from gogame.memory import MemoryBaseline


def make_data(n_players=40, bags_per=6, D=16, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_players, D))
    pid = np.array([f"p{i}" for i in range(n_players) for _ in range(bags_per)])
    rank = np.array([i % 10 for i in range(n_players) for _ in range(bags_per)])
    X = np.repeat(centers, bags_per, axis=0) + noise * rng.normal(size=(n_players * bags_per, D))
    return X, pid, rank, centers


def test_memory_baseline_self_match():
    X, pid, rank, _ = make_data()
    m = MemoryBaseline().fit(X, pid, rank)
    pred, sim = m.predict(X)
    assert (pred == rank).mean() > 0.95 and sim.mean() > 0.6


def test_memory_baseline_oof_lower_than_self():
    X, pid, rank, centers = make_data(noise=0.5)
    # unseen players: fresh bags from OTHER players -> rank is unrelated to nearest training player
    rng = np.random.default_rng(9)
    Xn = rng.normal(size=(400, X.shape[1])) * 1.0
    rn = rng.integers(0, 10, size=400)
    m = MemoryBaseline().fit(X, pid, rank)
    self_acc = (m.predict(X)[0] == rank).mean()
    new_pred, new_sim = m.predict(Xn)
    assert (new_pred == rn).mean() < self_acc - 0.5
    assert new_sim.mean() < m.predict(X)[1].mean()


def test_memory_baseline_k_vote_hand_computed():
    # 1-D features after z-scoring are signs; use 2-D unit-like directions with exact cosines
    X = np.array([[1, 0], [0.9, 0.1], [0.8, 0.3], [0.0, 1.0], [-1.0, 0.2]])
    pid = np.array(["a", "b", "c", "d", "e"])
    rank = np.array([2, 5, 5, 7, 9])
    m = MemoryBaseline(standardize=False).fit(X, pid, rank)
    q = np.array([[1.0, 0.05]])
    assert m.predict(q, k=1)[0][0] == 2                 # nearest is 'a' (rank 2)
    p3, _ = m.predict(q, k=3)                           # top-3 = a(2), b(5), c(5) -> vote 5
    assert p3[0] == 5


def test_memory_baseline_vote_tie_goes_to_closest_then_lowest_rank():
    X = np.array([[1, 0], [0.9, 0.2], [0, 1.0]])
    m = MemoryBaseline(standardize=False).fit(X, np.array(["a", "b", "c"]), np.array([4, 6, 1]))
    pred, _ = m.predict(np.array([[1.0, 0.1]]), k=3)    # three different ranks, 1 vote each -> highest summed cosine wins
    assert pred[0] == 4
    # exact tie in votes and similarity -> lowest rank id
    m2 = MemoryBaseline(standardize=False).fit(np.array([[1.0, 0], [-1.0, 0]]), np.array(["a", "b"]), np.array([7, 3]))
    assert m2.predict(np.array([[0.0, 1.0]]), k=2)[0][0] == 3


def test_memory_baseline_player_centroid_averages_bags_and_colors():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    m = MemoryBaseline(standardize=False).fit(X, np.array(["a", "a", "b"]), np.array([3, 3, 8]))
    assert m.n_centroids == 2
    assert m.predict(np.array([[1.0, 1.0]]))[0][0] == 3
    assert m.predict(np.array([[-1.0, 0.1]]))[0][0] == 8


def test_memory_baseline_rejects_inconsistent_rank_and_bad_k():
    with pytest.raises(ValueError):
        MemoryBaseline().fit(np.zeros((2, 3)), np.array(["a", "a"]), np.array([1, 2]))
    m = MemoryBaseline().fit(np.eye(3), np.array(["a", "b", "c"]), np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        m.predict(np.eye(3), k=0)


def test_models_reexports_the_single_implementation():
    from gogame import models
    assert models.MemoryBaseline is MemoryBaseline


def test_memory_baseline_colors_make_per_player_color_centroids_and_nan_is_zero():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, 1.0]])
    m = MemoryBaseline(standardize=False).fit(X, np.array(["a", "a", "a"]), np.array([3, 3, 3]), colors=np.array([0, 1, 1]))
    assert m.n_centroids == 2
    pred, sim = m.predict(np.array([[np.nan, 1.0]]))
    assert pred.dtype == np.int64 and pred[0] == 3 and np.isfinite(sim).all()
