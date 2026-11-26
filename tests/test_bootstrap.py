import numpy as np
import pytest

from gogame.metrics import BootResult, balanced_weights, paired_bootstrap


def _groups(n):
    return np.array([f"p{i}" for i in range(n)])


def test_bootstrap_identical_models():
    rng = np.random.default_rng(0)
    sa = rng.random(50)
    r = paired_bootstrap(sa, sa.copy(), _groups(50), n_boot=200, seed=1)
    assert r.delta == 0 and r.ci_low == 0 and r.ci_high == 0
    assert r.p_gt[0.0] == 0.0


def test_bootstrap_constant_shift():
    rng = np.random.default_rng(0)
    sb = rng.random(60)
    groups = np.array([f"p{i // 3}" for i in range(60)])
    r = paired_bootstrap(sb + 0.1, sb, groups, n_boot=300, seed=1)
    assert r.delta == pytest.approx(0.1, abs=1e-12)
    assert r.ci_low == pytest.approx(0.1, abs=1e-12)
    assert r.ci_high == pytest.approx(0.1, abs=1e-12)
    assert r.n_groups == 20 and r.n_boot == 300


def test_bootstrap_deterministic():
    rng = np.random.default_rng(0)
    sa, sb = rng.random(40), rng.random(40)
    g = _groups(40)
    r1 = paired_bootstrap(sa, sb, g, n_boot=300, seed=7)
    r2 = paired_bootstrap(sa, sb, g, n_boot=300, seed=7)
    r3 = paired_bootstrap(sa, sb, g, n_boot=300, seed=8)
    assert r1 == r2
    assert (r1.ci_low, r1.ci_high) != (r3.ci_low, r3.ci_high)


def test_bootstrap_resamples_by_group():
    # two groups with delta +1 / -1 (n_g = 2 each). Each resample is one of (2,0),(1,1),(0,2) -> +1, 0, -1.
    sa = np.array([1.0, 1.0, 0.0, 0.0])
    sb = np.array([0.0, 0.0, 1.0, 1.0])
    groups = np.array(["a", "a", "b", "b"])
    r = paired_bootstrap(sa, sb, groups, n_boot=400, seed=3)
    assert r.n_groups == 2
    assert r.delta == 0
    assert r.ci_low == -1.0 and r.ci_high == 1.0  # all three outcomes occur; extremes reachable
    assert 0.0 < r.p_gt[0.0] < 1.0  # only the (2,0) outcome is > 0 (probability 1/4)
    assert r.p_gt[0.0] == pytest.approx(0.25, abs=0.08)


def test_bootstrap_weights_balanced():
    rng = np.random.default_rng(0)
    rank_ids = np.array([0, 0, 0, 1, 2, 2])
    sa, sb = rng.random(6), rng.random(6)
    w = balanced_weights(rank_ids)
    r = paired_bootstrap(sa, sb, _groups(6), n_boot=50, seed=1, weights=w)
    assert r.delta == pytest.approx(float(np.sum(w * (sa - sb)) / np.sum(w)))


def test_bootstrap_p_gt_thresholds():
    rng = np.random.default_rng(0)
    r = paired_bootstrap(rng.random(30), rng.random(30), _groups(30), n_boot=100, seed=1)
    assert set(r.p_gt) == {0.0, 0.015}
    assert all(0.0 <= v <= 1.0 for v in r.p_gt.values())
    assert isinstance(r, BootResult)


def test_bootstrap_records_weights_mode():
    sa, sb = np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
    g = _groups(3)
    assert paired_bootstrap(sa, sb, g, n_boot=10, seed=1).weights_mode == "raw"
    w = balanced_weights(np.array([0, 1, 1]))
    assert paired_bootstrap(sa, sb, g, n_boot=10, seed=1, weights=w).weights_mode == "balanced"


def test_bootstrap_default_seed_is_named_stream():
    sa, sb = np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0])
    g = _groups(4)
    from gogame.seeds import derive_seed
    assert paired_bootstrap(sa, sb, g, n_boot=50) == paired_bootstrap(sa, sb, g, n_boot=50, seed=derive_seed(2026, "bootstrap"))


def test_bootstrap_input_validation():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4), _groups(3), n_boot=5, seed=1)
