import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / ".scratch/e07/site"))
pytest.importorskip("torch", reason="torch lives in .scratch/e07/site")
from scripts.experiments.e07 import lib  # noqa: E402
import torch  # noqa: E402


def test_identity_init_equals_cosine():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 50)).astype(np.float32)
    m = lib.Metric(50)
    E = lib.embed_all(m, X)
    a, b = lib.unit(E[:5].mean(0)), lib.unit(E[5:10].mean(0))
    c, d = lib.unit(X[:5].mean(0)), lib.unit(X[5:10].mean(0))
    assert abs(a @ b - c @ d) < 1e-5


def test_episode_disjoint_and_loss_backward():
    rng = np.random.default_rng(1)
    rows = [np.arange(i * 40, i * 40 + 30 + i % 5) for i in range(20)]
    pl, q, l, mk = lib.sample_episode(rng, rows, 8, 10, 12)
    for j in range(8):
        assert not set(q[j]) & set(l[j][mk[j]])
        assert set(q[j]) <= set(rows[pl[j]])
    Z = torch.randn(900, 50)
    m = lib.Metric(50)
    m.calibrate(Z[:200])
    loss = lib.centroid_loss(m, Z, q, l, mk)
    loss.backward()
    assert torch.isfinite(loss) and m.alpha.grad is not None and m.alpha.grad.abs() > 0
