import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch only in system python3 / Colab")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "experiments" / "e08"))
sys.path.insert(0, str(ROOT / "src"))

from student import Student, blocked_mask, gather_stats, make_planes, profile_corr, sym_perm, sym_planes  # noqa: E402
import prep  # noqa: E402
from gogame.board import iter_own_positions  # noqa: E402
from gogame.planes import encode_position, to_float  # noqa: E402
from gogame.sgf import parse_sgf  # noqa: E402

SGF = "(;SZ[19];B[qd];W[dd];B[pq];W[dp];B[oc];W[pd];B[pc];W[od];B[nd];W[oe];B[ne];W[of])"


def _rows():
    return prep.one_game(("u", SGF, 0))[1]


def test_prep_planes_match_encoder():
    g = parse_sgf(SGF)
    rows = _rows()
    pos = list(iter_own_positions(g, 0))
    assert len(rows) == len(pos)
    code = torch.from_numpy(np.stack([r[0] for r in rows]))
    t = lambda i: torch.tensor([r[i] for r in rows])  # noqa: E731
    x = make_planes(code, t(1), t(2), t(3), t(7), t(6)).numpy()
    for i, p in enumerate(pos):
        ref = to_float(encode_position(p))[:8].astype(np.float32)
        assert np.allclose(x[i, :8], ref, atol=1e-3), i
        assert np.allclose(x[i, 8], p.ply / 120.0)


def test_sym_perm_consistent_with_planes():
    for k in range(8):
        pt = (3, 14)
        a = torch.zeros(1, 1, 19, 19)
        a[0, 0, pt[0], pt[1]] = 1
        b = sym_planes(a, k)
        assert int(b.flatten().argmax()) == int(sym_perm(k)[pt[0] * 19 + pt[1]])
        assert sorted(sym_perm(k).tolist()) == list(range(362))


def test_student_policy_normalised_and_masked():
    torch.manual_seed(0)
    rows = _rows()
    code = torch.from_numpy(np.stack([r[0] for r in rows]))
    t = lambda i: torch.tensor([r[i] for r in rows])  # noqa: E731
    x = make_planes(code, t(1), t(2), t(3), t(7), t(6))
    blk = blocked_mask(code, t(3))
    net = Student(2, 8).eval()
    lp = net(x, blk)
    assert lp.shape == (len(rows), 11, 362)
    assert torch.allclose(lp.exp().sum(2), torch.ones(len(rows), 11), atol=1e-4)
    assert float(lp.detach().exp()[:, :, :361][blk.unsqueeze(1).expand(-1, 11, -1)].max()) < 1e-6
    v, r, e = gather_stats(lp, t(4))
    assert v.shape == r.shape == e.shape == (len(rows), 11) and int(r.min()) >= 1 and float(e.min()) >= 0


def test_profile_corr_detects_constant_output():
    rng = np.random.default_rng(0)
    teacher = rng.normal(size=(500, 11))
    good = profile_corr(teacher + 0.1 * rng.normal(size=teacher.shape), teacher)
    assert min(good["raw"]) > 0.9 and min(good["centred"]) > 0.8
    const = np.ones_like(teacher) * 0.5 + 1e-9 * rng.normal(size=teacher.shape)
    assert max(profile_corr(const, teacher)["student_std"]) < 1e-6
