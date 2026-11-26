import itertools

import numpy as np
import pytest

from fixtures.handmade import HANDMADE
from gogame.sgf import parse_sgf
from gogame.symmetry import (
    SHAPE_INDEX, SHAPE_VOCAB, apply_sym, apply_sym_grid, canon_ring, canonical_open_sym, ring_code,
)


def _plies_of(name):
    return [tuple(m[1:]) for m in parse_sgf(HANDMADE[name]).moves.tolist()]


def _transform(k, plies):
    return [(-1, -1) if r < 0 else apply_sym(k, r, c) for r, c in plies]


def _ring_perms(ring):
    """All 8 dihedral images of a ring: rotations r[(i+2k)%8], mirrors r[(-i+2k)%8]."""
    out = []
    for k in range(4):
        out.append([ring[(i + 2 * k) % 8] for i in range(8)])
        out.append([ring[(-i + 2 * k) % 8] for i in range(8)])
    return out


def test_eight_syms_are_distinct_group():
    pts = [(3, 15), (1, 7), (9, 2)]
    # tdd 4.6 names (3,15) as an "asymmetric point", but pd lies on the anti-diagonal (r + c == 18), so it has only
    # 4 distinct images (k=3 and k=4 coincide).  The 8-distinct check therefore uses (3,14).  See report.
    assert len({apply_sym(k, 3, 15) for k in range(8)}) == 4
    assert len({apply_sym(k, 3, 14) for k in range(8)}) == 8
    # hand-computed from the SDD table for (3,14): k -> (r,c)
    want = [(3, 14), (3, 4), (15, 14), (15, 4), (14, 3), (14, 15), (4, 3), (4, 15)]
    assert [apply_sym(k, 3, 14) for k in range(8)] == want
    # closure: composition of any two members is again exactly one member (on several points at once)
    for k1, k2 in itertools.product(range(8), repeat=2):
        found = [k3 for k3 in range(8)
                 if all(apply_sym(k3, *p) == apply_sym(k2, *apply_sym(k1, *p)) for p in pts)]
        assert len(found) == 1


def test_sym_grid_matches_point():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 3, size=(19, 19)).astype(np.int8)
    for k in range(8):
        out = apply_sym_grid(k, g)
        assert out.shape == (19, 19) and out.dtype == g.dtype
        for _ in range(60):
            r, c = int(rng.integers(19)), int(rng.integers(19))
            assert out[apply_sym(k, r, c)] == g[r, c]


def test_ring_canon_invariant_under_dihedral():
    rng = np.random.default_rng(1)
    for _ in range(200):
        ring = rng.integers(0, 4, size=8).tolist()
        codes = {canon_ring(r) for r in _ring_perms(ring)}
        assert len(codes) == 1
        assert canon_ring(ring) == min(ring_code(r) for r in _ring_perms(ring))
        assert 0 <= canon_ring(ring) < 65536


def test_ring_vocab_size_8740():
    assert len(SHAPE_VOCAB) == 8740
    assert SHAPE_VOCAB.dtype == np.int32
    assert list(SHAPE_VOCAB) == sorted(set(SHAPE_VOCAB.tolist()))
    assert len(SHAPE_INDEX) == 8740 and SHAPE_INDEX[int(SHAPE_VOCAB[100])] == 100
    # every ring of all 65536 maps into the vocabulary (Burnside count of D4 orbits = 8740)
    assert {canon_ring([(c >> (2 * (7 - i))) & 3 for i in range(8)]) for c in range(65536)} == set(SHAPE_VOCAB.tolist())


def test_known_canonical_codes():
    empty = [0] * 8
    assert canon_ring(empty) == 0 and SHAPE_INDEX[0] == 0
    corner = [3, 3, 0, 0, 0, 3, 3, 3]
    assert canon_ring(corner) == 1023 and SHAPE_INDEX[1023] == 519
    east_opp = [0, 0, 2, 0, 0, 0, 0, 0]
    assert canon_ring(east_opp) == 8 and SHAPE_INDEX[8] == 8
    for pos in (0, 2, 4, 6):   # N, E, S, W opponent: the four orthogonal neighbours are one class
        r = [0] * 8
        r[pos] = 2
        assert canon_ring(r) == 8
    north_own = [1, 0, 0, 0, 0, 0, 0, 0]
    east_own = [0, 0, 1, 0, 0, 0, 0, 0]
    assert canon_ring(north_own) == 4 and canon_ring(east_own) == 4


def test_ring_order_clockwise_from_north():
    north_opp = [2, 0, 0, 0, 0, 0, 0, 0]
    south_opp = [0, 0, 0, 0, 2, 0, 0, 0]
    assert ring_code(north_opp) == 2 * 4 ** 7          # index 0 (N) is the most significant digit
    assert ring_code(south_opp) == 2 * 4 ** 3
    assert ring_code(south_opp) != ring_code(north_opp)
    assert canon_ring(south_opp) == canon_ring(north_opp)
    assert ring_code([0, 0, 0, 0, 0, 0, 0, 1]) == 1    # NW is the least significant digit


def test_open_sym_canonical_for_all_8_versions():
    rng = np.random.default_rng(2)
    seqs = []
    for _ in range(20):
        pts = rng.choice(361, size=24, replace=False)
        seqs.append([(int(p) // 19, int(p) % 19) for p in pts])
    seqs.append(_plies_of("mini_B"))
    seqs.append([(3, 15), (-1, -1), (15, 3), (9, 9), (-1, -1)] + [(r, 2) for r in range(1, 10)])
    for seq in seqs:
        outs = set()
        for k in range(8):
            variant = _transform(k, seq)
            kc = canonical_open_sym(variant)
            outs.add(tuple(_transform(kc, variant)))
        assert len(outs) == 1


def test_open_sym_tie_break_min_k():
    assert canonical_open_sym([(9, 9)]) == 0
    assert canonical_open_sym([]) == 0
    # a sequence symmetric under the transpose has ties; the smallest k wins
    seq = [(3, 3), (15, 15)]
    k = canonical_open_sym(seq)
    assert k == min(kk for kk in range(8) if _transform(kk, seq) == _transform(k, seq))


def test_open_sym_ignores_color():
    assert canonical_open_sym(_plies_of("mini_B")) == canonical_open_sym(_plies_of("white_first"))
    assert canonical_open_sym(_plies_of("mini_B")) == 6   # tdd 6.5: tau is k=6 for mini_B


def test_open_sym_pass_sentinel():
    seq = [(3, 15), (-1, -1), (15, 3)]
    k = canonical_open_sym(seq)
    assert 0 <= k < 8
    seq2 = [(-1, -1)] * 5
    assert canonical_open_sym(seq2) == 0


def test_open_sym_uses_only_first_24():
    base = [(3, 15), (15, 3)] * 12
    a = canonical_open_sym(base + [(1, 1)])
    b = canonical_open_sym(base + [(1, 17)])
    assert a == b
