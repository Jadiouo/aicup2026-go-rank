"""tdd 4.8: features_l0 (hand-computed expectations from tdd 6.5)."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from fixtures.handmade import HANDMADE
from gogame.features_l0 import L0Batch, compute_l0, compute_l0_many, l0_schema
from gogame.io import GameRecord
from gogame.sgf import parse_sgf
from gogame.symmetry import SHAPE_INDEX, apply_sym
from helpers import make_corpus, sgf

GOLDEN_CSV = Path(__file__).parent / "fixtures" / "golden" / "games_50.csv"
SNAPSHOT = Path(__file__).parent / "fixtures" / "golden" / "golden_l0_snapshot.json"

DENSE_NAMES = (
    "n_plies", "n_own", "n_own_pass", "cap_own", "cap_lost",
    "line_1", "line_2", "line_3", "line_4", "line_c",
    "lineo_1", "lineo_2", "lineo_3", "lineo_4", "lineo_c",
    "iso_frac", "contact_opp_frac",
    "dopp_1", "dopp_2", "dopp_3", "dopp_4_6", "dopp_7p", "dopp_na", "dopp_mean",
    "dself_1", "dself_2", "dself_3", "dself_4_6", "dself_7p", "dself_na", "dself_mean",
)


def game(name):
    return parse_sgf(HANDMADE.get(name, name))


def L(name, color=0):
    g = compute_l0(game(name), color)
    names = l0_schema().dense_names
    return g, dict(zip(names, g.dense.tolist()))


def golden_games():
    df = pd.read_csv(GOLDEN_CSV, dtype=str, keep_default_na=False)
    return [(parse_sgf(s), 0 if c == "B" else 1) for s, c in zip(df["sgf_content"], df["color"])]


def test_schema_fixed_length_names():
    s = l0_schema()
    assert len(s.dense_names) == 31
    assert s.dense_names == DENSE_NAMES
    assert s.sparse_size == 10125
    assert s.groups == {"open_pos": (0, 361), "open_bigram": (361, 1385), "shape": (1385, 10125)}


def test_mini_B_dense_counts():
    g, d = L("mini_B")
    assert g.dense.dtype == np.float32 and g.dense.shape == (31,)
    assert (d["n_plies"], d["n_own"], d["n_own_pass"], d["cap_own"], d["cap_lost"]) == (5, 3, 0, 0, 0)


def test_mini_B_lines():
    _, d = L("mini_B")
    assert d["line_4"] == pytest.approx(2 / 3) and d["line_c"] == pytest.approx(1 / 3)
    assert d["line_1"] == d["line_2"] == d["line_3"] == 0
    for k in ("1", "2", "3", "4", "c"):
        assert d["lineo_" + k] == pytest.approx(d["line_" + k])


def test_mini_B_iso_contact():
    _, d = L("mini_B")
    assert d["iso_frac"] == 1.0 and d["contact_opp_frac"] == 0.0


def test_mini_B_dist_opp():
    _, d = L("mini_B")
    assert d["dopp_7p"] == 1.0
    assert d["dopp_1"] == d["dopp_2"] == d["dopp_3"] == d["dopp_4_6"] == 0
    assert d["dopp_na"] == pytest.approx(1 / 3)
    assert d["dopp_mean"] == pytest.approx(9.5)


def test_mini_B_dist_self():
    _, d = L("mini_B")
    assert d["dself_4_6"] == pytest.approx(0.5) and d["dself_7p"] == pytest.approx(0.5)
    assert d["dself_na"] == pytest.approx(1 / 3)
    assert d["dself_mean"] == pytest.approx(9.0)


def test_mini_B_sparse_exact():
    g, _ = L("mini_B")
    assert g.sp_idx.tolist() == [60, 72, 161, 628, 954, 1385]
    assert g.sp_val.tolist() == [1.0] * 6
    assert g.sp_idx.dtype == np.int32 and g.sp_val.dtype == np.float32


def test_corner_move_shape_code():
    g, _ = L(sgf("B:aa"))
    assert g.sp_idx.tolist()[-1] == 1385 + 519
    assert 519 == SHAPE_INDEX[1023]


def test_shape_uses_board_before_move():
    # W plays first at jj (9,9); B then plays at kj (9,10), adjacent east of the enemy stone.
    g, d = L(sgf("W:jj B:kj"), 0)
    # ring of kj: W neighbour (col 9) is enemy => code with a single 2 => canonical code 8 => index 8
    assert g.sp_idx.tolist()[-1] == 1385 + 8
    assert d["contact_opp_frac"] == 1.0 and d["iso_frac"] == 0.0
    # the mover's own stone is not part of the ring: B stones in ring would give a 1
    g2, _ = L(sgf("B:jj W:kj"), 1)   # white annotated, black neighbour is the enemy
    assert g2.sp_idx.tolist()[-1] == 1385 + 8


def test_shape_counts_after_capture():
    # capture_single: B captures at jk (ply 6); then B plays jj (the captured point) as an extra own move.
    s = HANDMADE["capture_single"][:-1] + ";W[aj];B[jj])"
    g = parse_sgf(s)
    r = compute_l0(g, 0)
    # last B move at (9,9): neighbours N (8,9)=jk? use ring: N=(8,9) empty, E=(9,10) B own, S=(10,9) B own, W=(9,8) B own
    # NE (8,10) empty? ji=(8,9): B played ji => N own.  We only assert that the ring contains no enemy stone
    # (the white stone at jj was captured) => last own move is not a contact move.
    d = dict(zip(l0_schema().dense_names, r.dense.tolist()))
    assert d["contact_opp_frac"] < 1.0
    n_np = int(d["n_own"])
    assert n_np == 5
    # before the capture B's 4th move (kj) touches W at jj => contact
    assert d["contact_opp_frac"] > 0.0


def test_capture_counts_features():
    _, d = L("capture_single", 0)
    assert (d["cap_own"], d["cap_lost"]) == (1, 0)
    _, d = L("capture_single", 1)
    assert (d["cap_own"], d["cap_lost"]) == (0, 1)


def test_cap_lost_counts_trailing_enemy_capture():
    # W's last move captures; annotated B must see it as lost even though B has no later move.
    _, d = L("capture_corner", 0)   # B:ba W:aa B:ab  -> B captures last (own)
    assert (d["cap_own"], d["cap_lost"]) == (1, 0)
    _, d = L("capture_corner", 1)
    assert (d["cap_own"], d["cap_lost"]) == (0, 1)
    # trailing: B plays last? W annotated, W's last move is own; check opposite: last move by enemy
    _, d = L("capture_single", 0)   # last move jk is B (own)
    _, d2 = L(sgf("B:sa W:ba B:sb W:ab B:aa W:ss"), 1)   # suicide of B's aa, harmless
    assert d2["cap_own"] == 0
    s = sgf("W:ba B:aa W:ss B:sa W:ab")   # W captures B corner stone with last move ab? aa has nbrs ba(W), ab(W)
    _, dw = L(s, 0)
    assert dw["cap_lost"] == 1 and dw["cap_own"] == 0


def test_pass_own_counted_not_in_denominators():
    _, d = L("pass_variants", 1)
    assert d["n_own"] == 2 and d["n_own_pass"] == 2
    assert d["line_1"] == d["line_c"] == 0 and d["iso_frac"] == 0 and d["dopp_na"] == 0
    assert np.isnan(d["dopp_mean"]) and np.isnan(d["dself_mean"])
    g, _ = L("pass_variants", 1)
    assert g.sp_idx.size == 0


def test_empty_game():
    g, d = L("empty_game", 0)
    assert d["n_plies"] == 0 and d["n_own"] == 0
    assert g.sp_idx.size == 0 and np.isnan(d["dopp_mean"])


def _sym_game(text, k):
    g = parse_sgf(text)
    mv = g.moves.copy()
    for i in range(len(mv)):
        if mv[i, 1] >= 0:
            mv[i, 1], mv[i, 2] = apply_sym(k, int(mv[i, 1]), int(mv[i, 2]))
    return type(g)(moves=mv, setup=g.setup, flags=g.flags)


def test_all_features_invariant_under_8_syms():
    texts = [HANDMADE["mini_B"], HANDMADE["capture_single"], HANDMADE["ko"]]
    gold = golden_games()[:6]
    for text in texts:
        for color in (0, 1):
            base = compute_l0(parse_sgf(text), color)
            for k in range(1, 8):
                o = compute_l0(_sym_game(text, k), color)
                np.testing.assert_array_equal(base.dense, o.dense)
                np.testing.assert_array_equal(base.sp_idx, o.sp_idx)
                np.testing.assert_array_equal(base.sp_val, o.sp_val)
    for g, color in gold:
        base = compute_l0(g, color)
        for k in range(1, 8):
            mv = g.moves.copy()
            for i in range(len(mv)):
                if mv[i, 1] >= 0:
                    mv[i, 1], mv[i, 2] = apply_sym(k, int(mv[i, 1]), int(mv[i, 2]))
            o = compute_l0(type(g)(moves=mv, setup=g.setup, flags=g.flags), color)
            np.testing.assert_array_equal(base.dense, o.dense)
            np.testing.assert_array_equal(base.sp_idx, o.sp_idx)
            np.testing.assert_array_equal(base.sp_val, o.sp_val)


def test_white_annotated_mirror_equals_black():
    a = compute_l0(game("white_first"), 1)
    b = compute_l0(game("mini_B"), 0)
    np.testing.assert_array_equal(a.dense, b.dense)
    np.testing.assert_array_equal(a.sp_idx, b.sp_idx)
    np.testing.assert_array_equal(a.sp_val, b.sp_val)


def test_white_annotated_mirror_on_real_games():
    for g, color in golden_games()[:10]:
        sw = type(g)(moves=np.column_stack([1 - g.moves[:, 0], g.moves[:, 1:]]).astype(g.moves.dtype),
                     setup=g.setup, flags=g.flags)
        a = compute_l0(g, color)
        b = compute_l0(sw, 1 - color)
        np.testing.assert_array_equal(a.dense, b.dense)
        np.testing.assert_array_equal(a.sp_idx, b.sp_idx)


def test_white_annotated_real_orientation():
    b = L("mini_B", 0)[1]
    w = L("mini_B", 1)[1]            # W owns dp(ply1) and dd(ply3); both have a defined prev_opp
    assert b["dopp_na"] == pytest.approx(1 / 3)
    assert w["dopp_na"] == 0.0 and w["n_own"] == 2


def test_only_annotated_side_moves_used():
    text = sgf("B:pd W:aa B:pp W:ab B:dp W:ac")      # B on line 4, W on line 1
    _, b = L(text, 0)
    _, w = L(text, 1)
    assert b["line_4"] == 1.0 and b["line_1"] == 0.0
    assert w["line_1"] == 1.0 and w["line_4"] == 0.0


def test_no_inf_nan_only_allowed():
    names = l0_schema().dense_names
    for g, color in golden_games():
        r = compute_l0(g, color)
        assert not np.isinf(r.dense).any()
        nan = np.isnan(r.dense)
        for i in np.where(nan)[0]:
            assert names[i].endswith("_mean")


def test_opening_window_boundary():
    # 30 plies: far-apart corner/edge points so bigram keys are distinct. Own = B on even plies.
    pts = ["pd", "dp", "pp", "dd", "jj", "pj", "dj", "jd", "jp", "qc", "cq", "qq", "cc", "mc", "mq", "cm", "qm",
           "oc", "oq", "co", "qo", "nc", "nq", "cn", "qn", "mm", "kk", "ll", "ii", "hh"]
    def sg(n):
        return sgf(" ".join(f"{'B' if i % 2 == 0 else 'W'}:{pts[i]}" for i in range(n)))
    a = compute_l0(parse_sgf(sg(24)), 0)
    b = compute_l0(parse_sgf(sg(25)), 0)      # 25th ply (index 24) is an own move whose bigram must be excluded
    ga = a.sp_idx[(a.sp_idx >= 361) & (a.sp_idx < 1385)]
    gb = b.sp_idx[(b.sp_idx >= 361) & (b.sp_idx < 1385)]
    assert a.sp_val[(a.sp_idx >= 361) & (a.sp_idx < 1385)].sum() == 11    # own plies 2,4,...,22
    assert b.sp_val[(b.sp_idx >= 361) & (b.sp_idx < 1385)].sum() == 11
    # ply index 23 is W (not own), so bigram total is unchanged; own ply 24 excluded.  Check with W annotated:
    aw = compute_l0(parse_sgf(sg(24)), 1)    # W own plies 1,3,...,23  -> bigrams at t=1..23 -> 12
    bw = compute_l0(parse_sgf(sg(26)), 1)    # adds W ply 25 (t=25 > 23) -> excluded
    sa = aw.sp_val[(aw.sp_idx >= 361) & (aw.sp_idx < 1385)].sum()
    sb = bw.sp_val[(bw.sp_idx >= 361) & (bw.sp_idx < 1385)].sum()
    assert sa == 12 and sb == 12


def test_open_pos_first_12_own_only():
    pts = ["pd", "dp", "pp", "dd", "jj", "pj", "dj", "jd", "jp", "qc", "cq", "qq", "cc", "mc", "mq", "cm", "qm",
           "oc", "oq", "co", "qo", "nc", "nq", "cn", "qn", "mm", "kk", "ll", "ii", "hh"]
    text = sgf(" ".join(f"{'B' if i % 2 == 0 else 'W'}:{pts[i]}" for i in range(30)))
    r = compute_l0(parse_sgf(text), 0)
    n_open = int(((r.sp_idx < 361)).sum())
    assert n_open == 12                              # 15 own moves, only the first 12 count


def test_sparse_indices_strictly_increasing():
    for g, color in golden_games():
        r = compute_l0(g, color)
        assert (np.diff(r.sp_idx) > 0).all()
        assert (r.sp_val > 0).all()
        assert r.sp_idx.size == 0 or (r.sp_idx.min() >= 0 and r.sp_idx.max() < 10125)


def _records(n=12):
    df = make_corpus(3, n_ranks=2, players_per_rank=3, games_per_player=2)
    return [GameRecord(uid=f"{r}_{g}", player_id=p, rank_id=0, color=0 if c == "B" else 1, sgf=s, source="train")
            for p, g, r, c, s in zip(df.player_id, df.game_id, df["rank"], df.color, df.sgf_content)][:n]


def test_compute_many_matches_single():
    recs = _records()
    b1 = compute_l0_many(recs, n_jobs=1)
    b2 = compute_l0_many(recs, n_jobs=2)
    assert isinstance(b1, L0Batch)
    assert b1.uids == [r.uid for r in recs] == list(b2.uids)
    np.testing.assert_array_equal(b1.dense, b2.dense)
    assert (b1.sparse != b2.sparse).nnz == 0
    assert b1.dense.shape == (len(recs), 31) and b1.sparse.shape == (len(recs), 10125)
    for i, r in enumerate(recs):
        single = compute_l0(parse_sgf(r.sgf), r.color)
        np.testing.assert_array_equal(b1.dense[i], single.dense)
        row = b1.sparse.getrow(i)
        np.testing.assert_array_equal(row.indices, single.sp_idx)


def test_golden_snapshot():
    import json
    snap = json.loads(SNAPSHOT.read_text())
    games = golden_games()
    for key, entry in snap.items():
        g, color = games[int(key)]
        r = compute_l0(g, color)
        assert [None if np.isnan(x) else x for x in r.dense.tolist()] == entry["dense"]
        assert r.sp_idx.tolist() == entry["sp_idx"]
        assert r.sp_val.tolist() == entry["sp_val"]


def test_dense_simple_features_match_naive_reference():
    """Independent re-computation of counts/lines/distances straight from the move list."""
    names = l0_schema().dense_names
    for g, color in golden_games():
        mv = g.moves.tolist()
        own = [(i, r, c) for i, (col, r, c) in enumerate(mv) if col == color]
        np_own = [(i, r, c) for i, r, c in own if r >= 0]
        r = dict(zip(names, compute_l0(g, color).dense.tolist()))
        assert r["n_plies"] == len(mv) and r["n_own"] == len(own) and r["n_own_pass"] == len(own) - len(np_own)
        lines = [min(rr, cc, 18 - rr, 18 - cc) + 1 for _, rr, cc in np_own]
        for k in (1, 2, 3, 4):
            assert r[f"line_{k}"] == pytest.approx(sum(l == k for l in lines) / max(len(lines), 1), abs=1e-6)
        assert r["line_c"] == pytest.approx(sum(l >= 5 for l in lines) / max(len(lines), 1), abs=1e-6)
        first = lines[:30]
        assert r["lineo_3"] == pytest.approx(sum(l == 3 for l in first) / max(len(first), 1), abs=1e-6)
        # dopp
        ds, na = [], 0
        for i, rr, cc in np_own:
            if i > 0 and mv[i - 1][0] != color and mv[i - 1][1] >= 0:
                ds.append(max(abs(rr - mv[i - 1][1]), abs(cc - mv[i - 1][2])))
            else:
                na += 1
        assert r["dopp_na"] == pytest.approx(na / max(len(np_own), 1), abs=1e-6)
        if ds:
            assert r["dopp_mean"] == pytest.approx(np.mean(ds), abs=1e-4)
            assert r["dopp_7p"] == pytest.approx(sum(d >= 7 for d in ds) / len(ds), abs=1e-6)
        ds, na = [], 0
        for j, (i, rr, cc) in enumerate(np_own):
            if j > 0:
                ds.append(max(abs(rr - np_own[j - 1][1]), abs(cc - np_own[j - 1][2])))
            else:
                na += 1
        assert r["dself_na"] == pytest.approx(na / max(len(np_own), 1), abs=1e-6)
        if ds:
            assert r["dself_mean"] == pytest.approx(np.mean(ds), abs=1e-4)


@pytest.mark.slow
def test_signal_line_feature_vs_rank():
    from scipy.stats import spearmanr
    df = make_corpus(1, n_ranks=10, players_per_rank=6, games_per_player=14)
    rank_id = {r: i for i, r in enumerate(["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"])}
    names = list(l0_schema().dense_names)
    x, y = [], []
    for s, c, rk in zip(df.sgf_content, df.color, df["rank"]):
        d = compute_l0(parse_sgf(s), 0 if c == "B" else 1).dense
        x.append(d[names.index("line_3")] + d[names.index("line_4")])
        y.append(rank_id[rk])
    assert abs(spearmanr(x, y)[0]) > 0.5


@pytest.mark.real_data
def test_real_games_no_inf(training_dir):
    from gogame.io import iter_train_games
    recs = []
    for r in iter_train_games(training_dir, ranks=("A",)):
        recs.append(r)
        if len(recs) >= 100:
            break
    b = compute_l0_many(recs)
    assert not np.isinf(b.dense).any()
