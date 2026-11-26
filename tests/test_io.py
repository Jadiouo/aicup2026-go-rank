import numpy as np
import pandas as pd
import pytest

from gogame import io as gio
from gogame.errors import DataFormatError
from helpers import make_contest, make_corpus, sgf

SPLITS = ("pub", "priv")


def _write_train(tmp_path, rank, rows):
    df = pd.DataFrame(rows, columns=["player_id", "game_id", "rank", "color", "sgf_content"])
    p = tmp_path / f"train_{rank}.csv"
    df.to_csv(p, index=False)
    return p


def test_read_train_csv_keeps_commas(tmp_path):
    tricky = '(;SZ[19]C[hello, "quoted", x,y];B[pd];W[dp])'
    p = _write_train(tmp_path, "D", [("pl1", "g_0000001", "D", "B", tricky)])
    df = gio.read_train_csv(p)
    assert df.loc[0, "sgf_content"] == tricky


def test_uid_unique_across_files(tmp_path):
    pd_ = _write_train(tmp_path, "D", [("a", "g_0000001", "D", "B", sgf("B:pd"))])
    p6 = _write_train(tmp_path, "6D", [("b", "g_0000001", "6D", "W", sgf("B:pd"))])
    d, d6 = gio.read_train_csv(pd_), gio.read_train_csv(p6)
    assert d.loc[0, "uid"] == "D_g_0000001"
    assert d6.loc[0, "uid"] == "6D_g_0000001"
    assert d.loc[0, "uid"] != d6.loc[0, "uid"]
    assert d6.loc[0, "color_id"] == 1 and d.loc[0, "color_id"] == 0
    assert d.rank_id.dtype == np.int8 and d.color_id.dtype == np.int8


def test_rank_id_mapping(tmp_path):
    for rank, want in [("D", 0), ("A", 3), ("1D", 4), ("6D", 9)]:
        p = _write_train(tmp_path, rank, [("a", "g_0000001", rank, "B", sgf("B:pd"))])
        assert gio.read_train_csv(p).loc[0, "rank_id"] == want


def test_rank_matches_filename(tmp_path):
    p = _write_train(tmp_path, "D", [("a", "g_0000001", "C", "B", sgf("B:pd"))])
    with pytest.raises(DataFormatError):
        gio.read_train_csv(p)


def test_iter_train_games(tmp_path):
    corpus = make_corpus(1, n_ranks=10, players_per_rank=2, games_per_player=2)
    for rank, g in corpus.groupby("rank"):
        g.to_csv(tmp_path / f"train_{rank}.csv", index=False)
    recs = list(gio.iter_train_games(tmp_path))
    assert len(recs) == len(corpus)
    assert len({r.uid for r in recs}) == len(recs)
    assert recs[0].source == "train" and recs[0].rank_id == 0
    assert {r.color for r in recs} == {0, 1}


def test_cross_rank_players_counts(tmp_path):
    frames = [
        pd.DataFrame({"player_id": ["a", "b"], "rank_id": [0, 0]}),
        pd.DataFrame({"player_id": ["b", "c"], "rank_id": [1, 1]}),
    ]
    assert gio.cross_rank_players(frames) == {"b"}


@pytest.mark.real_data
@pytest.mark.slow
def test_player_not_across_ranks(training_dir):
    frames = [gio.read_train_csv(training_dir / f"train_{r}.csv", usecols=["player_id", "game_id", "rank", "color"])
              for r in gio.RANKS]
    assert all(len(f) == 100_000 for f in frames)
    assert gio.cross_rank_players(frames) == set()
    assert len(set().union(*[set(f.player_id) for f in frames])) == 8007
    assert len(pd.concat(frames).uid.unique()) == 1_000_000


def test_read_questions_empty_cells_not_nan(tmp_path):
    c = make_contest(make_corpus(1), n_per_n=1)
    p = tmp_path / "q.csv"
    c["rank_public"].to_csv(p, index=False)
    df = gio.read_questions_csv(p, "pub")
    row = df.iloc[0]  # N=5
    assert row["sgf_6"] == "" and row["color_6"] == ""
    assert not df.isna().any().any()


def test_validate_questions_ok():
    c = make_contest(make_corpus(1), n_per_n=2)
    assert gio.validate_questions_frame(c["rank_public"], "rank") == []
    assert gio.validate_questions_frame(c["player_private"], "player") == []


def test_validate_questions_num_games_mismatch():
    df = make_contest(make_corpus(1), n_per_n=1)["rank_public"].copy()
    df.loc[1, "num_games"] = "10"   # row 1 has N=10; make it 15 sgf-wise mismatch
    df.loc[1, "sgf_10"] = ""
    df.loc[1, "color_10"] = ""
    codes = {i.code for i in gio.validate_questions_frame(df, "rank")}
    assert "Q_NUM_MISMATCH" in codes


def test_validate_questions_gap_in_sgf_columns():
    df = make_contest(make_corpus(1), n_per_n=1)["rank_public"].copy()
    df.loc[0, "sgf_3"] = ""
    codes = {i.code for i in gio.validate_questions_frame(df, "rank")}
    assert "Q_GAP" in codes


def test_validate_questions_mixed_colors():
    df = make_contest(make_corpus(1), n_per_n=1)["rank_public"].copy()
    df.loc[0, "color_1"] = "B"
    df.loc[0, "color_2"] = "W"
    codes = {i.code for i in gio.validate_questions_frame(df, "rank")}
    assert "Q_MIXED_COLOR" in codes


def test_validate_player_questions_requires_B():
    df = make_contest(make_corpus(1), n_per_n=1)["player_public"].copy()
    df.loc[0, "color_1"] = "W"
    codes = {i.code for i in gio.validate_questions_frame(df, "player")}
    assert "Q_NOT_BLACK" in codes


def test_validate_questions_duplicate_qid():
    df = make_contest(make_corpus(1), n_per_n=1)["rank_public"].copy()
    df.loc[1, "question_id"] = df.loc[0, "question_id"]
    codes = {i.code for i in gio.validate_questions_frame(df, "rank")}
    assert "Q_DUP_QID" in codes


def test_validate_questions_bad_columns():
    df = make_contest(make_corpus(1), n_per_n=1)["rank_public"].copy()
    codes = {i.code for i in gio.validate_questions_frame(df.drop(columns=["color_20"]), "rank")}
    assert "Q_COLUMNS" in codes


def test_normalize_qid_with_prefix():
    assert gio.normalize_qid("pub_q_0001", "pub") == "pub_q_0001"
    with pytest.raises(DataFormatError):
        gio.normalize_qid("priv_q_0001", "pub")


def test_normalize_qid_without_prefix():
    assert gio.normalize_qid("q_0001", "pub") == "pub_q_0001"
    assert gio.normalize_qid("q_0001", "priv") == "priv_q_0001"


def test_normalize_pid_both_forms():
    assert gio.normalize_pid("pub_p_0203", "pub") == "pub_p_0203"
    assert gio.normalize_pid("p_0203", "pub") == "pub_p_0203"
    with pytest.raises(DataFormatError):
        gio.normalize_pid("priv_p_0203", "pub")
    assert gio.normalize_pid("pub_p_0203 ", "pub") == "pub_p_0203 "   # no strip
    assert gio.normalize_pid("pub_p_0203 ", "pub") != "pub_p_0203"


@pytest.mark.parametrize("split,name", [("pub", "public"), ("priv", "private")])
def test_read_candidates_ids_normalized(tmp_path, split, name):
    corp = make_corpus(1)
    a = make_contest(corp, with_prefix=False, seed=5)
    b = make_contest(corp, with_prefix=True, seed=5)
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a[f"cand_{name}"].to_csv(pa, index=False)
    b[f"cand_{name}"].to_csv(pb, index=False)
    da, db = gio.read_candidates_csv(pa, split), gio.read_candidates_csv(pb, split)
    assert set(da.player_id) == set(db.player_id)
    assert all(p.startswith(split + "_") for p in da.player_id)
    assert da.uid.iloc[0] == f"cand_{split}:{da.game_id.iloc[0]}"


def test_cand_uid_rule():
    assert gio.cand_uid("g_0000001", "pub") == "cand_pub:g_0000001"
    assert gio.cand_uid("g_0000001", "priv") == "cand_priv:g_0000001"
    train_uid, test_uid = "D_g_0000001", "pub_q_0001#3"
    assert gio.cand_uid("g_0000001", "pub") not in (train_uid, test_uid)
    assert gio.cand_uid("g_0000001", "pub").startswith("cand_")
    assert not train_uid.startswith("cand_") and not test_uid.startswith("cand_")


def test_explode_question_uid_normalized():
    c = make_contest(make_corpus(1), n_per_n=1, with_prefix=False)
    row = c["rank_public"].iloc[0]
    recs = gio.explode_question(row, "pub")
    assert len(recs) == 5
    assert recs[0].uid == "pub_q_0001#1" and recs[4].uid == "pub_q_0001#5"
    assert recs[0].source == "test_pub" and recs[0].rank_id is None
    assert recs[0].color in (0, 1)
    # with-prefix input gives identical uids
    c2 = make_contest(make_corpus(1), n_per_n=1, with_prefix=True)
    assert gio.explode_question(c2["rank_public"].iloc[0], "pub")[0].uid == "pub_q_0001#1"
