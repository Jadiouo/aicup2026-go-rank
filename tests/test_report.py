import numpy as np
import pandas as pd
import pytest

from gogame.overlap import distribution_report
from helpers import make_contest, make_corpus


def qrow(qid, games, color="B", n=None):
    n = len(games) if n is None else n
    row = {"question_id": qid, "num_games": str(n)}
    for k in range(20):
        row[f"sgf_{k + 1}"] = games[k] if k < len(games) else ""
    for k in range(20):
        row[f"color_{k + 1}"] = color if k < len(games) else ""
    return row


def g(plies):
    pts = [f"{chr(97 + i % 19)}{chr(97 + (i * 5 + i // 19) % 19)}" for i in range(plies)]
    return "(;SZ[19]" + "".join(f";{'BW'[i % 2]}[{p}]" for i, p in enumerate(pts)) + ")"


def test_distribution_report_fields():
    g5 = [g(10), g(20), g(30), g(40), g(50)]
    dup = [g(10), g(10), g(30), g(40), g(50)]                     # 1 duplicated game in this question
    ten = [g(10 + i) for i in range(10)]
    pas = "(;SZ[19];B[pd];W[];B[dp];W[tt])"                        # 2 passes
    q = pd.DataFrame([qrow("pub_q_0001", g5, "B"), qrow("pub_q_0002", dup, "W"), qrow("q_0003", ten, "W"),
                      qrow("pub_q_0004", [pas, g(10), g(10), g(10), g(10)], "B")])
    rep = distribution_report(q, "rank")
    assert rep["n_questions"] == 4
    assert rep["n_dist"] == {5: 3, 10: 1}
    assert rep["color1_dist"] == {"B": 2, "W": 2}
    assert rep["n_games"] == 5 + 5 + 10 + 5
    assert rep["qid_prefix"] == {"n_prefixed": 3, "n_total": 4, "state": "mixed"}
    assert rep["flags"]["n_pass"] == 2
    assert rep["flags"]["n_bad_size"] == 0 and rep["flags"]["n_has_setup"] == 0 and rep["flags"]["n_nonalt"] == 0
    assert rep["illegal"] == {"suicide": 0, "occupied": 0, "ko_violation": 0}
    # duplicates: q2 has g(10) twice (1 extra); q4 has g(10) four times (3 extra)
    assert rep["dup_games"] == {"n_extra_copies": 4, "n_questions_with_dup": 2, "frac": 4 / 25}
    plies = np.array([10, 20, 30, 40, 50, 10, 10, 30, 40, 50] + [10 + i for i in range(10)] + [4, 10, 10, 10, 10])
    assert rep["ply"]["min"] == 4 and rep["ply"]["max"] == 50
    assert rep["ply"]["p50"] == pytest.approx(np.quantile(plies, 0.5))
    assert rep["ply"]["p95"] == pytest.approx(np.quantile(plies, 0.95))


def test_distribution_report_player_candidates_and_ks():
    q = pd.DataFrame([qrow("priv_q_1", [g(10)] * 5, "B")])
    cand = pd.DataFrame({"game_id": [f"g{i}" for i in range(7)], "player_id": ["a"] * 4 + ["b"] * 3,
                         "sgf_content": [g(20)] * 7, "color": ["B"] * 7})
    rep = distribution_report(q, "player", cand_df=cand, train_plies=np.arange(5, 200))
    assert rep["all_black"] is True
    assert rep["candidates"] == {"n_players": 2, "games_min": 3, "games_max": 4, "n_not_100": 2, "all_black": True}
    assert rep["ks_vs_train"]["stat"] > 0.5 and rep["ks_vs_train"]["pvalue"] < 0.01
    q2 = pd.DataFrame([qrow("priv_q_1", [g(10)] * 5, "W")])
    assert distribution_report(q2, "player")["all_black"] is False


def test_distribution_report_flags_and_illegal():
    bad = "(;SZ[9];B[aa];W[bb])"                                   # bad size
    occ = "(;SZ[19];B[pd];W[pd];B[dp])"                             # occupied point
    q = pd.DataFrame([qrow("pub_q_1", [bad, occ, g(10), g(10), g(12)], "B")])
    rep = distribution_report(q, "rank")
    assert rep["flags"]["n_bad_size"] == 1
    assert rep["illegal"]["occupied"] == 1


def test_distribution_report_on_fake_contest_runs():
    c = make_contest(make_corpus(1, players_per_rank=2, games_per_player=2), n_per_n=2, pool_size=4, lib_games=3)
    rep = distribution_report(c["rank_public"], "rank")
    assert rep["n_dist"] == {5: 2, 10: 2, 15: 2, 20: 2} and rep["qid_prefix"]["state"] == "all"
    rep = distribution_report(c["player_public"], "player", cand_df=c["cand_public"])
    assert rep["all_black"] and rep["candidates"]["n_players"] == 4 and rep["candidates"]["games_min"] == 3
    c0 = make_contest(make_corpus(1, players_per_rank=2, games_per_player=2), n_per_n=2, pool_size=4, lib_games=3, with_prefix=False)
    assert distribution_report(c0["rank_public"], "rank")["qid_prefix"]["state"] == "none"


def test_report_no_train_overlap_when_disjoint():
    from gogame.overlap import build_prefix_index, build_train_index, exact_overlap, prefix_overlap
    from gogame.io import GameRecord
    rng = np.random.default_rng(0)
    from helpers import random_game
    tr = [random_game(rng, 0.5, 70) for _ in range(20)]
    te = [GameRecord(f"u{i}", "", None, 0, random_game(rng, 0.5, 70), "test_pub") for i in range(20)]
    idx, col = build_train_index(tr, [0] * 20)
    assert exact_overlap(te, idx, col).n_exact == 0
    assert prefix_overlap(te, build_prefix_index(tr, (30,)), k=30).n_match[30] == 0
    assert prefix_overlap(te, build_prefix_index(tr, (24, 30)), k=30, ref_ks=(24,)).n_match[24] == 0
