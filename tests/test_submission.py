import csv
import io as _io
import subprocess
import sys

import numpy as np
import pytest

from gogame.constants import RANKS
from gogame.errors import Issue
from gogame.io import normalize_pid, normalize_qid
from gogame.submission import (ALL_CODES, RANK_LEGAL, check_player_submission, check_rank_submission,
                               write_player_submission, write_rank_submission)
from helpers import make_corpus, make_contest

CORPUS = make_corpus(1, n_ranks=10, players_per_rank=2, games_per_player=2)
CONTEST = make_contest(CORPUS, n_per_n=2, pool_size=8, lib_games=2)       # 8 questions per split


def ids(contest=CONTEST, key="rank"):
    pub = [normalize_qid(q, "pub") for q in contest[f"{key}_public"]["question_id"]]
    prv = [normalize_qid(q, "priv") for q in contest[f"{key}_private"]["question_id"]]
    return pub, prv


PUB, PRV = ids()
CAND_PUB = {normalize_pid(p, "pub") for p in CONTEST["cand_public"]["player_id"]}
CAND_PRV = {normalize_pid(p, "priv") for p in CONTEST["cand_private"]["player_id"]}
assert len(PUB) == len(PRV) == 8 and len(CAND_PUB) == 8


def rank_rows():
    return [[q, RANKS[i % 10]] for i, q in enumerate(PUB + PRV)]


def player_rows():
    pp, pr = sorted(CAND_PUB), sorted(CAND_PRV)
    rows = []
    for i, q in enumerate(PUB):
        rows.append([q] + [pp[(i + j) % 8] for j in range(5)])
    for i, q in enumerate(PRV):
        rows.append([q] + [pr[(i + j) % 8] for j in range(5)])
    return rows


RANK_HDR = ["question_id", "pred_rank"]
PLAYER_HDR = ["question_id"] + [f"top_{i}" for i in range(1, 6)]


def save(tmp_path, header, rows, name="s.csv"):
    buf = _io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    p = tmp_path / name
    p.write_bytes(buf.getvalue().encode("utf-8"))
    return p


def codes(issues):
    return {i.code for i in issues}


def chk_rank(tmp_path, rows, header=RANK_HDR, **kw):
    return codes(check_rank_submission(save(tmp_path, header, rows), PUB, PRV, **kw))


def chk_player(tmp_path, rows, header=PLAYER_HDR, **kw):
    return codes(check_player_submission(save(tmp_path, header, rows), PUB, PRV, CAND_PUB, CAND_PRV, **kw))


# ------------------------------------------------------------------------------ Rank
def test_rank_valid_passes(tmp_path):
    assert check_rank_submission(save(tmp_path, RANK_HDR, rank_rows()), PUB, PRV) == []
    assert RANK_LEGAL == frozenset(RANKS)


def test_rank_missing_question(tmp_path):
    assert chk_rank(tmp_path, rank_rows()[1:]) == {"E_ROWS", "E_MISSING"}


def test_rank_duplicate_question(tmp_path):
    r = rank_rows()
    r[1][0] = r[0][0]
    assert chk_rank(tmp_path, r) == {"E_DUP", "E_MISSING"}


def test_rank_unknown_question(tmp_path):
    r = rank_rows()
    r[0][0] = "pub_q_9999"
    assert chk_rank(tmp_path, r) == {"E_UNKNOWN", "E_MISSING"}


def test_rank_prefix_rewritten_dash(tmp_path):
    r = rank_rows()
    r[0][0] = r[0][0].replace("pub_", "pub-")
    assert chk_rank(tmp_path, r) == {"E_QPREFIX", "E_MISSING"}


def test_rank_prefix_stripped(tmp_path):
    r = rank_rows()
    r[0][0] = r[0][0][len("pub_"):]
    assert chk_rank(tmp_path, r) == {"E_QPREFIX", "E_MISSING"}


def test_rank_prefix_wrong_split(tmp_path):
    r = rank_rows()
    r[0][0] = "priv_" + r[0][0][len("pub_"):]
    assert chk_rank(tmp_path, r) == {"E_QPREFIX", "E_MISSING"}


def test_rank_whitespace_in_qid_is_qprefix(tmp_path):
    r = rank_rows()
    r[0][0] = " " + r[0][0]
    assert chk_rank(tmp_path, r) == {"E_QPREFIX", "E_MISSING"}


def test_rank_illegal_grade_lowercase(tmp_path):
    r = rank_rows()
    r[0][1] = "3d"
    assert chk_rank(tmp_path, r) == {"E_RANK"}


@pytest.mark.parametrize("bad", ["7D", "E", "1d", "", " A", "A ", "d", "6d", "10D"])
def test_rank_illegal_grade_other(tmp_path, bad):
    r = rank_rows()
    r[3][1] = bad
    assert chk_rank(tmp_path, r) == {"E_RANK"}


def test_rank_header_wrong(tmp_path):
    assert chk_rank(tmp_path, rank_rows(), header=["question_id", "rank"]) == {"E_HEADER"}
    assert chk_rank(tmp_path, [[b, a] for a, b in rank_rows()], header=["pred_rank", "question_id"]) == {"E_HEADER"}
    assert chk_rank(tmp_path, [r + ["x"] for r in rank_rows()], header=RANK_HDR + ["extra"]) == {"E_HEADER"}


def test_rank_not_utf8_or_unparseable(tmp_path):
    p = tmp_path / "bin.csv"
    p.write_bytes(b"\xff\xfe\x00\x81\x82garbage\xff")
    assert codes(check_rank_submission(p, PUB, PRV)) == {"E_PARSE"}
    p.write_bytes(b'question_id,pred_rank\n"pub_q_0001,3D\n')
    assert codes(check_rank_submission(p, PUB, PRV)) == {"E_PARSE"}
    p.write_bytes(b"")
    assert codes(check_rank_submission(p, PUB, PRV)) == {"E_PARSE"}
    assert codes(check_rank_submission(tmp_path / "nonexistent.csv", PUB, PRV)) == {"E_PARSE"}


def test_rank_bom_is_rejected_as_header(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes("﻿question_id,pred_rank\n".encode() + b"".join(f"{q},A\n".encode() for q, _ in rank_rows()))
    assert codes(check_rank_submission(p, PUB, PRV)) == {"E_HEADER"}


def test_rank_wrong_row_count_param(tmp_path):
    rows = rank_rows()
    assert chk_rank(tmp_path, rows, expected_rows=16) == set()
    assert chk_rank(tmp_path, rows, expected_rows=800) == {"E_ROWS"}
    assert chk_rank(tmp_path, rows + [["pub_q_0001", "A"]]) == {"E_ROWS", "E_DUP"}


def test_rank_reports_all_issues(tmp_path):
    r = rank_rows()
    del r[0]                                           # A: missing
    r[0][0] = r[0][0][len("pub_"):]                    # B: prefix stripped
    r[1][1] = r[1][1].lower() if r[1][1] != "6D" else "6d"   # C: lower-case grade
    assert chk_rank(tmp_path, r) == {"E_ROWS", "E_MISSING", "E_QPREFIX", "E_RANK"}


def test_issue_objects_have_row_and_detail(tmp_path):
    r = rank_rows()
    r[2][1] = "zz"
    iss = check_rank_submission(save(tmp_path, RANK_HDR, r), PUB, PRV)
    assert len(iss) == 1 and isinstance(iss[0], Issue) and iss[0].code == "E_RANK" and iss[0].row == 2 and "zz" in iss[0].detail


def test_write_rank_roundtrip_and_sha(tmp_path):
    qids = PUB + PRV
    preds = np.arange(len(qids)) % 10
    sha1 = write_rank_submission(qids, preds, tmp_path / "a.csv")
    sha2 = write_rank_submission(qids, preds, tmp_path / "b.csv")
    assert sha1 == sha2 and len(sha1) == 64
    assert check_rank_submission(tmp_path / "a.csv", PUB, PRV) == []
    raw = (tmp_path / "a.csv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    assert raw.startswith(b"question_id,pred_rank\npub_q_0001,D\n")
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == sha1
    write_rank_submission(qids, (preds + 1) % 10, tmp_path / "c.csv")
    assert (tmp_path / "c.csv").read_bytes() != raw


def test_write_rank_rejects_bad_input(tmp_path):
    with pytest.raises(Exception):
        write_rank_submission(["a", "b"], np.array([1, 12]), tmp_path / "x.csv")
    with pytest.raises(Exception):
        write_rank_submission(["a"], np.array([1, 2]), tmp_path / "x.csv")


# ---------------------------------------------------------------------------- Player
def test_player_valid_passes(tmp_path):
    assert check_player_submission(save(tmp_path, PLAYER_HDR, player_rows()), PUB, PRV, CAND_PUB, CAND_PRV) == []


def test_player_missing_duplicate_unknown_question(tmp_path):
    assert chk_player(tmp_path, player_rows()[1:]) == {"E_ROWS", "E_MISSING"}
    r = player_rows()
    r[1][0] = r[0][0]
    assert chk_player(tmp_path, r) == {"E_DUP", "E_MISSING"}
    r = player_rows()
    r[0][0] = "pub_q_9999"
    assert chk_player(tmp_path, r) == {"E_UNKNOWN", "E_MISSING"}


def test_player_prefix_question(tmp_path):
    r = player_rows()
    r[0][0] = r[0][0][len("pub_"):]
    assert chk_player(tmp_path, r) == {"E_QPREFIX", "E_MISSING"}


def test_player_top5_fewer_than_five(tmp_path):
    r = player_rows()
    r[0][5] = ""
    assert chk_player(tmp_path, r) == {"E_TOPN"}
    r = player_rows()
    r[0][3] = r[0][4] = r[0][5] = ""
    assert chk_player(tmp_path, r) == {"E_TOPN"}


def test_player_top5_duplicate_id(tmp_path):
    r = player_rows()
    r[0][2] = r[0][1]
    assert chk_player(tmp_path, r) == {"E_TOPDUP"}


def test_player_empty_slots_dont_count_as_duplicates(tmp_path):
    r = player_rows()
    r[0][4] = r[0][5] = ""
    assert chk_player(tmp_path, r) == {"E_TOPN"}


def test_player_id_not_in_pool(tmp_path):
    r = player_rows()
    r[0][1] = "pub_p_9999"
    assert chk_player(tmp_path, r) == {"E_POOL"}


def test_player_id_from_other_split_pool(tmp_path):
    r = player_rows()
    r[0][1] = sorted(CAND_PRV)[0]
    assert chk_player(tmp_path, r) == {"E_POOL", "E_PPREFIX"}


def test_player_prefix_mismatch_with_question(tmp_path):
    r = player_rows()
    r[0][1] = r[0][1][len("pub_"):]
    assert chk_player(tmp_path, r) == {"E_POOL", "E_PPREFIX"}


def test_player_header_wrong(tmp_path):
    assert chk_player(tmp_path, [r[:5] for r in player_rows()], header=PLAYER_HDR[:5]) == {"E_HEADER"}
    assert chk_player(tmp_path, player_rows(), header=["question_id"] + [f"top{i}" for i in range(1, 6)]) == {"E_HEADER"}


def test_player_whitespace_in_id(tmp_path):
    r = player_rows()
    r[0][1] = r[0][1] + " "
    assert chk_player(tmp_path, r) == {"E_POOL"}


def test_player_candidate_ids_normalized_in_check(tmp_path):
    c = make_contest(CORPUS, n_per_n=2, pool_size=8, lib_games=2, with_prefix=False)
    pub, prv = ids(c, "rank")
    cp = {normalize_pid(p, "pub") for p in c["cand_public"]["player_id"]}
    cv = {normalize_pid(p, "priv") for p in c["cand_private"]["player_id"]}
    assert all(p.startswith("pub_") for p in cp) and all(p.startswith("priv_") for p in cv)
    a, b = sorted(cp), sorted(cv)
    rows = [[q] + [a[(i + j) % 8] for j in range(5)] for i, q in enumerate(pub)] + \
           [[q] + [b[(i + j) % 8] for j in range(5)] for i, q in enumerate(prv)]
    p = save(tmp_path, PLAYER_HDR, rows)
    assert check_player_submission(p, pub, prv, cp, cv) == []
    rows[0][1] = rows[0][1][len("pub_"):]          # un-prefixed p_xxxx
    p = save(tmp_path, PLAYER_HDR, rows, "s2.csv")
    assert codes(check_player_submission(p, pub, prv, cp, cv)) == {"E_POOL", "E_PPREFIX"}


def test_player_checks_only_rows_with_known_question(tmp_path):
    r = player_rows()
    r[0][0] = "pub_q_9999"
    r[0][1] = "garbage"
    assert chk_player(tmp_path, r) == {"E_UNKNOWN", "E_MISSING"}      # top_i of an unknown question is not judged


def test_player_output_sorted_by_similarity_ties_by_id(tmp_path):
    top = np.array([["pub_p_0003", "pub_p_0001", "pub_p_0002", "pub_p_0005", "pub_p_0004"]], dtype=object)
    sc = np.array([[0.5, 0.9, 0.5, 0.1, 0.5]])
    write_player_submission(["pub_q_0001"], top, tmp_path / "p.csv", scores=sc)
    line = (tmp_path / "p.csv").read_text().splitlines()[1].split(",")
    assert line == ["pub_q_0001", "pub_p_0001", "pub_p_0002", "pub_p_0003", "pub_p_0004", "pub_p_0005"]


def test_write_player_roundtrip_and_sha(tmp_path):
    rows = player_rows()
    qids = [r[0] for r in rows]
    top5 = np.array([r[1:] for r in rows], dtype=object)
    sha = write_player_submission(qids, top5, tmp_path / "p.csv")
    assert sha == write_player_submission(qids, top5, tmp_path / "p2.csv")
    assert check_player_submission(tmp_path / "p.csv", PUB, PRV, CAND_PUB, CAND_PRV) == []
    with pytest.raises(Exception):
        write_player_submission(qids, top5[:, :4], tmp_path / "bad.csv")


def test_every_code_covered(tmp_path):
    seen = set()
    r = rank_rows()
    seen |= chk_rank(tmp_path, r[1:])
    seen |= chk_rank(tmp_path, [[r[0][0], "x"]] + r[1:])
    rr = rank_rows(); rr[1][0] = rr[0][0]
    seen |= chk_rank(tmp_path, rr)
    rr = rank_rows(); rr[0][0] = "pub_q_9999"
    seen |= chk_rank(tmp_path, rr)
    rr = rank_rows(); rr[0][0] = rr[0][0][4:]
    seen |= chk_rank(tmp_path, rr)
    seen |= chk_rank(tmp_path, rank_rows(), header=["a", "b"])
    p = tmp_path / "bin.csv"; p.write_bytes(b"\xff\xfe\x81")
    seen |= codes(check_rank_submission(p, PUB, PRV))
    pr = player_rows(); pr[0][5] = ""
    seen |= chk_player(tmp_path, pr)
    pr = player_rows(); pr[0][2] = pr[0][1]
    seen |= chk_player(tmp_path, pr)
    pr = player_rows(); pr[0][1] = sorted(CAND_PRV)[0]
    seen |= chk_player(tmp_path, pr)
    assert seen == set(ALL_CODES) and len(ALL_CODES) == 12


def test_check_cli_exit_code(tmp_path, project_root):
    from helpers import write_contest
    d = tmp_path / "contest"
    write_contest(CONTEST, d)
    good = save(tmp_path, RANK_HDR, rank_rows(), "good.csv")
    bad_rows = rank_rows()
    bad_rows[0][1] = "3d"
    bad = save(tmp_path, RANK_HDR, bad_rows, "bad.csv")
    script = project_root / "scripts" / "check_submission.py"
    base = [sys.executable, str(script), "--task", "rank", "--data-dir", str(d), "--expected-rows", "16"]
    assert subprocess.run(base + [str(good)], capture_output=True).returncode == 0
    r = subprocess.run(base + [str(bad)], capture_output=True, text=True)
    assert r.returncode != 0 and "E_RANK" in (r.stdout + r.stderr)
