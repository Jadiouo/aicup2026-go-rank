"""Submission writers and strict format checkers (SDD 3.3, 5.13). The set of returned codes is a pure function of the input."""
from __future__ import annotations

import csv
import hashlib
import io as _io
from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import ID_TO_RANK, RANKS
from .errors import Issue, SubmissionError

__all__ = ["RANK_LEGAL", "ALL_CODES", "Issue", "write_rank_submission", "write_player_submission",
           "check_rank_submission", "check_player_submission"]

RANK_LEGAL: frozenset[str] = frozenset(RANKS)
ALL_CODES = ("E_PARSE", "E_HEADER", "E_ROWS", "E_MISSING", "E_DUP", "E_UNKNOWN", "E_QPREFIX", "E_RANK", "E_TOPN",
             "E_TOPDUP", "E_POOL", "E_PPREFIX")
RANK_HEADER = ["question_id", "pred_rank"]
PLAYER_HEADER = ["question_id"] + [f"top_{i}" for i in range(1, 6)]
_PREFIXES = ("pub_", "priv_")


# ----------------------------------------------------------------------------- writers
def _write(path, header: list[str], rows: Sequence[Sequence[str]]) -> str:
    buf = _io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_rank_submission(qids: Sequence[str], pred_ids: np.ndarray, path) -> str:
    """qids in output order (public then private, test-CSV order); pred_ids int rank ids 0..9. Returns sha256."""
    pred = np.asarray(pred_ids)
    if pred.ndim != 1 or len(pred) != len(qids):
        raise SubmissionError(f"{len(qids)} question ids but predictions of shape {pred.shape}")
    rows = []
    for q, r in zip(qids, pred):
        if int(r) != r or int(r) not in ID_TO_RANK:
            raise SubmissionError(f"illegal rank id {r!r} for {q}")
        rows.append([q, ID_TO_RANK[int(r)]])
    return _write(path, RANK_HEADER, rows)


def write_player_submission(qids: Sequence[str], top5: np.ndarray, path, scores: np.ndarray | None = None) -> str:
    """top5: [Q,5] normalised (prefixed) player ids. With scores [Q,5], each row is sorted by score descending,
    ties by player id ascending (deterministic). Without scores the given order is kept."""
    t = np.asarray(top5, dtype=object)
    if t.ndim != 2 or t.shape != (len(qids), 5):
        raise SubmissionError(f"top5 must have shape ({len(qids)}, 5), got {t.shape}")
    rows = []
    for i, q in enumerate(qids):
        ids = [str(x) for x in t[i]]
        if scores is not None:
            s = np.asarray(scores)[i]
            order = sorted(range(5), key=lambda j: (-float(s[j]), ids[j]))
            ids = [ids[j] for j in order]
        rows.append([q] + ids)
    return _write(path, PLAYER_HEADER, rows)


# ----------------------------------------------------------------------------- checkers
def _read_rows(path) -> list[list[str]] | None:
    try:
        text = Path(path).read_bytes().decode("utf-8")
        rows = [r for r in csv.reader(_io.StringIO(text, newline=""), strict=True) if r]
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return rows or None


def _prefix(s: str) -> str | None:
    for p in _PREFIXES:
        if s.startswith(p):
            return p
    return None


def _swap(s: str) -> str | None:
    p = _prefix(s)
    if p == "pub_":
        return "priv_" + s[4:]
    if p == "priv_":
        return "pub_" + s[5:]
    return None


def _check_common(rows, header, public_ids, private_ids, expected_rows) -> tuple[list[Issue], list[list[str]], set[str]] | list[Issue]:
    if rows is None:
        return [Issue("E_PARSE", None, "file is not a UTF-8 CSV that can be parsed (or is empty)")]
    if rows[0] != header:
        return [Issue("E_HEADER", None, f"header {rows[0]} != {header}")]
    body = rows[1:]
    if any(len(r) != len(header) for r in body):
        bad = next(i for i, r in enumerate(body) if len(r) != len(header))
        return [Issue("E_PARSE", bad, f"row has {len(body[bad])} fields, expected {len(header)}")]
    expected = set(public_ids) | set(private_ids)
    n_exp = len(public_ids) + len(private_ids) if expected_rows is None else expected_rows
    issues: list[Issue] = []
    if len(body) != n_exp:
        issues.append(Issue("E_ROWS", None, f"{len(body)} data rows, expected {n_exp}"))
    seen: dict[str, int] = {}
    for i, r in enumerate(body):
        q = r[0]
        seen[q] = seen.get(q, 0) + 1
        if seen[q] == 2:
            issues.append(Issue("E_DUP", i, f"question_id {q!r} appears more than once"))
        if q not in expected:
            sw = _swap(q)
            if _prefix(q) is not None and (sw is None or sw not in expected):
                issues.append(Issue("E_UNKNOWN", i, f"unknown question_id {q!r}"))
            else:
                issues.append(Issue("E_QPREFIX", i, f"question_id {q!r} has a missing, rewritten or wrong prefix"))
    missing = sorted(expected - set(seen))
    if missing:
        issues.append(Issue("E_MISSING", None, f"{len(missing)} expected question_id missing, e.g. {missing[:3]}"))
    return issues, body, expected


def check_rank_submission(path, public_ids: Sequence[str], private_ids: Sequence[str],
                          expected_rows: int | None = None) -> list[Issue]:
    res = _check_common(_read_rows(path), RANK_HEADER, public_ids, private_ids, expected_rows)
    if isinstance(res, list):
        return res
    issues, body, _ = res
    for i, r in enumerate(body):
        if r[1] not in RANK_LEGAL:
            issues.append(Issue("E_RANK", i, f"illegal pred_rank {r[1]!r}"))
    return issues


def check_player_submission(path, public_ids: Sequence[str], private_ids: Sequence[str],
                            cand_pub: set[str], cand_priv: set[str], expected_rows: int | None = None) -> list[Issue]:
    res = _check_common(_read_rows(path), PLAYER_HEADER, public_ids, private_ids, expected_rows)
    if isinstance(res, list):
        return res
    issues, body, _ = res
    pub, priv = set(public_ids), set(private_ids)
    for i, r in enumerate(body):
        q = r[0]
        if q in pub:
            pool, qp = cand_pub, "pub_"
        elif q in priv:
            pool, qp = cand_priv, "priv_"
        else:
            continue
        tops = r[1:]
        nonempty = [t for t in tops if t != ""]
        if len(nonempty) < 5:
            issues.append(Issue("E_TOPN", i, f"{len(nonempty)} of 5 candidates given"))
        if len(set(nonempty)) != len(nonempty):
            issues.append(Issue("E_TOPDUP", i, "duplicate candidate ids in one row"))
        if any(t not in pool for t in nonempty):
            bad = next(t for t in nonempty if t not in pool)
            issues.append(Issue("E_POOL", i, f"candidate {bad!r} not in the candidate list of this question's test set"))
        if any(_prefix(t) != qp for t in nonempty):
            bad = next(t for t in nonempty if _prefix(t) != qp)
            issues.append(Issue("E_PPREFIX", i, f"candidate {bad!r} prefix does not match question prefix {qp!r}"))
    return issues
