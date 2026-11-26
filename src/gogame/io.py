"""CSV reading, uid rules, question validation and ID normalisation (SDD 5.2)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import pandas as pd

from .constants import RANK_TO_ID, RANKS
from .errors import DataFormatError, Issue

__all__ = [
    "RANKS", "GameRecord", "read_train_csv", "iter_train_games", "read_questions_csv", "validate_questions_frame",
    "read_candidates_csv", "iter_candidate_games", "explode_question", "normalize_qid", "normalize_pid",
    "cand_uid", "cross_rank_players", "Issue",
]

TRAIN_COLS = ["player_id", "game_id", "rank", "color", "sgf_content"]
QUESTION_COLS = ["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)] + [f"color_{i}" for i in range(1, 21)]
CAND_COLS = ["game_id", "player_id", "sgf_content", "color"]
VALID_NS = (5, 10, 15, 20)


@dataclass(frozen=True)
class GameRecord:
    uid: str
    player_id: str
    rank_id: int | None   # None for test / candidate games
    color: int            # 0=B, 1=W
    sgf: str
    source: str           # "train"|"cand_pub"|"cand_priv"|"test_pub"|"test_priv"


def _color_id(c: str) -> int:
    if c == "B":
        return 0
    if c == "W":
        return 1
    raise DataFormatError(f"bad color value: {c!r}")


def read_train_csv(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read one train_<rank>.csv. Adds rank_id:int8, uid:str, color_id:int8."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)
    for col in ("player_id", "game_id", "rank", "color"):
        if col not in df.columns:
            raise DataFormatError(f"{path}: missing column {col}")
    bad = sorted(set(df["rank"]) - set(RANKS))
    if bad:
        raise DataFormatError(f"{path}: unknown rank values {bad}")
    m = re.fullmatch(r"train_(.+)\.csv", path.name)
    if m and m.group(1) in RANK_TO_ID:
        if set(df["rank"]) - {m.group(1)}:
            raise DataFormatError(f"{path}: rank column disagrees with file name ({sorted(set(df['rank']))})")
    badc = sorted(set(df["color"]) - {"B", "W"})
    if badc:
        raise DataFormatError(f"{path}: bad color values {badc}")
    df["rank_id"] = df["rank"].map(RANK_TO_ID).astype("int8")
    df["uid"] = df["rank"] + "_" + df["game_id"]
    df["color_id"] = (df["color"] == "W").astype("int8")
    return df


def iter_train_games(data_dir: str | Path, ranks: Iterable[str] = RANKS) -> Iterator[GameRecord]:
    for r in ranks:
        df = read_train_csv(Path(data_dir) / f"train_{r}.csv")
        for pid, uid, rid, cid, sgf in zip(df["player_id"], df["uid"], df["rank_id"], df["color_id"], df["sgf_content"]):
            yield GameRecord(uid=uid, player_id=pid, rank_id=int(rid), color=int(cid), sgf=sgf, source="train")


def cross_rank_players(frames: Iterable[pd.DataFrame]) -> set[str]:
    """Players that appear under more than one rank. Frames need player_id and rank_id (or rank)."""
    seen: dict[str, set] = {}
    for f in frames:
        key = "rank_id" if "rank_id" in f.columns else "rank"
        for pid, rk in f[["player_id", key]].drop_duplicates().itertuples(index=False):
            seen.setdefault(pid, set()).add(rk)
    return {p for p, s in seen.items() if len(s) > 1}


def read_questions_csv(path: str | Path, split: Literal["pub", "priv"]) -> pd.DataFrame:
    """Raw question frame (all str, empty cells stay ''). IDs are NOT rewritten; see normalize_qid."""
    _check_split(split)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _check_split(split: str) -> None:
    if split not in ("pub", "priv"):
        raise ValueError(f"split must be 'pub' or 'priv', got {split!r}")


def validate_questions_frame(df: pd.DataFrame, task: Literal["rank", "player"]) -> list[Issue]:
    issues: list[Issue] = []
    if list(df.columns) != QUESTION_COLS:
        return [Issue("Q_COLUMNS", None, f"columns differ from spec: {list(df.columns)[:6]}...")]
    dups = df["question_id"][df["question_id"].duplicated(keep=False)]
    for qid in sorted(set(dups)):
        issues.append(Issue("Q_DUP_QID", None, f"duplicate question_id {qid}"))
    for i, row in enumerate(df.itertuples(index=False)):
        r = dict(zip(QUESTION_COLS, tuple(row)))
        try:
            n = int(r["num_games"])
        except ValueError:
            issues.append(Issue("Q_NUM_GAMES", i, f"num_games not an integer: {r['num_games']!r}"))
            continue
        if n not in VALID_NS:
            issues.append(Issue("W_NUM_GAMES", i, f"num_games={n} not in {VALID_NS}"))
        sgfs = [r[f"sgf_{k}"] for k in range(1, 21)]
        cols = [r[f"color_{k}"] for k in range(1, 21)]
        nonempty = [k for k, s in enumerate(sgfs) if s != ""]
        if len(nonempty) != n:
            issues.append(Issue("Q_NUM_MISMATCH", i, f"num_games={n} but {len(nonempty)} non-empty sgf columns"))
        if nonempty != list(range(len(nonempty))):
            issues.append(Issue("Q_GAP", i, "non-empty sgf columns are not a contiguous prefix"))
        used = [cols[k] for k in nonempty]
        if any(c not in ("B", "W") for c in used):
            issues.append(Issue("Q_COLOR_VALUE", i, f"invalid color values {sorted(set(used))}"))
        elif len(set(used)) > 1:
            issues.append(Issue("Q_MIXED_COLOR", i, "colors differ within one question"))
        if task == "player" and any(c != "B" for c in used):
            issues.append(Issue("Q_NOT_BLACK", i, "player question contains a non-B color"))
        if any(cols[k] != "" for k in range(20) if sgfs[k] == ""):
            issues.append(Issue("Q_COLOR_EXTRA", i, "color given for an empty sgf column"))
    return issues


def normalize_qid(raw: str, split: str) -> str:
    _check_split(split)
    for p in ("pub", "priv"):
        if raw.startswith(p + "_"):
            if p != split:
                raise DataFormatError(f"id {raw!r} has prefix {p!r} but belongs to split {split!r}")
            return raw
    return f"{split}_{raw}"


def normalize_pid(raw: str, split: str) -> str:
    """Same rule as normalize_qid; never strips whitespace."""
    return normalize_qid(raw, split)


def cand_uid(game_id: str, split: str) -> str:
    return f"cand_{split}:{game_id}"


def read_candidates_csv(path: str | Path, split: Literal["pub", "priv"]) -> pd.DataFrame:
    _check_split(split)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in CAND_COLS if c not in df.columns]
    if missing:
        raise DataFormatError(f"{path}: missing columns {missing}")
    df["player_id"] = [normalize_pid(p, split) for p in df["player_id"]]
    df["uid"] = [cand_uid(g, split) for g in df["game_id"]]
    return df


def iter_candidate_games(df: pd.DataFrame, split: str) -> Iterator[GameRecord]:
    for pid, uid, col, sgf in zip(df["player_id"], df["uid"], df["color"], df["sgf_content"]):
        yield GameRecord(uid=uid, player_id=pid, rank_id=None, color=_color_id(col), sgf=sgf, source=f"cand_{split}")


def explode_question(row, split: Literal["pub", "priv"]) -> list[GameRecord]:
    """First num_games non-empty sgf_i of a question row. uid = '<qid>#<i>' (i is 1-based)."""
    _check_split(split)
    qid = normalize_qid(str(row["question_id"]), split)
    n = int(row["num_games"])
    out: list[GameRecord] = []
    for i in range(1, 21):
        s = row[f"sgf_{i}"]
        if s == "":
            continue
        out.append(GameRecord(uid=f"{qid}#{i}", player_id="", rank_id=None, color=_color_id(row[f"color_{i}"]),
                              sgf=s, source=f"test_{split}"))
        if len(out) == n:
            break
    return out
