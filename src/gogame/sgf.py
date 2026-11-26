"""SGF parsing. The ONLY module that parses SGF text (SDD 4.2, 5.3).

Coordinate convention (anti-transposition): SGF "B[pd]" -> first letter is x = column, second letter is
y = row counted from the top.  row = ord(s[1]) - 97, col = ord(s[0]) - 97, board[row, col].
"""
from __future__ import annotations

import collections
import dataclasses
import os
import re
from dataclasses import dataclass

import numpy as np

from .errors import DataFormatError

_MOVE_RE = re.compile(r";\s*([BW])\[([a-z]{0,2})\]")
_SETUP_RE = re.compile(r"(?<![A-Za-z])(AB|AW)((?:\s*\[[a-z]{2}\])+)")
_SETUP_PT_RE = re.compile(r"\[([a-z]{2})\]")
_SIZE_RE = re.compile(r"(?<![A-Za-z])SZ\[\s*(\d+)\s*(?::\s*\d+\s*)?\]")


@dataclass(frozen=True)
class GameFlags:
    has_setup: bool
    bad_size: bool
    nonalt: bool
    bad_coord: int
    first_not_black: bool
    n_pass: int


@dataclass(frozen=True)
class ParsedGame:
    moves: np.ndarray   # int16[T,3]: (color 0=B/1=W, row, col); pass = (color,-1,-1)
    setup: np.ndarray   # int16[S,3]: (color,row,col) from AB/AW; shape (0,3) if none
    flags: GameFlags


def coord_to_rc(s: str) -> tuple[int, int]:
    """'pd' -> (row=3, col=15); '' or 'tt' -> (-1,-1). Raises DataFormatError for anything else."""
    if s == "" or s == "tt":
        return (-1, -1)
    if len(s) != 2 or not all("a" <= ch <= "s" for ch in s):
        raise DataFormatError(f"invalid SGF coordinate {s!r}")
    return (ord(s[1]) - 97, ord(s[0]) - 97)


def rc_to_coord(row: int, col: int) -> str:
    if row < 0 or col < 0:
        return ""
    if row > 18 or col > 18:
        raise DataFormatError(f"point off board: {(row, col)}")
    return chr(97 + col) + chr(97 + row)


def parse_sgf(text: str) -> ParsedGame:
    if not isinstance(text, str):
        raise DataFormatError(f"sgf must be str, got {type(text).__name__}")
    moves: list[tuple[int, int, int]] = []
    bad_coord = 0
    n_pass = 0
    for m in _MOVE_RE.finditer(text):
        color = 0 if m.group(1) == "B" else 1
        raw = m.group(2)
        if raw == "" or raw == "tt":
            moves.append((color, -1, -1))
            n_pass += 1
            continue
        try:
            r, c = coord_to_rc(raw)
        except DataFormatError:
            bad_coord += 1
            moves.append((color, -1, -1))   # treated as pass
            continue
        moves.append((color, r, c))
    setup: list[tuple[int, int, int]] = []
    for m in _SETUP_RE.finditer(text):
        color = 0 if m.group(1) == "AB" else 1
        for pt in _SETUP_PT_RE.findall(m.group(2)):
            try:
                r, c = coord_to_rc(pt)
            except DataFormatError:
                continue
            if r >= 0:
                setup.append((color, r, c))
    sz = _SIZE_RE.search(text)
    bad_size = bool(sz) and int(sz.group(1)) != 19
    colors = [mv[0] for mv in moves]
    nonalt = any(colors[i] == colors[i - 1] for i in range(1, len(colors)))
    flags = GameFlags(
        has_setup=bool(setup),
        bad_size=bad_size,
        nonalt=nonalt,
        bad_coord=bad_coord,
        first_not_black=bool(colors) and colors[0] != 0,
        n_pass=n_pass,
    )
    return ParsedGame(
        moves=np.array(moves, dtype=np.int16).reshape(-1, 3),
        setup=np.array(setup, dtype=np.int16).reshape(-1, 3),
        flags=flags,
    )


# ====================================================================================== format normalisation
# Official FAQ Q13/Q14 (2026-10-29, NOT final): test SGF root may hold only CA/HD/RE (no SZ), and handicap games
# (HD[2]..HD[9]) carry no stones and have B/W reversed.  Current-format games (SZ[19] only, no HD) pass through untouched.
# scripts/katago/kgio.py:normalize_game is the stdlib-only twin used by the KataGo runner (tests/test_sgf_format.py checks both).
HANDICAP_STONES: dict[int, tuple[str, ...]] = {
    2: ("pd", "dp"), 3: ("pd", "dp", "dd"), 4: ("pd", "dp", "dd", "pp"), 5: ("pd", "dp", "dd", "pp", "jj"),
    6: ("pd", "dp", "dd", "pp", "dj", "pj"), 7: ("pd", "dp", "dd", "pp", "dj", "pj", "jj"),
    8: ("pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp"), 9: ("pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp", "jj"),
}
COLOR_SEMANTICS = ("as_annotated", "actual")
DEFAULT_COLOR_SEMANTICS = "as_annotated"
KNOWN_ROOT_PROPS = ("SZ", "HD", "CA", "RE")
SKIP_WARN_FRAC = 0.01
_PROP_RE = re.compile(r"([A-Za-z]+)((?:\s*\[(?:\\.|[^\]\\])*\])+)", re.S)
_VAL_RE = re.compile(r"\[((?:\\.|[^\]\\])*)\]", re.S)
_SZ_ANY_RE = re.compile(r"(?<![A-Za-z])SZ\[([^\]]*)\]")


def get_color_semantics(value: str | None = None) -> str:
    """Explicit argument > env GOGAME_COLOR_SEMANTICS > default.  `as_annotated`: the color column uses the same B/W
    marking as the (possibly reversed) moves, so it is swapped together with them for HD>=2; `actual`: color is the real
    colour and is kept."""
    v = value or os.environ.get("GOGAME_COLOR_SEMANTICS") or DEFAULT_COLOR_SEMANTICS
    if v not in COLOR_SEMANTICS:
        raise ValueError(f"color_semantics must be one of {COLOR_SEMANTICS}, got {v!r}")
    return v


def root_props(sgf) -> dict[str, list[str]]:
    """Root node properties {NAME: [values]}; root node = text between the first ';' and the next ';' outside brackets."""
    if not isinstance(sgf, str):
        return {}
    i = sgf.find(";")
    if i < 0:
        return {}
    j, inside, esc = i + 1, False, False
    while j < len(sgf):
        ch = sgf[j]
        if esc:
            esc = False
        elif inside and ch == "\\":
            esc = True
        elif ch == "[":
            inside = True
        elif ch == "]":
            inside = False
        elif ch == ";" and not inside:
            break
        j += 1
    out: dict[str, list[str]] = {}
    for m in _PROP_RE.finditer(sgf[i + 1:j]):
        out.setdefault(m.group(1), []).extend(_VAL_RE.findall(m.group(2)))
    return out


@dataclass(frozen=True)
class NormalizedGame:
    parsed: ParsedGame      # handicap games: moves colour-swapped, setup = handicap stones; otherwise exactly parse_sgf(text)
    eff_color: int          # colour (0=B/1=W) of the annotated player INSIDE parsed.moves
    komi: float             # 6.5 for even games, 0.0 for HD>=1
    hd: int
    skip: str | None        # reason the KataGo runner skips this game (L0 does not skip); None = fine
    root: dict


def _skip_reason(text, root: dict) -> tuple[str | None, int]:
    """(reason, hd).  Mirrors kgio.normalize_game; hd is 0 whenever the reason is not None."""
    if not isinstance(text, str) or not text.strip():
        return "empty_sgf", 0
    if ";" not in text:
        return "no_nodes", 0
    sz = root.get("SZ") or _SZ_ANY_RE.findall(text)[:1]
    if sz:
        m = re.fullmatch(r"\s*(\d+)\s*(?::\s*(\d+)\s*)?", sz[0])
        if not m or int(m.group(1)) != 19 or (m.group(2) is not None and int(m.group(2)) != 19):
            return "bad_size", 0
    if "AB[" in text or "AW[" in text:
        return "has_setup", 0
    if "HD" in root:
        try:
            hd = int(root["HD"][0].strip())
        except ValueError:
            return "bad_hd", 0
        if hd < 0 or hd > 9:
            return "bad_hd", 0
        return None, hd
    return None, 0


def normalize_sgf(text: str, color: int, color_semantics: str | None = None) -> NormalizedGame:
    cs = get_color_semantics(color_semantics)
    color = int(color)
    base = parse_sgf(text)
    root = root_props(text)
    skip, hd = _skip_reason(text, root)
    if hd <= 1:
        return NormalizedGame(base, color, 6.5 if hd == 0 else 0.0, hd, skip, root)
    moves = base.moves.copy()
    moves[:, 0] = 1 - moves[:, 0]
    setup = np.array([(0, *coord_to_rc(x)) for x in HANDICAP_STONES[hd]], dtype=np.int16).reshape(-1, 3)
    flags = dataclasses.replace(base.flags, first_not_black=False)      # white opening is the normal case here
    eff = 1 - color if cs == "as_annotated" else color
    return NormalizedGame(ParsedGame(moves=moves, setup=setup, flags=flags), eff, 0.0, hd, skip, root)


# ---------------------------------------------------------------------------------------------- format report
def format_report(sgfs, colors=None, color_semantics: str | None = None, n_own_moves: int | None = None) -> dict:
    """Statistics of a batch of SGFs: root properties, HD distribution, unparseable games and every skip reason.
    `colors` (0/1 per game) enables the `no_own_moves` count.  Pure statistics, no model code involved."""
    cs = get_color_semantics(color_semantics)
    sgfs = list(sgfs)
    colors = list(colors) if colors is not None else [None] * len(sgfs)
    sz, hd_c, ca, res = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    other, skips = collections.Counter(), collections.Counter()
    n_ab = n_aw = unparse = n_hd_ge2 = 0
    for s, c in zip(sgfs, colors):
        root = root_props(s)
        if not isinstance(s, str) or not s.strip() or ";" not in s:
            unparse += 1
        sz[root["SZ"][0].strip() if "SZ" in root else "absent"] += 1
        hd_c[root["HD"][0].strip() if "HD" in root else "absent"] += 1
        if "CA" in root:
            ca[root["CA"][0]] += 1
        if "RE" in root:
            v = root["RE"][0].strip()
            res["B+" if v.startswith("B+") else "W+" if v.startswith("W+") else "0/draw" if v[:1] in ("0", "D") else "other"] += 1
        n_ab += "AB" in root or (isinstance(s, str) and "AB[" in s)
        n_aw += "AW" in root or (isinstance(s, str) and "AW[" in s)
        for k in root:
            if k not in KNOWN_ROOT_PROPS and k not in ("AB", "AW"):
                other[k] += 1
        skip, hd = _skip_reason(s, root)
        if skip is None and c is not None:
            mv = parse_sgf(s).moves
            eff = (1 - int(c) if hd >= 2 and cs == "as_annotated" else int(c))
            col = mv[:, 0] if hd < 2 else 1 - mv[:, 0]
            if not (col == eff).any():
                skip = "no_own_moves"
        if hd >= 2 and skip is None:
            n_hd_ge2 += 1
        if skip:
            skips[skip] += 1
    n = len(sgfs)
    n_skip = sum(skips.values())
    rep = {"n": n, "SZ": dict(sz), "HD": dict(hd_c), "CA": dict(ca), "RE": dict(res), "n_AB": int(n_ab), "n_AW": int(n_aw),
           "other_props": dict(other), "unparseable": unparse, "skip_reasons": dict(skips), "n_skipped": n_skip,
           "skip_frac": (n_skip / n) if n else 0.0, "n_handicap_ge2": n_hd_ge2, "color_semantics": cs}
    warns = []
    if other:
        warns.append(("WARN", f"unknown root properties {dict(other)} (known: {','.join(KNOWN_ROOT_PROPS)},AB,AW)"))
    if n and rep["skip_frac"] > SKIP_WARN_FRAC:
        warns.append(("WARN", f"skip rate {rep['skip_frac']:.2%} > {SKIP_WARN_FRAC:.0%}: {dict(skips)}"))
    if skips.get("bad_hd"):
        warns.append(("WARN", f"{skips['bad_hd']} games with unusable HD value"))
    if n_hd_ge2:
        warns.append(("INFO", f"{n_hd_ge2} handicap games (HD>=2) depend on the unconfirmed color definition; color_semantics={cs}"))
    rep["warnings"] = warns
    return rep


def format_report_text(rep: dict, title: str = "SGF format report") -> str:
    L = [f"== {title}: {rep['n']} games, color_semantics={rep['color_semantics']} =="]
    for k in ("SZ", "HD", "CA", "RE"):
        L.append(f"  {k:<3}: {rep[k]}")
    L.append(f"  AB games {rep['n_AB']}, AW games {rep['n_AW']}, other root props {rep['other_props'] or '{}'}, unparseable {rep['unparseable']}")
    L.append(f"  skipped {rep['n_skipped']} ({rep['skip_frac']:.2%}) by reason: {rep['skip_reasons'] or '{}'}; handicap (HD>=2) games kept: {rep['n_handicap_ge2']}")
    for lvl, msg in rep["warnings"]:
        L.append(("  !!!!!! WARNING: " if lvl == "WARN" else "  note: ") + msg)
    return "\n".join(L)


def format_gate(rep: dict, strict: bool = False, log=print, title: str = "SGF format report") -> None:
    """Print the report; raise DataFormatError on WARN-level findings only when strict."""
    log(format_report_text(rep, title))
    bad = [m for lvl, m in rep["warnings"] if lvl == "WARN"]
    if strict and bad:
        raise DataFormatError("--strict-format: " + "; ".join(bad))
