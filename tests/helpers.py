"""Test helpers (tdd 2.1)."""
from __future__ import annotations


def sgf(text: str) -> str:
    """sgf("B:pd W:dp B:pp") -> "(;SZ[19];B[pd];W[dp];B[pp])".  "W:" followed by space/end = pass."""
    parts = [f";{tok.split(':', 1)[0]}[{tok.split(':', 1)[1]}]" for tok in text.split(" ") if ":" in tok]
    return "(;SZ[19]" + "".join(parts) + ")"


# --------------------------------------------------------------------------------------
# Synthetic corpus / contest (tdd 2.4, 2.5)
# --------------------------------------------------------------------------------------
import numpy as np
import pandas as pd

from gogame.constants import RANKS

_LETTERS = "abcdefghijklmnopqrs"


def _line(r: int, c: int) -> int:
    return min(r, c, 18 - r, 18 - c) + 1


_LINE34 = [(r, c) for r in range(19) for c in range(19) if _line(r, c) in (3, 4)]


def random_game(rng: np.random.Generator, style: float, n_moves: int | None = None) -> str:
    """Random game: each point is used at most once (re-draw if occupied); with prob 0.6*style the move is on line 3/4."""
    if n_moves is None:
        n_moves = int(rng.integers(60, 140))
    used: set[tuple[int, int]] = set()
    parts = []
    for t in range(n_moves):
        while True:
            if rng.random() < 0.6 * style:
                r, c = _LINE34[int(rng.integers(len(_LINE34)))]
            else:
                r, c = int(rng.integers(19)), int(rng.integers(19))
            if (r, c) not in used:
                break
        used.add((r, c))
        parts.append(f";{'B' if t % 2 == 0 else 'W'}[{_LETTERS[c]}{_LETTERS[r]}]")
    return "(;SZ[19]" + "".join(parts) + ")"


def make_corpus(seed: int, n_ranks: int = 10, players_per_rank: int = 6, games_per_player: int = 14,
                color_mix: bool = True, heavy: bool = False, n_heavy: int = 4) -> pd.DataFrame:
    """5-column frame same shape as train_*.csv. game_id restarts at g_0000001 for every rank."""
    rng = np.random.default_rng(seed)
    rows = []
    for rid in range(n_ranks):
        style = rid / 9
        gnum = 0
        for i in range(players_per_rank):
            n_games = games_per_player
            if heavy and i < n_heavy:
                n_games = 260  # >= 120 black games each
            for g in range(n_games):
                gnum += 1
                color = "W" if (color_mix and g % 2 == 1) else "B"
                rows.append((f"p_{RANKS[rid]}_{i}", f"g_{gnum:07d}", RANKS[rid], color, random_game(rng, style)))
    return pd.DataFrame(rows, columns=["player_id", "game_id", "rank", "color", "sgf_content"])


def make_contest(corpus: pd.DataFrame, n_per_n: int = 5, pool_size: int = 12, lib_games: int = 30,
                 with_prefix: bool = True, seed: int = 0) -> dict:
    """Fake contest files (dict of DataFrames) plus 'truth'. ID suffixes of public/private are disjoint."""
    rng = np.random.default_rng(seed)
    out: dict = {"truth": {}}

    def qid(split, i):
        base = f"q_{(0 if split == 'pub' else 5000) + i:04d}"
        return f"{split}_{base}" if with_prefix else base

    def pid(split, i):
        base = f"p_{(0 if split == 'pub' else 5000) + i:04d}"
        return f"{split}_{base}" if with_prefix else base

    def qrow(q, games, color):
        row = {"question_id": q, "num_games": str(len(games))}
        for k in range(20):
            row[f"sgf_{k + 1}"] = games[k] if k < len(games) else ""
        for k in range(20):
            row[f"color_{k + 1}"] = color if k < len(games) else ""
        return row

    cols = ["question_id", "num_games"] + [f"sgf_{k}" for k in range(1, 21)] + [f"color_{k}" for k in range(1, 21)]
    ranks = corpus["rank"].unique() if len(corpus) else RANKS
    for split, name in (("pub", "public"), ("priv", "private")):
        rank_rows, player_rows, cand_rows = [], [], []
        qi = 0
        for n in (5, 10, 15, 20):
            for _ in range(n_per_n):
                qi += 1
                rid = int(rng.integers(len(RANKS)))
                color = "B" if rng.random() < 0.5 else "W"
                games = [random_game(rng, rid / 9) for _ in range(n)]
                rank_rows.append(qrow(qid(split, qi), games, color))
        pool = [pid(split, i + 1) for i in range(pool_size)]
        gnum = 0
        styles = {p: float(rng.random()) for p in pool}
        for p in pool:
            for _ in range(lib_games):
                gnum += 1
                cand_rows.append({"game_id": f"g_{gnum:07d}", "player_id": p,
                                  "sgf_content": random_game(rng, styles[p]), "color": "B"})
        qi = 0
        for n in (5, 10, 15, 20):
            for _ in range(n_per_n):
                qi += 1
                p = pool[int(rng.integers(pool_size))]
                games = [random_game(rng, styles[p]) for _ in range(n)]
                q = qid(split, qi)
                player_rows.append(qrow(q, games, "B"))
                out["truth"][q] = p if with_prefix else f"{split}_{p}"
        out[f"rank_{name}"] = pd.DataFrame(rank_rows, columns=cols)
        out[f"cand_{name}"] = pd.DataFrame(cand_rows, columns=["game_id", "player_id", "sgf_content", "color"])
        out[f"player_{name}"] = pd.DataFrame(player_rows, columns=cols)
    return out


def write_contest(contest: dict, directory) -> None:
    from pathlib import Path
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name in ("public", "private"):
        contest[f"rank_{name}"].to_csv(d / f"rank_prediction_test_{name}.csv", index=False)
        contest[f"cand_{name}"].to_csv(d / f"player_identification_candidates_{name}.csv", index=False)
        contest[f"player_{name}"].to_csv(d / f"player_identification_test_{name}.csv", index=False)
