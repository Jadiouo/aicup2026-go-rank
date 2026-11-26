"""Small helpers for split / questions / overlap tests (no heavy deps)."""
from __future__ import annotations

import pandas as pd

from gogame.constants import RANK_TO_ID


def corpus_meta(corpus: pd.DataFrame) -> pd.DataFrame:
    """uid, player_id, rank_id, color_id (0=B,1=W) from a make_corpus frame (no sgf)."""
    return pd.DataFrame({
        "uid": corpus["rank"] + "_" + corpus["game_id"],
        "player_id": corpus["player_id"],
        "rank_id": corpus["rank"].map(RANK_TO_ID).astype("int8"),
        "color_id": (corpus["color"] == "W").astype("int8"),
    })


def players_table(corpus: pd.DataFrame) -> pd.DataFrame:
    """player_id, rank_id, n_black."""
    m = corpus_meta(corpus)
    g = m.groupby("player_id").agg(rank_id=("rank_id", "first"), n_black=("color_id", lambda s: int((s == 0).sum())))
    return g.reset_index()


from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=None)
def heavy_players_table() -> pd.DataFrame:
    from helpers import make_corpus
    return players_table(make_corpus(2, players_per_rank=10, games_per_player=14, heavy=True, n_heavy=4))
