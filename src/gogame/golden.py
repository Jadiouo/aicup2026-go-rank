"""Golden-sample selection (tdd 2.3). Used by scripts/make_golden.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import RANKS, SEED
from .io import read_train_csv
from .seeds import derive_seed


def select_golden(data_dir: str | Path, n: int = 50) -> pd.DataFrame:
    """n/10 games per rank file; per file ceil(60%) black-annotated and the rest white-annotated.

    With n=50: 3 B + 2 W per file -> 20 W games in total (>= 15 required).
    """
    per = n // len(RANKS)
    n_w = max(2, int(round(per * 0.4)))
    n_b = per - n_w
    rng = np.random.default_rng(derive_seed(SEED, "golden"))
    parts = []
    for r in RANKS:
        df = read_train_csv(Path(data_dir) / f"train_{r}.csv")
        for color, k in (("B", n_b), ("W", n_w)):
            sub = df[df["color"] == color]
            idx = rng.choice(len(sub), size=k, replace=False)
            parts.append(sub.iloc[np.sort(idx)])
    out = pd.concat(parts, ignore_index=True)
    if (out["color"] == "W").sum() < 15 and n >= 50:
        raise RuntimeError("golden sample has fewer than 15 white games")
    return out[["player_id", "game_id", "rank", "color", "sgf_content"]]
