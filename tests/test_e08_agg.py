import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "experiments" / "e08"))
from agg import aggregate_moves  # noqa: E402

MOVES = ROOT / "data/cache/katago/pilot_hsl_moves.parquet"
GAMES = ROOT / "data/cache/katago/pilot_hsl_games.parquet"


@pytest.mark.skipif(not (MOVES.exists() and GAMES.exists()), reason="pilot HSL cache missing")
def test_aggregate_reproduces_teacher_game_features():
    mv = pd.read_parquet(MOVES)
    uids = sorted(mv["uid"].unique())[:300]
    sub = mv[mv["uid"].isin(uids)]
    got = aggregate_moves(sub).set_index("uid")
    ref = pd.read_parquet(GAMES).set_index("uid").loc[got.index]
    assert list(got.columns) == list(ref.columns)
    np.testing.assert_allclose(got.to_numpy(np.float64), ref.to_numpy(np.float64), rtol=1e-4, atol=1e-5)
