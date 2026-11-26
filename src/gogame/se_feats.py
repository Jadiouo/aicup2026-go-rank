"""E21 strength-estimator (SE) bag columns for the optional Rank variant `hsl_lgbm_se` (experiments/E21_strength_estimator).

`se_feats` is a verbatim copy of scripts/experiments/e21/run.py `se_feats` (tests/test_predict_se.py compares the two on the
same bags, bit for bit).  Do not "improve" it: any difference from the experiment silently removes the E21 gain.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SE_COLS = ["se_mean_mean", "se_mean_std", "se_mean_p10", "se_mean_p90", "se_mean_min", "se_mean_max", "se_std_mean"]
SE_TABLE_COLS = ("uid", "se_mean", "se_std", "se_n")
SE_N_POSITIONS = 32          # extract.py default --n; per-game positions behind each se_mean


def se_feats(bags, sm, ss):
    """7 bag-level SE columns from the bag's own games only (sm / ss: dict uid -> se_mean / se_std; NaN ignored)."""
    out = np.full((len(bags), 7), np.nan)
    for i, b in enumerate(bags):
        v = np.array([sm[u] for u in b.uids], float); v = v[~np.isnan(v)]
        w = np.array([ss[u] for u in b.uids], float); w = w[~np.isnan(w)]
        if len(v):
            out[i, :6] = [v.mean(), v.std(), np.percentile(v, 10), np.percentile(v, 90), v.min(), v.max()]
        if len(w):
            out[i, 6] = w.mean()
    return out


def load_se_tables(paths: Sequence[str | Path]) -> tuple[dict, dict, dict]:
    """Union of SE tables (uid, se_mean, se_std, se_n) -> (sm, ss, info).  A uid in two tables must agree (else ValueError).
    A uid that is in no table is NOT in the dicts; `SeMaps` turns that into NaN (a game without SE)."""
    sm: dict = {}
    ss: dict = {}
    per = []
    for p in paths:
        df = pd.read_parquet(p)
        miss = [c for c in SE_TABLE_COLS if c not in df.columns]
        if miss:
            raise ValueError(f"{p}: SE table lacks columns {miss}")
        if df["uid"].duplicated().any():
            raise ValueError(f"{p}: duplicate uid in SE table")
        m, s = df["se_mean"].astype(float).to_numpy(), df["se_std"].astype(float).to_numpy()
        for u, a, b in zip(df["uid"].tolist(), m, s):
            if u in sm and not (np.array_equal([sm[u], ss[u]], [a, b], equal_nan=True)):
                raise ValueError(f"uid {u} appears in two SE tables with different values")
            sm[u], ss[u] = a, b
        per.append({"path": str(p), "rows": int(len(df)), "nan": int(np.isnan(m).sum())})
    return sm, ss, {"tables": per, "rows": len(sm)}


class SeMaps(dict):
    """dict uid -> value that answers NaN for a uid that has no SE (HD>=2 games are absent from the training tables)."""

    def __missing__(self, key):
        return np.nan


def maps(sm: dict, ss: dict) -> tuple[SeMaps, SeMaps]:
    return SeMaps(sm), SeMaps(ss)


def is_handicap(sgf: str) -> bool:
    """Root HD >= 2.  The training table has no SE for these games (E21 range: HD<2), so the test table must not have one either,
    also when the file carries no AB/AW setup stones (then the encoder alone would happily score the colour-swapped moves)."""
    from .sgf import root_props
    try:
        return int((root_props(sgf).get("HD") or ["0"])[0]) >= 2
    except (ValueError, TypeError):
        return False


def default_se_rounds(results_dir) -> tuple[int, dict]:
    """M03 rounds of the SE variant = final_num_rounds (round-half-up median) of the 5 E21 folds' best_iteration_prime
    (experiments/E21_strength_estimator/results/meta_fold{0..4}.json; the selection.json value 80 belongs to M03 WITHOUT SE)."""
    from .models import final_num_rounds
    import json
    its, files = [], []
    for k in range(5):
        f = Path(results_dir) / f"meta_fold{k}.json"
        its.append(int(json.loads(f.read_text())["best_iteration_prime"]))
        files.append(str(f))
    return final_num_rounds(its), {"rounds_source": "E21 best_iteration_prime median", "best_iterations": its, "files": files}
