"""Game-level HumanSL-style aggregation from per-move frames.  Same formulas as scripts/katago/build_pilot_features.py
(copied, not imported: that directory is frozen), so teacher moves reproduce data/cache/katago/pilot_hsl_games.parquet.
`mv` columns: uid, own_move_idx, logp_0..10, rank_0..10, ent_0..10."""
import numpy as np
import pandas as pd

PROFILES = ["rank_12k", "rank_9k", "rank_6k", "rank_3k", "rank_1k", "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d"]
P = len(PROFILES)


def aggregate_moves(mv: pd.DataFrame) -> pd.DataFrame:
    mv = mv.sort_values(["uid", "own_move_idx"]).reset_index(drop=True)
    L = mv[[f"logp_{i}" for i in range(P)]].to_numpy(np.float64)
    g = mv.groupby("uid", sort=True)

    def agg(mask=None):
        d = pd.DataFrame(L, columns=PROFILES)
        d["uid"] = mv.uid.values
        if mask is not None:
            d = d[mask]
        return d.groupby("uid", sort=True)

    a = agg()
    out = pd.DataFrame(index=a.mean().index)
    M, S, D = a.mean(), a.std(ddof=0), a.median()
    for p in PROFILES:
        out[f"ml_{p}"] = M[p]
    ml = M.to_numpy()
    for i, p in enumerate(PROFILES):
        out[f"mlc_{p}"] = ml[:, i] - ml.mean(axis=1)
    out["ml_argmax_idx"] = ml.argmax(axis=1).astype("int8")
    w = np.exp(ml - ml.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    out["ml_exp_idx"] = (w * np.arange(P)).sum(axis=1)
    for p in PROFILES:
        out[f"mlmed_{p}"] = D[p]
        out[f"mlstd_{p}"] = S[p]
    for name, mask in [("early", mv.own_move_idx.values < 20), ("late", mv.own_move_idx.values >= 20)]:
        m2 = agg(mask).mean()
        for p in PROFILES:
            out[f"ml{name}_{p}"] = m2[p]
        e = m2.to_numpy()
        ww = np.exp(e - e.max(axis=1, keepdims=True))
        ww /= ww.sum(axis=1, keepdims=True)
        out[f"ml_exp_idx_{name}"] = (ww * np.arange(P)).sum(axis=1)
    out["n_moves"] = g.size().astype("int16")
    R = mv[[f"rank_{i}" for i in range(P)]].to_numpy()
    E = mv[[f"ent_{i}" for i in range(P)]].to_numpy(np.float64)
    for nm, A in [("top1", (R == 1).astype(float)), ("lrank", np.log(R)), ("ent", E)]:
        d = pd.DataFrame(A, columns=PROFILES)
        d["uid"] = mv.uid.values
        m3 = d.groupby("uid", sort=True).mean()
        for p in PROFILES:
            out[f"{nm}_{p}"] = m3[p]
    out = out.reset_index()
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype("float32")
    return out
