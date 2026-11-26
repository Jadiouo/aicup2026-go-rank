"""E34: W120 feature matrix.  HSL 104 columns of every game that has new rows = games_from_moves(stored own moves 0..59 + new own moves 60..119);
games without new rows (<= 60 own moves, or skipped) keep the W60 columns (nothing more to aggregate).  Other 35 columns = W60 matrix.
  PYTHONPATH=src:scripts nice -n 10 .venv/bin/python -m scripts.experiments.e34.build_x120   -> .scratch/e34/bundle/X120.npy + x120_manifest.json
Asserts the ledger is complete (24/24 done) and that stored and new rows tile each game exactly (no overlap, no gap)."""
from __future__ import annotations
import json, sys, time
import numpy as np, pandas as pd

from scripts.experiments.e34.common import *               # noqa: F403
sys.path.insert(0, str(KATAGO))
from scan_runner import games_from_moves                    # noqa: E402
from kgio import PROFILES                                   # noqa: E402


def main():
    t0 = time.time()
    led = pd.read_parquet(W2 / "ledger_w2.parquet"); assert (led.status == "done").all(), led[["shard_id", "status", "note"]]
    a = np.load(BUNDLE / "aux.npz"); uid = a["uid"].astype(str); names = a["names"].astype(str).tolist(); X60 = np.load(BUNDLE / "X60.npy")
    old = pd.read_parquet(S34 / "moves60.parquet")
    new = pd.concat([pd.read_parquet(W2 / "results" / f"{s}_moves.parquet") for s in sorted(led.shard_id)], ignore_index=True)
    assert set(new.uid) <= set(uid) and not new.duplicated(["uid", "own_move_idx"]).any()
    assert new.own_move_idx.min() == RANGE[0] and new.own_move_idx.max() < RANGE[1]
    oc = old.groupby("uid").size(); nc = new.groupby("uid").size()
    assert (oc.loc[nc.index] == RANGE[0]).all(), "a game has new rows but fewer than 60 stored rows"
    mx = new.groupby("uid").own_move_idx.agg(["min", "max", "size"]); assert ((mx["min"] == RANGE[0]) & (mx["max"] - mx["min"] + 1 == mx["size"])).all()
    both = pd.concat([old[old.uid.isin(nc.index)], new], ignore_index=True)
    ids = both.uid.unique(); X120 = X60.copy(); row_of = {u: i for i, u in enumerate(uid)}; parts = np.array_split(ids, 8)
    seam_max = 0.0
    for p in parts:
        ps = set(p)
        g = games_from_moves(both[both.uid.isin(ps)], PROFILES).set_index("uid"); assert list(g.columns) == names[:104]
        # guard 2: for games WITH new rows, aggregating only their stored 0..59 rows must reproduce the W60 HSL columns exactly (same function, same rows)
        g0 = games_from_moves(old[old.uid.isin(ps)], PROFILES).set_index("uid"); assert list(g0.columns) == names[:104] and g0.index.equals(g.index)
        w60 = X60[[row_of[u] for u in g0.index], :104]; r60 = g0[names[:104]].to_numpy(np.float32)
        assert np.array_equal(w60, r60, equal_nan=True), "W60 columns of games with new rows are not reproduced by the stored 1-60 rows"
        seam_max = max(seam_max, float(np.nanmax(np.abs(w60 - r60))))
        X120[[row_of[u] for u in g.index], :104] = g[names[:104]].to_numpy(np.float32)
    nm = names.index("n_moves"); n_own = X60[:, names.index("l0_n_own")]
    exp_new = np.clip(n_own - RANGE[0], 0, RANGE[1] - RANGE[0])
    got_new = pd.Series(0, index=uid).add(nc, fill_value=0).reindex(uid).fillna(0).to_numpy()
    assert (X120[:, nm] <= RANGE[1]).all() and (X120[:, nm] >= X60[:, nm]).all()
    # guard 1: games without new rows must be BIT-identical to W60.  Compare the raw float32 bit patterns (uint32 view), not values:
    # X60 legitimately holds NaN in se_mean/se_std (SE missing, flagged by a missing-indicator column) and NaN != NaN, so np.array_equal(X120, X60)
    # is False for any row with a missing SE even though the row is untouched.  The bit comparison is strictly stronger than equal_nan (same NaN payload too).
    keep = ~np.isin(uid, ids)
    same = np.array_equal(X120[keep].view(np.uint32), X60[keep].view(np.uint32))
    assert same, "games without new rows changed"
    nan_keep = int(np.isnan(X60[keep]).sum()); nan_new_changed = int((np.isnan(X120[~keep]) != np.isnan(X60[~keep])).sum())
    assert not np.isnan(X120[:, :104]).any(), "NaN in the HSL columns"
    np.save(BUNDLE / "X120.npy", X120)
    skips = {}
    for s in led.shard_id:
        for k, v in json.load(open(W2 / "results" / f"{s}_meta.json"))["skip_reasons"].items(): skips[k] = skips.get(k, 0) + v
    man = {"games": int(len(uid)), "games_with_new_rows": int(len(ids)), "games_without_new_rows": int(len(uid) - len(ids)), "skip_reasons": skips,
           "new_rows": int(len(new)), "mean_new_rows_per_game": float(len(new) / len(uid)), "mean_n_moves_W60": float(X60[:, nm].mean()), "mean_n_moves_W120": float(X120[:, nm].mean()),
           "share_new_rows_equal_l0_expectation": float((got_new == exp_new).mean()), "games_new_rows_differ_from_l0_expectation": int((got_new != exp_new).sum()),
           "max_abs_expectation_diff": float(np.abs(got_new - exp_new).max()), "nan_cells_in_untouched_games": nan_keep, "nan_pattern_changes_in_new_games": nan_new_changed, "w60_seam_max_abs_diff": seam_max, "x120_sha256": sha256_file(BUNDLE / "X120.npy"), "seconds": round(time.time() - t0)}
    (BUNDLE / "x120_manifest.json").write_text(json.dumps(man, indent=1)); print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
