"""E35: W120 feature matrices (training S rows, and the Rank test set) with the E34 guards.
  PYTHONPATH=src:scripts nice -n 10 systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 .venv/bin/python -m scripts.experiments.e35.build_x120 old|new|train|test
  old   : stored per-move rows 0..59 of the S games, split into NPART uid-blocks (.scratch/e35/parts/old_p*.parquet)       [can run any time]
  new   : scanned rows 60..119 (E34 pilot results for the pilot games + E35 results), split the same way                      [needs the ledger complete]
  train : X120.npy for the S rows (+ x120_manifest.json) with the E34 guards, all games.
  test  : test_X120.npy (+ manifest) for the 10,000 Rank test games with bit-level checks against the stored 1-60 test features.
HSL 104 columns = scan_runner.games_from_moves(stored 0..59 rows + new 60..119 rows)  -- the very function that produced the stored W60 columns; the other 35 columns are copied.
"""
from __future__ import annotations
import json, sys, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

from scripts.experiments.e35.common import *               # noqa: F403
sys.path.insert(0, str(KATAGO))
from scan_runner import games_from_moves                    # noqa: E402
from kgio import PROFILES                                   # noqa: E402

TEST = S35 / "test"


def part_of(uids_sorted):
    """uid -> partition index (contiguous blocks of the bundle's uid order)."""
    n = len(uids_sorted)
    return pd.Series(np.arange(n) * NPART // n, index=uids_sorted)


def split_moves(files, pmap, prefix, schema=None):
    PART.mkdir(parents=True, exist_ok=True)
    writers = {}; n_rows = 0
    try:
        for f in files:
            t = pq.read_table(f)
            if schema is None: schema = pa.schema([(fld.name, pa.string() if fld.name == "uid" else fld.type) for fld in t.schema])
            t = t.select(schema.names).cast(schema)
            uid = t.column("uid").to_pandas()
            p = uid.map(pmap)
            keep = p.notna().to_numpy()
            if not keep.any(): continue
            p = p.to_numpy()
            for k in np.unique(p[keep]):
                sub = t.filter(pa.array(keep & (p == k)))
                if k not in writers: writers[k] = pq.ParquetWriter(PART / f"{prefix}_p{int(k)}.parquet", schema, compression="zstd")
                writers[k].write_table(sub); n_rows += sub.num_rows
    finally:
        for w in writers.values(): w.close()
    return n_rows


def cmd_old():
    t0 = time.time()
    a = np.load(BUNDLE / "aux.npz"); uid = a["uid"].astype(str); pmap = part_of(uid)
    man = pd.read_parquet(ROOT / "data/scan/manifest.parquet", columns=["uid", "shard_id"]).set_index("uid").loc[uid, "shard_id"]
    files = [ROOT / "data/cache/katago/pilot_hsl_moves.parquet" if s == "P0000" else ROOT / f"data/scan/results/{s}_moves.parquet" for s in sorted(set(man))]
    for p in PART.glob("old_p*.parquet"): p.unlink()
    n = split_moves(files, pmap, "old")
    print(f"old rows {n} from {len(files)} stored shards, {time.time() - t0:.0f}s", flush=True)


def cmd_new():
    t0 = time.time()
    led = pd.read_parquet(W35 / "ledger_e35.parquet"); tr = led[led.family == "train"]
    assert (tr.status == "done").all(), tr[tr.status != "done"][["shard_id", "status", "note"]]
    led34 = pd.read_parquet(E34W2 / "ledger_w2.parquet"); assert (led34.status == "done").all()
    a = np.load(BUNDLE / "aux.npz"); uid = a["uid"].astype(str); pmap = part_of(uid)
    files = [E34W2 / "results" / f"{s}_moves.parquet" for s in sorted(led34.shard_id)] + [W35 / "results" / f"{s}_moves.parquet" for s in sorted(tr.shard_id)]
    for p in PART.glob("new_p*.parquet"): p.unlink()
    n = split_moves(files, pmap, "new")
    print(f"new rows {n} from {len(files)} result shards (E34 pilot {len(led34)} + E35 {len(tr)}), {time.time() - t0:.0f}s", flush=True)


def bits(x):
    return np.ascontiguousarray(x, np.float32).view(np.uint32)


def cmd_checkold():
    """Early guard (any time after `old`): the stored 0..59 rows of ALL S games reproduce the stored W60 HSL columns bit-exactly (same games_from_moves)."""
    t0 = time.time(); a = np.load(BUNDLE / "aux.npz"); uid = a["uid"].astype(str); names = a["names"].astype(str).tolist(); X60 = np.load(BUNDLE / "X60.npy", mmap_mode="r")
    pidx = part_of(uid).to_numpy(); mx = 0.0
    for p in range(NPART):
        rows = np.flatnonzero(pidx == p); pu = uid[rows]; old = pd.read_parquet(PART / f"old_p{p}.parquet")
        g0 = games_from_moves(old, PROFILES).set_index("uid"); assert list(g0.columns) == names[:104] and set(g0.index) == set(pu)
        r60 = g0.loc[pu, names[:104]].to_numpy(np.float32); w60 = np.asarray(X60[rows, :104])
        assert np.array_equal(bits(w60), bits(r60)), f"part {p}: stored rows do not reproduce W60 columns bit-exactly"
        mx = max(mx, float(np.abs(w60 - r60).max())); print(f"checkold part {p} ok ({len(rows)} games) {time.time() - t0:.0f}s", flush=True)
    (S35 / "checkold.json").write_text(json.dumps({"games": int(len(uid)), "bitwise_equal": True, "max_abs_diff": mx}))


def cmd_train():
    t0 = time.time()
    a = np.load(BUNDLE / "aux.npz"); uid = a["uid"].astype(str); names = a["names"].astype(str).tolist(); X60 = np.load(BUNDLE / "X60.npy")
    sel = pd.read_parquet(BUNDLE / "selection.parquet").set_index("uid").loc[uid]
    assert len(uid) == len(X60) == len(sel)
    pmap = part_of(uid); pidx = pmap.to_numpy()
    X120 = X60.copy(); nm = names.index("n_moves"); n_own = X60[:, names.index("l0_n_own")]
    got_new = np.zeros(len(uid)); with_new = np.zeros(len(uid), bool); seam_max = 0.0; n_new_rows = 0; skips = {}
    for p in range(NPART):
        rows = np.flatnonzero(pidx == p); pu = uid[rows]; row_of = {u: i for i, u in zip(rows, pu)}
        old = pd.read_parquet(PART / f"old_p{p}.parquet"); new = pd.read_parquet(PART / f"new_p{p}.parquet") if (PART / f"new_p{p}.parquet").exists() else old.iloc[:0]
        assert set(old.uid) <= set(pu) and set(new.uid) <= set(pu) and not new.duplicated(["uid", "own_move_idx"]).any() and not old.duplicated(["uid", "own_move_idx"]).any()
        # guard 3a: for ALL games of the block, the stored 0..59 rows reproduce the stored W60 HSL columns (same function) -- bit-exact (NaN-safe via equal_nan; HSL has no NaN)
        g0 = games_from_moves(old, PROFILES).set_index("uid"); assert list(g0.columns) == names[:104], "HSL column names/order"
        assert set(g0.index) == set(pu), "a game without stored rows"
        g0 = g0.loc[pu]; w60 = X60[rows, :104]; r60 = g0[names[:104]].to_numpy(np.float32)
        assert np.array_equal(bits(w60), bits(r60)) or np.array_equal(w60, r60, equal_nan=True), f"part {p}: stored rows do not reproduce W60 columns"
        seam_max = max(seam_max, float(np.nanmax(np.abs(w60 - r60))))
        if len(new):
            assert new.own_move_idx.min() == RANGE[0] and new.own_move_idx.max() < RANGE[1]
            oc = old.groupby("uid").size(); nc = new.groupby("uid").size()
            assert (oc.loc[nc.index] == RANGE[0]).all(), "a game has new rows but not exactly 60 stored rows"
            mx = new.groupby("uid").own_move_idx.agg(["min", "max", "size"]); assert ((mx["min"] == RANGE[0]) & (mx["max"] - mx["min"] + 1 == mx["size"])).all()
            ids = nc.index.to_numpy()
            both = pd.concat([old[old.uid.isin(set(ids))], new], ignore_index=True)
            g = games_from_moves(both, PROFILES).set_index("uid"); assert list(g.columns) == names[:104]
            ii = np.array([row_of[u] for u in g.index]); X120[ii, :104] = g[names[:104]].to_numpy(np.float32); with_new[ii] = True
            got_new[[row_of[u] for u in nc.index]] = nc.to_numpy(); n_new_rows += len(new)
        print(f"part {p}: games {len(rows)} with new rows {int(with_new[rows].sum())}  {time.time() - t0:.0f}s", flush=True)
    exp_new = np.clip(n_own - RANGE[0], 0, RANGE[1] - RANGE[0])
    assert (X120[:, nm] <= RANGE[1]).all() and (X120[:, nm] >= X60[:, nm]).all()
    # guard 3b: games without new rows are BIT-identical to W60 (uint32 view: X60 holds NaN in se_mean/se_std, NaN != NaN)
    keep = ~with_new
    assert np.array_equal(bits(X120[keep]), bits(X60[keep])), "games without new rows changed"
    assert np.array_equal(bits(X120[:, 104:]), bits(X60[:, 104:])), "non-HSL columns changed"
    assert not np.isnan(X120[:, :104]).any(), "NaN in the HSL columns"
    nan_keep = int(np.isnan(X60[keep]).sum())
    share_eq = float((got_new == exp_new).mean()); n_diff = int((got_new != exp_new).sum())
    assert n_diff == 0, f"{n_diff} games whose new row count differs from the L0 expectation"
    # guard 3c: for the pilot games our X120 equals E34's (independent build, same function)
    a34 = np.load(E34B / "aux.npz"); u34 = a34["uid"].astype(str); X34 = np.load(E34B / "X120.npy")
    pos = pd.Series(np.arange(len(uid)), index=uid).loc[u34].to_numpy()
    assert np.array_equal(bits(X120[pos]), bits(X34)), "pilot games: E35 X120 != E34 X120"
    np.save(BUNDLE / "X120.npy", X120)
    man = {"games": int(len(uid)), "games_with_new_rows": int(with_new.sum()), "games_without_new_rows": int((~with_new).sum()), "new_rows": int(n_new_rows),
           "share_new_rows_equal_l0_expectation": share_eq, "w60_seam_max_abs_diff": seam_max, "nan_cells_in_untouched_games": nan_keep,
           "pilot_games_equal_E34_X120": int(len(u34)), "mean_n_moves_W60": float(X60[:, nm].mean()), "mean_n_moves_W120": float(X120[:, nm].mean()),
           "x120_sha256": sha256_file(BUNDLE / "X120.npy"), "x60_sha256": sha256_file(BUNDLE / "X60.npy"), "seconds": round(time.time() - t0)}
    (BUNDLE / "x120_manifest.json").write_text(json.dumps(man, indent=1)); print(json.dumps(man, indent=1))


def cmd_testold():
    """Early guard (before the test scan finishes): rank_v4's test features == stored 1-60 test table, and the stored 0..59 test rows reproduce them bit-exactly with games_from_moves."""
    a = np.load(E29B / "aux.npz"); names = a["names"].astype(str).tolist(); Xt = np.load(E29T / "test_X.npy"); uid = np.load(E29T / "test_bags.npz")["uids"].astype(str)
    rep = json.load(open(E29T / "features_report.json")); assert sha256_file(E29T / "test_X.npy") == rep["x_sha256"], "test_X.npy changed since rank_v4"
    hsl = pd.read_parquet(ROOT / "data/test_scan/hsl_test_games.parquet").set_index("uid").reindex(uid); assert list(hsl.columns) == names[:104]
    assert np.array_equal(bits(Xt[:, :104]), bits(hsl[names[:104]].to_numpy(np.float32))), "test_X.npy HSL columns != hsl_test_games.parquet"
    old = pd.concat([pd.read_parquet(ROOT / f"data/test_scan/results/S900{i}_moves.parquet") for i in (1, 2, 3)], ignore_index=True)
    g0 = games_from_moves(old, PROFILES).set_index("uid"); assert list(g0.columns) == names[:104] and set(g0.index) == set(uid)
    r0 = g0.loc[uid, names[:104]].to_numpy(np.float32)
    assert np.array_equal(bits(r0), bits(Xt[:, :104])), "test: stored rows 0..59 do not reproduce the stored W60 HSL columns bit-exactly"
    nm = names.index("l0_n_own"); n_own = Xt[:, nm]; oc = old.groupby("uid").size().reindex(uid)
    out = {"games": int(len(uid)), "stored_rows_reproduce_test_X_hsl_bitwise": True, "test_X_hsl_equals_hsl_test_games_bitwise": True, "nan_cells_test_X": int(np.isnan(Xt).sum()),
           "games_with_more_than_60_own_moves": int((n_own > 60).sum()), "stored_rows_per_game_eq_min(60,l0_n_own)": bool((oc.to_numpy() == np.minimum(n_own, 60)).all()), "max_abs_diff": float(np.abs(r0 - Xt[:, :104]).max())}
    (S35 / "testold.json").write_text(json.dumps(out)); print(json.dumps(out))


def cmd_test():
    t0 = time.time(); TEST.mkdir(parents=True, exist_ok=True)
    a = np.load(E29B / "aux.npz"); names = a["names"].astype(str).tolist()
    Xt = np.load(E29T / "test_X.npy"); bg = np.load(E29T / "test_bags.npz"); uid = bg["uids"].astype(str)
    rep = json.load(open(E29T / "features_report.json")); assert sha256_file(E29T / "test_X.npy") == rep["x_sha256"], "test_X.npy changed since rank_v4"
    assert Xt.shape == (len(uid), 139) and len(set(uid)) == len(uid) == 10000
    hsl = pd.read_parquet(ROOT / "data/test_scan/hsl_test_games.parquet").set_index("uid").reindex(uid); assert list(hsl.columns) == names[:104]
    # guard: rank_v4 test features' HSL columns == the stored 1-60 test table (bit level)
    assert np.array_equal(bits(Xt[:, :104]), bits(hsl[names[:104]].to_numpy(np.float32))), "test_X.npy HSL columns != hsl_test_games.parquet"
    led = pd.read_parquet(W35 / "ledger_e35.parquet"); te = led[led.family == "test"]; assert len(te) == 3 and (te.status == "done").all(), te[["shard_id", "status", "note"]]
    old = pd.concat([pd.read_parquet(ROOT / f"data/test_scan/results/S900{i}_moves.parquet") for i in (1, 2, 3)], ignore_index=True)
    new = pd.concat([pd.read_parquet(W35 / "results" / f"{s}_moves.parquet") for s in sorted(te.shard_id)], ignore_index=True)
    assert set(old.uid) == set(uid) and set(new.uid) <= set(uid) and not new.duplicated(["uid", "own_move_idx"]).any()
    # guard: the stored 1-60 rows re-aggregated reproduce BOTH stored tables bit-exactly (same function as the training build)
    g0 = games_from_moves(old, PROFILES).set_index("uid"); assert list(g0.columns) == names[:104]
    r0 = g0.loc[uid, names[:104]].to_numpy(np.float32)
    assert np.array_equal(bits(r0), bits(Xt[:, :104])), "test: stored rows 0..59 do not reproduce the stored W60 HSL columns bit-exactly"
    # new rows: contiguous from 60, exactly 60 stored rows for games with new rows, count = L0 expectation
    assert new.own_move_idx.min() == RANGE[0] and new.own_move_idx.max() < RANGE[1]
    oc = old.groupby("uid").size(); nc = new.groupby("uid").size(); assert (oc.loc[nc.index] == RANGE[0]).all()
    mx = new.groupby("uid").own_move_idx.agg(["min", "max", "size"]); assert ((mx["min"] == RANGE[0]) & (mx["max"] - mx["min"] + 1 == mx["size"])).all()
    n_own = Xt[:, names.index("l0_n_own")]; exp_new = np.clip(n_own - RANGE[0], 0, RANGE[1] - RANGE[0])
    got = pd.Series(0, index=uid).add(nc, fill_value=0).reindex(uid).fillna(0).to_numpy()
    assert (got == exp_new).all(), f"{int((got != exp_new).sum())} test games whose new row count differs from the L0 expectation"
    ids = nc.index.to_numpy(); both = pd.concat([old[old.uid.isin(set(ids))], new], ignore_index=True)
    g = games_from_moves(both, PROFILES).set_index("uid"); assert list(g.columns) == names[:104]
    X120 = Xt.copy(); row_of = {u: i for i, u in enumerate(uid)}; ii = np.array([row_of[u] for u in g.index]); X120[ii, :104] = g[names[:104]].to_numpy(np.float32)
    keep = np.ones(len(uid), bool); keep[ii] = False
    assert np.array_equal(bits(X120[keep]), bits(Xt[keep])), "test games without new rows changed"
    assert np.array_equal(bits(X120[:, 104:]), bits(Xt[:, 104:])), "test non-HSL columns changed"
    nm = names.index("n_moves"); assert (X120[:, nm] <= RANGE[1]).all() and (X120[:, nm] >= Xt[:, nm]).all() and not np.isnan(X120[:, :104]).any()
    meta = [json.load(open(W35 / "results" / f"{s}_meta.json")) for s in sorted(te.shard_id)]
    assert all(m["color_semantics"] == SEM_TEST and m["config_hash"] == CONFIG_HASH_W2 for m in meta)
    np.save(TEST / "test_X120.npy", X120)
    man = {"games": int(len(uid)), "games_with_new_rows": int(len(ids)), "games_without_new_rows": int(keep.sum()), "new_rows": int(len(new)), "skip_reasons": {k: sum(m["skip_reasons"].get(k, 0) for m in meta) for k in set().union(*[m["skip_reasons"] for m in meta])},
           "stored_w60_reproduced_bitwise": True, "new_rows_equal_l0_expectation_all": True, "mean_n_moves_W60": float(Xt[:, nm].mean()), "mean_n_moves_W120": float(X120[:, nm].mean()),
           "color_semantics_of_new_scan": SEM_TEST, "config_hash": CONFIG_HASH_W2, "test_x_sha256": sha256_file(E29T / "test_X.npy"), "test_x120_sha256": sha256_file(TEST / "test_X120.npy"), "seconds": round(time.time() - t0)}
    (TEST / "test_x120_manifest.json").write_text(json.dumps(man, indent=1)); print(json.dumps(man, indent=1))


if __name__ == "__main__":
    {"old": cmd_old, "new": cmd_new, "checkold": cmd_checkold, "testold": cmd_testold, "train": cmd_train, "test": cmd_test}[sys.argv[1]]()
