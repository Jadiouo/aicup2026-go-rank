"""E36: full W120 feature table = E34 pilot + E35 + E36 scans over all 712,945 non-H bundle games (PLAN section 8).  No training, no scoring.
  PYTHONPATH=src:scripts nice -n 10 systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 .venv/bin/python -m scripts.experiments.e36.build_w120_full old|new|train [--allow-partial]
  old   : stored per-move rows 0..59 of ALL bundle games, split into NPART uid-blocks (.scratch/e36/parts/old_p*.parquet)
  new   : scanned rows 60..119 (E34 pilot results, E35 results, E36 results), split the same way
  train : X120_full.npy (+ x120_full_manifest.json) with the guards below; --allow-partial = only games whose late window has been scanned (mask file), otherwise ALL games are required.
Guards (any failure aborts; compare bit patterns via a uint32 view so NaN positions must match too -- never np.array_equal on float arrays holding NaN):
  1 first 60 own moves: HSL 104 columns recomputed from stored rows == X60 columns, bit-exact, for every game of every block
  2 games without new rows: whole row bit-identical to X60;  3 the 35 non-HSL columns bit-identical for all rows;  4 no NaN in HSL columns
  5 games with new rows have exactly 60 stored rows, new rows contiguous from own_move_idx 60;  6 new-row count == clip(l0_n_own-60, 0, 60) for every scanned game
  7 games already in E35's X120.npy (if present): bit-identical;  8 every game exactly once, no H game;  9 (full mode) every game belongs to a finished shard / pilot.
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

from scripts.experiments.e36.common import *               # noqa: F403
sys.path.insert(0, str(KATAGO))
from scan_runner import games_from_moves                    # noqa: E402
from kgio import PROFILES                                   # noqa: E402


def bits(x): return np.ascontiguousarray(x, np.float32).view(np.uint32)
def bit_equal(a, b): return a.shape == b.shape and bool(np.array_equal(bits(a), bits(b)))


def part_of(uids):
    n = len(uids); return pd.Series(np.arange(n) * NPART // n, index=uids)


def split_moves(files, pmap, prefix):
    PART.mkdir(parents=True, exist_ok=True); writers = {}; n_rows = 0; schema = None
    try:
        for f in files:
            t = pq.read_table(f)
            if schema is None: schema = pa.schema([(fld.name, pa.string() if fld.name == "uid" else fld.type) for fld in t.schema])
            t = t.select(schema.names).cast(schema)
            p = t.column("uid").to_pandas().map(pmap); keep = p.notna().to_numpy()
            if not keep.any(): continue
            p = p.to_numpy()
            for k in np.unique(p[keep]):
                sub = t.filter(pa.array(keep & (p == k)))
                if k not in writers: writers[k] = pq.ParquetWriter(PART / f"{prefix}_p{int(k)}.parquet", schema, compression="zstd")
                writers[k].write_table(sub); n_rows += sub.num_rows
    finally:
        for w in writers.values(): w.close()
    return n_rows


def load_aux():
    a = np.load(E29B / "aux.npz"); return a["uid"].astype(str), a["names"].astype(str).tolist()


def cmd_old():
    t0 = time.time(); uid, _ = load_aux(); pmap = part_of(uid)
    man = pd.read_parquet(ROOT / "data/scan/manifest.parquet", columns=["uid", "shard_id"]).set_index("uid").loc[uid, "shard_id"]
    files = [ROOT / "data/cache/katago/pilot_hsl_moves.parquet" if s == "P0000" else ROOT / f"data/scan/results/{s}_moves.parquet" for s in sorted(set(man))]
    for p in PART.glob("old_p*.parquet"): p.unlink()
    n = split_moves(files, pmap, "old"); print(f"old rows {n} from {len(files)} stored shards, {time.time() - t0:.0f}s", flush=True)


def new_files(allow_partial):
    led34 = pd.read_parquet(E34W2 / "ledger_w2.parquet"); assert (led34.status == "done").all()
    led35 = pd.read_parquet(W35 / "ledger_e35.parquet"); t35 = led35[led35.family == "train"]
    led36 = pd.read_parquet(W36 / "ledger_e36.parquet"); t36 = led36[led36.family == "train"]
    if not allow_partial:
        assert (t35.status == "done").all(), f"E35 not complete: {int((t35.status != 'done').sum())} shards"
        assert (t36.status == "done").all(), f"E36 not complete: {int((t36.status != 'done').sum())} shards"
    d35, d36 = t35[t35.status == "done"], t36[t36.status == "done"]
    files = [E34W2 / "results" / f"{s}_moves.parquet" for s in sorted(led34.shard_id)] + [W35 / "results" / f"{s}_moves.parquet" for s in sorted(d35.shard_id)] + [W36 / "results" / f"{s}_moves.parquet" for s in sorted(d36.shard_id)]
    return files, (len(led34), len(d35), len(d36))


def cmd_new(allow_partial):
    t0 = time.time(); uid, _ = load_aux(); pmap = part_of(uid); files, cnt = new_files(allow_partial)
    for p in PART.glob("new_p*.parquet"): p.unlink()
    n = split_moves(files, pmap, "new"); print(f"new rows {n} from {len(files)} result shards (E34 {cnt[0]}, E35 {cnt[1]}, E36 {cnt[2]}), {time.time() - t0:.0f}s", flush=True)


def scanned_uids(allow_partial):
    """uids whose late window has been scanned (or skipped by the runner): E34 pilot selection, finished E35 shards, finished E36 shards."""
    s = set(pd.read_parquet(S35 / "bundle/selection.parquet", columns=["uid", "in_pilot"]).query("in_pilot").uid)
    for W, fam in ((W35, "ledger_e35.parquet"), (W36, "ledger_e36.parquet")):
        led = pd.read_parquet(W / fam); led = led[(led.family == "train") & (led.status == "done")]
        for sid in led.shard_id: s |= set(pd.read_parquet(W / "sgf" / f"{sid}.parquet", columns=["uid"]).uid)
    return s


def cmd_train(allow_partial):
    t0 = time.time(); uid, names = load_aux(); X60 = np.load(E29B / "X.npy"); assert X60.shape == (len(uid), 139) and names[:104] == names[:104]
    assert len(set(uid)) == len(uid)
    split = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(split.loc[split.is_holdout, "player_id"])
    gm = pd.read_parquet(ROOT / "data/cache/games_meta.parquet", columns=["uid", "player_id"]).set_index("uid")
    assert not (set(gm.loc[uid, "player_id"]) & H), "guard 8: H game in the bundle"
    sc = scanned_uids(allow_partial); is_sc = np.fromiter((u in sc for u in uid), bool, len(uid))
    if not allow_partial: assert is_sc.all(), f"guard 9: {int((~is_sc).sum())} games not covered by a finished shard"
    pidx = part_of(uid).to_numpy(); nm = names.index("n_moves"); n_own = X60[:, names.index("l0_n_own")]
    X120 = X60.copy(); got_new = np.zeros(len(uid)); with_new = np.zeros(len(uid), bool); n_new_rows = 0; seam_max = 0.0
    for p in range(NPART):
        rows = np.flatnonzero(pidx == p); pu = uid[rows]; row_of = {u: i for i, u in zip(rows, pu)}
        old = pd.read_parquet(PART / f"old_p{p}.parquet"); newf = PART / f"new_p{p}.parquet"; new = pd.read_parquet(newf) if newf.exists() else old.iloc[:0]
        assert set(old.uid) <= set(pu) and set(new.uid) <= set(pu) and not new.duplicated(["uid", "own_move_idx"]).any() and not old.duplicated(["uid", "own_move_idx"]).any()
        g0 = games_from_moves(old, PROFILES).set_index("uid"); assert list(g0.columns) == names[:104], "HSL column names/order"; assert set(g0.index) == set(pu)
        g0 = g0.loc[pu]; w60 = X60[rows, :104]; r60 = g0[names[:104]].to_numpy(np.float32)
        assert bit_equal(w60, r60), f"guard 1 part {p}: stored rows 0..59 do not reproduce the X60 HSL columns bit-exactly"
        seam_max = max(seam_max, float(np.nanmax(np.abs(w60 - r60))))
        if len(new):
            assert new.own_move_idx.min() == RANGE[0] and new.own_move_idx.max() < RANGE[1]
            oc = old.groupby("uid").size(); nc = new.groupby("uid").size()
            assert (oc.loc[nc.index] == RANGE[0]).all(), "guard 5: a game has new rows but not exactly 60 stored rows"
            mx = new.groupby("uid").own_move_idx.agg(["min", "max", "size"]); assert ((mx["min"] == RANGE[0]) & (mx["max"] - mx["min"] + 1 == mx["size"])).all(), "guard 5: new rows not contiguous"
            both = pd.concat([old[old.uid.isin(set(nc.index))], new], ignore_index=True)
            g = games_from_moves(both, PROFILES).set_index("uid"); assert list(g.columns) == names[:104]
            ii = np.array([row_of[u] for u in g.index]); X120[ii, :104] = g[names[:104]].to_numpy(np.float32); with_new[ii] = True
            got_new[[row_of[u] for u in nc.index]] = nc.to_numpy(); n_new_rows += len(new)
        print(f"part {p}: games {len(rows)} with new rows {int(with_new[rows].sum())}  {time.time() - t0:.0f}s", flush=True)
    exp_new = np.clip(n_own - RANGE[0], 0, RANGE[1] - RANGE[0])
    chk = is_sc if allow_partial else np.ones(len(uid), bool)
    n_diff = int((got_new[chk] != exp_new[chk]).sum()); assert n_diff == 0, f"guard 6: {n_diff} scanned games whose new row count differs from the L0 expectation"
    keep = ~with_new
    assert bit_equal(X120[keep], X60[keep]), "guard 2: games without new rows changed"
    assert bit_equal(X120[:, 104:], X60[:, 104:]), "guard 3: non-HSL columns changed"
    assert not np.isnan(X120[:, :104]).any(), "guard 4: NaN in the HSL columns"
    assert (X120[:, nm] <= RANGE[1]).all() and (X120[:, nm] >= X60[:, nm]).all()
    e35x = S35 / "bundle/X120.npy"; g7 = "e35 X120.npy not present"
    if e35x.exists():
        a35 = np.load(S35 / "bundle/aux.npz"); u35 = a35["uid"].astype(str); pos = pd.Series(np.arange(len(uid)), index=uid).loc[u35].to_numpy()
        assert bit_equal(X120[pos], np.load(e35x)), "guard 7: games in E35's X120 differ from this build"; g7 = f"{len(u35)} games bit-identical to E35 X120.npy"
    sel = np.ones(len(uid), bool) if not allow_partial else is_sc
    out = X120[sel]; np.save(BUNDLE / ("X120_full.npy" if not allow_partial else "X120_partial.npy"), out)
    if allow_partial: np.save(BUNDLE / "X120_partial_rows.npy", np.flatnonzero(sel))
    man = {"mode": "partial" if allow_partial else "full", "games_in_bundle": int(len(uid)), "games_in_table": int(sel.sum()), "games_with_new_rows": int(with_new.sum()), "games_without_new_rows_scanned": int((is_sc & ~with_new).sum()),
           "games_not_scanned": int((~is_sc).sum()), "new_rows": int(n_new_rows), "guards_passed": ["1 W60 bit-exact", "2 no-new-row games == X60", "3 non-HSL cols == X60", "4 no NaN in HSL", "5 60 old rows + contiguous", "6 new rows == L0 expectation",
                                                                                                    "7 " + g7, "8 once each, no H", "9 coverage" + (" (skipped: partial)" if allow_partial else "")],
           "w60_seam_max_abs_diff": seam_max, "mean_n_moves_W60": float(X60[sel, nm].mean()), "mean_n_moves_W120": float(out[:, nm].mean()), "seconds": round(time.time() - t0)}
    fn = BUNDLE / ("x120_full_manifest.json" if not allow_partial else "x120_partial_manifest.json"); man["sha256"] = sha256_file(BUNDLE / ("X120_full.npy" if not allow_partial else "X120_partial.npy"))
    fn.write_text(json.dumps(man, indent=1)); print(json.dumps(man, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["old", "new", "train"]); ap.add_argument("--allow-partial", action="store_true"); a = ap.parse_args()
    cmd_old() if a.cmd == "old" else cmd_new(a.allow_partial) if a.cmd == "new" else cmd_train(a.allow_partial)
