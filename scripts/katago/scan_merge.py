"""Ledger + validated ingest/merge for the full scan.

  scan_merge.py init                      create data/scan/ledger.parquet from shards.parquet (P0000 = pilot, done)
  scan_merge.py claim S0001,S0002 PLATFORM   mark shards running
  scan_merge.py ingest DIR [--cfg analysis.cfg]   validate every *_meta.json in DIR, accept (copy to data/scan/results,
                                          status=done) or reject (status=failed + reason in ledger; files not copied)
  scan_merge.py release PLATFORM          running shards of PLATFORM that were not ingested -> todo
  scan_merge.py reconcile                 results/ vs ledger: record valid-but-unrecorded shards as done, report the rest
  scan_merge.py status                    counts per status/platform, games done, tier-1 progress
  scan_merge.py merge [--moves]           concatenate accepted shards (+ pilot) -> data/cache/katago/full_hsl_{games,moves}.parquet
"""
import argparse, glob, json, os, shutil, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCAN = f"{ROOT}/data/scan"; LEDGER = f"{SCAN}/ledger.parquet"; RES = f"{SCAN}/results"
sys.path.insert(0, HERE)
from scan_runner import config_hash  # noqa: E402
import scan_ledger  # noqa: E402  (locked, atomic ledger writes; module globals SCAN/LEDGER/RES below stay the redirect points)
PINS = json.load(open(f"{HERE}/scan_pins.json"))


def asset_hash_errors(meta):
    """Accept old archive provenance or checked runtime content, never a mixed partial claim."""
    errs = []
    content_fields = (("katago_zip", "katago_executable_sha256", "executable_sha256"),
                      ("main_net", "main_net_content_sha256", "content_sha256"),
                      ("human_net", "human_net_content_sha256", "content_sha256"))
    if any(field in meta for _, field, _ in content_fields):
        for key, field, expected in content_fields:
            if meta.get(field) != PINS[key][expected]: errs.append(f"{key} content sha")
        for key, field in (("katago_zip", "katago_zip_sha256"),
                           ("main_net", "main_net_sha256"), ("human_net", "human_net_sha256")):
            if meta.get(field) is not None and meta[field] != PINS[key]["sha256"]:
                errs.append(f"{key} archive sha")
    else:  # Existing shards recorded and verified the original compressed archives.
        for key, field in (("katago_zip", "katago_zip_sha256"),
                           ("main_net", "main_net_sha256"), ("human_net", "human_net_sha256")):
            if meta.get(field) != PINS[key]["sha256"]: errs.append(f"{key} archive sha")
    return errs


def load():
    return pd.read_parquet(LEDGER)


def save(l):
    l.to_parquet(LEDGER + ".tmp", index=False); os.replace(LEDGER + ".tmp", LEDGER)


def cmd_init():
    sh = pd.read_parquet(f"{SCAN}/shards.parquet")
    l = sh[["shard_id", "tier", "n_games", "content_hash"]].copy()
    for c in ["status", "platform", "gpu", "config_hash", "note"]:
        l[c] = ""
    for c in ["started", "finished", "wall_sec", "games_per_sec"]:
        l[c] = np.nan
    l["status"] = "todo"
    p = l.shard_id == "P0000"
    l.loc[p, ["status", "platform", "gpu", "note"]] = ["done", "colab", "NVIDIA L4",
        "pilot (rootNumSymmetriesToSample=1 random, same cfg); files data/cache/katago/pilot_hsl_*.parquet"]
    save(l); print(l.status.value_counts())


def validate(sid, d, led_row, man_uids, cfg):
    errs = []
    mp_, gp_, jp = f"{d}/{sid}_moves.parquet", f"{d}/{sid}_games.parquet", f"{d}/{sid}_meta.json"
    for f in (mp_, gp_, jp):
        if not os.path.exists(f):
            return [f"missing {os.path.basename(f)}"]
    meta = json.load(open(jp))
    fm = json.loads(pq.read_schema(mp_).metadata[b"scan_meta"]); fg = json.loads(pq.read_schema(gp_).metadata[b"scan_meta"])
    if fm != meta or fg != meta:
        errs.append("footer meta differs from sidecar json")
    if meta["shard_id"] != sid: errs.append("shard_id mismatch")
    if meta["content_hash"] != led_row.content_hash: errs.append("content_hash != manifest")
    if meta.get("fake_engine") or meta.get("limit_games"): errs.append("test run (fake engine / limit_games)")
    if meta["katago_version"] != PINS["katago_version"]: errs.append("katago version")
    errs.extend(asset_hash_errors(meta))
    if meta["config_hash"] != config_hash(cfg, PINS): errs.append("config_hash differs from pinned setting")
    if meta["n_games_in"] != led_row.n_games: errs.append(f"n_games_in {meta['n_games_in']} != {led_row.n_games}")
    if meta["n_games_out"] + meta["n_skipped"] != meta["n_games_in"]: errs.append("games out + skipped != in")
    mv = pd.read_parquet(mp_, columns=["uid", "own_move_idx", "n_illegal", "logp_0", "logp_10"])
    gm = pd.read_parquet(gp_, columns=["uid", "n_moves", "ml_rank_12k"])
    if mv.duplicated(["uid", "own_move_idx"]).any(): errs.append("duplicate (uid, own_move_idx)")
    if gm.uid.duplicated().any(): errs.append("duplicate games uid")
    if set(gm.uid) != set(mv.uid): errs.append("games/moves uid sets differ")
    if not set(gm.uid) <= man_uids: errs.append("uids not in shard manifest")
    if mv[["logp_0", "logp_10"]].isna().any().any() or gm.ml_rank_12k.isna().any(): errs.append("NaN")
    if len(mv) != meta["n_move_rows"] or len(gm) != meta["n_games_out"]: errs.append("row counts != meta")
    if (gm.set_index("uid").n_moves != mv.groupby("uid").size()).any(): errs.append("n_moves != rows per uid")
    return errs


def cmd_ingest(d, cfg):
    """Validate every S*_meta.json in d OUTSIDE the lock, then enter the ledger lock per shard: re-read, skip if already
    done (first valid result wins), else copy files into results/ (tmp -> rename) and record done in the same lock."""
    man = pd.read_parquet(f"{SCAN}/manifest.parquet", columns=["uid", "shard_id"]); os.makedirs(RES, exist_ok=True)
    for jp in sorted(glob.glob(f"{d}/S*_meta.json")):
        sid = os.path.basename(jp).split("_")[0]
        l = load(); i = l.index[l.shard_id == sid]
        if len(i) != 1:
            print("REJECT", sid, "not in ledger"); continue
        row = l.loc[i[0]]
        errs = validate(sid, d, row, set(man.uid[man.shard_id == sid]), cfg)
        if errs:
            if scan_ledger.mark_failed(LEDGER, sid, "; ".join(errs), owner="ingest"): print("REJECT", sid, errs)
            else: print("already done", sid)
            continue
        meta = json.load(open(jp))

        def install(sid=sid, d=d):
            for nm, ext in (("moves", "parquet"), ("games", "parquet"), ("meta", "json")):
                shutil.copy(f"{d}/{sid}_{nm}.{ext}", f"{RES}/{sid}_{nm}.{ext}.tmp")
            for nm, ext in (("moves", "parquet"), ("games", "parquet"), ("meta", "json")):
                os.replace(f"{RES}/{sid}_{nm}.{ext}.tmp", f"{RES}/{sid}_{nm}.{ext}")
        if scan_ledger.mark_done(LEDGER, sid, meta, validation_errors=errs, install=install, owner="ingest"):
            print("ACCEPT", sid, f"{meta['games_per_sec_wall']:.2f} games/s")
        else:
            print("already done", sid)


def cmd_reconcile(cfg):
    """results/ vs ledger: valid result files whose shard is not done get recorded; reports invalid / missing."""
    man = pd.read_parquet(f"{SCAN}/manifest.parquet", columns=["uid", "shard_id"])
    def vf(sid, row, res_dir): return validate(sid, res_dir, row, set(man.uid[man.shard_id == sid]), cfg)
    r = scan_ledger.reconcile(LEDGER, RES, vf)
    print("reconcile: repaired", r["repaired"], "invalid", r["invalid"], "missing_files", r["missing"]); return r


def cmd_status():
    l = load()
    print(l.groupby("status").agg(shards=("shard_id", "count"), games=("n_games", "sum")))
    print(l[l.status == "done"].groupby(["platform", "gpu"]).agg(shards=("shard_id", "count"), games=("n_games", "sum"), mean_gps=("games_per_sec", "mean")))
    t1 = l[l.tier == 1]; print("tier1 done games", t1[t1.status == "done"].n_games.sum(), "of", t1.n_games.sum())


def cmd_merge(moves):
    l = load(); done = l[(l.status == "done") & (l.shard_id != "P0000")].shard_id.tolist()
    out = f"{ROOT}/data/cache/katago"
    gs = [pd.read_parquet(f"{out}/pilot_hsl_games.parquet")] + [pd.read_parquet(f"{RES}/{s}_games.parquet") for s in done]
    g = pd.concat(gs, ignore_index=True); assert not g.uid.duplicated().any()
    g.to_parquet(f"{out}/full_hsl_games.parquet", index=False); print("games", g.shape)
    if moves:
        ms = [pd.read_parquet(f"{out}/pilot_hsl_moves.parquet")] + [pd.read_parquet(f"{RES}/{s}_moves.parquet") for s in done]
        m = pd.concat(ms, ignore_index=True); m.to_parquet(f"{out}/full_hsl_moves.parquet", index=False); print("moves", m.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("args", nargs="*")
    ap.add_argument("--cfg", default=f"{HERE}/analysis.cfg"); ap.add_argument("--moves", action="store_true")
    a = ap.parse_args()
    if a.cmd == "init": cmd_init()
    elif a.cmd == "claim":
        want = set(a.args[0].split(","))
        def _claim(l):    # done shards are never re-claimed
            m = l.shard_id.isin(want) & (l.status != "done")
            l.loc[m, ["status", "platform", "started"]] = ["running", a.args[1], time.time()]; return l.shard_id[m].tolist()
        scan_ledger.rmw(LEDGER, _claim, op="cli-claim")
    elif a.cmd == "release":    # legacy platform-string release (tier-1 style); multi-owner code uses scan_ledger.release(owner)
        def _rel(l):
            m = (l.status == "running") & (l.platform == a.args[0]); l.loc[m, "status"] = "todo"; return l.shard_id[m].tolist()
        scan_ledger.rmw(LEDGER, _rel, op="cli-release")
    elif a.cmd == "reconcile": cmd_reconcile(a.cfg)
    elif a.cmd == "ingest": cmd_ingest(a.args[0], a.cfg)
    elif a.cmd == "status": cmd_status()
    elif a.cmd == "merge": cmd_merge(a.moves)
