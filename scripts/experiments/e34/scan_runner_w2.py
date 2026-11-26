"""E34 scanner for a later window of own moves (default own_move_idx 60..119) -- a separate copy of the tier-1/2 scan setting.

Reuses scan_runner.py (games_from_moves, cast_moves, FakeEngine, shard_stream, sha256_file, content_hash, gpu_info) and kgio.py unchanged.
Differences from scan_runner.process_chunk / main (nothing else):
  * own moves are own_all[lo:hi] (pins["query"]["own_move_range"]); the query still sends the full prefix mv[:own[-1]+1] with analyzeTurns=own,
  * emitted `own_move_idx` is ABSOLUTE (index among all own moves), `ply` as before,
  * games with no own move in the range are skipped (reason `no_own_moves_in_range`) -- their window-120 features equal window-60,
  * config_hash_w2 = sha256(cfg || json[engine_overrides, query_w2]); meta also carries parent_config_hash and parent_hash_check.
Layout and resume behaviour as scan_runner (chunk files in <work>/<shard>/, per-shard out files + meta).

  python scan_runner_w2.py --shards E3401,E3402 --sgf-dir sgf --shards-table shards_w2.parquet --katago-dir katago --out out --work work \
      --platform school --gpus 0,0,0,0 --pins scan_pins_w2.json --pins-parent scan_pins_parent.json [--queue-file queue.txt]
"""
import argparse, collections, glob, json, math, multiprocessing as mp, os, platform as _pf, subprocess, sys, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "..", "katago")):
    if os.path.isfile(os.path.join(_p, "scan_runner.py")) and _p not in sys.path:
        sys.path.insert(0, _p)
from kgio import Engine, mk_query, move_idx, normalize_game, get_color_semantics  # noqa: E402
import scan_runner as base  # noqa: E402
from scan_runner import games_from_moves, cast_moves, FakeEngine, shard_stream, sha256_file, content_hash, gpu_info, config_hash  # noqa: E402

SKIP_RANGE = "no_own_moves_in_range"


# ------------------------------------------------------------------ pins / hashes
def parent_query(pins_w2):
    """Query block of the original scan reconstructed from the w2 pins: own_move_range removed, n_own_moves=60."""
    q = dict(pins_w2["query"]); q.pop("own_move_range"); q["n_own_moves"] = 60
    return q


def same_origin_errors(pins_w2, pins_parent):
    """[] if pins_w2 == pins_parent in everything except the own-move window."""
    errs = []
    for k in sorted(set(pins_w2) | set(pins_parent)):
        if k != "query" and pins_w2.get(k) != pins_parent.get(k): errs.append(f"pins key {k} differs")
    if parent_query(pins_w2) != pins_parent["query"]: errs.append("query (minus own-move window) differs from parent")
    return errs


def config_hash_w2(cfg_path, pins_w2):
    return config_hash(cfg_path, pins_w2)                # same formula, query_w2 contains own_move_range instead of n_own_moves


def parent_hashes(cfg_path, pins_w2, pins_parent):
    """-> (parent_config_hash from the real parent pins, hash recomputed from query_w2 with the window removed)."""
    return config_hash(cfg_path, pins_parent), config_hash(cfg_path, {"engine_overrides": pins_w2["engine_overrides"], "query": parent_query(pins_w2)})


# ------------------------------------------------------------------ engine work
def process_chunk_w2(e, sub, pins):
    """sub: DataFrame(uid,color,sgf) -> (moves DataFrame, n_eval, engine_sec, skipped list).  Same as scan_runner.process_chunk except the own-move window."""
    q = pins["query"]; prof = q["profiles"]; lo, hi = q["own_move_range"]; nsym = q["root_num_symmetries"]
    qs = []; meta = {}; skipped = []; cs = get_color_semantics()
    for u, c, s in zip(sub.uid, sub.color, sub.sgf):
        ng = normalize_game(s, int(c), cs)
        if ng.skip:
            skipped.append((u, ng.skip)); continue
        if ng.truncated:
            print(f"truncated_illegal {u} -> {len(ng.moves)} moves", flush=True)
        mv = ng.moves; col = "BW"[ng.eff_color]
        own_all = [t for t in range(len(mv)) if mv[t][0] == col]
        own = own_all[lo:hi]
        if not own:
            skipped.append((u, "no_own_moves" if not own_all else SKIP_RANGE)); continue
        meta[u] = (mv, own, lo)
        komi = q["komi"] if ng.komi is None else ng.komi
        for p in prof:
            qs.append(mk_query(f"{u}|{p}", mv[:own[-1] + 1], own, q["max_visits"], komi, q["rules"], profile=p, policy=True, nsym=nsym,
                               initial_stones=ng.initial_stones, initial_player=ng.initial_player))
    res, dt = e.run(qs) if qs else ([], 0.0)
    bad_uids = {i.rsplit("|", 1)[0] for i in getattr(e, "last_errors", {})} if qs else set()
    if bad_uids:
        for u in sorted(bad_uids):
            skipped.append((u, "engine_error")); meta.pop(u, None)
        res = [r for r in res if r["id"].rsplit("|", 1)[0] not in bad_uids]
    acc = {}; n_eval = 0
    for r in res:
        uid, p = r["id"].rsplit("|", 1); t = r["turnNumber"]; mv, own, _ = meta[uid]
        pol = np.asarray(r["humanPolicy"], dtype=np.float64)
        pa_ = pol[move_idx(mv[t][1])]
        pos = np.clip(pol, 0, None); qq = pos[pos > 0] / pos.sum()
        ent = float(-(qq * np.log(qq)).sum())
        rk = int((pol > pa_).sum()) + 1
        acc.setdefault((uid, t), {})[p] = (math.log(max(pa_, 1e-9)), rk, ent, pa_ < 0)
        n_eval += 1
    expected_turns = {(uid, t) for uid, (_, own, _) in meta.items() for t in own}
    assert set(acc) == expected_turns, f"missing or unexpected turn responses: {sorted(expected_turns - set(acc))[:5]}"
    rows = []
    for (uid, t), d in acc.items():
        mv, own, lo_ = meta[uid]
        assert len(d) == len(prof), (uid, t, len(d))
        row = {"uid": uid, "own_move_idx": lo_ + own.index(t), "ply": t, "n_illegal": int(sum(d[p][3] for p in prof))}
        for i, p in enumerate(prof):
            row[f"logp_{i}"] = d[p][0]; row[f"rank_{i}"] = d[p][1]; row[f"ent_{i}"] = d[p][2]
        rows.append(row)
    return pd.DataFrame(rows), n_eval, dt, skipped


def reason_for(sgf, color, rng):
    """Why a game gave no rows (used for chunks resumed from an earlier run, whose skip list is lost)."""
    ng = normalize_game(sgf, int(color), get_color_semantics())
    if ng.skip: return ng.skip
    col = "BW"[ng.eff_color]; n_own = sum(1 for m in ng.moves if m[0] == col)
    return "no_own_moves" if n_own == 0 else (SKIP_RANGE if len(ng.moves) and n_own <= rng[0] else "engine_error")


def worker_w2(gpu, tq, rq, a, pins):
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu != "" else {}
    kd = a.katago_dir
    if os.environ.get("SCAN_FAKE_ENGINE"):
        e = FakeEngine()
    else:
        os.makedirs(a.work, exist_ok=True)
        e = Engine(f"{kd}/katago", f"{kd}/{pins['main_net']['name'][:-3]}/model.bin", f"{kd}/{pins['human_net']['name'][:-3]}/model.bin",
                   a.cfg, pins["engine_overrides"], log=f"{a.work}/katago_stderr_gpu{gpu or 0}.log", cu12=a.cu12, env_extra=env)
    rq.put(("ready", gpu))
    P = len(pins["query"]["profiles"])
    while True:
        task = tq.get()
        if task is None:
            break
        sid, ci, fn, sub = task
        if os.path.exists(fn):
            rq.put(("chunk", sid, ci, 0, 0.0, [], gpu)); continue
        mvdf, n_eval, dt, skipped = process_chunk_w2(e, sub, pins)
        if len(mvdf):
            cast_moves(mvdf, P)
        else:
            mvdf = pd.DataFrame({"uid": []})
        mvdf.to_parquet(fn + ".tmp"); os.replace(fn + ".tmp", fn)
        rq.put(("chunk", sid, ci, n_eval, dt, skipped, gpu))
    e.close()


def main():
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="comma list; '' with --queue-file")
    ap.add_argument("--sgf-dir", required=True); ap.add_argument("--shards-table", required=True)
    ap.add_argument("--katago-dir", required=True); ap.add_argument("--cfg", default=os.path.join(HERE, "analysis.cfg"))
    ap.add_argument("--pins", default=os.path.join(HERE, "scan_pins_w2.json")); ap.add_argument("--pins-parent", default=os.path.join(HERE, "scan_pins_parent.json"))
    ap.add_argument("--out", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--platform", required=True); ap.add_argument("--gpus", default="auto"); ap.add_argument("--cu12", default=os.environ.get("CU12_DIR"))
    ap.add_argument("--queue-file", default=None); ap.add_argument("--idle-minutes", type=float, default=10.0)
    ap.add_argument("--limit-games", type=int, default=0, help="testing only: shard NOT valid for ingest")
    a = ap.parse_args()
    pins = json.load(open(a.pins)); pins_parent = json.load(open(a.pins_parent))
    errs = same_origin_errors(pins, pins_parent)
    assert not errs, errs
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.work, exist_ok=True)
    fake = bool(os.environ.get("SCAN_FAKE_ENGINE")); ginfo = gpu_info()
    gpus = ([str(i) for i in range(len(ginfo))] or [""]) if a.gpus == "auto" else a.gpus.split(",")

    def archive_sha(key):
        path = f"{a.katago_dir}/{pins[key]['name']}"
        return sha256_file(path) if os.path.isfile(path) else None
    ph, ph_re = parent_hashes(a.cfg, pins, pins_parent)
    prov = {"scan_version": pins["scan_version"], "scan_kind": "e34_w2", "platform": a.platform, "gpu": "; ".join(sorted(set(x.split(",")[0] for x in ginfo))) or "none",
            "n_gpus_used": len(gpus), "gpu_raw": ginfo, "katago_version": pins["katago_version"],
            "katago_zip_sha256": archive_sha("katago_zip") if not fake else None,
            "katago_executable_sha256": sha256_file(f"{a.katago_dir}/katago") if not fake else pins["katago_zip"]["executable_sha256"],
            "main_net": pins["main_net"]["name"], "main_net_sha256": archive_sha("main_net") if not fake else None,
            "main_net_content_sha256": sha256_file(f"{a.katago_dir}/{pins['main_net']['name'][:-3]}/model.bin") if not fake else pins["main_net"]["content_sha256"],
            "human_net": pins["human_net"]["name"], "human_net_sha256": archive_sha("human_net") if not fake else None,
            "human_net_content_sha256": sha256_file(f"{a.katago_dir}/{pins['human_net']['name'][:-3]}/model.bin") if not fake else pins["human_net"]["content_sha256"],
            "config_hash": config_hash_w2(a.cfg, pins), "parent_config_hash": ph, "parent_hash_check": bool(ph == ph_re),
            "own_move_range": pins["query"]["own_move_range"], "host": _pf.platform(), "fake_engine": fake, "limit_games": a.limit_games,
            "code_sha256": {f: sha256_file(os.path.join(HERE, f)) for f in ("scan_runner_w2.py",) if os.path.isfile(os.path.join(HERE, f))}}
    assert prov["parent_hash_check"], "parent_config_hash does not reproduce from query_w2"
    for key, af, cf, ef in (("katago_zip", "katago_zip_sha256", "katago_executable_sha256", "executable_sha256"),
                            ("main_net", "main_net_sha256", "main_net_content_sha256", "content_sha256"),
                            ("human_net", "human_net_sha256", "human_net_content_sha256", "content_sha256")):
        assert prov[cf] == pins[key][ef], f"{key} content sha256 mismatch"
        assert prov[af] is None or prov[af] == pins[key]["sha256"], f"{key} archive sha256 mismatch"
    if not fake:
        v = subprocess.run([os.path.join(a.katago_dir, "katago"), "version"], capture_output=True, text=True,
                           env={**os.environ, "LD_LIBRARY_PATH": ":".join(glob.glob((a.cu12 or "/content/cu12") + "/nvidia/*/lib")) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}).stdout
        assert pins["katago_version"].lstrip("v") in v, f"KataGo version mismatch: {v[:120]}"
    print("PROVENANCE", json.dumps({k: v for k, v in prov.items() if k != "gpu_raw"}), flush=True)
    stab = pd.read_parquet(a.shards_table).set_index("shard_id")
    ctx = mp.get_context("spawn"); tq, rq = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker_w2, args=(g, tq, rq, a, pins)) for g in gpus]
    for p in procs: p.start()
    for _ in procs:
        m = rq.get(timeout=900); assert m[0] == "ready"
    print("engines ready", len(procs), flush=True)
    CH = pins["chunk_games"]; P = len(pins["query"]["profiles"]); prof = pins["query"]["profiles"]; done_list = []
    for sid in shard_stream([x for x in a.shards.split(",") if x], a.queue_file, a.idle_minutes):
        if os.path.exists(f"{a.out}/{sid}_meta.json") and not a.limit_games:
            print("already done", sid); done_list.append(sid); continue
        sg = pd.read_parquet(f"{a.sgf_dir}/{sid}.parquet")
        ch = content_hash(sg.uid, sg.color, sg.sgf)
        assert ch == stab.loc[sid, "content_hash"] and len(sg) == stab.loc[sid, "n_games"], f"{sid}: content hash/size mismatch"
        if a.limit_games: sg = sg.iloc[:a.limit_games]
        t0 = time.time(); wd = f"{a.work}/{sid}"; os.makedirs(wd, exist_ok=True); nch = int(np.ceil(len(sg) / CH))
        for ci in range(nch):
            tq.put((sid, ci, f"{wd}/chunk_{ci:04d}.parquet", sg.iloc[ci * CH:(ci + 1) * CH].reset_index(drop=True)))
        n_eval = 0; busy = {}; skipped = []; got = 0
        while got < nch:
            _, _, ci, ne, dt, sk, g = rq.get(); got += 1; n_eval += ne; busy[g] = busy.get(g, 0) + dt; skipped += sk
            if got % 10 == 0 or got == nch:
                print(f"{sid} chunks {got}/{nch} evals {n_eval} wall {time.time() - t0:.0f}s", flush=True)
        parts = [pd.read_parquet(f"{wd}/chunk_{ci:04d}.parquet") for ci in range(nch)]
        mv = pd.concat([p for p in parts if len(p)], ignore_index=True)
        mv = cast_moves(mv, P).sort_values(["uid", "own_move_idx"]).reset_index(drop=True)
        gm = games_from_moves(mv, prof); wall = time.time() - t0
        # chunks resumed from an earlier run carry no skip list: derive skipped games from what is missing
        sk_known = {u for u, _ in skipped}
        missing = sorted(set(sg.uid) - set(mv.uid) - sk_known)
        if missing:
            sm = sg.set_index("uid")
            skipped += [(u, reason_for(sm.loc[u, "sgf"], sm.loc[u, "color"], pins["query"]["own_move_range"])) for u in missing]
        meta = {**prov, "shard_id": sid, "content_hash": ch, "n_games_in": len(sg), "n_games_out": len(gm), "n_move_rows": len(mv), "n_skipped": len(skipped),
                "skipped": skipped[:50], "skip_reasons": dict(collections.Counter(r for _, r in skipped)), "color_semantics": get_color_semantics(), "n_eval": n_eval,
                "wall_sec": wall, "engine_sec_per_gpu": busy, "evals_per_sec_wall": n_eval / wall, "games_per_sec_wall": len(sg) / wall,
                "t_start": t0, "t_end": time.time(), "pins_query": pins["query"], "engine_overrides": pins["engine_overrides"]}
        md = json.dumps(meta, default=str); suffix = "" if not a.limit_games else "_TEST"
        for nm, df in (("moves", mv), ("games", gm)):
            tb = pa.Table.from_pandas(df, preserve_index=False)
            tb = tb.replace_schema_metadata({**(tb.schema.metadata or {}), b"scan_meta": md.encode()})
            fn = f"{a.out}/{sid}_{nm}{suffix}.parquet"; pq.write_table(tb, fn + ".tmp", compression="zstd"); os.replace(fn + ".tmp", fn)
        with open(f"{a.out}/{sid}_meta{suffix}.json.tmp", "w") as f: f.write(md)
        os.replace(f"{a.out}/{sid}_meta{suffix}.json.tmp", f"{a.out}/{sid}_meta{suffix}.json")
        print(f"SHARD_DONE {sid} games {len(gm)}/{len(sg)} moves {len(mv)} evals {n_eval} wall {wall:.0f}s {n_eval / wall:.0f} evals/s {len(sg) / wall:.2f} games/s", flush=True)
        done_list.append(sid)
    for _ in procs: tq.put(None)
    for p in procs: p.join(60)
    print("RUNNER_DONE", done_list, flush=True)


if __name__ == "__main__":
    main()
