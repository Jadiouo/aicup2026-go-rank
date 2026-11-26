"""Portable resumable shard runner for the full KataGo HumanSL scan (any platform, any GPU count).

Per shard it writes <out>/<shard>_moves.parquet and <out>/<shard>_games.parquet (same columns as the pilot's
pilot_hsl_moves/games.parquet) plus <shard>_meta.json; the same provenance is stored in the parquet footer
(key "scan_meta"). Work is checkpointed per chunk of 50 games in <work>/<shard>/chunk_NNNN.parquet, so a
re-run (same --work) resumes. One engine process per GPU (CUDA_VISIBLE_DEVICES), chunks via a shared queue.

  python scan_runner.py --shards S0001,S0002 --sgf-dir DIR --shards-table shards.parquet \
      --katago-dir ROOT/katago --out OUT --work WORK --platform kaggle [--gpus 0,1] [--max-minutes 600]
"""
import argparse, collections, glob, hashlib, json, math, multiprocessing as mp, os, platform as _pf, subprocess, sys, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kgio import Engine, parse_sgf_moves, mk_query, move_idx, normalize_game, get_color_semantics  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def content_hash(uids, colors, sgfs):
    h = hashlib.sha256()
    for u, c, s in zip(uids, colors, sgfs):
        h.update(f"{u}\t{int(c)}\t{s}\n".encode())
    return h.hexdigest()


def config_hash(cfg_path, pins):
    """One hash over everything that influences the numbers: cfg file, engine overrides, query settings."""
    h = hashlib.sha256()
    h.update(open(cfg_path, "rb").read())
    h.update(json.dumps([pins["engine_overrides"], pins["query"]], sort_keys=True).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- per-game features (same as build_pilot_features.py)
def games_from_moves(mv, profiles):
    P = len(profiles)
    mv = mv.sort_values(["uid", "own_move_idx"]).reset_index(drop=True)
    L = mv[[f"logp_{i}" for i in range(P)]].to_numpy(np.float64)

    def agg(mask=None):
        d = pd.DataFrame(L, columns=profiles); d["uid"] = mv.uid.values
        if mask is not None:
            d = d[mask]
        return d.groupby("uid", sort=True)
    g = mv.groupby("uid", sort=True)
    a = agg(); out = pd.DataFrame(index=a.mean().index)
    M = a.mean(); S = a.std(ddof=0); D = a.median()
    for p in profiles:
        out[f"ml_{p}"] = M[p]
    ml = M.to_numpy()
    for i, p in enumerate(profiles):
        out[f"mlc_{p}"] = ml[:, i] - ml.mean(axis=1)
    out["ml_argmax_idx"] = ml.argmax(axis=1).astype("int8")
    w = np.exp(ml - ml.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True)
    out["ml_exp_idx"] = (w * np.arange(P)).sum(axis=1)
    for p in profiles:
        out[f"mlmed_{p}"] = D[p]; out[f"mlstd_{p}"] = S[p]
    for name, mask in [("early", mv.own_move_idx.values < 20), ("late", mv.own_move_idx.values >= 20)]:
        m2 = agg(mask).mean().reindex(out.index)
        m2 = m2.fillna(M)         # game with <=20 own moves has no 'late' rows: fall back to the whole-game mean (was: length-mismatch crash of the whole shard)
        for p in profiles:
            out[f"ml{name}_{p}"] = m2[p]
        e = m2.to_numpy(); ww = np.exp(e - e.max(axis=1, keepdims=True)); ww /= ww.sum(axis=1, keepdims=True)
        out[f"ml_exp_idx_{name}"] = (ww * np.arange(P)).sum(axis=1)
    out["n_moves"] = g.size().astype("int16")
    R = mv[[f"rank_{i}" for i in range(P)]].to_numpy(); E = mv[[f"ent_{i}" for i in range(P)]].to_numpy(np.float64)
    for nm, A in [("top1", (R == 1).astype(float)), ("lrank", np.log(R)), ("ent", E)]:
        d = pd.DataFrame(A, columns=profiles); d["uid"] = mv.uid.values
        m3 = d.groupby("uid", sort=True).mean()
        for p in profiles:
            out[f"{nm}_{p}"] = m3[p]
    out = out.reset_index()
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype("float32")
    return out


# ---------------------------------------------------------------- engine work
class FakeEngine:
    """Deterministic stand-in for tests without a GPU (SCAN_FAKE_ENGINE=1)."""
    def __init__(self, *a, **k): pass
    def run(self, queries):
        res = []
        for q in queries:
            for t in q["analyzeTurns"]:
                rng = np.random.default_rng(abs(hash((q["id"], t))) % (2**32))
                p = rng.dirichlet(np.ones(362) * 0.05)
                res.append({"id": q["id"], "turnNumber": t, "humanPolicy": p.tolist()})
        return res, 0.001
    def close(self): pass


def process_chunk(e, sub, pins):
    """sub: DataFrame(uid,color,sgf). -> (moves DataFrame, n_eval, engine_sec, skipped list)"""
    prof = pins["query"]["profiles"]; NOWN = pins["query"]["n_own_moves"]; nsym = pins["query"]["root_num_symmetries"]
    qs = []; meta = {}; skipped = []; cs = get_color_semantics()
    for u, c, s in zip(sub.uid, sub.color, sub.sgf):
        ng = normalize_game(s, int(c), cs)      # current format: same moves/colour/komi as before; HD>=2: stones + swapped moves
        if ng.skip:
            skipped.append((u, ng.skip)); continue
        if ng.truncated:
            print(f"truncated_illegal {u} -> {len(ng.moves)} moves", flush=True)
        mv = ng.moves; col = "BW"[ng.eff_color]
        own = [t for t in range(len(mv)) if mv[t][0] == col][:NOWN]
        if not own:
            skipped.append((u, "no_own_moves")); continue
        meta[u] = (mv, own)
        komi = pins["query"]["komi"] if ng.komi is None else ng.komi
        for p in prof:
            qs.append(mk_query(f"{u}|{p}", mv[:own[-1] + 1], own, pins["query"]["max_visits"], komi,
                               pins["query"]["rules"], profile=p, policy=True, nsym=nsym,
                               initial_stones=ng.initial_stones, initial_player=ng.initial_player))
    res, dt = e.run(qs) if qs else ([], 0.0)
    bad_uids = {i.rsplit("|", 1)[0] for i in getattr(e, "last_errors", {})} if qs else set()     # engine rejected the query
    if bad_uids:
        for u in sorted(bad_uids):
            skipped.append((u, "engine_error")); meta.pop(u, None)
        res = [r for r in res if r["id"].rsplit("|", 1)[0] not in bad_uids]
    acc = {}; n_eval = 0
    for r in res:
        uid, p = r["id"].rsplit("|", 1); t = r["turnNumber"]; mv, own = meta[uid]
        pol = np.asarray(r["humanPolicy"], dtype=np.float64)
        pa = pol[move_idx(mv[t][1])]
        pos = np.clip(pol, 0, None); q = pos[pos > 0] / pos.sum()
        ent = float(-(q * np.log(q)).sum())
        rk = int((pol > pa).sum()) + 1
        acc.setdefault((uid, t), {})[p] = (math.log(max(pa, 1e-9)), rk, ent, pa < 0)
        n_eval += 1
    expected_turns = {(uid, t) for uid, (_, own) in meta.items() for t in own}
    assert set(acc) == expected_turns, f"missing or unexpected turn responses: {sorted(expected_turns - set(acc))[:5]}"
    rows = []
    for (uid, t), d in acc.items():
        mv, own = meta[uid]
        assert len(d) == len(prof), (uid, t, len(d))
        row = {"uid": uid, "own_move_idx": own.index(t), "ply": t, "n_illegal": int(sum(d[p][3] for p in prof))}
        for i, p in enumerate(prof):
            row[f"logp_{i}"] = d[p][0]; row[f"rank_{i}"] = d[p][1]; row[f"ent_{i}"] = d[p][2]
        rows.append(row)
    return pd.DataFrame(rows), n_eval, dt, skipped


def cast_moves(out, P):
    for i in range(P):
        out[f"logp_{i}"] = out[f"logp_{i}"].astype("float32"); out[f"ent_{i}"] = out[f"ent_{i}"].astype("float32")
        out[f"rank_{i}"] = out[f"rank_{i}"].astype("int16")
    out["own_move_idx"] = out.own_move_idx.astype("int16"); out["ply"] = out.ply.astype("int16")
    out["n_illegal"] = out.n_illegal.astype("int64")
    return out


def worker(gpu, tq, rq, a, pins):
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu != "" else {}
    kd = a.katago_dir
    if os.environ.get("SCAN_FAKE_ENGINE"):
        e = FakeEngine()
    else:
        os.makedirs(a.work, exist_ok=True)
        e = Engine(f"{kd}/katago", f"{kd}/{pins['main_net']['name'][:-3]}/model.bin", f"{kd}/{pins['human_net']['name'][:-3]}/model.bin",
                   a.cfg, pins["engine_overrides"], log=f"{a.work}/katago_stderr_gpu{gpu or 0}.log",
                   cu12=a.cu12, env_extra=env)
    rq.put(("ready", gpu))
    P = len(pins["query"]["profiles"])
    while True:
        task = tq.get()
        if task is None:
            break
        sid, ci, fn, sub = task
        if os.path.exists(fn):
            rq.put(("chunk", sid, ci, 0, 0.0, [], gpu)); continue
        mvdf, n_eval, dt, skipped = process_chunk(e, sub, pins)
        if len(mvdf):
            cast_moves(mvdf, P)
        else:
            mvdf = pd.DataFrame({"uid": []})
        mvdf.to_parquet(fn + ".tmp"); os.replace(fn + ".tmp", fn)
        rq.put(("chunk", sid, ci, n_eval, dt, skipped, gpu))
    e.close()


def gpu_info():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True)
        return [x.strip() for x in o.strip().splitlines()]
    except Exception:
        return []


def shard_stream(initial, queue_file=None, idle_minutes=10.0, poll_sec=5.0, sleep=time.sleep, clock=time.time):
    """Yield the initial shard ids, then (with queue_file) shard ids appended to that file, one per line.  A line `END`
    finishes the stream once every id before it was handed out; so does no new id for idle_minutes.  Without queue_file
    this is exactly the old behaviour (iterate the list)."""
    seen = set()
    for s in initial:
        if s and s not in seen: seen.add(s); yield s
    if not queue_file: return
    last_new = clock()
    while True:
        try: lines = [x.strip() for x in open(queue_file).read().splitlines() if x.strip()]
        except OSError: lines = []
        new = [x for x in lines if x != "END" and x not in seen]
        for s in new: seen.add(s); last_new = clock(); yield s
        if new: continue
        if "END" in lines or clock() - last_new > idle_minutes * 60: return
        sleep(poll_sec)


def main():
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="comma list (S0001,S0002) or 'auto' = claim todo shards from --ledger in priority order")
    ap.add_argument("--ledger", default=None, help="ledger.parquet (needed for --shards auto)")
    ap.add_argument("--range", default=None, help="auto mode: only shards lo:hi, e.g. S0001:S0047 (avoid collisions between platforms)")
    ap.add_argument("--skip", default="", help="auto mode: comma list of shard ids to skip")
    ap.add_argument("--time-budget-hours", type=float, default=0, help="total wall budget; no new shard is started when the remainder < reserve or < 1.1x last shard time")
    ap.add_argument("--elapsed-before-min", type=float, default=0, help="minutes already spent (setup) that count against the budget")
    ap.add_argument("--reserve-minutes", type=float, default=20)
    ap.add_argument("--sgf-dir", required=True); ap.add_argument("--shards-table", required=True)
    ap.add_argument("--katago-dir", required=True); ap.add_argument("--cfg", default=os.path.join(HERE, "analysis.cfg"))
    ap.add_argument("--pins", default=os.path.join(HERE, "scan_pins.json"))
    ap.add_argument("--out", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--platform", required=True); ap.add_argument("--gpus", default="auto")
    ap.add_argument("--cu12", default=os.environ.get("CU12_DIR"))
    ap.add_argument("--queue-file", default=None, help="after --shards, keep reading shard ids (one per line) appended to this file; line END (or --idle-minutes without news) finishes")
    ap.add_argument("--idle-minutes", type=float, default=10.0)
    ap.add_argument("--limit-games", type=int, default=0, help="testing only: use first N games (shard NOT valid for merge)")
    a = ap.parse_args()
    pins = json.load(open(a.pins)); os.makedirs(a.out, exist_ok=True); os.makedirs(a.work, exist_ok=True)
    fake = bool(os.environ.get("SCAN_FAKE_ENGINE"))
    ginfo = gpu_info()
    if a.gpus == "auto":
        gpus = [str(i) for i in range(len(ginfo))] or [""]
    else:
        gpus = a.gpus.split(",")
    def archive_sha(key):
        path = f"{a.katago_dir}/{pins[key]['name']}"
        return sha256_file(path) if os.path.isfile(path) else None

    prov = {"scan_version": pins["scan_version"], "platform": a.platform, "gpu": "; ".join(sorted(set(x.split(",")[0] for x in ginfo))) or "none",
            "n_gpus_used": len(gpus), "gpu_raw": ginfo, "katago_version": pins["katago_version"],
            "katago_zip_sha256": archive_sha("katago_zip") if not fake else None,
            "katago_executable_sha256": sha256_file(f"{a.katago_dir}/katago") if not fake else pins["katago_zip"]["executable_sha256"],
            "main_net": pins["main_net"]["name"],
            "main_net_sha256": archive_sha("main_net") if not fake else None,
            "main_net_content_sha256": sha256_file(f"{a.katago_dir}/{pins['main_net']['name'][:-3]}/model.bin") if not fake else pins["main_net"]["content_sha256"],
            "human_net": pins["human_net"]["name"],
            "human_net_sha256": archive_sha("human_net") if not fake else None,
            "human_net_content_sha256": sha256_file(f"{a.katago_dir}/{pins['human_net']['name'][:-3]}/model.bin") if not fake else pins["human_net"]["content_sha256"],
            "config_hash": config_hash(a.cfg, pins), "host": _pf.node()[:0] + _pf.platform(), "fake_engine": fake,
            "limit_games": a.limit_games}
    for key, archive_field, content_field, expected_field in (
        ("katago_zip", "katago_zip_sha256", "katago_executable_sha256", "executable_sha256"),
        ("main_net", "main_net_sha256", "main_net_content_sha256", "content_sha256"),
        ("human_net", "human_net_sha256", "human_net_content_sha256", "content_sha256"),
    ):
        assert prov[content_field] == pins[key][expected_field], f"{key} content sha256 mismatch"
        assert prov[archive_field] is None or prov[archive_field] == pins[key]["sha256"], f"{key} archive sha256 mismatch"
    kz = os.path.join(a.katago_dir, "katago")
    if not fake:
        v = subprocess.run([kz, "version"], capture_output=True, text=True, env={**os.environ, "LD_LIBRARY_PATH": ":".join(glob.glob((a.cu12 or "/content/cu12") + "/nvidia/*/lib")) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}).stdout
        assert pins["katago_version"].lstrip("v") in v, f"KataGo version mismatch: {v[:120]}"
    print("PROVENANCE", json.dumps({k: v for k, v in prov.items() if k != "gpu_raw"}), flush=True)
    stab = pd.read_parquet(a.shards_table).set_index("shard_id")
    ctx = mp.get_context("spawn")
    tq, rq = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(g, tq, rq, a, pins)) for g in gpus]
    for p in procs:
        p.start()
    for _ in procs:
        m = rq.get(timeout=900); assert m[0] == "ready"
    print("engines ready", len(procs), flush=True)
    t_run = time.time(); CH = pins["chunk_games"]; P = len(pins["query"]["profiles"]); prof = pins["query"]["profiles"]
    done_list = []; t_prog = T0 - a.elapsed_before_min * 60; shard_secs = []
    if a.shards == "auto":
        led = pd.read_parquet(a.ledger)
        todo = led[led.status.isin(["todo", "failed"])].shard_id.tolist()
        skip = set(filter(None, a.skip.split(",")))
        if a.range:
            lo, hi = a.range.split(":"); todo = [s for s in todo if lo <= s <= hi]
        shard_iter = [s for s in sorted(todo) if s not in skip]
        print("auto shards:", len(shard_iter), shard_iter[:5], "...", flush=True)
    else:
        shard_iter = shard_stream(a.shards.split(","), a.queue_file, a.idle_minutes)
    budget = a.time_budget_hours * 3600
    for sid in shard_iter:
        if budget:
            remain = budget - (time.time() - t_prog)
            need = max(a.reserve_minutes * 60, 1.1 * max(shard_secs) if shard_secs else 0)
            if remain < need:
                print(f"time budget: remaining {remain / 60:.0f} min < {need / 60:.0f} min needed; stop before {sid}", flush=True); break
        if os.path.exists(f"{a.out}/{sid}_meta.json") and not a.limit_games:
            print("already done", sid); done_list.append(sid); continue
        sg = pd.read_parquet(f"{a.sgf_dir}/{sid}.parquet")
        ch = content_hash(sg.uid, sg.color, sg.sgf)
        assert ch == stab.loc[sid, "content_hash"] and len(sg) == stab.loc[sid, "n_games"], f"{sid}: content hash/size mismatch"
        if a.limit_games:
            sg = sg.iloc[:a.limit_games]
        t0 = time.time(); wd = f"{a.work}/{sid}"; os.makedirs(wd, exist_ok=True)
        nch = int(np.ceil(len(sg) / CH))
        for ci in range(nch):
            tq.put((sid, ci, f"{wd}/chunk_{ci:04d}.parquet", sg.iloc[ci * CH:(ci + 1) * CH].reset_index(drop=True)))
        n_eval = 0; busy = {}; skipped = []; got = 0
        while got < nch:
            m = rq.get()
            _, _, ci, ne, dt, sk, g = m; got += 1; n_eval += ne; busy[g] = busy.get(g, 0) + dt; skipped += sk
            if got % 10 == 0 or got == nch:
                print(f"{sid} chunks {got}/{nch} evals {n_eval} wall {time.time() - t0:.0f}s", flush=True)
        parts = [pd.read_parquet(f"{wd}/chunk_{ci:04d}.parquet") for ci in range(nch)]
        mv = pd.concat([p for p in parts if len(p)], ignore_index=True)
        mv = cast_moves(mv, P).sort_values(["uid", "own_move_idx"]).reset_index(drop=True)
        gm = games_from_moves(mv, prof)
        wall = time.time() - t0
        meta = {**prov, "shard_id": sid, "content_hash": ch, "n_games_in": len(sg), "n_games_out": len(gm), "n_move_rows": len(mv),
                "n_skipped": len(skipped), "skipped": skipped[:50],
                "skip_reasons": dict(collections.Counter(r for _, r in skipped)), "color_semantics": get_color_semantics(), "n_eval": n_eval, "wall_sec": wall,
                "engine_sec_per_gpu": busy, "evals_per_sec_wall": n_eval / wall, "games_per_sec_wall": len(sg) / wall,
                "t_start": t0, "t_end": time.time(), "pins_query": pins["query"], "engine_overrides": pins["engine_overrides"]}
        md = json.dumps(meta, default=str)
        suffix = "" if not a.limit_games else "_TEST"
        for nm, df in (("moves", mv), ("games", gm)):
            tb = pa.Table.from_pandas(df, preserve_index=False)
            tb = tb.replace_schema_metadata({**(tb.schema.metadata or {}), b"scan_meta": md.encode()})
            fn = f"{a.out}/{sid}_{nm}{suffix}.parquet"
            pq.write_table(tb, fn + ".tmp", compression="zstd"); os.replace(fn + ".tmp", fn)
        with open(f"{a.out}/{sid}_meta{suffix}.json.tmp", "w") as f:
            f.write(md)
        os.replace(f"{a.out}/{sid}_meta{suffix}.json.tmp", f"{a.out}/{sid}_meta{suffix}.json")
        print(f"SHARD_DONE {sid} games {len(gm)}/{len(sg)} moves {len(mv)} evals {n_eval} wall {wall:.0f}s "
              f"{n_eval / wall:.0f} evals/s {len(sg) / wall:.2f} games/s", flush=True)
        done_list.append(sid); shard_secs.append(wall)
    for _ in procs:
        tq.put(None)
    for p in procs:
        p.join(60)
    print("RUNNER_DONE", done_list, flush=True)


if __name__ == "__main__":
    main()
