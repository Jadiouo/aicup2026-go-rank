"""E37 Dse arm: the two Dse legs (M03_SE LightGBM and M04 CORAL, equal weights, EV decode downstream) retrained with the HSL columns taken from a given X file (W60 or W120).
Plan: experiments/E37_rank_v6_full/PLAN.md section 2 and "Dse W120 arm" (appendix).  Everything except the 104 per-game HSL columns is byte-for-byte what built the saved v4 Dse:
  * M03_SE leg = scripts/experiments/e21/run.py `fit` (L0 aggregate + HSL aggregate without bag_n/is_white + 7 SE bag columns; LightGBM lr .1, max_bin 63, 4 threads, derive_seed(SEED,"lgbm"),
    inv_players weights, early stopping on the E10 ES bags; num_boost_round / early_stopping_rounds = gogame.models.DEFAULT_PARAMS);
  * M04 leg = scripts/experiments/e10/pipeline.py stage_oof candidate "A" (CORAL on the 522-column HSL bag aggregate, chunk=None, inv_players weights);
  * the same training bags (CANDS["A"], train_seed(k)) and ES bags (make_es_bags) feed both legs;  Dse = lib.ensemble_probs(M03, M04) 0.5/0.5 on the Q2 validation questions of the fold.
Inputs: --home (build_dse_home.py; non-H caches + v4 reference), --x / --aux (per-game HSL = X[:, :104], rows in aux uid order), --label (names the output dir).
Outputs <out>/<label>/oof_{Dse,M03SE,M04}_fold{k}.parquet, meta_fold{k}.json (+ repro_fold{k}.json when checked).  File names equal E21's, so scripts/experiments/e37/score.py --dse120 <dir> reads them.
  PYTHONPATH=src:scripts systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 nice -n 10 .venv/bin/python -m scripts.experiments.e37.dse_w120 --label W60 --x .scratch/e29/bundle/X.npy --folds 0
Reproduction check (--repro auto = on for label W60, formal runs): the W60 outputs must equal the saved v4 values (E21 oof_Dse / oof_M03SE, E10 oof_A on its Q2 rows): max abs diff <= 1e-6 on praw_* / p_*, equal LightGBM best_iteration, equal CORAL iterations; MISMATCH stops the run.
--allow-partial: X may hold only some bundle rows (X120_partial.npy + --rows X120_partial_rows.npy, other games keep their W60 HSL from --x60); output label gets a `_partial` suffix and a PARTIAL.json marker; for smoke runs only, never judged.
--max-players N: smoke subset of the non-H players (E10 smoke seed); skips the n_train_bags / reference checks.  Never touches H (the home tree has none) and never reads h_eval_ledger.jsonl."""
from __future__ import annotations
import argparse, hashlib, json, os, resource, subprocess, sys, threading, time, traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"          # identical to E10 / E21 / E28 (<= 4 threads; LightGBM num_threads is 4 as well)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
CODE_ROOT = Path(__file__).resolve().parents[3]          # repo root locally, <host root>/code on a host (same relative layout)
for _p in (CODE_ROOT / "scripts", CODE_ROOT / "src", CODE_ROOT):
    sys.path.insert(0, str(_p))
import numpy as np, pandas as pd

from scripts.experiments.e10 import lib
from gogame.aggregate import L0Store, aggregate_many, fit_sparse_selector, make_spec
from gogame.constants import SEED
from gogame.coral import fit_coral
from gogame.gametable import load_table
from gogame.models import fit_lgbm
from gogame.questions import bag_class_weights, sample_train_bags
from gogame.seeds import derive_seed
from gogame.split import read_split
import run_e01 as e01

PRM = {"num_threads": 4, "learning_rate": 0.1, "max_bin": 63, "log_every": 0}      # == e21/run.py PRM (also e28)
TOL = 1e-6
PC = lib.PCOLS
RC = [f"praw_{j}" for j in range(10)]
SE_COLS = ["se_mean_mean", "se_mean_std", "se_mean_p10", "se_mean_p90", "se_mean_min", "se_mean_max", "se_std_mean"]
N_HSL = 104
HB = {"phase": "init", "t0": time.time()}
LOG = None


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


STOP_HB = threading.Event()


def heartbeat(period):
    while not STOP_HB.wait(period):
        log(f"hb {HB.get('phase')} {time.time() - HB['t0']:.0f}s peak_rss {rss_mb()}MB")


def sha_arr(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def se_feats(bags, sm, ss):
    """Copy of scripts/experiments/e21/run.py se_feats (7 bag-level SE columns from the bag's own games only)."""
    out = np.full((len(bags), 7), np.nan)
    for i, b in enumerate(bags):
        v = np.array([sm[u] for u in b.uids], float); v = v[~np.isnan(v)]
        w = np.array([ss[u] for u in b.uids], float); w = w[~np.isnan(w)]
        if len(v):
            out[i, :6] = [v.mean(), v.std(), np.percentile(v, 10), np.percentile(v, 90), v.min(), v.max()]
        if len(w):
            out[i, 6] = w.mean()
    return out


# ---------------------------------------------------------------------------------------------- inputs
class Ctx:
    pass


def load_hsl(a, uid, names):
    """float32 [n, 104] per-game HSL in aux uid order from the X file (formal: all rows; partial: rows from --rows, others from --x60)."""
    X = np.load(a.x, mmap_mode="r")
    assert X.ndim == 2 and X.shape[1] >= N_HSL and X.dtype == np.float32, (X.shape, X.dtype)
    if a.allow_partial:
        assert a.rows and a.x60, "--allow-partial needs --rows (bundle row index of every X row) and --x60 (full-pool W60 X for the other games)"
        rows = np.load(a.rows); assert rows.ndim == 1 and len(rows) == X.shape[0] and rows.min() >= 0 and rows.max() < len(uid) and len(set(rows.tolist())) == len(rows)
        X60 = np.load(a.x60, mmap_mode="r"); assert X60.shape[0] == len(uid)
        H = np.array(X60[:, :N_HSL], np.float32); H[rows] = X[:, :N_HSL]
        n_new = int((H.view(np.uint32) != np.asarray(X60[:, :N_HSL]).view(np.uint32)).any(axis=1).sum())
        return H, {"partial": True, "rows_from_x": int(len(rows)), "rows_changed_vs_w60": n_new}
    assert X.shape[0] == len(uid), f"{a.x}: {X.shape[0]} rows != {len(uid)} aux games (use --allow-partial with --rows for a partial table)"
    H = np.array(X[:, :N_HSL], np.float32)
    return H, {"partial": False, "rows_from_x": int(X.shape[0])}


def load_ctx(a):
    home = Path(a.home)
    hm = json.loads((home / "manifest.json").read_text())
    assert hm["h_check"]["split_holdout_rows"] == 0 and hm["h_check"]["meta_players_inter_H"] == 0, "home tree is not H-free"
    ctx = Ctx(); ctx.home = home
    aux = np.load(a.aux, allow_pickle=False)
    uid = aux["uid"].astype(str); names = aux["names"].astype(str).tolist()
    split = read_split(home / "data/splits/split.parquet")
    assert not split["is_holdout"].any()
    ctx.H = set()
    meta = pd.read_parquet(home / "data/cache/games_meta.parquet")
    assert set(meta["player_id"]) <= set(split["player_id"]) and set(meta["uid"]) == set(uid) and len(meta) == len(uid), "aux uid set != home games"
    table = load_table(home / "data/cache/l0")
    assert set(np.asarray(table.uids).astype(str)) == set(uid)
    ctx.l0 = L0Store(list(table.uids), table.dense, table.sparse, table.names); del table
    se = pd.read_parquet(home / "data/se_full.parquet")
    assert set(se["uid"]) == set(uid) and se["uid"].is_unique
    ctx.sm = dict(zip(se["uid"], se["se_mean"].astype(float))); ctx.ss = dict(zip(se["uid"], se["se_std"].astype(float)))
    H, hinfo = load_hsl(a, uid, names)
    assert np.isfinite(H).all(), "NaN/inf in the HSL columns"
    ctx.hnames = tuple(names[:N_HSL])
    ctx.hs = L0Store(list(uid), H, None, ctx.hnames)
    ctx.sp_h = make_spec(ctx.hnames, np.zeros(0, np.int32))
    ctx.hinfo = hinfo | {"hsl_sha256": sha_arr(H)}
    ctx.meta = meta
    ctx.index_all = lib.hsl_index(meta, list(uid))            # same call as pipeline.Ctx (every non-H game has HSL)
    nonh = split.reset_index(drop=True)
    if a.max_players:
        keep = np.random.default_rng(derive_seed(SEED, "e10", "smoke_players")).permutation(sorted(nonh["player_id"]))[:a.max_players]
        nonh = nonh[nonh["player_id"].isin(set(keep))]
    ctx.nonh = nonh.reset_index(drop=True)
    ctx.q2_all = e01.frame_to_bags(pd.read_parquet(home / "ref/questions_q2.parquet"))
    return ctx


# ---------------------------------------------------------------------------------------------- one fold
def compare(d, ref, cc=RC + PC):
    mine, r = d.set_index("qid"), ref.set_index("qid").loc[d["qid"]]
    assert list(r["rank_id"]) == list(mine["rank_id"])
    return float(np.abs(mine[cc].to_numpy() - r[cc].to_numpy()).max())


def run_fold(a, ctx, k, out):
    t0 = time.perf_counter(); HB["phase"] = f"fold{k}:setup"
    fin = out / f"oof_Dse_fold{k}.parquet"
    if fin.exists() and (out / f"meta_fold{k}.json").exists():
        log(f"fold {k}: cached"); return
    formal = not a.max_players
    ref = Path(ctx.home) / "ref"
    fs = lib.fold_sets(ctx.nonh, k)
    lib.assert_fold_disjoint(fs, ctx.H)
    val = [b for b in ctx.q2_all if b.player_id in fs["val"]]
    game_rep = lib.assert_no_game_leak(ctx.meta, [u for b in val for u in b.uids], fs["train_all"])
    c = lib.CANDS["A"]
    fit_in = sorted(p for p in fs["fit"] if p in ctx.index_all.games)
    tb = sample_train_bags(ctx.index_all, fit_in, c.bags_per_pc, c.n_range, seed=c.train_seed(k))
    es = lib.make_es_bags(ctx.index_all, fs["es"], k, False)
    refm = json.loads((ref / f"meta_D_fold{k}.json").read_text())
    if formal:
        a_q2 = pd.read_parquet(ref / f"oof_A_fold{k}.parquet").query("qset == 'q2'")
        assert set(a_q2["qid"]) == {b.qid for b in val}, "val Q2 question set != E10 oof_A"
        assert len(tb) == refm["n_train_bags"] and len(es) == refm["n_es_bags"], (len(tb), len(es), refm["n_train_bags"], refm["n_es_bags"])
    assert {b.player_id for b in tb} <= fs["fit"] and not ({b.player_id for b in tb} & (fs["val"] | fs["es"]))
    vu = {u for b in val for u in b.uids}
    assert not (vu & {u for b in tb + es for u in b.uids}), "validation game inside a training / ES bag"
    ytr, yes = np.array([b.rank_id for b in tb]), np.array([b.rank_id for b in es])
    wtr, wes = bag_class_weights(tb, "inv_players"), bag_class_weights(es, "inv_players")
    bags_sha = hashlib.sha256("|".join(",".join(b.uids) for b in tb + es + val).encode()).hexdigest()
    log(f"fold {k}: val {len(val)} train bags {len(tb)} es {len(es)} fit players {len(fit_in)}")
    # ---- M03_SE leg
    HB["phase"] = f"fold{k}:M03 features"
    used = sorted({u for b in tb for u in b.uids})
    cols = fit_sparse_selector(ctx.l0.sparse[ctx.l0.rows(used)], e01.CFG["min_df"]); del used
    spec_l0 = make_spec(ctx.l0.names, cols)
    mats, nonhsl = [], None
    for nm, bags in (("tr", tb), ("es", es), ("val", val)):
        XL = aggregate_many(bags, ctx.l0, spec_l0)
        XH = aggregate_many(bags, ctx.hs, ctx.sp_h)
        SE = se_feats(bags, ctx.sm, ctx.ss)
        if nm == "tr":
            nonhsl = sha_arr(XL, SE); hsl_blk = sha_arr(XH[:, :-2]); n_l0 = XL.shape[1]
        mats.append(np.hstack([XL, XH[:, :-2], SE])); del XL, XH
    Xtr, Xes, Xv = mats; del mats
    HB["phase"] = f"fold{k}:M03 fit"; t1 = time.perf_counter()
    m3 = fit_lgbm(Xtr, ytr, Xes, yes, PRM, derive_seed(SEED, "lgbm"), wtr, wes)
    pr, p = m3.predict_proba(Xv, raw=True), m3.predict_proba(Xv)
    assert np.isfinite(pr).all() and np.isfinite(p).all()
    d3 = e01.probs_frame(val, k, pr, p); d3["qset"] = "q2"
    sec_m03 = round(time.perf_counter() - t1); n_feat = int(Xtr.shape[1]); best3 = int(m3.best_iteration)
    del Xtr, Xes, Xv, m3
    # ---- M04 leg (CORAL, HSL aggregate only)
    HB["phase"] = f"fold{k}:M04 fit"; t1 = time.perf_counter()
    Xh = aggregate_many(tb, ctx.hs, ctx.sp_h)
    m4 = fit_coral(Xh, ytr, wtr, chunk=lib.CORAL_CHUNK); del Xh
    Xhv = aggregate_many(val, ctx.hs, ctx.sp_h)
    pr4, p4 = m4.predict_proba(Xhv, raw=True), m4.predict_proba(Xhv)
    assert np.isfinite(pr4).all() and np.isfinite(p4).all()
    d4 = e01.probs_frame(val, k, pr4, p4); d4["qset"] = "q2"
    sec_m04 = round(time.perf_counter() - t1); coral = dict(m4.info); del m4, Xhv
    dd = d3.copy()
    dd[PC] = lib.ensemble_probs(d3[PC].to_numpy(), d4[PC].to_numpy())
    dd[RC] = lib.ensemble_probs(d3[RC].to_numpy(), d4[RC].to_numpy())
    info = {"fold": k, "label": a.label, "x": str(a.x), "hsl_input": ctx.hinfo, "n_train_bags": len(tb), "n_es_bags": len(es), "n_val": len(val), "n_fit_players": len(fit_in),
            "n_features_m03": n_feat, "n_l0_cols": int(n_l0), "se_cols": SE_COLS, "best_iteration_m03": best3, "coral": coral, "game_leak_check": game_rep,
            "non_hsl_train_sha256": nonhsl, "hsl_train_block_sha256": hsl_blk, "bags_sha256": bags_sha, "seconds_m03": sec_m03, "seconds_m04": sec_m04,
            "seconds_total": round(time.perf_counter() - t0), "peak_rss_mb": rss_mb(), "num_threads": PRM["num_threads"], "n_H_in_any_set": 0, "subset_players": a.max_players or None}
    rep_on = formal and (a.repro == "on" or (a.repro == "auto" and a.label.startswith("W60") and not ctx.hinfo["partial"]))
    if rep_on:
        e21m = json.loads((ref / f"meta_E21_fold{k}.json").read_text()); a_meta = json.loads((ref / f"meta_A_fold{k}.json").read_text())
        diffs = {"M03SE": compare(d3, pd.read_parquet(ref / f"oof_M03SE_fold{k}.parquet")), "M04": compare(d4, a_q2), "Dse": compare(dd, pd.read_parquet(ref / f"oof_Dse_fold{k}.parquet"))}
        ok = max(diffs.values()) <= TOL and best3 == int(e21m["best_iteration_prime"]) and int(coral["n_iterations"]) == int(a_meta["coral"]["n_iterations"])
        rec = {"status": "OK" if ok else "MISMATCH", "max_abs_diff": diffs, "tol": TOL, "best_iteration_m03": best3, "e21_best_iteration": int(e21m["best_iteration_prime"]),
               "coral_n_iterations": int(coral["n_iterations"]), "e10_coral_n_iterations": int(a_meta["coral"]["n_iterations"]), "fold": k, "peak_rss_mb": rss_mb()}
        (out / f"repro_fold{k}.json").write_text(json.dumps(rec, indent=1)); info["repro"] = rec["status"]; log("repro", json.dumps(rec))
        if not ok:
            (out / f"meta_fold{k}.json").write_text(json.dumps(info, default=str))
            raise SystemExit(f"STOP: {a.label} does not reproduce the saved v4 Dse on fold {k} (see {out}/repro_fold{k}.json)")
    for nm, df in (("M03SE", d3), ("M04", d4), ("Dse", dd)):
        tmp = out / f"oof_{nm}_fold{k}.parquet.tmp"; df.to_parquet(tmp, index=False); tmp.replace(out / f"oof_{nm}_fold{k}.parquet")
    (out / f"meta_fold{k}.json").write_text(json.dumps(info, default=str))
    log(f"fold {k} done {info['seconds_total']}s (M03 {sec_m03}s iters {best3}, M04 {sec_m04}s iters {coral.get('n_iterations')}) peak {info['peak_rss_mb']}MB")


# ---------------------------------------------------------------------------------------------- driver
def finish(a, ok, msg=""):
    if not a.root:
        return
    R = Path(a.root)
    if ok:
        if a.final:
            (R / "DONE").write_text(time.strftime("%F %T") + "\n")
    else:
        (R / "FAILED").write_text(msg)


def make_tasks(a):
    """(label, x, fold, repro) per unit of work.  --repro-x adds the platform check first: the W60 X refit of fold --repro-fold, which must reproduce the saved v4 Dse."""
    t = []
    if a.repro_x:
        t.append((a.repro_label, a.repro_x, a.repro_fold, "on"))
    return t + [(a.label, a.x, k, a.repro) for k in a.folds]


def out_for(a, label):
    return Path(a.out) / (label + ("_partial" if a.allow_partial and not label.endswith("_partial") else ""))


def run_sequential(a, tasks):
    cur, ctx = None, None
    for label, x, k, repro in tasks:
        if cur != (label, x):
            ctx = None; ctx = load_ctx(argparse.Namespace(**{**vars(a), "x": x, "label": label})); cur = (label, x)
            log(f"label {label}: games {len(ctx.meta)} players {len(ctx.nonh)} hsl {ctx.hinfo}")
        o = out_for(a, label); o.mkdir(parents=True, exist_ok=True)
        if a.allow_partial:
            (o / "PARTIAL.json").write_text(json.dumps({"partial": True, "note": "smoke only; never judge or fuse"}))
        (o / "input_manifest.json").write_text(json.dumps({"label": label, "x": str(x), "hsl_input": ctx.hinfo,
                                                           "home_manifest_sha256": hashlib.sha256((Path(a.home) / "manifest.json").read_bytes()).hexdigest()}, default=str))
        run_fold(argparse.Namespace(**{**vars(a), "label": label, "repro": repro}), ctx, k, o)
    return 0


def child_cmd(a, label, x, k, repro):
    c = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--label", label, "--x", str(x), "--folds", str(k), "--repro", repro, "--home", a.home, "--aux", a.aux,
         "--out", a.out, "--hb-s", str(a.hb_s)]
    if a.allow_partial:
        c += ["--allow-partial", "--rows", a.rows, "--x60", a.x60]
    if a.max_players:
        c += ["--max-players", str(a.max_players)]
    if a.root:
        c += ["--root", a.root]
    return c


def run_parallel(a, tasks):
    pending, procs, rc = list(tasks), {}, 0
    while pending or procs:
        while pending and len(procs) < a.jobs and not rc:
            t = pending.pop(0)
            procs[t] = subprocess.Popen(child_cmd(a, *t))
            log(f"started {t[0]} fold {t[2]} pid {procs[t].pid}")
        time.sleep(5)
        for t, p in list(procs.items()):
            r = p.poll()
            if r is not None:
                del procs[t]
                if r != 0:
                    rc = r; pending.clear()
                    for q in procs.values():
                        q.terminate()
    if rc:      # no FAILED flag here: a child that raised wrote it itself; a child killed by a signal (OOM, kill) must look like a dead job so the controller watchdog restarts and resumes
        log(f"a fold child exited with {rc}"); return rc
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(S37_HOME)); ap.add_argument("--x", required=True); ap.add_argument("--aux", default=str(CODE_ROOT / ".scratch/e29/bundle/aux.npz"))
    ap.add_argument("--label", required=True, help="W60 | W120 (names the output dir; W60* labels get the v4 reproduction check)")
    ap.add_argument("--out", default=str(CODE_ROOT / ".scratch/e37/dse_out")); ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--repro", choices=["auto", "on", "off"], default="auto"); ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--rows"); ap.add_argument("--x60"); ap.add_argument("--max-players", type=int, default=0)
    ap.add_argument("--repro-x", help="platform check: ALSO refit --repro-fold with this W60 X (label --repro-label) and require it to reproduce the saved v4 Dse (run before the main label)")
    ap.add_argument("--repro-fold", type=int, default=0); ap.add_argument("--repro-label", default="W60")
    ap.add_argument("--jobs", type=int, default=1, help="tasks (folds) in parallel as child processes, 4 threads each; ~7-9 GB RSS per task")
    ap.add_argument("--root", help="host job root: writes FAILED on error, DONE after success when --final"); ap.add_argument("--final", action="store_true")
    ap.add_argument("--hb-s", type=float, default=60.0); ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    hb = threading.Thread(target=heartbeat, args=(a.hb_s,), daemon=True); hb.start()
    tasks = make_tasks(a)
    try:
        return _main(a, tasks)
    finally:                          # stop the heartbeat before interpreter shutdown (a daemon thread printing while finalizing aborts the process: seen in the first fold-0 run)
        STOP_HB.set(); hb.join(5)


def _main(a, tasks):
    try:
        rc = run_parallel(a, tasks) if (a.jobs > 1 and len(tasks) > 1 and not a.child) else run_sequential(a, tasks)
        if rc == 0:
            log("ALLDONE", [(t[0], t[2]) for t in tasks]); finish(a, True)
        return rc
    except SystemExit as e:
        finish(a, False, str(e)); raise
    except BaseException:                                       # noqa: BLE001  (record any failure for the controller)
        tb = traceback.format_exc(); log("FAILED", tb); finish(a, False, tb); raise


S37_HOME = CODE_ROOT / ".scratch/e37/dse_home"

if __name__ == "__main__":
    sys.exit(main())
