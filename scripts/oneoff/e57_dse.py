"""E57 CPU legs (run on remote-vm, `~/gogame_e57/code`; same relative layout as the repo).  One process = one (variant, fold) task; derived from e55_dse.py / scripts/experiments/e37/dse_w120.py `run_fold` (same bags, ES bags, seeds, PRM, early stopping).
Variants:  M03off  (E37 W120 M03SE refit, no trunk columns; platform repro check vs the saved W120 M03SE)
           M04off  (E37 W120 M04 CORAL refit, HSL aggregate only; platform repro check vs the saved W120 M04)
           M03T    (E55 M03T: + 7 SE cols + 240 trunk PCA bag-aggregate columns)
           M03TS   (M03T + 55 S2-head bag-aggregate columns from .scratch/e57/head/H_fold{k}.npy)
           M04T    (CORAL on 522 HSL aggregate cols + the 240 trunk PCA aggregate cols = 762)
Out: <out>/<variant>/oof_<variant>_fold{k}.parquet (+ meta).  Never touches H (the home tree has none) / h_eval_ledger.
  .../venv/bin/python scripts/oneoff/e57_dse.py --tasks M04T:0 M03TS:1 ..."""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np, pandas as pd
import scripts.experiments.e37.dse_w120 as D
from scripts.experiments.e10 import lib
from gogame.aggregate import aggregate_many, fit_sparse_selector, make_spec, L0Store
from gogame.coral import fit_coral
from gogame.models import fit_lgbm
from gogame.questions import bag_class_weights, sample_train_bags
import run_e01 as e01
S = ROOT / ".scratch/e55"; S57 = ROOT / ".scratch/e57"; SAVED = ROOT / ".scratch/e37/runs/v6_dse/out/W120"; PN = tuple(f"tk{j:02d}" for j in range(48)); HN = tuple(f"hd{j:02d}" for j in range(11))
def project(G, pca, chunk=100_000):
    out = np.zeros((len(G), 48), np.float32); mu, sd, C = pca["mu"], pca["sd"], pca["comps"]
    for a in range(0, len(G), chunk):
        z = (np.asarray(G[a:a + chunk], np.float32) - mu) / sd; bad = ~np.isfinite(z).all(1); z[bad] = 0; out[a:a + chunk] = z @ C.T
    return out
def run(ctx, k, var, out, aux):
    fin = out / f"oof_{var}_fold{k}.parquet"
    if fin.exists(): D.log(var, k, "cached"); return
    t0 = time.perf_counter(); fs = lib.fold_sets(ctx.nonh, k); lib.assert_fold_disjoint(fs, ctx.H)
    pf = aux["player_fold"]; pid = aux["player_ids"].astype(str); assert fs["val"] == set(pid[pf == k])
    val = [b for b in ctx.q2_all if b.player_id in fs["val"]]; lib.assert_no_game_leak(ctx.meta, [u for b in val for u in b.uids], fs["train_all"])
    c = lib.CANDS["A"]; fit_in = sorted(p for p in fs["fit"] if p in ctx.index_all.games)
    tb = sample_train_bags(ctx.index_all, fit_in, c.bags_per_pc, c.n_range, seed=c.train_seed(k)); es = lib.make_es_bags(ctx.index_all, fs["es"], k, False)
    refm = json.loads((Path(ctx.home) / "ref" / f"meta_D_fold{k}.json").read_text()); assert len(tb) == refm["n_train_bags"] and len(es) == refm["n_es_bags"]
    assert {b.player_id for b in tb} <= fs["fit"] and not ({b.player_id for b in tb} & (fs["val"] | fs["es"]))
    ytr, yes = np.array([b.rank_id for b in tb]), np.array([b.rank_id for b in es]); wtr, wes = bag_class_weights(tb, "inv_players"), bag_class_weights(es, "inv_players")
    uid = aux["uid"].astype(str); info = {"fold": k, "variant": var}
    use_t = var in ("M03T", "M03TS", "M04T"); use_h = var == "M03TS"
    if use_t:
        Pk = project(np.load(S / "G_full.npy", mmap_mode="r"), dict(np.load(S / f"pca/fold{k}.npz"))); ts = L0Store(list(uid), Pk, None, PN); sp_t = make_spec(PN, np.zeros(0, np.int32))
    if use_h:
        Hk = np.load(S57 / f"head/H_fold{k}.npy"); assert Hk.shape == (len(uid), 11) and np.isfinite(Hk).all(); hs2 = L0Store(list(uid), Hk, None, HN); sp_hd = make_spec(HN, np.zeros(0, np.int32))
    if var.startswith("M03"):
        used = sorted({u for b in tb for u in b.uids}); cols = fit_sparse_selector(ctx.l0.sparse[ctx.l0.rows(used)], e01.CFG["min_df"]); del used; spec_l0 = make_spec(ctx.l0.names, cols); mats = []
        for nm, bags in (("tr", tb), ("es", es), ("val", val)):
            parts = [aggregate_many(bags, ctx.l0, spec_l0), aggregate_many(bags, ctx.hs, ctx.sp_h)[:, :-2], D.se_feats(bags, ctx.sm, ctx.ss)]
            if use_t: parts.append(aggregate_many(bags, ts, sp_t)[:, :-2])
            if use_h: parts.append(aggregate_many(bags, hs2, sp_hd)[:, :-2])
            mats.append(np.hstack(parts)); del parts
        Xtr, Xes, Xv = mats; del mats; D.HB["phase"] = f"{var}{k}:fit"; t1 = time.perf_counter()
        m = fit_lgbm(Xtr, ytr, Xes, yes, D.PRM, D.derive_seed(D.SEED, "lgbm"), wtr, wes); pr, p = m.predict_proba(Xv, raw=True), m.predict_proba(Xv); assert np.isfinite(pr).all() and np.isfinite(p).all()
        info.update(best_iteration=int(m.best_iteration), n_features=int(Xtr.shape[1])); df = e01.probs_frame(val, k, pr, p); df["qset"] = "q2"; ref = pd.read_parquet(SAVED / f"oof_M03SE_fold{k}.parquet")
        info["saved_best_iteration"] = json.loads((SAVED / f"meta_fold{k}.json").read_text()).get("best_iteration_m03")
    else:
        D.HB["phase"] = f"{var}{k}:fit"; t1 = time.perf_counter(); Xh = aggregate_many(tb, ctx.hs, ctx.sp_h)
        if use_t: Xh = np.hstack([Xh, aggregate_many(tb, ts, sp_t)[:, :-2]])
        m = fit_coral(Xh, ytr, wtr, chunk=lib.CORAL_CHUNK); del Xh; Xv = aggregate_many(val, ctx.hs, ctx.sp_h)
        if use_t: Xv = np.hstack([Xv, aggregate_many(val, ts, sp_t)[:, :-2]])
        pr, p = m.predict_proba(Xv, raw=True), m.predict_proba(Xv); assert np.isfinite(pr).all() and np.isfinite(p).all(); info["coral"] = dict(m.info); info["n_features"] = int(Xv.shape[1])
        df = e01.probs_frame(val, k, pr, p); df["qset"] = "q2"; ref = pd.read_parquet(SAVED / f"oof_M04_fold{k}.parquet")
    assert list(ref["qid"]) == list(df["qid"]); info["max_abs_diff_vs_saved"] = D.compare(df, ref); info["seconds"] = round(time.perf_counter() - t1); info["peak_rss_mb"] = D.rss_mb()
    if var in ("M03off", "M04off"): info["repro"] = "OK" if info["max_abs_diff_vs_saved"] <= D.TOL else "MISMATCH"
    tmp = out / f"oof_{var}_fold{k}.parquet.tmp"; df.to_parquet(tmp, index=False); tmp.replace(fin); (out / f"meta_{var}_fold{k}.json").write_text(json.dumps(info, indent=1, default=str)); D.log(var, k, "done", json.dumps({x: info[x] for x in info if x != "coral"}, default=str))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", nargs="+", required=True); ap.add_argument("--out", default=str(S57 / "dse_out")); ap.add_argument("--hb-s", type=float, default=300.0); a = ap.parse_args()
    import threading
    try: os.nice(10)
    except OSError: pass
    threading.Thread(target=D.heartbeat, args=(a.hb_s,), daemon=True).start()
    ns = argparse.Namespace(home=str(D.S37_HOME), x=str(ROOT / ".scratch/e36/bundle/X120_full.npy"), aux=str(ROOT / ".scratch/e29/bundle/aux.npz"), allow_pickle=False, allow_partial=False, rows=None, x60=None, max_players=0, label="W120")
    aux = np.load(ns.aux, allow_pickle=False); ctx = D.load_ctx(ns); D.log("ctx loaded", ctx.hinfo)
    for t in a.tasks:
        var, k = t.split(":"); o = Path(a.out) / var; o.mkdir(parents=True, exist_ok=True); run(ctx, int(k), var, o, aux)
    D.STOP_HB.set(); D.log("E57_DSE_DONE", a.tasks)
