"""E63 (copy of e59_dse.py; variants M03T = E55 recipe [tk cols only; repro check vs E55 saved], M03TL = tk + 48 LATE-window PCA columns lk00-47 (e63_pca.py, .scratch/e63/pca/fold{k}.npz on L_full.npy); tasks "M03T:0 M03TL:0 ...").  Original: E55 (a): D120 M03 LightGBM leg with the per-fold PCA-48 trunk columns appended (bag aggregate = `aggregate_many` with the same 5 statistics, bag_n/is_white dropped).  Everything else is
`scripts/experiments/e37/dse_w120.py run_fold` verbatim (imported helpers; same bags, ES bags, seeds, PRM, early stopping).  The M04 CORAL leg is NOT refit: D120T = lib.ensemble_probs(M03T, saved W120 M04).
Tasks: `--trunk off` refits M03 without trunk columns on fold --repro-fold and must reproduce the saved W120 M03SE (max abs diff <= 1e-6) before any trunk run (PLAN s1/s2 (v)).
  PYTHONPATH=src:scripts systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 nice -n 10 .venv/bin/python scripts/oneoff/e55_dse.py --repro-fold 0 --folds 0 1 2 3 4
Out: .scratch/e55/dse_out/{W120off|W120T}/oof_{M03SE,Dse}_fold{k}.parquet, meta_fold{k}.json.   Never touches H (home tree has none) / h_eval_ledger."""
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np, pandas as pd
import scripts.experiments.e37.dse_w120 as D
from scripts.experiments.e10 import lib
from gogame.aggregate import aggregate_many, fit_sparse_selector, make_spec, L0Store
from gogame.models import fit_lgbm
from gogame.questions import bag_class_weights, sample_train_bags
import run_e01 as e01
S = ROOT / ".scratch/e55"; S63 = ROOT / ".scratch/e63"; SAVED = ROOT / ".scratch/e37/runs/v6_dse/out/W120"; PNAMES = tuple(f"tk{j:02d}" for j in range(48)); ZNAMES = tuple(f"lk{j:02d}" for j in range(48))
def project(G, pca, chunk=100_000):
    out = np.zeros((len(G), 48), np.float32); mu, sd, C = pca["mu"], pca["sd"], pca["comps"]
    for a in range(0, len(G), chunk):
        z = (np.asarray(G[a:a + chunk], np.float32) - mu) / sd; bad = ~np.isfinite(z).all(1); z[bad] = 0; out[a:a + chunk] = z @ C.T
    return out
def run(a, ctx, k, var, out, aux):
    trunk = True
    if (out / f"oof_Dse_fold{k}.parquet").exists(): D.log(f"fold {k} {out.name}: cached"); return
    t0 = time.perf_counter(); fs = lib.fold_sets(ctx.nonh, k); lib.assert_fold_disjoint(fs, ctx.H)
    pf = aux["player_fold"]; pid = aux["player_ids"].astype(str); assert fs["val"] == set(pid[pf == k]), "val players != player_fold == k"
    val = [b for b in ctx.q2_all if b.player_id in fs["val"]]; lib.assert_no_game_leak(ctx.meta, [u for b in val for u in b.uids], fs["train_all"])
    c = lib.CANDS["A"]; fit_in = sorted(p for p in fs["fit"] if p in ctx.index_all.games)
    tb = sample_train_bags(ctx.index_all, fit_in, c.bags_per_pc, c.n_range, seed=c.train_seed(k)); es = lib.make_es_bags(ctx.index_all, fs["es"], k, False)
    refm = json.loads((Path(ctx.home) / "ref" / f"meta_D_fold{k}.json").read_text()); assert len(tb) == refm["n_train_bags"] and len(es) == refm["n_es_bags"]
    assert {b.player_id for b in tb} <= fs["fit"] and not ({b.player_id for b in tb} & (fs["val"] | fs["es"]))
    ytr, yes = np.array([b.rank_id for b in tb]), np.array([b.rank_id for b in es]); wtr, wes = bag_class_weights(tb, "inv_players"), bag_class_weights(es, "inv_players")
    if trunk:
        G = np.load(S / "G_full.npy", mmap_mode="r"); Pk = project(G, dict(np.load(S / f"pca/fold{k}.npz"))); uid = aux["uid"].astype(str)
        if var == "M03TL": Pz = project(np.load(S63 / "L_full.npy", mmap_mode="r"), dict(np.load(S63 / f"pca/fold{k}.npz"))); Pk = np.hstack([Pk, Pz]); nm_ = PNAMES + ZNAMES
        else: nm_ = PNAMES
        ts = L0Store(list(uid), Pk, None, nm_); sp_t = make_spec(nm_, np.zeros(0, np.int32)); D.log(f"fold {k} {var}: trunk columns {Pk.shape}")
    used = sorted({u for b in tb for u in b.uids}); cols = fit_sparse_selector(ctx.l0.sparse[ctx.l0.rows(used)], e01.CFG["min_df"]); del used
    spec_l0 = make_spec(ctx.l0.names, cols); mats = []
    for nm, bags in (("tr", tb), ("es", es), ("val", val)):
        parts = [aggregate_many(bags, ctx.l0, spec_l0), aggregate_many(bags, ctx.hs, ctx.sp_h)[:, :-2], D.se_feats(bags, ctx.sm, ctx.ss)]
        if trunk: parts.append(aggregate_many(bags, ts, sp_t)[:, :-2])
        mats.append(np.hstack(parts)); del parts
    Xtr, Xes, Xv = mats; del mats; D.HB["phase"] = f"fold{k}:M03 fit"; t1 = time.perf_counter()
    m3 = fit_lgbm(Xtr, ytr, Xes, yes, D.PRM, D.derive_seed(D.SEED, "lgbm"), wtr, wes); pr, p = m3.predict_proba(Xv, raw=True), m3.predict_proba(Xv); assert np.isfinite(pr).all() and np.isfinite(p).all()
    d3 = e01.probs_frame(val, k, pr, p); d3["qset"] = "q2"; best3 = int(m3.best_iteration); nfeat = int(Xtr.shape[1]); sec = round(time.perf_counter() - t1); del Xtr, Xes, Xv, m3
    ref3 = pd.read_parquet(SAVED / f"oof_M03SE_fold{k}.parquet"); ref4 = pd.read_parquet(SAVED / f"oof_M04_fold{k}.parquet"); refD = pd.read_parquet(SAVED / f"oof_Dse_fold{k}.parquet")
    assert list(ref3["qid"]) == list(d3["qid"]) == list(ref4["qid"]), "question order differs from the saved W120 arms"
    dd = d3.copy(); dd[D.PC] = lib.ensemble_probs(d3[D.PC].to_numpy(), ref4[D.PC].to_numpy()); dd[D.RC] = lib.ensemble_probs(d3[D.RC].to_numpy(), ref4[D.RC].to_numpy())
    info = {"fold": k, "variant": var, "n_features_m03": nfeat, "best_iteration_m03": best3, "seconds_m03_fit": sec, "seconds_total": round(time.perf_counter() - t0), "peak_rss_mb": D.rss_mb(), "n_train_bags": len(tb), "n_es_bags": len(es), "n_val": len(val),
            "saved_best_iteration": json.loads((SAVED / f"meta_fold{k}.json").read_text()).get("best_iteration_m03")}
    if var == "M03T":
        r55 = pd.read_parquet(S / "dse_out/W120T" / f"oof_M03SE_fold{k}.parquet"); assert list(r55["qid"]) == list(d3["qid"])
        diff = D.compare(d3, r55); b55 = json.loads((S / "dse_out/W120T" / f"meta_fold{k}.json").read_text())["best_iteration_m03"]
        info["repro_vs_e55_M03T_max_abs_diff"] = diff; info["e55_best_iteration"] = b55; info["repro"] = "OK" if (diff <= D.TOL and best3 == b55) else "MISMATCH"; D.log("repro vs E55 M03T", info["repro"], diff, best3, b55)
    for nm, df in (("M03SE", d3), ("Dse", dd)): tmp = out / f"oof_{nm}_fold{k}.parquet.tmp"; df.to_parquet(tmp, index=False); tmp.replace(out / f"oof_{nm}_fold{k}.parquet")
    (out / f"meta_fold{k}.json").write_text(json.dumps(info, indent=1)); D.log(f"fold {k} {out.name} done {info['seconds_total']}s best_iter {best3} feats {nfeat} peak {info['peak_rss_mb']}MB")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", nargs="+", required=True); ap.add_argument("--out", default=str(S63 / "dse_out")); ap.add_argument("--hb-s", type=float, default=120.0); a = ap.parse_args()
    import threading, os
    try: os.nice(10)
    except OSError: pass
    threading.Thread(target=D.heartbeat, args=(a.hb_s,), daemon=True).start()
    ns = argparse.Namespace(home=str(D.S37_HOME), x=str(ROOT / ".scratch/e36/bundle/X120_full.npy"), aux=str(ROOT / ".scratch/e29/bundle/aux.npz"), allow_pickle=False, allow_partial=False, rows=None, x60=None, max_players=0, label="W120")
    aux = np.load(ns.aux, allow_pickle=False); ctx = D.load_ctx(ns); D.log("ctx loaded", ctx.hinfo)
    for t in a.tasks:
        var, k = t.split(":"); k = int(k); assert var in ("M03T", "M03TL"); o = Path(a.out) / var; o.mkdir(parents=True, exist_ok=True); run(a, ctx, k, var, o, aux)
    D.STOP_HB.set(); D.log("E63_DSE_DONE")
