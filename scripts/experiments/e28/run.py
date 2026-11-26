"""E28: HSL window ablation (first 15/30/45/60 own moves) and HD>=2 exclusion, both on the M03 leg of Dse.  Plan: experiments/E28_hsl_window_hd/PLAN.md.
Step 1 (once): .venv/bin/python -m scripts.experiments.e28.windows        -> .scratch/e28/hsl_W{15,30,45,60}.parquet
Step 2: systemd-run ... nice -n 10 .venv/bin/python -m scripts.experiments.e28.run fit --folds 0 1 --variants w60 w45 w30 w15 hd reseed
Step 3: .venv/bin/python -m scripts.experiments.e28.run report --folds 0 1
Variants: wNN = M03SE leg with HSL aggregate from the first NN own moves (w60 = control, must reproduce E21 bit-for-bit);
          hd = control features/W60, train+ES bags drawn without HD>=2 games (validation questions unchanged);
          reseed = control, train bags drawn with the A_s1 sampling seed (bag-sampling noise reference, report only).
M04_A leg is E10 oof_A (fixed).  H never touched.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np, pandas as pd

from scripts.experiments.e10 import lib, pipeline
from gogame.aggregate import L0Store, aggregate_many, fit_sparse_selector, make_spec
from gogame.constants import SEED
from gogame.metrics import balanced_weights, rank_scores
from gogame.models import decide_ev, fit_lgbm
from gogame.questions import bag_class_weights, sample_train_bags
from gogame.seeds import derive_seed
import run_e01 as e01

ROOT = lib.ROOT
OUT = ROOT / "experiments" / "E28_hsl_window_hd" / "results"
SCR = ROOT / ".scratch" / "e28"
E21_RES = ROOT / "experiments" / "E21_strength_estimator" / "results"
E10_OOF = lib.OUT / "oof"
PC = lib.PCOLS
PRM = {"num_threads": 4, "learning_rate": 0.1, "max_bin": 63, "log_every": 0}
TOL = 1e-6
SEF = ROOT / ".scratch" / "e21" / "se_full.parquet"
HDF = ROOT / ".scratch" / "e19" / "games_hd.parquet"
VARIANTS = {"w60": dict(W=60, bags="std"), "w45": dict(W=45, bags="std"), "w30": dict(W=30, bags="std"), "w15": dict(W=15, bags="std"),
            "hd": dict(W=60, bags="hd"), "reseed": dict(W=60, bags="reseed")}


def se_feats(bags, sm, ss):
    out = np.full((len(bags), 7), np.nan)
    for i, b in enumerate(bags):
        v = np.array([sm[u] for u in b.uids], float); v = v[~np.isnan(v)]
        w = np.array([ss[u] for u in b.uids], float); w = w[~np.isnan(w)]
        if len(v):
            out[i, :6] = [v.mean(), v.std(), np.percentile(v, 10), np.percentile(v, 90), v.min(), v.max()]
        if len(w):
            out[i, 6] = w.mean()
    return out


def load_window(W, names_ref):
    df = pd.read_parquet(SCR / f"hsl_W{W}.parquet")
    names = tuple(c for c in df.columns if c != "uid")
    assert names == tuple(names_ref)
    uids = df["uid"].tolist()
    mat = df[list(names)].to_numpy(np.float32)
    del df
    return L0Store(uids, mat, None, names)


def hd_table():
    g = pd.read_parquet(HDF)
    g["uid"] = g["rank"] + "_" + g["game_id"]
    s = g.set_index("uid")["hd"].astype(int)
    assert s.index.is_unique and len(s) == 1_000_000
    return s


def check_repro(k, dfs, ref_meta_k, best_it):
    """w60 control vs E21 (M03SE and Dse), tolerance TOL on praw_*/p_*, equal best_iteration."""
    cc = [f"praw_{i}" for i in range(10)] + PC
    res = {}
    for nm, d in (("M03SE", dfs["M03"]), ("Dse", dfs["Dse"])):
        ref = pd.read_parquet(E21_RES / f"oof_{nm}_fold{k}.parquet").set_index("qid").loc[d["qid"]]
        mine = d.set_index("qid")
        assert list(ref["rank_id"]) == list(mine["rank_id"])
        res[nm] = float(np.abs(mine[cc].to_numpy() - ref[cc].to_numpy()).max())
    e21_it = json.loads((E21_RES / f"meta_fold{k}.json").read_text())["best_iteration_prime"]
    ok = max(res.values()) <= TOL and int(best_it) == int(e21_it)
    rec = {"status": "OK" if ok else "MISMATCH", "max_abs_diff": res, "tol": TOL, "best_iteration": int(best_it), "e21_best_iteration": int(e21_it)}
    (OUT / f"repro_w60_fold{k}.json").write_text(json.dumps(rec, indent=1))
    print("repro w60 fold", k, rec, flush=True)
    return ok


def fit(folds, variants):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = pipeline.default_cache(False)
    ctx = pipeline.Ctx(cache, False)
    assert not ctx.partial
    st = pd.read_parquet(SEF)
    nonh_uids = set(ctx.meta_all.loc[~ctx.meta_all["player_id"].isin(ctx.H), "uid"])
    assert set(st["uid"]) == nonh_uids and st["uid"].is_unique
    sm = dict(zip(st["uid"], st["se_mean"].astype(float))); ss = dict(zip(st["uid"], st["se_std"].astype(float)))
    hd = hd_table()
    hd2 = set(hd.index[hd >= 2])
    meta_hd = ctx.meta[~ctx.meta["uid"].isin(hd2)]
    index_hd = lib.hsl_index(meta_hd, meta_hd["uid"].tolist())
    print("HD>=2 games excluded:", len(hd2), flush=True)
    q2_all = e01.frame_to_bags(pd.read_parquet(E10_OOF / "questions_q2.parquet"))
    assert not ({b.player_id for b in q2_all} & ctx.H)
    c = lib.CANDS["A"]
    stores = {}
    for k in folds:
        fs = lib.fold_sets(ctx.nonh, k)
        lib.assert_fold_disjoint(fs, ctx.H)
        for key in ("fit", "es", "val"):
            assert not (fs[key] & ctx.H), key
        val = [b for b in q2_all if b.player_id in fs["val"]]
        a_q2 = pd.read_parquet(E10_OOF / f"oof_A_fold{k}.parquet").set_index("qid")
        a_q2 = a_q2[a_q2["qset"] == "q2"]
        assert set(a_q2.index) == {b.qid for b in val}
        game_rep = lib.assert_no_game_leak(ctx.meta_all, [u for b in val for u in b.uids], fs["train_all"])
        ref_meta = json.loads((E10_OOF / f"meta_D_fold{k}.json").read_text())
        vu = {u for b in val for u in b.uids}
        Xv_se = se_feats(val, sm, ss)
        ytr_cache = {}
        # bag sets per mode
        def bag_sets(mode):
            if mode == "hd":
                index = index_hd
                fit_in = sorted(p for p in fs["fit"] if p in index.games)
                tb = sample_train_bags(index, fit_in, c.bags_per_pc, c.n_range, seed=c.train_seed(k))
                es = lib.make_es_bags(index, fs["es"], k, False)
                assert not any(u in hd2 for b in tb + es for u in b.uids)
            else:
                index = ctx.index_all
                fit_in = sorted(p for p in fs["fit"] if p in index.games)
                seed = lib.CANDS["A_s1"].train_seed(k) if mode == "reseed" else c.train_seed(k)
                tb = sample_train_bags(index, fit_in, c.bags_per_pc, c.n_range, seed=seed)
                es = lib.make_es_bags(index, fs["es"], k, False)
                if mode == "std":
                    assert len(tb) == ref_meta["n_train_bags"], (len(tb), ref_meta["n_train_bags"])
                    assert len(es) == ref_meta["n_es_bags"]
            assert {b.player_id for b in tb} <= fs["fit"] and not ({b.player_id for b in tb} & (fs["val"] | fs["es"] | ctx.H))
            assert not (vu & {u for b in tb + es for u in b.uids})
            return tb, es

        base = {}      # mode -> dict(tb, es, spec, XL tr/es/v, SE tr/es)
        order = [v for v in VARIANTS if v in variants]
        for v in order:
            t0 = time.perf_counter()
            vs = VARIANTS[v]
            fm = OUT / f"oof_Dse_{v}_fold{k}.parquet"
            if fm.exists():
                print(f"fold {k} {v}: cached", flush=True)
                continue
            mode = vs["bags"]
            if mode not in base:
                base.clear()
                tb, es = bag_sets(mode)
                used = sorted({u for b in tb for u in b.uids})
                cols = fit_sparse_selector(ctx.l0.sparse[ctx.l0.rows(used)], e01.CFG["min_df"])
                spec_l0 = make_spec(ctx.l0.names, cols)
                del used
                base[mode] = dict(tb=tb, es=es, spec=spec_l0,
                                  XL=[aggregate_many(b_, ctx.l0, spec_l0) for b_ in (tb, es, val)],
                                  SE=[se_feats(tb, sm, ss), se_feats(es, sm, ss), Xv_se])
                print(f"fold {k}: mode {mode}: train bags {len(tb)} es {len(es)} val {len(val)} L0 agg {time.perf_counter()-t0:.0f}s", flush=True)
            B = base[mode]
            W = vs["W"]
            if W not in stores:
                stores.clear(); stores[W] = load_window(W, ctx.hnames)
            hs = stores[W]
            spec_h = make_spec(hs.names, np.zeros(0, np.int32))
            Xs = []
            for bags, xl, se in zip((B["tb"], B["es"], val), B["XL"], B["SE"]):
                xh = aggregate_many(bags, hs, spec_h)
                Xs.append(np.hstack([xl, xh[:, :-2], se]))
            Xtr, Xes, Xv = Xs
            ytr, yes = np.array([b.rank_id for b in B["tb"]]), np.array([b.rank_id for b in B["es"]])
            wtr, wes = bag_class_weights(B["tb"], "inv_players"), bag_class_weights(B["es"], "inv_players")
            m = fit_lgbm(Xtr, ytr, Xes, yes, PRM, derive_seed(SEED, "lgbm"), wtr, wes)
            pr, p = m.predict_proba(Xv, raw=True), m.predict_proba(Xv)
            assert np.isfinite(pr).all() and np.isfinite(p).all()
            d3 = e01.probs_frame(val, k, pr, p); d3["qset"] = "q2"
            ap = a_q2.loc[d3["qid"]]
            dd = d3.copy(); rc = [f"praw_{j}" for j in range(10)]
            dd[PC] = lib.ensemble_probs(d3[PC].to_numpy(), ap[PC].to_numpy())
            dd[rc] = lib.ensemble_probs(d3[rc].to_numpy(), ap[rc].to_numpy())
            info = {"fold": k, "variant": v, "W": W, "bags": mode, "n_train_bags": len(B["tb"]), "n_es_bags": len(B["es"]), "n_val": len(val),
                    "n_features": int(Xtr.shape[1]), "best_iteration": int(m.best_iteration), "game_leak_check": game_rep,
                    "seconds": round(time.perf_counter() - t0), "peak_rss_mb": round(pipeline.peak_rss_mb()), "n_H_in_any_set": 0}
            if v == "w60":
                ok = check_repro(k, {"M03": d3, "Dse": dd}, ref_meta, m.best_iteration)
                info["repro"] = "OK" if ok else "MISMATCH"
                if not ok:
                    (OUT / f"meta_{v}_fold{k}.json").write_text(json.dumps(info, default=str))
                    raise SystemExit(f"STOP: w60 control does not reproduce E21 on fold {k} (see repro_w60_fold{k}.json)")
            d3.to_parquet(OUT / f"oof_M03_{v}_fold{k}.parquet", index=False)
            dd.to_parquet(fm.with_suffix(".tmp"), index=False); fm.with_suffix(".tmp").replace(fm)
            (OUT / f"meta_{v}_fold{k}.json").write_text(json.dumps(info, default=str))
            print(f"fold {k} {v}: iters {m.best_iteration} {info['seconds']}s peak {info['peak_rss_mb']}MB", flush=True)
            del Xtr, Xes, Xv, Xs, m
        base.clear()


# ---------------------------------------------------------------------------------------------- report
NS = (5, 10, 15, 20)


def _boot(num_den, C):
    return np.mean([(C @ n) / (C @ d) for n, d in num_den], axis=0)


def report(folds):
    rows, drows = [], []
    E21 = pd.concat([pd.read_parquet(E21_RES / f"oof_Dse_fold{k}.parquet") for k in folds], ignore_index=True).sort_values("qid").reset_index(drop=True)
    E21m = pd.concat([pd.read_parquet(E21_RES / f"oof_M03SE_fold{k}.parquet") for k in folds], ignore_index=True)
    fr = {"E21": E21, "m_E21": E21m.set_index("qid").loc[E21["qid"]].reset_index()}
    for v in VARIANTS:
        fs_ = [OUT / f"oof_Dse_{v}_fold{k}.parquet" for k in folds]
        if all(f.exists() for f in fs_):
            fr[v] = pd.concat([pd.read_parquet(f) for f in fs_], ignore_index=True).set_index("qid").loc[E21["qid"]].reset_index()
            fr["m_" + v] = pd.concat([pd.read_parquet(OUT / f"oof_M03_{v}_fold{k}.parquet") for k in folds], ignore_index=True).set_index("qid").loc[E21["qid"]].reset_index()
    for nm in fr:
        assert list(fr[nm]["qid"]) == list(E21["qid"]) and list(fr[nm]["rank_id"]) == list(E21["rank_id"])
    truth, ns, fold, pl = E21["rank_id"].to_numpy(), E21["n"].to_numpy(), E21["fold"].to_numpy(), E21["player_id"].to_numpy()
    assert not (set(pl) & set(pd.read_parquet(ROOT / "data/splits/split.parquet").query("is_holdout")["player_id"]))
    sc = {nm: rank_scores(decide_ev(fr[nm][PC].to_numpy()), truth) for nm in fr}
    hit = {nm: (decide_ev(fr[nm][PC].to_numpy()) == truth).astype(float) for nm in fr}

    def ndp(v, kind, mask):
        u = np.unique(pl[mask]); P = len(u); out = []
        for n in NS:
            mk = mask & (ns == n)
            w = balanced_weights(truth[mk]) if kind == "balanced" else np.ones(mk.sum())
            g = np.searchsorted(u, pl[mk])
            out.append((np.bincount(g, weights=v[mk] * w, minlength=P), np.bincount(g, weights=w, minlength=P)))
        return out, P

    def summarize(label, a, b, mask, tag):
        if a not in fr or (b is not None and b not in fr) or not mask.any():
            return
        P = len(np.unique(pl[mask]))
        C = np.random.default_rng(derive_seed(SEED, "e28", "bootstrap", tag)).multinomial(P, np.full(P, 1.0 / P), size=4000).astype(np.float64)
        for kind in ("balanced", "raw", "hit"):
            va = hit if kind == "hit" else sc
            nd_a, _ = ndp(va[a], kind, mask)
            if b is None:
                for i, n in enumerate(NS):
                    bs = (C @ nd_a[i][0]) / (C @ nd_a[i][1])
                    rows.append({"set": label, "model": a, "N": n, "metric": kind, "value": nd_a[i][0].sum() / nd_a[i][1].sum(), "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975), "n_q": int((mask & (ns == n)).sum())})
                bb = _boot(nd_a, C)
                rows.append({"set": label, "model": a, "N": "eq", "metric": kind, "value": float(np.mean([x.sum() / y.sum() for x, y in nd_a])), "lo": np.quantile(bb, .025), "hi": np.quantile(bb, .975), "n_q": int(mask.sum())})
            else:
                nd_b, _ = ndp(va[b], kind, mask)
                dn = [(x[0] - y[0], x[1]) for x, y in zip(nd_a, nd_b)]
                for i, n in enumerate(NS):
                    bs = (C @ dn[i][0]) / (C @ dn[i][1])
                    drows.append({"set": label, "contrast": f"{a}-{b}", "N": n, "metric": kind, "delta": dn[i][0].sum() / dn[i][1].sum(), "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975), "players": P})
                bb = _boot(dn, C)
                drows.append({"set": label, "contrast": f"{a}-{b}", "N": "eq", "metric": kind, "delta": float(np.mean([x.sum() / y.sum() for x, y in dn])), "lo": np.quantile(bb, .025), "hi": np.quantile(bb, .975), "players": P})

    main = np.isin(fold, [0, 1])
    # strata
    q2 = pd.read_parquet(E10_OOF / "questions_q2.parquet").set_index("qid").loc[E21["qid"]]
    hd = hd_table()
    U = q2[[f"uid_{i}" for i in range(1, 21)]]
    has_hd2 = U.apply(lambda col: col.map(hd)).ge(2).any(axis=1).to_numpy()
    gm = pd.read_parquet(ROOT / "data/cache/games_meta.parquet", columns=["player_id"])["player_id"].value_counts()
    ng = gm.reindex(pl).to_numpy()
    pool = q2["pool"].to_numpy()
    strata = {"main": main, "light_20_80": main & (ng >= 20) & (ng <= 80), "heavy_gt80": main & (ng > 80),
              "pool_lt60": main & (pool < lib.POOL_SPLIT), "pool_ge60": main & (pool >= lib.POOL_SPLIT), "val_no_hd2": main & ~has_hd2}
    print("val questions with any HD>=2 game (fold0+1):", int((main & has_hd2).sum()), "of", int(main.sum()), flush=True)
    models = [m for m in fr if not m.startswith("m_")]
    for m in models:
        summarize("main", m, None, main, "main")
    contrasts = [("w60", "E21"), ("w60", "w30"), ("w60", "w45"), ("w45", "w30"), ("w30", "w15"), ("w60", "w15"), ("hd", "E21"), ("reseed", "E21"), ("hd", "reseed"),
                 ("m_w60", "m_E21"), ("m_w60", "m_w30"), ("m_w60", "m_w45"), ("m_w45", "m_w30"), ("m_w30", "m_w15"), ("m_hd", "m_E21"), ("m_reseed", "m_E21")]
    for a, b in contrasts:
        summarize("main", a, b, main, "main")
    for k in folds:
        for a, b in (("hd", "E21"), ("w60", "w30"), ("reseed", "E21")):
            summarize(f"fold{k}", a, b, fold == k, f"fold{k}")
    for lab, mk in strata.items():
        if lab == "main":
            continue
        for a, b in (("hd", "E21"), ("w60", "w30"), ("reseed", "E21"), ("w60", "w45")):
            summarize(lab, a, b, mk, lab)
        summarize(lab, "E21", None, mk, lab)
        if "hd" in fr:
            summarize(lab, "hd", None, mk, lab)
    if len(folds) > 2:
        for a, b in (("hd", "E21"),):
            summarize("all_folds", a, b, np.ones(len(E21), bool), "all")
    sdf, ddf = pd.DataFrame(rows), pd.DataFrame(drows)
    tag = "_".join(str(k) for k in folds)
    sdf.to_csv(OUT / f"scores_folds{tag}.csv", index=False); ddf.to_csv(OUT / f"deltas_folds{tag}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    print(sdf[(sdf.N == "eq") & (sdf.metric == "balanced")].to_string())
    print(ddf[(ddf.N == "eq") & (ddf.metric == "balanced")].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "report"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    fit(a.folds, set(a.variants)) if a.cmd == "fit" else report(a.folds)
