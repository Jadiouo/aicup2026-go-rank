"""E21 stage 3: SE (strength estimator) bag columns added to M03 inside candidate D.  Plan: experiments/E21_strength_estimator/PLAN.md.
School: E21_HOME=... E10_HOME=~/gogame_e10 python run.py fit --fold 0 1
Local : .venv/bin/python -m scripts.experiments.e21.run report --folds 0 1 [2 3 4]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
E10_HOME = os.environ.get("E10_HOME")
if E10_HOME:
    sys.path.insert(0, E10_HOME)
import numpy as np, pandas as pd, scipy.sparse as sps

from scripts.experiments.e10 import lib, pipeline
from gogame.aggregate import L0Store, aggregate_many, fit_sparse_selector, make_spec
from gogame.constants import RANKS, SEED
from gogame.metrics import balanced_weights, rank_scores
from gogame.models import decide_ev, fit_lgbm
from gogame.questions import bag_class_weights, sample_train_bags
from gogame.seeds import derive_seed
import run_e01 as e01

HOME = Path(os.environ.get("E21_HOME", lib.ROOT))
OUT = HOME / "out" if os.environ.get("E21_HOME") else lib.ROOT / "experiments" / "E21_strength_estimator" / "results"
E10_OOF = lib.OUT / "oof" if not os.environ.get("E21_E10_OOF") else Path(os.environ["E21_E10_OOF"])
PC = lib.PCOLS
PRM = {"num_threads": 4, "learning_rate": 0.1, "max_bin": 63, "log_every": 0}
TOL = 1e-6


SEF = Path(os.environ.get("E21_SE", lib.ROOT / ".scratch" / "e21" / "se_full.parquet"))
SE_COLS = ["se_mean_mean", "se_mean_std", "se_mean_p10", "se_mean_p90", "se_mean_min", "se_mean_max", "se_std_mean"]


def se_feats(bags, sm, ss):
    """7 bag-level SE columns from the bag's own games only (sm / ss: dict uid -> se_mean / se_std; NaN ignored)."""
    out = np.full((len(bags), 7), np.nan)
    for i, b in enumerate(bags):
        v = np.array([sm[u] for u in b.uids], float); v = v[~np.isnan(v)]
        w = np.array([ss[u] for u in b.uids], float); w = w[~np.isnan(w)]
        if len(v):
            out[i, :6] = [v.mean(), v.std(), np.percentile(v, 10), np.percentile(v, 90), v.min(), v.max()]
        if len(w):
            out[i, 6] = w.mean()
    return out


def fit(folds, skip_control):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = pipeline.default_cache(False)
    ctx = pipeline.Ctx(cache, False)
    assert not ctx.partial
    st = pd.read_parquet(SEF)
    nonh_uids = set(ctx.meta_all.loc[~ctx.meta_all["player_id"].isin(ctx.H), "uid"])
    assert set(st["uid"]) == nonh_uids and st["uid"].is_unique, "SE table must cover exactly the non-H games"
    assert not (set(ctx.meta_all.loc[ctx.meta_all["uid"].isin(set(st["uid"])), "player_id"]) & ctx.H)
    sm = dict(zip(st["uid"], st["se_mean"].astype(float))); ss = dict(zip(st["uid"], st["se_std"].astype(float)))
    print("SE table", len(st), "nan", int(st["se_mean"].isna().sum()), flush=True)
    q2f = pd.read_parquet(E10_OOF / "questions_q2.parquet")
    q2_all = e01.frame_to_bags(q2f)
    assert not ({b.player_id for b in q2_all} & ctx.H)
    c = lib.CANDS["A"]
    for k in folds:
        t0 = time.perf_counter()
        fs = lib.fold_sets(ctx.nonh, k)
        lib.assert_fold_disjoint(fs, ctx.H)
        for key in ("fit", "es", "val"):
            assert not (fs[key] & ctx.H), key
        val = [b for b in q2_all if b.player_id in fs["val"]]
        a = pd.read_parquet(E10_OOF / f"oof_A_fold{k}.parquet").set_index("qid")
        a_q2 = a[a["qset"] == "q2"]
        assert set(a_q2.index) == {b.qid for b in val}, "M04_A oof question set != Q2 val questions"
        game_rep = lib.assert_no_game_leak(ctx.meta_all, [u for b in val for u in b.uids], fs["train_all"])
        fit_in = sorted(p for p in fs["fit"] if p in ctx.index_all.games)
        tb = sample_train_bags(ctx.index_all, fit_in, c.bags_per_pc, c.n_range, seed=c.train_seed(k))
        es = lib.make_es_bags(ctx.index_all, fs["es"], k, False)
        ref_meta = json.loads((E10_OOF / f"meta_D_fold{k}.json").read_text())
        assert len(tb) == ref_meta["n_train_bags"], (len(tb), ref_meta["n_train_bags"])
        assert len(es) == ref_meta["n_es_bags"], (len(es), ref_meta["n_es_bags"])
        assert {b.player_id for b in tb} <= fs["fit"] and not ({b.player_id for b in tb} & (fs["val"] | fs["es"] | ctx.H))
        vu = {u for b in val for u in b.uids}
        assert not (vu & {u for b in tb + es for u in b.uids})
        vu = {u for b in val for u in b.uids}
        assert not (vu & {u for b in tb + es for u in b.uids})
        dup_q = []
        print(f"fold {k}: val {len(val)} train bags {len(tb)} es {len(es)}", flush=True)
        # features
        used = sorted({u for b in tb for u in b.uids})
        cols = fit_sparse_selector(ctx.l0.sparse[ctx.l0.rows(used)], e01.CFG["min_df"])
        spec_l0 = make_spec(ctx.l0.names, cols)
        del used

        def mat(bags):
            base = pipeline._m03_matrix(ctx, bags, spec_l0)
            return np.hstack([base, se_feats(bags, sm, ss)]), base.shape[1]
        Xtr, n_ctrl = mat(tb); Xes, _ = mat(es); Xv, _ = mat(val)
        print(f"features {time.perf_counter()-t0:.0f}s widths ctrl {n_ctrl} prime {Xtr.shape[1]}", flush=True)
        ytr, yes = np.array([b.rank_id for b in tb]), np.array([b.rank_id for b in es])
        wtr, wes = bag_class_weights(tb, "inv_players"), bag_class_weights(es, "inv_players")
        seed = derive_seed(SEED, "lgbm")
        info = {"fold": k, "n_train_bags": len(tb), "n_es_bags": len(es), "n_val": len(val), "dup_hash_questions": dup_q,
                "game_leak_check": game_rep, "n_ctrl_features": int(n_ctrl), "n_prime_features": int(Xtr.shape[1]),
                "se_cols": SE_COLS}
        if False:
            rf = OUT / "repro_M03_fold0.json"
            if not (rf.exists() and json.loads(rf.read_text()).get("status") == "OK"):
                t1 = time.perf_counter()
                m = fit_lgbm(np.ascontiguousarray(Xtr[:, :n_ctrl]), ytr, np.ascontiguousarray(Xes[:, :n_ctrl]), yes, PRM, seed, wtr, wes)
                pr, p = m.predict_proba(Xv[:, :n_ctrl], raw=True), m.predict_proba(Xv[:, :n_ctrl])
                mine = e01.probs_frame(val, k, pr, p).set_index("qid")
                ref = pd.read_parquet(E10_OOF / "oof_M03A_fold0.parquet").set_index("qid").loc[mine.index]
                cc = [f"praw_{i}" for i in range(10)] + PC
                d = float(np.abs(mine[cc].to_numpy() - ref[cc].to_numpy()).max())
                ok = d <= TOL and int(m.best_iteration) == int(ref_meta["best_iteration"])
                rec = {"status": "OK" if ok else "MISMATCH", "max_abs_diff": d, "tol": TOL, "best_iteration": int(m.best_iteration),
                       "e10_best_iteration": ref_meta["best_iteration"], "n_questions": len(mine), "seconds": round(time.perf_counter() - t1),
                       "peak_rss_mb": round(pipeline.peak_rss_mb())}
                rf.write_text(json.dumps(rec, indent=1))
                print("repro:", rec, flush=True)
                if not ok:
                    raise SystemExit("STOP: control M03 refit != E10 oof_M03A_fold0 (see repro_M03_fold0.json); not running M03'")
                del m
        t1 = time.perf_counter()
        m = fit_lgbm(Xtr, ytr, Xes, yes, PRM, seed, wtr, wes)
        pr, p = m.predict_proba(Xv, raw=True), m.predict_proba(Xv)
        assert np.isfinite(pr).all() and np.isfinite(p).all()
        d3 = e01.probs_frame(val, k, pr, p)
        d3["qset"] = "q2"
        imp = m.booster.feature_importance("gain"); tot = float(imp.sum()); order = np.argsort(-imp)
        rank_of = np.empty(len(imp), int); rank_of[order] = np.arange(1, len(imp) + 1)
        info["se_importance"] = {c: {"gain_share": float(imp[n_ctrl + j] / tot), "rank": int(rank_of[n_ctrl + j])} for j, c in enumerate(SE_COLS)}
        info["se_gain_share_total"] = float(imp[n_ctrl:].sum() / tot); info["n_features"] = int(len(imp))
        info["n_bag_all_nan"] = {"train": int(np.isnan(Xtr[:, n_ctrl]).sum()), "val": int(np.isnan(Xv[:, n_ctrl]).sum())}
        ap = a_q2.loc[d3["qid"]]
        dd = d3.copy()
        rc = [f"praw_{j}" for j in range(10)]
        dd[PC] = lib.ensemble_probs(d3[PC].to_numpy(), ap[PC].to_numpy())
        dd[rc] = lib.ensemble_probs(d3[rc].to_numpy(), ap[rc].to_numpy())
        for df, nm in ((d3, "M03SE"), (dd, "Dse")):
            df.to_parquet(OUT / f"oof_{nm}_fold{k}.parquet.tmp", index=False)
            (OUT / f"oof_{nm}_fold{k}.parquet.tmp").replace(OUT / f"oof_{nm}_fold{k}.parquet")
        info.update({"best_iteration_prime": int(m.best_iteration), "seconds_fit_prime": round(time.perf_counter() - t1),
                     "seconds_total": round(time.perf_counter() - t0), "peak_rss_mb": round(pipeline.peak_rss_mb()), "n_H_in_any_set": 0})
        (OUT / f"meta_fold{k}.json").write_text(json.dumps(info, default=str))
        print(f"fold {k} done {info['seconds_total']}s iters {m.best_iteration} peak {info['peak_rss_mb']}MB", flush=True)
        del Xtr, Xes, Xv, m


# ---------------------------------------------------------------------------------------------- report (local)
NS = (5, 10, 15, 20)


def _boot(num_den, C):
    return np.mean([(C @ n) / (C @ d) for n, d in num_den], axis=0)


def report(folds):
    rows, drows = [], []
    D = pd.concat([pd.read_parquet(E10_OOF / f"oof_D_fold{k}.parquet").query("qset == 'q2'") for k in folds], ignore_index=True)
    Dse = pd.concat([pd.read_parquet(OUT / f"oof_Dse_fold{k}.parquet") for k in folds], ignore_index=True)
    M3 = pd.concat([pd.read_parquet(E10_OOF / f"oof_M03A_fold{k}.parquet").query("qset == 'q2'") for k in folds], ignore_index=True)
    M3p = pd.concat([pd.read_parquet(OUT / f"oof_M03SE_fold{k}.parquet") for k in folds], ignore_index=True)
    dup = set(q for k in folds for q in json.loads((OUT / f"meta_fold{k}.json").read_text())["dup_hash_questions"])
    D = D.sort_values("qid").reset_index(drop=True)
    fr = {"D": D}
    for nm, df in (("Dse", Dse), ("M03A", M3), ("M03SE", M3p)):
        fr[nm] = df.set_index("qid").loc[D["qid"]].reset_index()
    for nm in fr:
        assert list(fr[nm]["qid"]) == list(D["qid"]) and list(fr[nm]["rank_id"]) == list(D["rank_id"])
    truth, ns, fold = D["rank_id"].to_numpy(), D["n"].to_numpy(), D["fold"].to_numpy()
    pl = D["player_id"].to_numpy()
    assert not (set(pl) & set(pd.read_parquet(lib.ROOT / "data/splits/split.parquet").query("is_holdout")["player_id"]))
    pred = {nm: decide_ev(fr[nm][PC].to_numpy()) for nm in fr}
    sc = {nm: rank_scores(pred[nm], truth) for nm in fr}
    hit = {nm: (pred[nm] == truth).astype(float) for nm in fr}

    def groups(mask):
        u, gi = np.unique(pl[mask], return_inverse=True)
        return gi, len(u)

    def ndp(v, kind, mask):
        gi, P = groups(mask)
        out = []
        for n in NS:
            mk = mask & (ns == n)
            w = balanced_weights(truth[mk]) if kind == "balanced" else np.ones(mk.sum())
            g = np.searchsorted(np.unique(pl[mask]), pl[mk])
            out.append((np.bincount(g, weights=v[mk] * w, minlength=P), np.bincount(g, weights=w, minlength=P)))
        return out, P

    def summarize(label, a, b, mask, seed_tag):
        gi, P = groups(mask)
        C = np.random.default_rng(derive_seed(SEED, "e21", "bootstrap", seed_tag)).multinomial(P, np.full(P, 1.0 / P), size=4000).astype(np.float64)
        for kind in ("balanced", "raw", "hit"):
            va = {"balanced": sc, "raw": sc, "hit": hit}[kind]
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
    allm = np.isin(fold, [0, 1])
    assert allm.any()
    summarize("main", "D", None, allm, "main"); summarize("main", "Dse", None, allm, "main")
    summarize("main", "Dse", "D", allm, "main"); summarize("main", "M03SE", "M03A", allm, "main")
    for k in folds:
        mk = fold == k
        summarize(f"fold{k}", "Dse", "D", mk, f"fold{k}")
        summarize(f"fold{k}", "M03SE", "M03A", mk, f"fold{k}")
    if len(folds) > 2:
        summarize("all_folds", "Dse", "D", np.ones(len(D), bool), "all")
    # pool-size strata (games in the question's player/colour pool), as in E10
    q2p = pd.read_parquet(E10_OOF / "questions_q2.parquet", columns=["qid", "pool"]).set_index("qid")["pool"]
    pool = q2p.reindex(D["qid"]).to_numpy()
    for lab, mk in (("pool_lt60", (pool < lib.POOL_SPLIT) & allm), ("pool_ge60", (pool >= lib.POOL_SPLIT) & allm)):
        summarize(lab, "Dse", "D", mk, lab)
    sdf, ddf = pd.DataFrame(rows), pd.DataFrame(drows)
    base = sdf[(sdf.set == "main") & (sdf.model == "D") & (sdf.N == "eq") & (sdf.metric == "balanced")]["value"].iloc[0]
    print("CONTROL D balanced fold0+1 =", round(base, 4), "(E20: .5878)", "OK" if abs(base - 0.5878) < 5e-4 else "MISMATCH")
    sdf.to_csv(OUT / "scores.csv", index=False); ddf.to_csv(OUT / "deltas.csv", index=False)
    (OUT / "report_meta.json").write_text(json.dumps({"folds": folds, "questions": len(D), "players": len(set(pl)), "dup_hash_questions": len(dup), "n_boot": 4000}))
    pd.set_option("display.width", 200)
    print(sdf[(sdf.N == "eq")].to_string()); print(ddf[(ddf.N == "eq")].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "report"]); ap.add_argument("--fold", "--folds", dest="fold", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--skip-control", action="store_true")
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    fit(a.fold, a.skip_control) if a.cmd == "fit" else report(a.fold)
