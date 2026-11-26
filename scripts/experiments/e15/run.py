"""E15 opponent-move L0 probe.  Plan: experiments/E15_opponent_moves/PLAN.md (pre-registered).
  .venv/bin/python -m scripts.experiments.e15.run fit --fold 0      /   ... report
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sps

from scripts.experiments.e10 import lib
from gogame.aggregate import L0Store, aggregate_many, fit_sparse_selector, make_spec
from gogame.constants import RANKS, SEED
from gogame.gametable import load_table
from gogame.metrics import balanced_weights, rank_scores
from gogame.models import decide_argmax, decide_ev, fit_lgbm
from gogame.questions import bag_class_weights, build_player_index, sample_train_bags
from gogame.seeds import derive_seed
from gogame.split import read_split
import run_e01 as e01

ROOT = lib.ROOT
OUT = ROOT / "experiments" / "E15_opponent_moves"
RES = OUT / "results"
OPP = ROOT / ".scratch" / "e15" / "opp_l0"
PC = lib.PCOLS
PRM = {"num_threads": 4, "learning_rate": 0.1, "max_bin": 63, "log_every": 0}


def load_opp():
    uids, dense, sp, mh = [], [], [], []
    for r in RANKS:
        assert (OPP / f"{r}.done").exists(), f"opp L0 rank {r} missing"
        m = pd.read_parquet(OPP / f"{r}_meta.parquet")
        uids += m["uid"].tolist(); mh += m["mhash"].tolist()
        dense.append(np.load(OPP / f"{r}_dense.npy")); sp.append(sps.load_npz(OPP / f"{r}_sparse.npz"))
    return uids, np.vstack(dense), sps.vstack(sp).tocsr(), np.array(mh)


def fit(k):
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    split = read_split(ROOT / "data/splits/split.parquet")
    H = set(split.loc[split["is_holdout"], "player_id"])
    nonh = split[~split["is_holdout"]].reset_index(drop=True)
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    meta = meta[~meta["player_id"].isin(H)].reset_index(drop=True)       # H games never enter
    assert not (set(meta["player_id"]) & H)
    index = build_player_index(meta.assign(color_id=meta["color"]))
    fs = lib.fold_sets(nonh, k)
    lib.assert_fold_disjoint(fs, H)
    for key in ("fit", "es", "val"):
        assert not (fs[key] & H), key
    # stores (own from the E10-era L0 cache, restricted to non-H uids; opp from extract_opp.py)
    ouids, odense, osp, mh = load_opp()
    assert set(ouids) == set(meta["uid"]), "opp store must cover exactly the non-H games"
    from gogame.features_l0 import DENSE_NAMES
    opp = L0Store(ouids, odense, osp, tuple("l0_" + n for n in DENSE_NAMES))
    tab = load_table(ROOT / "data/cache/l0")
    keep = np.nonzero(np.isin(np.asarray(tab.uids), np.asarray(ouids)))[0]
    own = L0Store(list(np.asarray(tab.uids)[keep]), tab.dense[keep], tab.sparse[keep], tab.names)
    del tab
    assert own.names == opp.names
    # bags
    fit_in = sorted(p for p in fs["fit"] if p in index.games)
    tb = sample_train_bags(index, fit_in, 8, (5, 20), seed=derive_seed(SEED, "e15", "train_bags", f"fold{k}"))
    es = lib.make_es_bags(index, fs["es"], k, False)
    q2f = pd.read_parquet(lib.OUT / "oof" / "questions_q2.parquet")
    val = [b for b in e01.frame_to_bags(q2f) if b.player_id in fs["val"]]
    assert val and not ({b.player_id for b in val} & (fs["train_all"] | H))
    assert {b.player_id for b in tb} <= fs["fit"] and {b.player_id for b in es} <= fs["es"]
    # leak checks: exact-uid, (rank,game_id) overlap, move-hash overlap
    vu = {u for b in val for u in b.uids}
    tu = {u for b in tb + es for u in b.uids}
    assert not (vu & tu)
    leak = lib.assert_no_game_leak(meta, vu, fs["train_all"])
    h2u = pd.Series(mh, index=ouids)
    trh = set(h2u[h2u.index.isin(set(meta.loc[meta["player_id"].isin(fs["train_all"]), "uid"]))])
    bad_u = {u for u in vu if h2u[u] in trh}
    val_ok = [b for b in val if not (set(b.uids) & bad_u)]
    print(f"fold {k}: val bags {len(val)} (dup-hash dropped {len(val) - len(val_ok)}), train bags {len(tb)}, es {len(es)}, leak {leak['n_overlap']}", flush=True)
    assert len(val) - len(val_ok) <= 0.01 * len(val), "more than 1% duplicate-game questions: stop and ask"
    val = val_ok
    # features
    used = sorted({u for b in tb for u in b.uids})
    spec_own = make_spec(own.names, fit_sparse_selector(own.sparse[own.rows(used)], 0.005))
    spec_opp = make_spec(opp.names, fit_sparse_selector(opp.sparse[opp.rows(used)], 0.005))
    del used
    def feats(bags):
        XA, XO = aggregate_many(bags, own, spec_own), aggregate_many(bags, opp, spec_opp)
        return {"A": XA, "B": np.hstack([XA, XO[:, :-2]]), "C": XO}
    Ftr, Fes, Fv = feats(tb), feats(es), feats(val)
    print(f"features {time.time()-t0:.0f}s, widths { {g: Ftr[g].shape[1] for g in Ftr} }", flush=True)
    ytr, yes = np.array([b.rank_id for b in tb]), np.array([b.rank_id for b in es])
    wtr, wes = bag_class_weights(tb, "inv_players"), bag_class_weights(es, "inv_players")
    for g in ("A", "B", "C"):
        f = RES / f"probs_{g}_fold{k}.parquet"
        if f.exists():
            continue
        m = fit_lgbm(Ftr[g], ytr, Fes[g], yes, PRM, derive_seed(SEED, "lgbm"), wtr, wes)
        p = m.predict_proba(Fv[g]); pr = m.predict_proba(Fv[g], raw=True)
        d = e01.probs_frame(val, k, pr, p)
        d.to_parquet(f, index=False)
        (RES / f"meta_{g}_fold{k}.json").write_text(json.dumps({"best_iteration": int(m.best_iteration), "n_features": int(Ftr[g].shape[1]), "seconds": round(time.time() - t0)}))
        print(f"{g} done best_iter {m.best_iteration} {time.time()-t0:.0f}s", flush=True)
    (RES / f"info_fold{k}.json").write_text(json.dumps({"n_val": len(val), "n_train_bags": len(tb), "n_es": len(es), "leak": leak, "dup_hash_dropped": len(val) - len(val_ok),
                                                       "n_val_players": len({b.player_id for b in val}), "n_H_in_any_set": 0}))


def _agg(v, w, gi, P):
    return np.bincount(gi, weights=v * w, minlength=P), np.bincount(gi, weights=w, minlength=P)


def report(folds):
    fr = {g: pd.concat([pd.read_parquet(RES / f"probs_{g}_fold{k}.parquet") for k in folds], ignore_index=True).sort_values("qid").reset_index(drop=True) for g in "ABC"}
    for g in "BC":
        assert list(fr[g]["qid"]) == list(fr["A"]["qid"])
    base = fr["A"]
    split = read_split(ROOT / "data/splits/split.parquet")
    H = set(split.loc[split["is_holdout"], "player_id"])
    assert not (set(base["player_id"]) & H)
    truth, ns = base["rank_id"].to_numpy(), base["n"].to_numpy()
    pl = sorted(set(base["player_id"])); pi = {p: i for i, p in enumerate(pl)}; gi = base["player_id"].map(pi).to_numpy(); P = len(pl)
    rng = np.random.default_rng(derive_seed(SEED, "e15", "bootstrap"))
    C = rng.multinomial(P, np.full(P, 1.0 / P), size=4000).astype(np.float64)
    pred = {g: decide_ev(fr[g][PC].to_numpy()) for g in "ABC"}
    sc = {g: rank_scores(pred[g], truth) for g in "ABC"}
    hit = {g: (pred[g] == truth).astype(float) for g in "ABC"}
    rows = []
    NS = (5, 10, 15, 20)
    def stat(v, kind, msk):
        w = balanced_weights(truth[msk]) if kind == "balanced" else np.ones(msk.sum())
        return _agg(v[msk], w, gi[msk], P)
    for g in "ABC":
        for kind, v in (("raw", sc[g]), ("balanced", sc[g]), ("hit", hit[g])):
            parts = {}
            for n in NS:
                msk = ns == n
                num, den = stat(v, kind, msk); parts[n] = (num, den)
                b = (C @ num) / (C @ den)
                rows.append({"model": g, "N": n, "metric": kind, "value": num.sum() / den.sum(), "lo": np.quantile(b, .025), "hi": np.quantile(b, .975), "n_q": int(msk.sum())})
            pt = np.mean([parts[n][0].sum() / parts[n][1].sum() for n in NS])
            bb = np.mean([(C @ parts[n][0]) / (C @ parts[n][1]) for n in NS], axis=0)
            rows.append({"model": g, "N": "eq", "metric": kind, "value": pt, "lo": np.quantile(bb, .025), "hi": np.quantile(bb, .975), "n_q": len(ns)})
    pd.DataFrame(rows).to_csv(RES / "scores.csv", index=False)
    drows = []
    def contrast(a, b, sel_folds=None, label=None):
        m = np.ones(len(ns), bool) if sel_folds is None else base["fold"].isin(sel_folds).to_numpy()
        pls = sorted(set(base.loc[m, "player_id"])); pj = {p: i for i, p in enumerate(pls)}; g_ = base["player_id"].map(pj).fillna(0).astype(int).to_numpy(); Q = len(pls)
        Cm = np.random.default_rng(derive_seed(SEED, "e15", "bootstrap_c", str(sel_folds))).multinomial(Q, np.full(Q, 1.0 / Q), size=4000).astype(np.float64) if sel_folds is not None else C
        for kind in ("balanced", "raw"):
            pts, bs = [], []
            for n in NS:
                mk = m & (ns == n)
                w = balanced_weights(truth[mk]) if kind == "balanced" else np.ones(mk.sum())
                d = (sc[a] - sc[b])[mk]
                num = np.bincount(g_[mk], weights=d * w, minlength=Q); den = np.bincount(g_[mk], weights=w, minlength=Q)
                pts.append(num.sum() / den.sum()); bs.append((Cm @ num) / (Cm @ den))
                drows.append({"contrast": label or f"{a}-{b}", "N": n, "metric": kind, "delta": pts[-1], "lo": np.quantile(bs[-1], .025), "hi": np.quantile(bs[-1], .975), "players": Q})
            bb = np.mean(bs, axis=0)
            drows.append({"contrast": label or f"{a}-{b}", "N": "eq", "metric": kind, "delta": float(np.mean(pts)), "lo": np.quantile(bb, .025), "hi": np.quantile(bb, .975), "players": Q})
    for a, b in (("B", "A"), ("C", "A"), ("B", "C")):
        contrast(a, b)
    for k in folds:
        contrast("B", "A", [k], f"B-A fold{k}")
    pd.DataFrame(drows).to_csv(RES / "deltas.csv", index=False)
    (RES / "report_meta.json").write_text(json.dumps({"folds": folds, "players": P, "questions": len(ns), "n_boot": 4000}))
    sdf = pd.DataFrame(rows); ddf = pd.DataFrame(drows)
    print(sdf[sdf.N == "eq"].to_string()); print(ddf[ddf.N == "eq"].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "report"]); ap.add_argument("--fold", type=int, nargs="*", default=[0, 1])
    a = ap.parse_args()
    if a.cmd == "fit":
        for k in a.fold:
            fit(k)
    else:
        report(a.fold)
