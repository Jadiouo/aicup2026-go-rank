"""E01: L0 hand-crafted features + LightGBM, player-grouped 5-fold OOF, H confirmation, controls, Player simulation.

Stages (each resumable, writes into experiments/E01_l0/):
  .venv/bin/python scripts/run_e01.py oof [--folds 0 1 2 3 4]   # per-fold models + controls (len-only, memory, random)
  .venv/bin/python scripts/run_e01.py final                    # all non-H players -> questions_H
  .venv/bin/python scripts/run_e01.py player                   # Player simulation (L0 centroid cosine) on H
  .venv/bin/python scripts/run_e01.py report                   # scores, CIs, per-N/rank, confusion, results.json
Prerequisite: scripts/build_l0_cache.py (data/cache/l0 + games_meta.parquet).  No GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gogame.aggregate import L0Store, aggregate_many, fit_sparse_selector, make_spec  # noqa: E402
from gogame.constants import SEED  # noqa: E402
from gogame.gametable import load_table  # noqa: E402
from gogame.memory import MemoryBaseline  # noqa: E402
from gogame.metrics import balanced_weights, paired_bootstrap, rank_scores  # noqa: E402
from gogame.models import CLASSES, decide_argmax, decide_ev, final_num_rounds, fit_lgbm  # noqa: E402
from gogame.questions import (Bag, bag_class_weights, build_player_index, build_player_pool_sim,  # noqa: E402
                              build_val_questions, flag_n_class_shift, n_class_report, sample_train_bags)
from gogame.seeds import derive_seed  # noqa: E402
from gogame.split import assert_no_player_leak, read_split  # noqa: E402

OUT = ROOT / "experiments" / "E01_l0"
CACHE = ROOT / "data" / "cache"
SPL = ROOT / "data" / "splits"
CFG = {"bags_per_player_color": 8, "n_range": [5, 20], "min_df": 0.005, "es_frac": 0.10, "ns": [5, 10, 15, 20],
       "lgbm_threads": 16, "length_cols": ["l0_n_plies", "l0_n_own", "l0_n_own_pass"], "target_prior": "uniform",
       "weights": "inv_players", "memory_k": [1, 5]}
PARAMS = {"num_threads": CFG["lgbm_threads"], "learning_rate": 0.1, "max_bin": 63, "log_every": 25}
CFG["lgbm_overrides"] = {"learning_rate": 0.1, "max_bin": 63}   # speed: spec start values (0.05, 255) took >1h per fold


def load_all():
    split = read_split(SPL / "split.parquet")
    meta = pd.read_parquet(CACHE / "games_meta.parquet")
    table = load_table(CACHE / "l0")
    assert list(table.uids) == meta["uid"].tolist()
    store = L0Store.from_table(table)
    index = build_player_index(meta.assign(color_id=meta["color"]))
    return split, meta, store, index


def frame_to_bags(df: pd.DataFrame) -> list[Bag]:
    ucols = [f"uid_{i}" for i in range(1, 21)]
    out = []
    for r in df.itertuples(index=False):
        n = int(r.n)
        out.append(Bag(r.qid, r.player_id, int(r.rank_id), int(r.color), n, tuple(getattr(r, c) for c in ucols[:n])))
    return out


def es_players(split_players: pd.DataFrame, frac: float) -> set:
    """10% of the players of every rank (stratified), named seed."""
    rng = np.random.default_rng(derive_seed(SEED, "es_split"))
    chosen = set()
    for rk, g in split_players.groupby("rank_id", sort=True):
        ids = sorted(g["player_id"])
        k = int(round(frac * len(ids)))
        chosen |= {ids[i] for i in rng.permutation(len(ids))[:k]}
    return chosen


def probs_frame(bags, fold, praw, p):
    d = {"qid": [b.qid for b in bags], "player_id": [b.player_id for b in bags], "rank_id": [b.rank_id for b in bags],
         "n": [b.n for b in bags], "color": [b.color for b in bags], "fold": fold}
    for i in range(CLASSES):
        d[f"praw_{i}"] = praw[:, i]
    for i in range(CLASSES):
        d[f"p_{i}"] = p[:, i]
    return pd.DataFrame(d)


def tune_threads():
    return dict(PARAMS)


# ------------------------------------------------------------------------------------------------ oof
def stage_oof(folds):
    split, meta, store, index = load_all()
    H = set(split.loc[split["is_holdout"], "player_id"])
    len_idx = [store.names.index(c) for c in CFG["length_cols"]]
    store_len = L0Store(store.uids, store.dense[:, len_idx], None, tuple(CFG["length_cols"]))
    for k in folds:
        done = OUT / f"oof_fold{k}.parquet"
        if done.exists():
            print(f"fold {k} cached", flush=True)
            continue
        t0 = time.perf_counter()
        pl = split[split["fold"] >= 0]
        train_all = set(pl.loc[pl["fold"] != k, "player_id"])
        val_df = pd.read_parquet(SPL / f"questions_fold{k}.parquet")
        val_bags = frame_to_bags(val_df)
        val_pl = set(pl.loc[pl["fold"] == k, "player_id"])
        assert {b.player_id for b in val_bags} <= val_pl
        es_pl = es_players(pl[pl["player_id"].isin(train_all)], CFG["es_frac"])
        fit_pl = train_all - es_pl
        for a, b in ((fit_pl, val_pl), (es_pl, val_pl), (fit_pl, es_pl), (fit_pl | es_pl, H), (val_pl, H)):
            assert_no_player_leak(a, b)
        tb = sample_train_bags(index, fit_pl, CFG["bags_per_player_color"], tuple(CFG["n_range"]),
                               seed=derive_seed(SEED, "train_bags", f"fold{k}"))
        eb = build_val_questions(index, es_pl, CFG["ns"], seed=derive_seed(SEED, "es_val", f"fold{k}"))
        w_tr = bag_class_weights(tb, "inv_players")
        w_es = bag_class_weights(eb, "inv_players")
        rep = n_class_report(tb)
        shift = flag_n_class_shift(rep)
        # sanity: no training / early-stopping game is also a validation game
        assert not ({u for b in tb for u in b.uids} & {u for b in val_bags for u in b.uids})
        ytr = np.array([b.rank_id for b in tb])
        yes = np.array([b.rank_id for b in eb])
        meta_out = {"fold": k, "n_fit_players": len(fit_pl), "n_es_players": len(es_pl), "n_val_players": len(val_pl),
                    "n_train_bags": len(tb), "n_es_bags": len(eb), "n_val_questions": len(val_bags),
                    "n_shift_flags": {str(n): bool(v) for n, v in shift.items()}}
        # ---- full L0 model
        used = sorted({u for b in tb for u in b.uids})
        cols = fit_sparse_selector(store.sparse[store.rows(used)], CFG["min_df"])
        spec = make_spec(store.names, cols)
        Xtr, Xes, Xva = (aggregate_many(b, store, spec) for b in (tb, eb, val_bags))
        print(f"fold {k}: bags {len(tb)} feat {Xtr.shape[1]} (sparse cols {len(cols)})  agg {time.perf_counter() - t0:.0f}s", flush=True)
        m = fit_lgbm(Xtr, ytr, Xes, yes, PARAMS, derive_seed(SEED, "lgbm"), w_tr, w_es)
        m.spec = spec
        praw, p = m.predict_proba(Xva, raw=True), m.predict_proba(Xva)
        probs_frame(val_bags, k, praw, p).to_parquet(done, index=False)
        meta_out.update(best_iteration=int(m.best_iteration), n_features=int(Xtr.shape[1]), n_sparse_cols=int(len(cols)),
                        train_prior=m.train_prior.tolist())
        print(f"fold {k}: lgbm best_iter {m.best_iteration}  {time.perf_counter() - t0:.0f}s", flush=True)
        # ---- controls on the same questions
        sp_len = make_spec(store_len.names, np.zeros(0, np.int32))
        Ltr, Les, Lva = (aggregate_many(b, store_len, sp_len) for b in (tb, eb, val_bags))
        ml = fit_lgbm(Ltr, ytr, Les, yes, PARAMS, derive_seed(SEED, "lgbm_len"), w_tr, w_es)
        ml.spec = sp_len
        probs_frame(val_bags, k, ml.predict_proba(Lva, raw=True), ml.predict_proba(Lva)).to_parquet(OUT / f"oof_len_fold{k}.parquet", index=False)
        meta_out["len_best_iteration"] = int(ml.best_iteration)
        pid_tr = np.array([b.player_id for b in tb])
        mem = {}
        mb = MemoryBaseline().fit(Xtr, pid_tr, ytr, colors=np.array([b.color for b in tb]))
        for kk in CFG["memory_k"]:
            pr, sim = mb.predict(Xva, k=kk)
            mem[f"pred_k{kk}"], mem[f"sim_k{kk}"] = pr, sim
        # positive control: the validation players ARE in the memory (fresh bags, other seed) -> must be far above chance
        cb = sample_train_bags(index, val_pl, CFG["bags_per_player_color"], tuple(CFG["n_range"]),
                               seed=derive_seed(SEED, "mem_control", f"fold{k}"))
        Xc = aggregate_many(cb, store, spec)
        mc = MemoryBaseline().fit(np.vstack([Xtr, Xc]), np.concatenate([pid_tr, [b.player_id for b in cb]]),
                                  np.concatenate([ytr, [b.rank_id for b in cb]]),
                                  colors=np.concatenate([[b.color for b in tb], [b.color for b in cb]]))
        mem["pred_control"], mem["sim_control"] = mc.predict(Xva, k=1)
        rng = np.random.default_rng(derive_seed(SEED, "random_baseline", f"fold{k}"))
        mem["pred_random"] = rng.integers(0, CLASSES, size=len(val_bags))
        md = {"qid": [b.qid for b in val_bags], "player_id": [b.player_id for b in val_bags],
              "rank_id": [b.rank_id for b in val_bags], "n": [b.n for b in val_bags], **mem}
        pd.DataFrame(md).to_parquet(OUT / f"oof_ctrl_fold{k}.parquet", index=False)
        meta_out["memory_n_centroids"] = int(mb.n_centroids)
        meta_out["seconds"] = round(time.perf_counter() - t0, 1)
        pd.DataFrame([meta_out]).to_json(OUT / f"oof_meta_fold{k}.json", orient="records", indent=1)
        rep.to_csv(OUT / f"train_bag_report_fold{k}.csv", index=False)
        print(f"fold {k} done {meta_out['seconds']}s", flush=True)


# ------------------------------------------------------------------------------------------------ final (H)
def stage_final():
    if (OUT / "h_probs.parquet").exists():
        print("final cached")
        return
    split, meta, store, index = load_all()
    H = set(split.loc[split["is_holdout"], "player_id"])
    best = [json.loads((OUT / f"oof_meta_fold{k}.json").read_text())[0]["best_iteration"] for k in range(5)]
    rounds = final_num_rounds(best)
    train_pl = set(split.loc[split["fold"] >= 0, "player_id"])
    assert_no_player_leak(train_pl, H)
    tb = sample_train_bags(index, train_pl, CFG["bags_per_player_color"], tuple(CFG["n_range"]),
                           seed=derive_seed(SEED, "train_bags", "final"))
    w = bag_class_weights(tb, "inv_players")
    used = sorted({u for b in tb for u in b.uids})
    cols = fit_sparse_selector(store.sparse[store.rows(used)], CFG["min_df"])
    spec = make_spec(store.names, cols)
    Xtr = aggregate_many(tb, store, spec)
    m = fit_lgbm(Xtr, np.array([b.rank_id for b in tb]), None, None, {**PARAMS, "num_boost_round": rounds},
                 derive_seed(SEED, "lgbm"), w, None)
    m.spec = spec
    hb = frame_to_bags(pd.read_parquet(SPL / "questions_H.parquet"))
    assert {b.player_id for b in hb} <= H
    Xh = aggregate_many(hb, store, spec)
    probs_frame(hb, -1, m.predict_proba(Xh, raw=True), m.predict_proba(Xh)).to_parquet(OUT / "h_probs.parquet", index=False)
    (OUT / "final_meta.json").write_text(json.dumps({"rounds": rounds, "oof_best_iterations": best, "n_train_players": len(train_pl),
                                                     "n_train_bags": len(tb), "n_h_questions": len(hb), "n_sparse_cols": int(len(cols))}, indent=1))
    print("final done", rounds)


# ------------------------------------------------------------------------------------------------ player sim
def stage_player():
    split, meta, store, index = load_all()
    H = sorted(split.loc[split["is_holdout"], "player_id"])
    Hset = set(H)
    nonH = ~meta["player_id"].isin(Hset).to_numpy()
    # standardiser + sparse-column selection fitted on non-H games only
    D = store.dense[nonH].astype(np.float64)
    mu, sd = np.nanmean(D, axis=0), np.nanstd(D, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    SP = store.sparse[np.flatnonzero(nonH)]
    cols = fit_sparse_selector(SP, CFG["min_df"])
    S = SP[:, cols]
    smu = np.asarray(S.mean(axis=0)).ravel()
    ssd = np.sqrt(np.maximum(np.asarray(S.multiply(S).mean(axis=0)).ravel() - smu ** 2, 0))
    ssd = np.where(ssd > 0, ssd, 1.0)
    del D, SP, S
    hrows = np.flatnonzero(~nonH & (meta["color"].to_numpy() == 0))
    Zd = np.nan_to_num((store.dense[hrows] - mu) / sd, nan=0.0)
    Zs = (store.sparse[hrows][:, cols].toarray() - smu) / ssd
    Z = np.hstack([Zd, Zs]).astype(np.float32)
    rowof = {u: i for i, u in enumerate(meta["uid"].to_numpy()[hrows])}
    print(f"player: {len(hrows)} H black games, vec dim {Z.shape[1]} ({len(cols)} sparse)", flush=True)

    def vec(uids):
        v = Z[[rowof[u] for u in uids]].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    res = {}
    for lib in (30, 50, 100):
        sims = build_player_pool_sim(index, H, 400, 20, lib, tuple(CFG["ns"]), holdout_set=Hset)
        draws = []
        for s in sims:
            pool = list(s.players)
            C = np.stack([vec(s.library[p]) for p in pool])
            Q = np.stack([vec(q.uids) for q in s.queries])
            cos = Q @ C.T
            top = np.argsort(-cos, axis=1, kind="stable")[:, :5]
            truth = np.array([pool.index(q.player_id) for q in s.queries])
            hit = top == truth[:, None]
            rpos = np.where(hit.any(1), hit.argmax(1) + 1, 0)
            sc = np.where(rpos > 0, np.exp(-(rpos - 1.0)), 0.0)
            ns = np.array([q.n for q in s.queries])
            draws.append({"score": sc.mean(), "top1": (rpos == 1).mean(), "top5": (rpos > 0).mean(),
                          **{f"score_n{n}": sc[ns == n].mean() for n in CFG["ns"]},
                          **{f"top1_n{n}": (rpos[ns == n] == 1).mean() for n in CFG["ns"]}, "n_queries": len(sc)})
        dd = pd.DataFrame(draws)
        res[str(lib)] = {c: {"mean": float(dd[c].mean()), "std": float(dd[c].std(ddof=1))} for c in dd.columns if c != "n_queries"}
        res[str(lib)]["n_queries_per_draw"] = float(dd["n_queries"].mean())
        res[str(lib)]["n_draws"] = len(dd)
        print(lib, res[str(lib)]["score"], flush=True)
    res["_meta"] = {"vector": "z-scored L0 dense(31) + sparse cols (min_df 0.005), standardiser fitted on non-H games",
                    "n_sparse_cols": int(len(cols)), "threshold": 0.2845, "pool_size": 400, "n_draws": 20}
    (OUT / "player_sim.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ report
def pcols(pre):
    return [f"{pre}_{i}" for i in range(CLASSES)]


def score_table(truth, pred):
    s = rank_scores(pred, truth)
    return s, float(s.mean()), float((s * balanced_weights(truth)).mean())


def ci_of(s, groups, w=None):
    b = paired_bootstrap(s, np.zeros_like(s), groups, weights=w)
    return {"mean": b.delta, "ci_low": b.ci_low, "ci_high": b.ci_high}


def delta(sa, sb, groups, truth):
    out = {}
    for mode, w in (("raw", None), ("balanced", balanced_weights(truth))):
        b = paired_bootstrap(sa, sb, groups, weights=w)
        out[mode] = {"delta": b.delta, "ci_low": b.ci_low, "ci_high": b.ci_high, "p_gt0": b.p_gt[0.0], "p_gt_0.015": b.p_gt[0.015]}
    return out


def expected_random(truth):
    M = np.eye(CLASSES) + np.exp(-1.0) * (np.eye(CLASSES, k=1) + np.eye(CLASSES, k=-1))
    per = M[truth].mean(axis=1)
    return float(per.mean()), float((per * balanced_weights(truth)).mean())


def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gogame.constants import RANKS
    cat = lambda pat: pd.concat([pd.read_parquet(OUT / pat.format(k=k)) for k in range(5)], ignore_index=True)
    oof, ln, ctrl = cat("oof_fold{k}.parquet"), cat("oof_len_fold{k}.parquet"), cat("oof_ctrl_fold{k}.parquet")
    assert oof["qid"].is_unique and (oof["qid"].to_numpy() == ln["qid"].to_numpy()).all() and (oof["qid"].to_numpy() == ctrl["qid"].to_numpy()).all()
    truth, groups, ns = oof["rank_id"].to_numpy(), oof["player_id"].to_numpy(), oof["n"].to_numpy()
    wb = balanced_weights(truth)
    R: dict = {"n_questions": len(oof), "n_players": int(oof["player_id"].nunique()), "config": CFG}

    P = {pk: oof[pcols(pk)].to_numpy() for pk in ("praw", "p")}
    variants = {}
    for dec_name, dec in (("argmax", decide_argmax), ("ev", decide_ev)):
        for pk in ("praw", "p"):
            s, raw, bal = score_table(truth, dec(P[pk]))
            variants[f"{dec_name}_{pk}"] = {"raw": raw, "balanced": bal}
    R["variants"] = variants
    pred = decide_argmax(P["p"])
    s_main = rank_scores(pred, truth)
    R["E01_argmax_p"] = {"raw": ci_of(s_main, groups), "balanced": ci_of(s_main, groups, wb),
                         "exact_acc_raw": float((pred == truth).mean()),
                         "exact_acc_balanced": float(((pred == truth) * wb).mean())}
    s_ev = rank_scores(decide_ev(P["p"]), truth)
    R["ev_vs_argmax"] = delta(s_ev, s_main, groups, truth)
    s_pr = rank_scores(decide_argmax(P["praw"]), truth)
    R["p_vs_praw_argmax"] = delta(s_main, s_pr, groups, truth)

    # controls on the same questions
    er, eb = expected_random(truth)
    ctl = {"random_expected_analytic": {"raw": er, "balanced": eb}}
    scores = {}
    for name, pr in (("random_uniform", ctrl["pred_random"].to_numpy()),
                     ("memory_k1", ctrl["pred_k1"].to_numpy()), ("memory_k5", ctrl["pred_k5"].to_numpy()),
                     ("memory_positive_control_k1", ctrl["pred_control"].to_numpy())):
        s = rank_scores(pr, truth)
        scores[name] = s
        ctl[name] = {"raw": ci_of(s, groups), "balanced": ci_of(s, groups, wb), "exact_acc_raw": float((pr == truth).mean())}
    Pl = ln[pcols("p")].to_numpy()
    s_len = rank_scores(decide_argmax(Pl), truth)
    scores["length_only_lgbm"] = s_len
    ctl["length_only_lgbm"] = {"raw": ci_of(s_len, groups), "balanced": ci_of(s_len, groups, wb),
                               "exact_acc_raw": float((decide_argmax(Pl) == truth).mean())}
    R["controls"] = ctl
    R["E01_minus_controls"] = {nm: delta(s_main, s, groups, truth) for nm, s in scores.items()}
    R["memory_sim_mean"] = {"k1": float(ctrl["sim_k1"].mean()), "control": float(ctrl["sim_control"].mean())}

    # per N / per rank
    perN = {}
    for n in sorted(set(ns)):
        m = ns == n
        w = balanced_weights(truth[m])
        perN[int(n)] = {"n_q": int(m.sum()), "raw": float(s_main[m].mean()), "balanced": float((s_main[m] * w).mean()),
                        "random_raw": float(expected_random(truth[m])[0]), "length_only_raw": float(s_len[m].mean()),
                        "memory_k1_raw": float(scores["memory_k1"][m].mean())}
    R["per_N"] = perN
    perR = {}
    for r in range(CLASSES):
        m = truth == r
        perR[RANKS[r]] = {"n_q": int(m.sum()), "score": float(s_main[m].mean()), "recall": float((pred[m] == r).mean()),
                          "mean_abs_rank_err": float(np.abs(pred[m] - r).mean()),
                          "pred_share": float((pred == r).mean()), "length_only_score": float(s_len[m].mean())}
    R["per_rank"] = perR

    cm = np.zeros((CLASSES, CLASSES), dtype=int)
    np.add.at(cm, (truth, pred), 1)
    pd.DataFrame(cm, index=RANKS, columns=RANKS).to_csv(OUT / "confusion_oof_argmax.csv")
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for a, M, ttl, fmt in ((ax[0], cm, "counts", "{:d}"), (ax[1], cmn, "row-normalised (recall)", "{:.2f}")):
        im = a.imshow(M, cmap="Blues")
        a.set_xticks(range(10), RANKS)
        a.set_yticks(range(10), RANKS)
        a.set_xlabel("predicted")
        a.set_ylabel("true")
        a.set_title(f"E01 OOF confusion ({ttl})")
        for i in range(10):
            for j in range(10):
                a.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                       color="white" if M[i, j] > M.max() * 0.6 else "black")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_oof.png", dpi=130)

    # H confirmation
    hp = pd.read_parquet(OUT / "h_probs.parquet")
    ht = hp["rank_id"].to_numpy()
    hpred = decide_argmax(hp[pcols("p")].to_numpy())
    hs = rank_scores(hpred, ht)
    hw = balanced_weights(ht)
    R["H"] = {"n_questions": len(hp), "n_players": int(hp["player_id"].nunique()),
              "argmax_p": {"raw": ci_of(hs, hp["player_id"].to_numpy()), "balanced": ci_of(hs, hp["player_id"].to_numpy(), hw)},
              "ev_p": {"raw": float(rank_scores(decide_ev(hp[pcols("p")].to_numpy()), ht).mean())},
              "random_expected": dict(zip(("raw", "balanced"), expected_random(ht))),
              "per_N": {int(n): {"raw": float(hs[hp["n"].to_numpy() == n].mean()), "n_q": int((hp["n"] == n).sum())} for n in sorted(set(hp["n"]))},
              "exact_acc_raw": float((hpred == ht).mean())}
    # OOF restricted to the same N-mix is not comparable (H is heavy players) -> per-N tables only
    meta_all = pd.concat([pd.read_json(OUT / f"oof_meta_fold{k}.json") for k in range(5)], ignore_index=True)
    R["oof_meta"] = meta_all[["fold", "n_train_bags", "n_features", "n_sparse_cols", "best_iteration", "len_best_iteration",
                              "memory_n_centroids", "seconds"]].to_dict(orient="records")
    R["n_shift_flags_any"] = bool(any(any(v.values()) if isinstance(v, dict) else False for v in meta_all["n_shift_flags"]))
    R["final_meta"] = json.loads((OUT / "final_meta.json").read_text())
    if (OUT / "player_sim.json").exists():
        R["player_sim"] = json.loads((OUT / "player_sim.json").read_text())
    (OUT / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps({k: R[k] for k in ("E01_argmax_p", "variants", "controls", "ev_vs_argmax", "per_N", "H")}, indent=1)[:6000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["oof", "final", "player", "report"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"oof": lambda: stage_oof(a.folds), "final": stage_final, "player": stage_player, "report": stage_report}[a.stage]()
