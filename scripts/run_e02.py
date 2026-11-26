"""E02: route-A pilot -- L0 vs HumanSL vs L0+HumanSL on the 1,600 pilot players (player-grouped 5-fold OOF).

  nice -n 10 .venv/bin/python scripts/run_e02.py oof      # 3 pilot-only models per fold (+ cached)
  nice -n 10 .venv/bin/python scripts/run_e02.py full     # reference M_L0_full (E01-style, non-pilot non-H players)
  nice -n 10 .venv/bin/python scripts/run_e02.py report   # scores, paired bootstrap, go/no-go, diagnostics, png
Same players / same questions / same folds (pilot_games.fold) / same LightGBM settings and weights for all models.
No GPU.  Outputs in experiments/E02_pilot_hsl/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gogame.aggregate import L0Store, aggregate_many, fit_sparse_selector, make_spec  # noqa: E402
from gogame.constants import RANKS, SEED  # noqa: E402
from gogame.metrics import balanced_weights, paired_bootstrap, rank_scores  # noqa: E402
from gogame.models import CLASSES, decide_argmax, decide_ev, fit_lgbm  # noqa: E402
from gogame.questions import Bag, bag_class_weights, build_val_questions, sample_train_bags  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402
from gogame.split import assert_no_player_leak  # noqa: E402
import run_e01 as e01  # noqa: E402  (reuse load_all / es_players / probs_frame / CFG / PARAMS)

OUT = ROOT / "experiments" / "E02_pilot_hsl"
PILOT = ROOT / "data" / "pilot" / "pilot_games.parquet"
HSL = ROOT / "data" / "cache" / "katago" / "pilot_hsl_games.parquet"
CFG = dict(e01.CFG)
CFG.update({"pilot_n_range": [5, 10], "val_ns": [5, 10], "val_groups": {"5": 2, "10": 1}, "bags_per_player_color": 8})
PARAMS = dict(e01.PARAMS)
MODELS = ("L0", "HSL", "L0_HSL")
PCOLS = lambda pre: [f"{pre}_{i}" for i in range(CLASSES)]  # noqa: E731


def load_pilot():
    split, meta, store, index = e01.load_all()
    pil = pd.read_parquet(PILOT)
    hsl = pd.read_parquet(HSL)
    pil_pl = pil.drop_duplicates("player_id").set_index("player_id")
    assert (split.set_index("player_id").loc[pil_pl.index, "fold"].to_numpy() == pil_pl["fold"].to_numpy()).all()
    assert not split.set_index("player_id").loc[pil_pl.index, "is_holdout"].any()
    assert set(hsl["uid"]) == set(pil["uid"])
    hcols = [c for c in hsl.columns if c != "uid"]
    hstore = L0Store(hsl["uid"].tolist(), hsl[hcols].to_numpy(np.float32), None, tuple(hcols))
    return split, meta, store, index, pil, pil_pl, hstore


def pilot_questions(pil: pd.DataFrame) -> list[Bag]:
    """Per player (10 games, one colour): N=5 x2 disjoint groups + N=10 x1.  Fixed named seed."""
    out = []
    for pid, g in pil.groupby("player_id", sort=True):
        uids = sorted(g["uid"])
        assert len(uids) == 10 and g["color"].nunique() == 1
        rk, col = int(g["rank_id"].iloc[0]), int(g["color"].iloc[0])
        perm = np.random.default_rng(derive_seed(SEED, "e02_questions", pid)).permutation(10)
        for j in range(2):
            u = tuple(sorted(uids[i] for i in perm[5 * j:5 * j + 5]))
            out.append(Bag(f"q_{pid}_5_{j}", pid, rk, col, 5, u))
        out.append(Bag(f"q_{pid}_10_0", pid, rk, col, 10, tuple(uids)))
    return out


def feats(bags, store, hstore, spec_l0, spec_h):
    XL = aggregate_many(bags, store, spec_l0)
    XH = aggregate_many(bags, hstore, spec_h)
    return {"L0": XL, "HSL": XH, "L0_HSL": np.hstack([XL, XH[:, :-2]])}   # drop duplicate bag_n / is_white


def save(path, bags, praw, p, fold):
    e01.probs_frame(bags, fold, praw, p).to_parquet(path, index=False)


# ------------------------------------------------------------------------------------------------ oof
def stage_oof(folds):
    split, meta, store, index, pil, pil_pl, hstore = load_pilot()
    allq = pilot_questions(pil)
    pd.DataFrame({"qid": [b.qid for b in allq], "player_id": [b.player_id for b in allq], "n": [b.n for b in allq]}) \
        .to_parquet(OUT / "questions.parquet", index=False)
    pidx = {pid: {c: sorted(g.loc[g["color"] == c, "uid"]) for c in (0, 1)} for pid, g in pil.groupby("player_id")}  # the 10 pilot games only
    index_p = type(index)(pidx, {p: int(pil_pl.loc[p, "rank_id"]) for p in pidx})
    sp_h = make_spec(hstore.names, np.zeros(0, np.int32))
    for k in folds:
        if all((OUT / f"oof_{m}_fold{k}.parquet").exists() for m in MODELS):
            print(f"fold {k} cached", flush=True)
            continue
        t0 = time.perf_counter()
        train_all = set(pil_pl.index[pil_pl["fold"] != k])
        val_pl = set(pil_pl.index[pil_pl["fold"] == k])
        val_bags = [b for b in allq if b.player_id in val_pl]
        assert_no_player_leak(train_all, val_pl)
        es_pl = e01.es_players(split[split["player_id"].isin(train_all)], CFG["es_frac"])
        fit_pl = train_all - es_pl
        assert_no_player_leak(fit_pl, es_pl)
        tb = sample_train_bags(index_p, fit_pl, CFG["bags_per_player_color"], tuple(CFG["pilot_n_range"]),
                               seed=derive_seed(SEED, "e02_train_bags", f"fold{k}"))
        eb = build_val_questions(index_p, es_pl, CFG["val_ns"], seed=derive_seed(SEED, "e02_es_val", f"fold{k}"))
        assert not ({u for b in tb for u in b.uids} & {u for b in val_bags for u in b.uids})
        w_tr, w_es = bag_class_weights(tb, "inv_players"), bag_class_weights(eb, "inv_players")
        ytr, yes = np.array([b.rank_id for b in tb]), np.array([b.rank_id for b in eb])
        cols = fit_sparse_selector(store.sparse[store.rows(sorted({u for b in tb for u in b.uids}))], CFG["min_df"])
        sp_l0 = make_spec(store.names, cols)
        Ftr, Fes, Fva = (feats(b, store, hstore, sp_l0, sp_h) for b in (tb, eb, val_bags))
        info = {"fold": k, "n_fit_players": len(fit_pl), "n_es_players": len(es_pl), "n_val_players": len(val_pl),
                "n_train_bags": len(tb), "n_es_bags": len(eb), "n_val_questions": len(val_bags)}
        for m in MODELS:
            if (OUT / f"oof_{m}_fold{k}.parquet").exists():
                continue
            mdl = fit_lgbm(Ftr[m], ytr, Fes[m], yes, PARAMS, derive_seed(SEED, "lgbm"), w_tr, w_es)
            save(OUT / f"oof_{m}_fold{k}.parquet", val_bags, mdl.predict_proba(Fva[m], raw=True), mdl.predict_proba(Fva[m]), k)
            info[f"{m}_best_iter"] = int(mdl.best_iteration)
            info[f"{m}_n_features"] = int(Ftr[m].shape[1])
            print(f"fold {k} {m}: {Ftr[m].shape[1]} feats, best_iter {mdl.best_iteration}, {time.perf_counter() - t0:.0f}s", flush=True)
            (OUT / f"meta_{m}_fold{k}.json").write_text(json.dumps(info))


# ------------------------------------------------------------------------------------------------ full (reference)
def stage_full():
    done = OUT / "oof_L0_full.parquet"
    if done.exists():
        print("full cached")
        return
    split, meta, store, index, pil, pil_pl, hstore = load_pilot()
    allq = pilot_questions(pil)
    H = set(split.loc[split["is_holdout"], "player_id"])
    pilot_set = set(pil_pl.index)
    train_all = set(split.loc[split["fold"] >= 0, "player_id"]) - pilot_set
    assert_no_player_leak(train_all, pilot_set)
    assert_no_player_leak(train_all, H)
    es_pl = e01.es_players(split[split["player_id"].isin(train_all)], CFG["es_frac"])
    fit_pl = train_all - es_pl
    tb = sample_train_bags(index, fit_pl, CFG["bags_per_player_color"], tuple(CFG["n_range"]), seed=derive_seed(SEED, "e02_full_bags"))
    eb = build_val_questions(index, es_pl, CFG["ns"], seed=derive_seed(SEED, "e02_full_es"))
    assert not ({b.player_id for b in tb} & pilot_set)
    w_tr, w_es = bag_class_weights(tb, "inv_players"), bag_class_weights(eb, "inv_players")
    cols = fit_sparse_selector(store.sparse[store.rows(sorted({u for b in tb for u in b.uids}))], CFG["min_df"])
    spec = make_spec(store.names, cols)
    Xtr, Xes, Xva = (aggregate_many(b, store, spec) for b in (tb, eb, allq))
    mdl = fit_lgbm(Xtr, [b.rank_id for b in tb], Xes, [b.rank_id for b in eb], PARAMS, derive_seed(SEED, "lgbm"), w_tr, w_es)
    save(done, allq, mdl.predict_proba(Xva, raw=True), mdl.predict_proba(Xva), -1)
    (OUT / "meta_full.json").write_text(json.dumps({"n_fit_players": len(fit_pl), "n_es_players": len(es_pl), "n_train_bags": len(tb),
                                                    "best_iter": int(mdl.best_iteration), "n_features": int(Xtr.shape[1]),
                                                    "pilot_players_in_train": 0}))
    print("full done", mdl.best_iteration)


# ------------------------------------------------------------------------------------------------ report
def go_no_go(delta, ci_low):
    """SDD 11.4 stage 1, balanced only."""
    if delta < 0:
        return "NO_GO"
    if ci_low > 0.015:
        return "GO"
    return "EXPAND"


def boot(sa, sb, groups, truth, balanced):
    w = balanced_weights(truth) if balanced else None
    b = paired_bootstrap(sa, sb, groups, weights=w)
    return {"delta": b.delta, "ci_low": b.ci_low, "ci_high": b.ci_high, "p_gt0": b.p_gt[0.0], "p_gt_0.015": b.p_gt[0.015]}


def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    P = {m: pd.concat([pd.read_parquet(OUT / f"oof_{m}_fold{k}.parquet") for k in range(5)], ignore_index=True) for m in MODELS}
    P["L0_full"] = pd.read_parquet(OUT / "oof_L0_full.parquet")
    base = P["L0"]
    for m, d in P.items():     # identical questions across models (reorder the full model by qid)
        P[m] = d.set_index("qid").loc[base["qid"]].reset_index()
        assert (P[m]["qid"].to_numpy() == base["qid"].to_numpy()).all()
    truth, groups, ns = base["rank_id"].to_numpy(), base["player_id"].to_numpy(), base["n"].to_numpy()
    R = {"n_questions": len(base), "n_players": int(base["player_id"].nunique()), "config": CFG}
    dec = {"argmax": decide_argmax, "ev": decide_ev}
    pred, S = {}, {}
    for m, d in P.items():
        for dn, f in dec.items():
            pred[m, dn] = f(d[PCOLS("p")].to_numpy())
            S[m, dn] = rank_scores(pred[m, dn], truth)
    wb = balanced_weights(truth)

    def ci(s, bal):
        b = paired_bootstrap(s, np.zeros_like(s), groups, weights=wb if bal else None)
        return {"mean": b.delta, "ci_low": b.ci_low, "ci_high": b.ci_high}

    R["scores"] = {f"{m}|{dn}": {"raw": ci(S[m, dn], False), "balanced": ci(S[m, dn], True),
                                 "exact_acc": float((pred[m, dn] == truth).mean())} for m in P for dn in dec}
    R["scores"]["random_expected"] = dict(zip(("raw", "balanced"), e01.expected_random(truth)))
    # per N / per rank
    R["per_N"], R["per_rank"] = {}, {}
    for m in P:
        for dn in dec:
            R["per_N"][f"{m}|{dn}"] = {}
            for n in sorted(set(ns)):
                k = ns == n
                w = balanced_weights(truth[k])
                R["per_N"][f"{m}|{dn}"][int(n)] = {"raw": float(S[m, dn][k].mean()), "balanced": float((S[m, dn][k] * w).mean())}
        R["per_rank"][m] = {RANKS[r]: {"score": float(S[m, "argmax"][truth == r].mean()),
                                       "recall": float((pred[m, "argmax"][truth == r] == r).mean()),
                                       "mean_abs_err": float(np.abs(pred[m, "argmax"][truth == r] - r).mean())} for r in range(CLASSES)}
    # deltas (paired bootstrap, player = unit) + verdict
    R["deltas"], R["verdict"] = {}, {}
    for a, b in (("L0_HSL", "L0"), ("HSL", "L0"), ("L0_HSL", "HSL")):
        for dn in dec:
            key = f"{a}-{b}|{dn}"
            R["deltas"][key] = {"raw": boot(S[a, dn], S[b, dn], groups, truth, False), "balanced": boot(S[a, dn], S[b, dn], groups, truth, True)}
            R["deltas"][key]["per_N_balanced"] = {}
            for n in sorted(set(ns)):
                k = ns == n
                R["deltas"][key]["per_N_balanced"][int(n)] = boot(S[a, dn][k], S[b, dn][k], groups[k], truth[k], True)
    for dn in dec:
        for a in ("L0_HSL", "HSL"):
            d = R["deltas"][f"{a}-L0|{dn}"]["balanced"]
            R["verdict"][f"{a}-L0|{dn}"] = go_no_go(d["delta"], d["ci_low"])
    R["verdict"]["primary"] = {"rule": "SDD 11.4 stage 1, balanced, argmax+p, M_L0+HSL - M_L0", "result": R["verdict"]["L0_HSL-L0|argmax"]}
    # reference: pilot-only L0 vs full-data L0 (not part of verdict)
    R["reference_full"] = {dn: {"L0_full-L0": boot(S["L0_full", dn], S["L0", dn], groups, truth, True),
                                "L0_HSL-L0_full": boot(S["L0_HSL", dn], S["L0_full", dn], groups, truth, True)} for dn in dec}

    # ---- mid-level segments
    segs = {"A_1D_2D": [3, 4, 5], "3D_5D": [6, 7, 8]}
    R["segments"] = {}
    for sn, ids in segs.items():
        R["segments"][sn] = {}
        for m in P:
            pr, k = pred[m, "argmax"], np.isin(truth, ids)
            R["segments"][sn][m] = {"n_q": int(k.sum()), "score": float(S[m, "argmax"][k].mean()),
                                    "exact_acc": float((pr[k] == truth[k]).mean()), "mean_abs_err": float(np.abs(pr[k] - truth[k]).mean()),
                                    "pred_inside_segment": float(np.isin(pr[k], ids).mean()),
                                    "recall": {RANKS[r]: float((pr[truth == r] == r).mean()) for r in ids}}
        d = boot(S["L0_HSL", "argmax"][np.isin(truth, ids)], S["L0", "argmax"][np.isin(truth, ids)], groups[np.isin(truth, ids)], truth[np.isin(truth, ids)], True)
        R["segments"][sn]["delta_L0_HSL_minus_L0_balanced"] = d
    # ---- confusion matrices
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    R["confusion"] = {}
    for a, m in zip(ax, ("L0", "HSL", "L0_HSL")):
        cm = np.zeros((CLASSES, CLASSES), int)
        np.add.at(cm, (truth, pred[m, "argmax"]), 1)
        pd.DataFrame(cm, index=RANKS, columns=RANKS).to_csv(OUT / f"confusion_{m}.csv")
        R["confusion"][m] = cm.tolist()
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = a.imshow(cmn, cmap="Blues", vmin=0, vmax=0.8)
        a.set_xticks(range(10), RANKS)
        a.set_yticks(range(10), RANKS)
        a.set_xlabel("predicted")
        a.set_ylabel("true")
        a.set_title(f"E02 {m} (argmax, row-normalised)  bal={R['scores'][m + '|argmax']['balanced']['mean']:.3f}")
        for i in range(10):
            for j in range(10):
                a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.45 else "black")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_E02.png", dpi=120)
    # ---- HSL column vs rank Spearman (per game, and per player mean)
    pil = pd.read_parquet(PILOT)
    hsl = pd.read_parquet(HSL).set_index("uid").loc[pil["uid"]]
    y = pil["rank_id"].to_numpy()
    rows = []
    for c in hsl.columns:
        x = hsl[c].to_numpy()
        rho = spearmanr(x, y).statistic if np.nanstd(x) > 0 else np.nan
        pm = hsl[c].groupby(pil["player_id"].to_numpy()).mean()
        yp = pil.groupby("player_id")["rank_id"].first().loc[pm.index].to_numpy()
        rp = spearmanr(pm.to_numpy(), yp).statistic if pm.std() > 0 else np.nan
        rows.append((c, rho, rp))
    sp = pd.DataFrame(rows, columns=["col", "rho_game", "rho_player_mean"])
    sp["abs"] = sp["rho_game"].abs()
    sp = sp.sort_values("abs", ascending=False).drop(columns="abs")
    sp.to_csv(OUT / "hsl_spearman.csv", index=False)
    R["hsl_spearman_top15"] = sp.head(15).round(4).to_dict(orient="records")
    # ---- player-level paired diff counts
    (OUT / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    s = lambda k, b="balanced": R["scores"][k][b]  # noqa: E731
    for m in list(P):
        print(m, "argmax bal %.4f [%.4f,%.4f] raw %.4f | ev bal %.4f raw %.4f" % (
            s(m + "|argmax")["mean"], s(m + "|argmax")["ci_low"], s(m + "|argmax")["ci_high"], s(m + "|argmax", "raw")["mean"],
            s(m + "|ev")["mean"], s(m + "|ev", "raw")["mean"]))
    print(json.dumps({k: v for k, v in R["deltas"].items()}, indent=0)[:200])
    print(json.dumps(R["verdict"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["oof", "full", "report"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    PARAMS["num_threads"] = a.threads
    OUT.mkdir(parents=True, exist_ok=True)
    {"oof": lambda: stage_oof(a.folds), "full": stage_full, "report": stage_report}[a.stage]()
