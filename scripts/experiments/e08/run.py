"""E08 analysis (local CPU, <=4 threads):  python scripts/experiments/e08/run.py sanity | oof | report
Reads Colab student outputs in .scratch/e08/out/pair_{a}_{b}.npz.  Re-uses E02 questions / folds / bags / LightGBM settings (scripts/run_e02.py)
and E02's saved OOF for L0 and HSL (teacher).  New models: STU (104 student aggregate columns), L0_STU.
Nested fold-out: for outer fold k, LightGBM-train fold j uses pair {j,k}; evaluation fold k uses pair {k,(k+1)%5}.
"""
import argparse, json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_e02 as e02  # noqa: E402
import run_e01 as e01  # noqa: E402
from agg import PROFILES, aggregate_moves  # noqa: E402
from gogame.aggregate import L0Store, fit_sparse_selector, make_spec, aggregate_many  # noqa: E402
from gogame.constants import SEED  # noqa: E402
from gogame.metrics import balanced_weights, rank_scores  # noqa: E402
from gogame.models import CLASSES, decide_argmax, decide_ev, fit_lgbm  # noqa: E402
from gogame.questions import bag_class_weights, build_val_questions, sample_train_bags  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402
from gogame.split import assert_no_player_leak  # noqa: E402

W = ROOT / ".scratch" / "e08"
OUTD = W / "out"
RES = ROOT / "experiments" / "E08_distill_pilot"
MODELS = ("STU", "L0_STU")


def pair_features(a, b, meta, cache={}):
    key = (a, b)
    if key in cache:
        return cache[key]
    z = np.load(OUTD / f"pair_{a}_{b}.npz")
    rows = z["rows"]
    df = pd.DataFrame({"uid": meta["uid"].to_numpy()[rows], "own_move_idx": meta["own_move_idx"].to_numpy()[rows]})
    L, R, E = z["logp"].astype(np.float32), z["rank"], z["ent"].astype(np.float32)
    for i in range(11):
        df[f"logp_{i}"] = L[:, i]
        df[f"rank_{i}"] = R[:, i]
        df[f"ent_{i}"] = E[:, i]
    cache[key] = aggregate_moves(df).set_index("uid")
    return cache[key]


def load_meta():
    return pd.read_parquet(W / "prep" / "meta.parquet")


def stu_store(k, pil, meta):
    """Student features for outer fold k as an L0Store over all 16,000 pilot uids."""
    fold = pil.set_index("uid")["fold"]
    parts = []
    for j in range(5):
        pair = tuple(sorted((j, k))) if j != k else tuple(sorted((k, (k + 1) % 5)))
        f = pair_features(*pair, meta)
        parts.append(f.loc[[u for u in f.index if fold[u] == j]])
    allf = pd.concat(parts).loc[pil["uid"].tolist()]
    cols = list(allf.columns)
    return L0Store(allf.index.tolist(), allf.to_numpy(np.float32), None, tuple(cols))


def assert_nested(k):
    for j in range(5):
        pair = tuple(sorted((j, k))) if j != k else tuple(sorted((k, (k + 1) % 5)))
        assert j in pair and k in pair, (k, j, pair)       # student predicting fold j never trained on j or k


def stage_sanity():
    meta = load_meta()
    pil = pd.read_parquet(e02.PILOT)
    teacher = pd.read_parquet(e02.HSL).set_index("uid")
    from scipy.stats import spearmanr
    rows, rep = [], {"pairs": {}}
    for a, b in combinations(range(5), 2):
        js = json.loads((OUTD / f"pair_{a}_{b}.json").read_text())
        c = js["heldout_corr"]
        f = pair_features(a, b, meta)
        t = teacher.loc[f.index]
        rid = pil.set_index("uid").loc[f.index, "rank_id"]
        rep["pairs"][f"{a}_{b}"] = {
            "raw_corr_mean": float(np.mean(c["raw"])), "raw_corr_min": float(np.min(c["raw"])),
            "centred_corr_mean": float(np.mean(c["centred"])), "centred_corr_min": float(np.min(c["centred"])),
            "student_std_min": float(np.min(c["student_std"])), "teacher_std_mean": float(np.mean(c["teacher_std"])),
            "game_exp_idx_spearman_vs_teacher": float(spearmanr(f["ml_exp_idx"], t["ml_exp_idx"]).statistic),
            "game_exp_idx_spearman_vs_rank_student": float(spearmanr(f["ml_exp_idx"], rid).statistic),
            "game_exp_idx_spearman_vs_rank_teacher": float(spearmanr(t["ml_exp_idx"], rid).statistic),
            "game_mlc_corr_vs_teacher_mean": float(np.mean([np.corrcoef(f[f"mlc_{p}"], t[f"mlc_{p}"])[0, 1] for p in PROFILES])),
            "final_huber": js["log"][-1]["huber"], "sec": js["sec"]}
    ok = all(v["student_std_min"] > 0.05 and v["centred_corr_mean"] > 0.1 for v in rep["pairs"].values())
    rep["sanity_pass"] = bool(ok)
    (RES / "sanity.json").write_text(json.dumps(rep, indent=1))
    print(pd.DataFrame(rep["pairs"]).T.round(3).to_string())
    print("sanity_pass", ok)


def stage_oof(folds, threads):
    e02.PARAMS["num_threads"] = threads
    meta = load_meta()
    split, _m, store, index, pil, pil_pl, _h = e02.load_pilot()
    allq = e02.pilot_questions(pil)
    pidx = {pid: {c: sorted(g.loc[g["color"] == c, "uid"]) for c in (0, 1)} for pid, g in pil.groupby("player_id")}
    index_p = type(index)(pidx, {p: int(pil_pl.loc[p, "rank_id"]) for p in pidx})
    for k in folds:
        if all((RES / f"oof_{m}_fold{k}.parquet").exists() for m in MODELS):
            print(f"fold {k} cached")
            continue
        assert_nested(k)
        t0 = time.perf_counter()
        hstore = stu_store(k, pil, meta)
        sp_h = make_spec(hstore.names, np.zeros(0, np.int32))
        train_all = set(pil_pl.index[pil_pl["fold"] != k])
        val_pl = set(pil_pl.index[pil_pl["fold"] == k])
        val_bags = [b for b in allq if b.player_id in val_pl]
        assert_no_player_leak(train_all, val_pl)
        es_pl = e01.es_players(split[split["player_id"].isin(train_all)], e02.CFG["es_frac"])
        fit_pl = train_all - es_pl
        tb = sample_train_bags(index_p, fit_pl, e02.CFG["bags_per_player_color"], tuple(e02.CFG["pilot_n_range"]),
                               seed=derive_seed(SEED, "e02_train_bags", f"fold{k}"))
        eb = build_val_questions(index_p, es_pl, e02.CFG["val_ns"], seed=derive_seed(SEED, "e02_es_val", f"fold{k}"))
        w_tr, w_es = bag_class_weights(tb, "inv_players"), bag_class_weights(eb, "inv_players")
        ytr, yes = np.array([b.rank_id for b in tb]), np.array([b.rank_id for b in eb])
        cols = fit_sparse_selector(store.sparse[store.rows(sorted({u for b in tb for u in b.uids}))], e02.CFG["min_df"])
        sp_l0 = make_spec(store.names, cols)
        Ftr, Fes, Fva = (_feats(b, store, hstore, sp_l0, sp_h) for b in (tb, eb, val_bags))
        for m in MODELS:
            mdl = fit_lgbm(Ftr[m], ytr, Fes[m], yes, e02.PARAMS, derive_seed(SEED, "lgbm"), w_tr, w_es)
            e02.save(RES / f"oof_{m}_fold{k}.parquet", val_bags, mdl.predict_proba(Fva[m], raw=True), mdl.predict_proba(Fva[m]), k)
            print(f"fold {k} {m}: {Ftr[m].shape[1]} feats, best_iter {mdl.best_iteration}, {time.perf_counter() - t0:.0f}s", flush=True)


def _feats(bags, store, hstore, spec_l0, spec_h):
    XL = aggregate_many(bags, store, spec_l0)
    XH = aggregate_many(bags, hstore, spec_h)
    return {"STU": XH, "L0_STU": np.hstack([XL, XH[:, :-2]])}


def stage_report(n_boot=4000):
    PC = lambda pre: [f"{pre}_{i}" for i in range(CLASSES)]  # noqa: E731
    P = {"L0": None, "HSL": None, "STU": None, "L0_STU": None}
    for m in P:
        src = e02.OUT if m in ("L0", "HSL") else RES
        P[m] = pd.concat([pd.read_parquet(src / f"oof_{m}_fold{k}.parquet") for k in range(5)], ignore_index=True)
    base = P["L0"]
    for m in P:
        P[m] = P[m].set_index("qid").loc[base["qid"]].reset_index()
    truth, groups, ns = base["rank_id"].to_numpy(), base["player_id"].to_numpy(), base["n"].to_numpy()
    out = {"n_questions": len(base), "n_players": int(len(np.unique(groups))), "n_boot": n_boot}
    for dn, dec in (("argmax", decide_argmax), ("ev", decide_ev)):
        S = {m: rank_scores(dec(P[m][PC("p")].to_numpy()), truth) for m in P}
        res = {"scores": {}, "per_N": {}}

        def boot(mask):
            w = balanced_weights(truth[mask])
            gi = np.unique(groups[mask], return_inverse=True)[1]
            Pn = gi.max() + 1
            num = {m: np.bincount(gi, weights=w * S[m][mask], minlength=Pn) for m in S}
            den = np.bincount(gi, weights=w, minlength=Pn)
            rng = np.random.default_rng(derive_seed(SEED, "e08_boot"))
            c = rng.multinomial(Pn, np.full(Pn, 1.0 / Pn), size=n_boot).astype(np.float64)
            sc = {m: (c @ num[m]) / (c @ den) for m in S}
            pt = {m: num[m].sum() / den.sum() for m in S}
            q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]  # noqa: E731
            r = {"scores": {m: {"point": float(pt[m]), "ci": q(sc[m])} for m in S}}
            for nm, (a, b) in {"S-L0": ("STU", "L0"), "H-L0": ("HSL", "L0"), "LS-L0": ("L0_STU", "L0"), "H-S": ("HSL", "STU")}.items():
                d = sc[a] - sc[b]
                r[f"delta_{nm}"] = {"point": float(pt[a] - pt[b]), "ci": q(d)}
            dS, dH = sc["STU"] - sc["L0"], sc["HSL"] - sc["L0"]
            ratio = dS / dH
            r["ratio_S_over_H"] = {"point": float((pt["STU"] - pt["L0"]) / (pt["HSL"] - pt["L0"])), "ci": q(ratio),
                                   "p_ratio_ge_0.70": float(np.mean(ratio >= 0.70)), "p_dS_gt0": float(np.mean(dS > 0))}
            return r
        res["all"] = boot(np.ones(len(truth), bool))
        for n in sorted(set(ns)):
            res["per_N"][int(n)] = boot(ns == n)
        out[dn] = res
    a = out["argmax"]["all"]
    out["verdict"] = {"rule": "ratio point >= 0.70 and delta(STU-L0) ci_low > 0 (balanced, argmax+p)",
                      "ratio": a["ratio_S_over_H"]["point"], "dS_ci_low": a["delta_S-L0"]["ci"][0],
                      "result": "POSITIVE" if (a["ratio_S_over_H"]["point"] >= 0.70 and a["delta_S-L0"]["ci"][0] > 0) else "NEGATIVE"}
    (RES / "results.json").write_text(json.dumps(out, indent=1))
    for dn in ("argmax", "ev"):
        r = out[dn]["all"]
        print(dn, {m: round(v["point"], 4) for m, v in r["scores"].items()}, "dS", r["delta_S-L0"], "dH", r["delta_H-L0"], "ratio", r["ratio_S_over_H"])
        for n, rn in out[dn]["per_N"].items():
            print("  N", n, "dS", round(rn["delta_S-L0"]["point"], 4), "dH", round(rn["delta_H-L0"]["point"], 4), "ratio", rn["ratio_S_over_H"])
    print(out["verdict"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sanity", "oof", "report"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"sanity": stage_sanity, "oof": lambda: stage_oof(a.folds, a.threads), "report": stage_report}[a.stage]()
