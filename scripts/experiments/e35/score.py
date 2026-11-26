"""E35 scoring (local, light).  Plan: experiments/E35_late_hsl_full/PLAN.md §3-4.  Scores are computed HERE from per-question probabilities.
  PYTHONPATH=src:scripts .venv/bin/python -m scripts.experiments.e35.score --out experiments/E35_late_hsl_full/results/final.json [--repro-only]
Arms: A1 = W120 (reduced pool S), C = W60 (same S, control), A0 = rank_v4 method OOF (E29 DeepSets W60, full pool; probabilities of the models that built rank_v4).
All fused with Dse (E21 oof_Dse) as 0.5/0.5 (lib.ensemble_probs), EV decode, per-N class-balanced mean, N = 5/10/15/20 equal weight, 5 folds pooled; ONE player-cluster
bootstrap resample (4,000, named seed) shared by all arms / N, so every difference is paired.  Main criterion: Delta(A1F - A0F) >= .012, ci_low > 0, all folds > 0, > 2*sigma(.0025)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

from scripts.experiments.e35.common import *               # noqa: F403
from scripts.experiments.e10 import lib
from gogame.metrics import balanced_weights, rank_scores
from gogame.models import decide_ev

NB = 4000; SIGMA = 0.0025; MIN_DELTA = 0.012
PC = [f"p_{i}" for i in range(10)]
E21 = ROOT / "experiments/E21_strength_estimator/results"
SCH2 = ROOT / ".scratch/e29/school2"
OUT35 = S35 / "out"
REPRO = {"Dse": 0.6084, "A0set": 0.6427, "A0F": 0.6399}      # E29 RESULTS 5-fold pooled balanced; tolerance .0005


class Frame:
    def __init__(self):
        a = np.load(BUNDLE / "aux.npz"); self.aux = a; parts = []
        for k in range(NFOLD):
            parts.append(pd.DataFrame({"qid": a[f"val_qid_f{k}"].astype(str), "player": a[f"val_player_f{k}"], "rank_id": a[f"val_rank_f{k}"].astype(int), "n": a[f"val_n_f{k}"].astype(int), "fold": k}))
        self.q = pd.concat(parts, ignore_index=True)
        self.truth = self.q.rank_id.to_numpy(); self.ns = self.q.n.to_numpy(); self.pl = self.q.player.to_numpy(); self.fold = self.q.fold.to_numpy()
        hd = pd.read_parquet(ROOT / ".scratch/e19/games_hd.parquet"); hd["uid"] = hd["rank"] + "_" + hd["game_id"]; hd = hd.set_index("uid")["hd"].astype(int)
        hdg = hd.reindex(a["uid"].astype(str)).fillna(0).to_numpy(); h2 = []
        for k in range(NFOLD):
            for r in a[f"val_rows_f{k}"]: h2.append(bool((hdg[r[r >= 0]] >= 2).any()))
        self.has_hd2 = np.array(h2)
        split = pd.read_parquet(ROOT / "data/splits/split.parquet")
        assert not (set(a["player_ids"].astype(str)) & set(split.loc[split["is_holdout"], "player_id"])), "H player in bundle"
        assert len(self.q) == sum(len(a[f"val_qid_f{k}"]) for k in range(NFOLD))

    def align(self, df):
        d = df.set_index("qid").loc[self.q.qid]; assert (d["rank_id"].to_numpy() == self.truth).all(); return d[PC].to_numpy(np.float64)

    def arm_probs(self, arm, seed):
        """E35 arm A1/C: per fold chosen width -> setprobs for the seed, concatenated in the aux fold order."""
        out = []
        for k in range(NFOLD):
            W = json.loads((OUT35 / arm / f"select_ds_fold{k}.json").read_text())["chosen_W"]
            z = np.load(OUT35 / arm / f"setprobs_ds_W{W}_fold{k}_s{seed}.npz", allow_pickle=False)
            assert (z["qid"].astype(str) == self.q.qid[self.q.fold == k].to_numpy()).all(), f"{arm} fold {k}: qid order"
            out.append(z["p"].astype(np.float64))
        return np.concatenate(out)

    def a0_probs(self, seed):
        out = []
        for k in range(NFOLD):
            d = SCH2 / ("out_f0save" if k == 0 else "out"); W = json.loads((d / f"select_ds_fold{k}.json").read_text())["chosen_W"]
            z = np.load(d / f"setprobs_ds_W{W}_fold{k}_s{seed}.npz", allow_pickle=False)
            assert (z["qid"].astype(str) == self.q.qid[self.q.fold == k].to_numpy()).all(), f"A0 fold {k}: qid order"
            out.append(z["p"].astype(np.float64))
        return np.concatenate(out)

    def dse(self):
        return self.align(pd.concat([pd.read_parquet(E21 / f"oof_Dse_fold{k}.parquet") for k in range(NFOLD)], ignore_index=True))

    def status(self, arm):
        return [json.loads(str(np.load(OUT35 / arm / f"setprobs_ds_W{json.loads((OUT35 / arm / f'select_ds_fold{k}.json').read_text())['chosen_W']}_fold{k}_s{s}.npz", allow_pickle=False)["info"]))["status"]
                for k in range(NFOLD) for s in (0, 1, 2)]


def per_player(sc, F, mask, kind):
    u = np.unique(F.pl[mask]); P = len(u); out = []
    for n in NS:
        mk = mask & (F.ns == n)
        if not mk.any(): out.append((np.zeros(P), np.zeros(P))); continue
        w = balanced_weights(F.truth[mk]) if kind == "balanced" else np.ones(mk.sum())
        g = np.searchsorted(u, F.pl[mk])
        out.append((np.bincount(g, weights=sc[mk] * w, minlength=P), np.bincount(g, weights=w, minlength=P)))
    return out, P


def boot(scs, F, mask, kind, tag):
    items = {nm: per_player(sc, F, mask, kind)[0] for nm, sc in scs.items()}
    P = per_player(next(iter(scs.values())), F, mask, kind)[1]
    ok = [i for i, (_, d) in enumerate(next(iter(items.values()))) if d.sum() > 0]
    C = np.random.default_rng(sd("bootstrap", tag)).multinomial(P, np.full(P, 1.0 / P), size=NB).astype(np.float64)
    res = {}
    for nm, nd in items.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = np.mean([(C @ nd[i][0]) / (C @ nd[i][1]) for i in ok], axis=0)
        res[nm] = (float(np.mean([nd[i][0].sum() / nd[i][1].sum() for i in ok])), {NS[i]: float(nd[i][0].sum() / nd[i][1].sum()) for i in ok}, bs)
    return res


def ci(v): return {"lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975))}


def diff(res, a, b):
    d = res[a][2] - res[b][2]
    return {"delta": res[a][0] - res[b][0], **ci(d), "boot_sd": float(d.std()), "frac_boot_gt_0": float((d > 0).mean())}


def analyse(repro_only=False):
    F = Frame(); allm = np.ones(len(F.q), bool)
    dse = F.dse(); a0 = [F.a0_probs(s) for s in (0, 1, 2)]; a0m = np.mean(a0, 0)
    P = {"Dse": dse, "A0set": a0m, "A0F": lib.ensemble_probs(dse, a0m, 0.5)}
    arms = {}
    if not repro_only:
        for arm in ("A1", "C"):
            ps = [F.arm_probs(arm, s) for s in (0, 1, 2)]; arms[arm] = ps; m = np.mean(ps, 0)
            P[f"{arm}set"] = m; P[f"{arm}F"] = lib.ensemble_probs(dse, m, 0.5)
    sc = {nm: rank_scores(decide_ev(p), F.truth) for nm, p in P.items()}
    hit = {nm: (decide_ev(p) == F.truth).astype(float) for nm, p in P.items()}
    out = {"n_questions": int(len(F.q)), "n_players": int(len(np.unique(F.pl))), "n_questions_with_hd2": int(F.has_hd2.sum()), "scores": {}, "delta": {}}
    res = boot(sc, F, allm, "balanced", "main")
    repro = {k: res[k][0] for k in REPRO}
    out["repro"] = {k: {"got": repro[k], "expected": REPRO[k], "ok": bool(abs(repro[k] - REPRO[k]) <= 0.0005)} for k in REPRO}
    print("REPRO", json.dumps(out["repro"]))
    if repro_only:
        assert all(v["ok"] for v in out["repro"].values()), "A0 does not reproduce E29's reported OOF numbers"; return out
    pairs = [("A1F", "A0F"), ("A1set", "A0set"), ("A1F", "CF"), ("A1set", "Cset"), ("CF", "A0F"), ("Cset", "A0set"), ("A1F", "Dse"), ("A0F", "Dse")]
    out["status"] = {a: F.status(a) for a in ("A1", "C")}
    chk = {a: [json.loads((OUT35 / a / f"init_check_fold{k}.json").read_text()) for k in range(NFOLD)] for a in ("A1", "C")}
    out["init_check"] = {a: {"all_ok": all(c["ok"] for c in v), "nll": [c["init_es_nll_balanced"] for c in v], "n_val": [c["n_val"] for c in v], "n_fit_games": [c["n_fit_games"] for c in v]} for a, v in chk.items()}
    out["width"] = {a: {str(k): json.loads((OUT35 / a / f"select_ds_fold{k}.json").read_text())["chosen_W"] for k in range(NFOLD)} for a in ("A1", "C")}
    out["width"]["A0"] = {str(k): json.loads(((SCH2 / ("out_f0save" if k == 0 else "out")) / f"select_ds_fold{k}.json").read_text())["chosen_W"] for k in range(NFOLD)}
    strata = {"all": allm, "no_hd2": ~F.has_hd2}
    for nm, mk in strata.items():
        for kind in ("balanced", "raw"):
            if kind == "raw" and nm != "all": continue
            r = boot(sc, F, mk, kind, f"{nm}_{kind}"); key = f"{nm}|{kind}"
            out["scores"][key] = {a: {"score": r[a][0], **ci(r[a][2]), "per_N": r[a][1]} for a in P}
            out["delta"][key] = {f"{x}-{y}": diff(r, x, y) for x, y in pairs}
    hr = boot(hit, F, allm, "balanced", "hit"); out["scores"]["all|balanced|hit"] = {a: hr[a][0] for a in P}
    out["delta_perN"] = {}; out["delta_perfold"] = {}; out["score_perfold"] = {}
    for n in NS:
        r2 = boot(sc, F, F.ns == n, "balanced", f"N{n}"); out["delta_perN"][str(n)] = {f"{x}-{y}": diff(r2, x, y) for x, y in pairs}
    for k in range(NFOLD):
        r3 = boot(sc, F, F.fold == k, "balanced", f"fold{k}"); out["delta_perfold"][str(k)] = {f"{x}-{y}": diff(r3, x, y) for x, y in pairs}; out["score_perfold"][str(k)] = {a: r3[a][0] for a in P}
    seed = {}
    for arm, ps in arms.items():
        seed[arm] = [boot({"s": rank_scores(decide_ev(p), F.truth)}, F, allm, "balanced", "seed")["s"][0] for p in ps]
    seed["A0"] = [boot({"s": rank_scores(decide_ev(p), F.truth)}, F, allm, "balanced", "seed")["s"][0] for p in a0]
    out["seed_scores"] = seed; out["seed_sd"] = {a: float(np.std(v, ddof=1)) for a, v in seed.items()}
    # sensitivity (report only): Dse weight 2/3
    for arm in ("A1",):
        w13 = lib.ensemble_probs(dse, P[f"{arm}set"], 1 / 3); w13_0 = lib.ensemble_probs(dse, a0m, 1 / 3)
        r4 = boot({"A1F_w": rank_scores(decide_ev(w13), F.truth), "A0F_w": rank_scores(decide_ev(w13_0), F.truth)}, F, allm, "balanced", "w13")
        out["sensitivity_set_weight_2_3"] = diff(r4, "A1F_w", "A0F_w")
    # ---- main criterion (PLAN §4)
    m = out["delta"]["all|balanced"]["A1F-A0F"]; folds = [out["delta_perfold"][str(k)]["A1F-A0F"]["delta"] for k in range(NFOLD)]
    crit = {"a_delta_ge_.012": bool(m["delta"] >= MIN_DELTA), "b_ci_low_gt_0": bool(m["lo"] > 0), "c_all_folds_positive": bool(all(f > 0 for f in folds)), "d_delta_gt_2sigma": bool(m["delta"] > 2 * SIGMA),
            "delta": m["delta"], "lo": m["lo"], "hi": m["hi"], "per_fold": folds, "sigma": SIGMA}
    crit["PASS"] = bool(crit["a_delta_ge_.012"] and crit["b_ci_low_gt_0"] and crit["c_all_folds_positive"] and crit["d_delta_gt_2sigma"])
    ok_train = all(s == "OK" for a in ("A1", "C") for s in out["status"][a]) and all(v["all_ok"] for v in out["init_check"].values())
    crit["training_checks_ok"] = bool(ok_train); crit["PASS"] = bool(crit["PASS"] and ok_train)
    out["criterion"] = crit
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--repro-only", action="store_true"); a = ap.parse_args()
    res = analyse(a.repro_only)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    if not a.repro_only:
        for key in ("all|balanced", "no_hd2|balanced", "all|raw"):
            print(key, {k: round(v["score"], 4) for k, v in res["scores"][key].items()})
            print("  delta", {k: f"{v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]" for k, v in res["delta"][key].items()})
        print("CRITERION", json.dumps(res["criterion"]))


if __name__ == "__main__":
    main()
