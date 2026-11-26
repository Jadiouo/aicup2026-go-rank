"""E14 Rank information-ceiling curve.  Plan: experiments/E14_rank_ceiling/PLAN.md (pre-registered)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.e10 import lib, pipeline
from scripts.experiments.e10.lib import CANDS
from gogame.aggregate import aggregate_many
from gogame.constants import SEED
from gogame.coral import fit_coral
from gogame.metrics import balanced_weights, rank_scores
from gogame.models import decide_argmax, decide_ev, fit_lgbm
from gogame.questions import Bag, bag_class_weights, sample_train_bags
from gogame.seeds import derive_seed
import run_e01 as e01

OUT = lib.ROOT / "experiments" / "E14_rank_ceiling"
RES = OUT / "results"
EVAL_NS = (5, 10, 15, 20, 50, 100)
MIN_POOL = 120
REPS = 3
PC = lib.PCOLS


# ------------------------------------------------------------------ questions
def nested_bags(index, players, ns, reps, min_pool, tag, color_mode="max") -> list[Bag]:
    out = []
    for p in sorted(players):
        if p not in index.games:
            continue
        pools = {c: index.games[p][c] for c in (0, 1)}
        c = 0 if len(pools[0]) >= len(pools[1]) else 1
        if len(pools[c]) < min_pool:
            continue
        if color_mode == "random":      # E10-style: random colour among those with >=5 games (diagnostic only)
            ok = [cc for cc in (0, 1) if len(pools[cc]) >= 5]
            c = ok[int(np.random.default_rng(derive_seed(SEED, "e14", tag, "color", p)).integers(len(ok)))]
        g = pools[c]
        for r in range(reps):
            perm = np.random.default_rng(derive_seed(SEED, "e14", tag, p, str(r))).permutation(len(g))
            for n in ns:
                if n <= len(g):
                    out.append(Bag(f"e14{tag}_{p}_r{r}_n{n}", p, index.rank[p], c, n, tuple(sorted(g[i] for i in perm[:n]))))
    return out


def wide_bags(index, players, seed, per_group=4, lo=16, hi=100) -> list[Bag]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    for p in sorted(players):
        for c in (0, 1):
            g = index.games[p][c]
            if len(g) < 20:
                continue
            top = min(len(g), hi)
            for _ in range(per_group):
                n = int(round(np.exp(rng.uniform(np.log(lo), np.log(top)))))
                n = max(lo, min(n, top))
                idx = np.sort(rng.choice(len(g), size=n, replace=False))
                uids = tuple(g[i] for i in idx)
                if uids in seen:
                    continue
                seen.add(uids)
                out.append(Bag(f"wb_{len(out):07d}", p, index.rank[p], c, n, uids))
    return out


def wide_es_bags(index, es_players, fold) -> list[Bag]:
    out = []
    for p in sorted(es_players):
        if p not in index.games:
            continue
        for c in (0, 1):
            g = index.games[p][c]
            for n in (20, 50, 100):
                if len(g) >= n:
                    rng = np.random.default_rng(derive_seed(SEED, "e14", "wes", p, str(c), str(n), str(fold)))
                    idx = np.sort(rng.choice(len(g), size=n, replace=False))
                    out.append(Bag(f"wes_{p}_{c}_{n}", p, index.rank[p], c, n, tuple(g[i] for i in idx)))
    return out


def _pdf(bags, fold, p, praw=None, src="e14"):
    d = e01.probs_frame(bags, fold, praw if praw is not None else p, p)
    d["src"] = src
    d["rep"] = [int(b.qid.split("_r")[-1].split("_")[0]) if b.qid.startswith("e14") else -1 for b in bags]
    return d


class Fold:
    def __init__(self, ctx, k):
        self.ctx, self.k = ctx, k
        self.fs = lib.fold_sets(ctx.nonh, k)
        lib.assert_fold_disjoint(self.fs, ctx.H)
        self.index = ctx.index_all
        self.eval = nested_bags(self.index, self.fs["val"], EVAL_NS, REPS, MIN_POOL, "ev")
        pl = {b.player_id for b in self.eval}
        assert pl and not (pl & ctx.H) and not (pl & (self.fs["train_all"] | self.fs["es"] | self.fs["fit"]))
        q2f = pd.read_parquet(lib.OUT / "oof" / "questions_q2.parquet")
        self.q2 = [b for b in e01.frame_to_bags(q2f) if b.player_id in self.fs["val"]]
        self.val = self.eval + self.q2
        self.src = np.array(["e14"] * len(self.eval) + ["e10q2"] * len(self.q2))
        self.fit_in = sorted(p for p in self.fs["fit"] if p in self.index.games)
        self.tb15 = sample_train_bags(self.index, self.fit_in, 8, (5, 15), seed=CANDS["A"].train_seed(k))
        self.tbW = self.tb15 + wide_bags(self.index, self.fit_in, derive_seed(SEED, "e14", "wide_bags", f"fold{k}"))
        for tb in (self.tb15, self.tbW):
            assert {b.player_id for b in tb} <= self.fs["fit"]
            assert not ({u for b in tb for u in b.uids} & {u for b in self.val for u in b.uids})
        print(f"fold {k}: eval players {len(pl)}, eval bags {len(self.eval)}, e10q2 bags {len(self.q2)}, tb15 {len(self.tb15)}, tbW {len(self.tbW)}", flush=True)

    def save(self, name, p, praw=None):
        df = _pdf(self.val, self.k, p, praw)
        df["src"] = self.src
        RES.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RES / f"probs_{name}_fold{self.k}.parquet", index=False)

    def repro(self, name, p, ref_name):
        f = lib.OUT / "oof" / f"oof_{ref_name}_fold{self.k}.parquet"
        ref = pd.read_parquet(f).set_index("qid")
        q = [b.qid for b in self.q2]
        pm = p[len(self.eval):]
        d = float(np.abs(pm - ref.loc[q, PC].to_numpy()).max())
        rec = {"model": name, "ref": f.name, "n": len(q), "max_abs_diff": d}
        (RES / f"repro_{name}_fold{self.k}.json").write_text(json.dumps(rec))
        print("repro", rec, flush=True)


def fit(k, models):
    t0 = time.time()
    ctx = pipeline.Ctx(lib.CACHE, False, None)
    F = Fold(ctx, k)
    RES.mkdir(parents=True, exist_ok=True)
    if "CORAL" in models:
        Xv = aggregate_many(F.val, ctx.hs, ctx.sp_h)
        for name, tb in (("CORAL15", F.tb15), ("CORALW", F.tbW)):
            if (RES / f"probs_{name}_fold{k}.parquet").exists():
                continue
            X = aggregate_many(tb, ctx.hs, ctx.sp_h)
            m = fit_coral(X, np.array([b.rank_id for b in tb]), bag_class_weights(tb, "inv_players"), chunk=None)
            del X
            praw, p = m.predict_proba(Xv, raw=True), m.predict_proba(Xv)
            F.save(name, p, praw)
            if name == "CORAL15":
                F.repro(name, p, "A")
            print(f"{name} done {time.time() - t0:.0f}s", flush=True)
    if "BAYES" in models:
        bayes(ctx, F)
    for tag, tb in (("LGBM15", F.tb15), ("LGBMW", F.tbW)):
        if tag in models and not (RES / f"probs_{tag}_fold{k}.parquet").exists():
            es = lib.make_es_bags(F.index, F.fs["es"], k, False)
            if tag == "LGBMW":
                es = es + wide_es_bags(F.index, F.fs["es"], k)
            ((praw, p),), info = pipeline.fit_m03(ctx, tb, es, [F.val])
            F.save(tag, p, praw)
            if tag == "LGBM15":
                F.repro(tag, p, "M03A")
            (RES / f"meta_{tag}_fold{k}.json").write_text(json.dumps({**info, "n_train_bags": len(tb), "n_es_bags": len(es), "seconds": round(time.time() - t0)}))
            print(f"{tag} done {info} {time.time() - t0:.0f}s", flush=True)
    print(f"fold {k} models {models} finished in {time.time() - t0:.0f}s, peak RSS {pipeline.peak_rss_mb():.0f} MB", flush=True)


# ------------------------------------------------------------------ BAYES
def _game_X(ctx, uids, color_of):
    rows = ctx.hs.rows(uids)
    return np.hstack([ctx.hs.dense[rows], np.array([color_of[u] for u in uids], np.float32)[:, None]])


def bayes(ctx, F):
    k = F.k
    if (RES / f"probs_BAYES1_fold{k}.parquet").exists():
        return
    t0 = time.time()
    meta = ctx.meta.set_index("uid")
    color_of = {u: int(c) for u, c in zip(meta.index, meta["color"])}
    rng = np.random.default_rng(derive_seed(SEED, "e14", "bayes_games", f"fold{k}"))

    def games_of(players, cap):
        out = []
        for p in sorted(players):
            if p not in F.index.games:
                continue
            g = F.index.games[p][0] + F.index.games[p][1]
            if len(g) > cap:
                g = [g[i] for i in rng.choice(len(g), size=cap, replace=False)]
            out.append((p, g))
        return out
    tr = games_of(F.fs["fit"], 80)
    es = games_of(F.fs["es"], 30)
    ppr = {}
    for p, _ in tr:
        ppr.setdefault(F.index.rank[p], set()).add(p)
    ppr = {r: len(s) for r, s in ppr.items()}
    tu = [u for p, g in tr for u in g]
    ty = np.array([F.index.rank[p] for p, g in tr for _ in g])
    tw = np.array([1.0 / (ppr[F.index.rank[p]] * len(g)) for p, g in tr for _ in g])
    tw /= tw.mean()
    eu = [u for p, g in es for u in g]
    ey = np.array([F.index.rank[p] for p, g in es for _ in g])
    ew = np.array([1.0 / len(g) for p, g in es for _ in g])
    ew /= ew.mean()
    assert not ({u for u in tu + eu} & {u for b in F.val for u in b.uids})
    Xtr, Xes = _game_X(ctx, tu, color_of), _game_X(ctx, eu, color_of)
    m = fit_lgbm(Xtr, ty, Xes, ey, {"num_threads": 4, "learning_rate": 0.1, "max_bin": 63, "num_boost_round": 400, "early_stopping_rounds": 30, "log_every": 0},
                 derive_seed(SEED, "e14", "bayes_lgbm", f"fold{k}"), tw, ew)
    print(f"bayes game model: {len(tu)} train games, {len(eu)} es games, best_iter {m.best_iteration}, {time.time() - t0:.0f}s", flush=True)
    del Xtr

    def bag_logsum(bags):
        uu = sorted({u for b in bags for u in b.uids})
        lp = np.log(np.clip(m.predict_proba(_game_X(ctx, uu, color_of)).astype(np.float64), 1e-9, None))
        pos = {u: i for i, u in enumerate(uu)}
        return np.stack([lp[[pos[u] for u in b.uids]].sum(axis=0) for b in bags])
    # calibration of alpha_N on es players (nested bags, R=2)
    cal = nested_bags(F.index, F.fs["es"], EVAL_NS, 2, 5, "cal")
    S = bag_logsum(cal)
    y = np.array([b.rank_id for b in cal])
    nn = np.array([b.n for b in cal])
    grid = np.logspace(-3, 0.5, 70)
    alpha = {}
    for n in EVAL_NS:
        s, yy = S[nn == n], y[nn == n]
        if len(yy) < 10:
            alpha[n] = float("nan")
            continue
        nll = []
        for a in grid:
            z = a * s
            z = z - z.max(1, keepdims=True)
            lse = np.log(np.exp(z).sum(1))
            nll.append(float(np.mean(lse - z[np.arange(len(yy)), yy])))
        alpha[n] = float(grid[int(np.argmin(nll))])
        alpha[f"n_cal_{n}"] = int(len(yy))
    print("alpha_N:", alpha, flush=True)
    rc = nested_bags(F.index, {b.player_id for b in F.eval}, (5, 10, 15, 20), 10, MIN_POOL, "rc", "random")
    Src = bag_logsum(rc)
    arc = np.array([alpha[b.n] for b in rc])[:, None]
    z = arc * Src
    z = z - z.max(1, keepdims=True)
    prc = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    dfrc = e01.probs_frame(rc, k, prc, prc)
    dfrc["src"] = "e14rc"
    dfrc["rep"] = -1
    dfrc.to_parquet(RES / f"probs_BAYEScRC_fold{k}.parquet", index=False)
    Sv = bag_logsum(F.val)
    nv = np.array([b.n for b in F.val])

    def soft(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    F.save("BAYES1", soft(Sv))
    # alpha for N values outside EVAL_NS (e10q2 bags): nearest-N alpha (only used for src e10q2; not reported)
    pc = np.empty_like(Sv)
    for i, n in enumerate(nv):
        a = alpha.get(int(n), None)
        if a is None or not np.isfinite(a):
            a = alpha[min((q for q in EVAL_NS if np.isfinite(alpha[q])), key=lambda q: abs(q - n))]
        pc[i] = a * Sv[i]
    F.save("BAYESc", soft(pc))
    (RES / f"meta_BAYES_fold{k}.json").write_text(json.dumps({"alpha": {str(a): v for a, v in alpha.items()}, "best_iteration": int(m.best_iteration),
                                                             "n_train_games": len(tu), "n_es_games": len(eu), "seconds": round(time.time() - t0)}))


# ------------------------------------------------------------------ report
def _boot_counts(P, n_boot=4000):
    rng = np.random.default_rng(derive_seed(SEED, "e14", "bootstrap"))
    return rng.multinomial(P, np.full(P, 1.0 / P), size=n_boot).astype(np.float64)


def _agg(vals, w, gi, P):
    return np.bincount(gi, weights=vals * w, minlength=P), np.bincount(gi, weights=w, minlength=P)


def _ci(num, den, C):
    point = num.sum() / den.sum()
    b = (C @ num) / (C @ den)
    return float(point), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def report():
    models = ["CORAL15", "LGBM15", "D15", "CORALW", "LGBMW", "DW", "BAYES1", "BAYESc"]
    frames = {}
    for m in ("CORAL15", "LGBM15", "CORALW", "LGBMW", "BAYES1", "BAYESc"):
        fs = sorted(RES.glob(f"probs_{m}_fold*.parquet"))
        if fs:
            frames[m] = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    for d, a, b in (("D15", "LGBM15", "CORAL15"), ("DW", "LGBMW", "CORALW")):
        if a in frames and b in frames:
            A, B = frames[a], frames[b]
            common = sorted(set(A["qid"]) & set(B["qid"]))
            A, B = A.set_index("qid").loc[common], B.set_index("qid").loc[common]
            D = A.copy()
            D[PC] = lib.ensemble_probs(A[PC].to_numpy(), B[PC].to_numpy())
            frames[d] = D.reset_index()
    folds_of = {m: sorted(set(f["fold"])) for m, f in frames.items()}
    print("folds per model:", folds_of)
    common_folds = sorted(set.intersection(*[set(v) for m, v in folds_of.items() if m in ("CORAL15", "LGBM15", "CORALW", "LGBMW", "BAYESc")] or [set()]))
    print("common folds (all main models):", common_folds)
    rows, deltas = [], []
    pop = {}
    for scope, fl in (("common", common_folds), ("fold0", [0])):
        if not fl:
            continue
        ev = {m: f[(f["src"] == "e14") & f["fold"].isin(fl)].sort_values("qid").reset_index(drop=True) for m, f in frames.items()}
        ev = {m: f for m, f in ev.items() if len(f)}
        base = next(iter(ev.values()))
        players = sorted(set(base["player_id"]))
        pidx = {p: i for i, p in enumerate(players)}
        P = len(players)
        C = _boot_counts(P)
        pop[scope] = P
        gi_all = base["player_id"].map(pidx).to_numpy()
        truth_all = base["rank_id"].to_numpy()
        ns_all = base["n"].to_numpy()
        per = {}
        for m, f in ev.items():
            if list(f["qid"]) != list(base["qid"]):
                keep = sorted(set(f["qid"]) & set(base["qid"]))
                f = f.set_index("qid").loc[keep].reset_index()
                assert len(f) == len(base), f"{m}: question set differs"
            p = f[PC].to_numpy()
            for dec, fn in (("ev", decide_ev), ("argmax", decide_argmax)):
                pred = fn(p)
                for n in EVAL_NS:
                    msk = ns_all == n
                    s = rank_scores(pred[msk], truth_all[msk])
                    hit = (pred[msk] == truth_all[msk]).astype(float)
                    gi = gi_all[msk]
                    bw = balanced_weights(truth_all[msk])
                    one = np.ones(msk.sum())
                    rec = {"scope": scope, "model": m, "decode": dec, "N": n, "n_players": P, "n_questions": int(msk.sum())}
                    store = {}
                    for key, v, w in (("raw", s, one), ("balanced", s, bw), ("hit", hit, one)):
                        num, den = _agg(v, w, gi, P)
                        store[key] = (num, den)
                        pt, lo, hi = _ci(num, den, C)
                        rec[key], rec[key + "_lo"], rec[key + "_hi"] = pt, lo, hi
                    rows.append(rec)
                    per[(m, dec, n)] = store
        # curve increments and key contrasts (EV decode)
        def delta(label, a, b, key):
            na, da = per[a][key]
            nb, db = per[b][key]
            # difference of two ratios: resample once
            pa, pb = (C @ na) / (C @ da), (C @ nb) / (C @ db)
            pt = na.sum() / da.sum() - nb.sum() / db.sum()
            dd = pa - pb
            deltas.append({"scope": scope, "contrast": label, "metric": key, "delta": float(pt), "lo": float(np.quantile(dd, 0.025)), "hi": float(np.quantile(dd, 0.975))})
        for m in ev:
            for key in ("raw", "balanced"):
                delta(f"{m}: N=100 - N=20", (m, "ev", 100), (m, "ev", 20), key)
                delta(f"{m}: N=100 - N=50", (m, "ev", 100), (m, "ev", 50), key)
                delta(f"{m}: N=20 - N=5", (m, "ev", 20), (m, "ev", 5), key)
        for n in EVAL_NS:
            for a, b in (("BAYESc", "D15"), ("DW", "D15"), ("BAYESc", "DW"), ("CORALW", "CORAL15"), ("LGBMW", "LGBM15")):
                if a in ev and b in ev:
                    delta(f"{a} - {b} @N={n}", (a, "ev", n), (b, "ev", n), "balanced")
        # reference line vs E10 D on E10 Q2 (same players)
        ref = []
        dfs = [pd.read_parquet(lib.OUT / "oof" / f"oof_D_fold{k}.parquet") for k in fl]
        e10 = pd.concat(dfs)
        e10 = e10[(e10["qset"] == "q2") & e10["player_id"].isin(set(players))].sort_values("qid")
        if "D15" in ev:
            mine = ev["D15"]
            for n in (5, 10, 15):
                a = e10[e10["n"] == n]
                pa = decide_ev(a[PC].to_numpy())
                sa = rank_scores(pa, a["rank_id"].to_numpy())
                ma = mine[mine["n"] == n]
                sb = rank_scores(decide_ev(ma[PC].to_numpy()), ma["rank_id"].to_numpy())
                for key in ("raw", "balanced"):
                    wa = np.ones(len(a)) if key == "raw" else balanced_weights(a["rank_id"].to_numpy())
                    wb = np.ones(len(ma)) if key == "raw" else balanced_weights(ma["rank_id"].to_numpy())
                    na, da = _agg(sa, wa, a["player_id"].map(pidx).to_numpy(), P)
                    nb, db = _agg(sb, wb, ma["player_id"].map(pidx).to_numpy(), P)
                    ptA, loA, hiA = _ci(na, da, C)
                    ptB, loB, hiB = _ci(nb, db, C)
                    dd = (C @ nb) / (C @ db) - (C @ na) / (C @ da)
                    ref.append({"scope": scope, "N": n, "metric": key, "E10_D_on_E10Q2": ptA, "E10_lo": loA, "E10_hi": hiA, "n_E10": int(len(a)),
                                "E14_D15_on_E14": ptB, "E14_lo": loB, "E14_hi": hiB, "diff_E14_minus_E10": ptB - ptA,
                                "diff_lo": float(np.quantile(dd, 0.025)), "diff_hi": float(np.quantile(dd, 0.975))})
        pd.DataFrame(ref).to_csv(RES / f"reference_{scope}.csv", index=False)
    # E10 D on all validation players (not only heavy), for context
    allp = pd.concat([pd.read_parquet(lib.OUT / "oof" / f"oof_D_fold{k}.parquet") for k in common_folds or [0]])
    allp = allp[allp["qset"] == "q2"]
    ctxrows = []
    for n in (5, 10, 15, 20):
        a = allp[allp["n"] == n]
        s = rank_scores(decide_ev(a[PC].to_numpy()), a["rank_id"].to_numpy())
        ctxrows.append({"N": n, "n_questions": len(a), "raw": float(s.mean()), "balanced": float((s * balanced_weights(a["rank_id"].to_numpy())).mean())})
    pd.DataFrame(ctxrows).to_csv(RES / "e10_D_all_players_context.csv", index=False)
    pd.DataFrame(rows).to_csv(RES / "curve.csv", index=False)
    pd.DataFrame(deltas).to_csv(RES / "deltas.csv", index=False)
    (RES / "report_meta.json").write_text(json.dumps({"folds_per_model": folds_of, "common_folds": common_folds, "n_players": pop, "n_boot": 4000}))
    print("report written", pop)
