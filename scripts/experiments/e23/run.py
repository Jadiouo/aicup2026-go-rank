"""E23: similarity normalisation (CSLS / cohort z-norm / k-reciprocal) on Player M01 and E04v2.  CPU only.
Protocol frozen in experiments/E23_player_norm/PLAN.md.
Stages: prep (Z of tune players + consistency check) -> tune (all configs, writes selection.json) -> main (selected configs only) -> report.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from gogame.aggregate import L0Store, fit_sparse_selector  # noqa: E402
from gogame.gametable import load_table  # noqa: E402
from gogame.metrics import paired_bootstrap, player_scores  # noqa: E402
from gogame import player_rerank as prr  # noqa: E402
from gogame.questions import build_player_index, build_player_pool_sim  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402
from gogame.split import read_split  # noqa: E402

E05 = ROOT / ".scratch" / "e05"
SCR = ROOT / ".scratch" / "e23"
OUT = ROOT / "experiments" / "E23_player_norm" / "results"
CACHE = ROOT / "data" / "cache"
NS = (5, 10, 15, 20)
K = prr.K
ND = prr.N_DENSE
BETAS = (0.25, 0.5, 1.0, 2.0, 100.0)


def S(*k):
    return derive_seed(2026, "E23", *k)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ------------------------------------------------------------------ configs
def all_configs():
    cfg = []
    for k, a in itertools.product((5, 10, 20), (0.25, 0.5, 1.0)):
        cfg.append(("csls_cc", k, a)); 
    for k, a in itertools.product((5, 10, 20), (0.25, 0.5, 1.0)):
        cfg.append(("csls_qq", k, a))
    for m in (20, 50, 399):
        cfg.append(("znorm", m, 0.0))
    for k, l in itertools.product((5, 10, 20), (0.3, 0.5, 0.8)):
        cfg.append(("krecip", k, l))
    return cfg


def norm_score(fam, p1, p2, C, G, ns):
    """Normalised score matrix (nq,P), higher = better."""
    P = C.shape[1]
    Gd = np.where(np.eye(P, dtype=bool), -np.inf, G)
    if fam == "csls_cc":
        r = -np.sort(-Gd, axis=1)[:, :p1].mean(1)
        return C - p2 * r[None, :]
    if fam == "csls_qq":
        S_ = np.empty_like(C)
        for n in np.unique(ns):
            idx = np.flatnonzero(ns == n)
            Cn = C[idx]
            T = -np.sort(-Cn, axis=0)[:p1 + 1]                   # (k+1,P)
            sk, sk1, kth = T[:p1].sum(0), T.sum(0), T[p1 - 1]
            incl = Cn >= kth[None, :]
            r = np.where(incl, (sk1[None, :] - Cn) / p1, sk[None, :] / p1)   # leave-one-out top-k mean
            S_[idx] = Cn - p2 * r
        return S_
    if fam == "znorm":
        srt = -np.sort(-Gd, axis=1)[:, :p1]
        mu, sd = srt.mean(1), srt.std(1) + 1e-9
        return (C - mu[None, :]) / sd[None, :]
    if fam == "krecip":
        k, lam = p1, p2
        kth = -np.sort(-Gd, axis=1)[:, k - 1]                     # c's k-th nearest cand-cand cosine
        A = Gd >= kth[:, None]
        R = ((A & A.T) | np.eye(P, dtype=bool)).astype(np.float32)
        ordq = np.argsort(-C, axis=1, kind="stable")[:, :k]
        topq = np.zeros(C.shape, bool)
        np.put_along_axis(topq, ordq, True, axis=1)
        M = topq & (C >= kth[None, :])
        Mf = M.astype(np.float32)
        inter = Mf @ R.T + Mf                                      # + element q itself (in both iff mutual)
        szq = Mf.sum(1, keepdims=True) + 1.0
        szc = R.sum(1)[None, :] + Mf
        dj = 1.0 - inter / np.maximum(szq + szc - inter, 1e-9)
        return -(lam * dj + (1 - lam) * (1.0 - C))
    raise ValueError(fam)


def zrow(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def w_rank(r):
    return np.where(r <= 5, np.exp(-(r - 1.0)), 0.0)


def rank_of_truth(Smat, truth):
    st = Smat[np.arange(len(truth)), truth]
    return 1 + (Smat > st[:, None]).sum(1)


# ------------------------------------------------------------------ data
class Ds:
    def __init__(self, Z, uids, sel, meta):
        self.Z, self.sel = Z, sel
        self.rowof = {u: i for i, u in enumerate(uids)}
        m = meta[meta.uid.isin(self.rowof)]
        self.index = build_player_index(m.assign(color_id=m["color"]))

    def rows(self, uids):
        return np.fromiter((self.rowof[u] for u in uids), dtype=np.int64, count=len(uids))


def sets():
    sp = read_split(ROOT / "data/splits/split.parquet")
    e5 = json.loads((E05 / "sets_players.json").read_text())
    H, dev, fit = set(e5["H"]), set(e5["dev"]), set(e5["fit"])
    assert set(sp.loc[sp.is_holdout, "player_id"]) == H
    nh = sp[~sp.is_holdout]
    tune = sorted(set(nh.loc[nh.n_black >= 55, "player_id"]) - fit - dev)
    assert not (set(tune) & H) and not (set(dev) & H) and not (set(tune) & dev) and not (set(tune) & fit)
    return {"H": H, "dev": sorted(dev), "fit": fit, "tune": tune}


def stage_prep():
    SCR.mkdir(parents=True, exist_ok=True)
    st = sets()
    meta = pd.read_parquet(CACHE / "games_meta.parquet")
    table = load_table(CACHE / "l0")
    assert list(table.uids) == meta["uid"].tolist()
    store = L0Store.from_table(table)
    nonH = ~meta["player_id"].isin(st["H"]).to_numpy()
    D = store.dense[nonH].astype(np.float64)
    mu, sd = np.nanmean(D, axis=0), np.nanstd(D, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    SP = store.sparse[np.flatnonzero(nonH)]
    cols = fit_sparse_selector(SP, 0.005)
    Sp = SP[:, cols]
    smu = np.asarray(Sp.mean(axis=0)).ravel()
    ssd = np.sqrt(np.maximum(np.asarray(Sp.multiply(Sp).mean(axis=0)).ravel() - smu ** 2, 0))
    ssd = np.where(ssd > 0, ssd, 1.0)
    del D, SP, Sp
    pid, col = meta["player_id"].to_numpy(), meta["color"].to_numpy()
    rows = np.flatnonzero(np.isin(pid, st["tune"]) & (col == 0))
    # consistency probe: 3000 dev rows must equal E05's Z.npy
    z5 = np.load(E05 / "Z.npy", mmap_mode="r")
    u5 = json.loads((E05 / "uids.json").read_text())
    pos5 = {u: i for i, u in enumerate(u5)}
    drows = np.flatnonzero(np.isin(pid, st["dev"]) & (col == 0))
    drows = drows[np.random.default_rng(0).permutation(len(drows))[:3000]]
    allr = np.concatenate([rows, drows])
    Z = np.empty((len(allr), ND + len(cols)), np.float32)
    for a in range(0, len(allr), 20000):
        r = allr[a:a + 20000]
        Zd = np.nan_to_num((store.dense[r] - mu) / sd, nan=0.0)
        Zs = (store.sparse[r][:, cols].toarray() - smu) / ssd
        Z[a:a + 20000] = np.hstack([Zd, Zs])
    ref = np.stack([z5[pos5[u]] for u in meta["uid"].to_numpy()[drows]])
    err = float(np.abs(Z[len(rows):] - ref).max())
    assert err < 1e-4, f"Z mismatch vs E05: {err}"
    np.save(SCR / "Z_tune.npy", Z[:len(rows)])
    (SCR / "uids_tune.json").write_text(json.dumps(meta["uid"].to_numpy()[rows].tolist()))
    print("prep ok", Z[:len(rows)].shape, "max|dZ| vs E05", err, "tune players", len(st["tune"]), flush=True)


def get_ds(which):
    meta = pd.read_parquet(CACHE / "games_meta.parquet")
    sel = np.load(E05 / "sel.npy")
    if which == "tune":
        return Ds(np.load(SCR / "Z_tune.npy"), json.loads((SCR / "uids_tune.json").read_text()), sel, meta)
    return Ds(np.load(E05 / "Z.npy", mmap_mode="r"), json.loads((E05 / "uids.json").read_text()), sel, meta)


# ------------------------------------------------------------------ evaluation
def eval_pool(ds, sim, model, cfgs, fams_for_e04, st):
    pool = list(sim.players)
    qs = sim.queries
    assert all(p in st for p in pool)
    for q in qs:
        assert not (set(q.uids) & set(sim.library[q.player_id]))
    lib_z = [np.asarray(ds.Z[ds.rows(sim.library[p])]) for p in pool]
    q_z = [np.asarray(ds.Z[ds.rows(q.uids)]) for q in qs]
    pos = {p: i for i, p in enumerate(pool)}
    truth = np.array([pos[q.player_id] for q in qs])
    f = prr.candidate_features(lib_z, q_z, ds.sel, K, ND)
    C, top, ns = f["cos"], f["top"], f["ns"]
    Cen = unit(np.stack([z.mean(0) for z in lib_z]))
    G = Cen @ Cen.T
    e = model.predict(f["X"], num_threads=4).reshape(len(qs), K)
    truth_ids = np.asarray(pool)[truth]
    out = {"qplayer": np.array([q.player_id for q in qs]), "n": ns,
           "recall": (top == truth[:, None]).any(1).astype(float)}
    # baselines (also checked against the official scorer)
    out["m01|base"] = w_rank(rank_of_truth(C, truth))
    o = np.argsort(-e, axis=1, kind="stable")
    rk_e = np.where(top == truth[:, None], 1, 0)
    ordd = np.take_along_axis(top, o, axis=1)
    hit = ordd == truth[:, None]
    out["e04v2|base"] = np.where(hit.any(1), np.exp(-hit.argmax(1).astype(float)), 0.0) * (hit.argmax(1) < 5)
    chk1 = player_scores(np.asarray(pool)[prr.top5_indices(top, None)], truth_ids)
    chk2 = player_scores(np.asarray(pool)[prr.top5_indices(top, e.reshape(-1))], truth_ids)
    assert np.allclose(out["m01|base"], chk1) and np.allclose(out["e04v2|base"], chk2), "scorer mismatch"
    ze = zrow(e)
    for c in cfgs:
        Sm = norm_score(*c, C, G, ns)
        out[f"m01|{c}"] = w_rank(rank_of_truth(Sm, truth))
        if c in fams_for_e04:
            zs = zrow(np.take_along_axis(Sm, top, axis=1))
            for b in BETAS:
                fin = ze + b * zs
                o2 = np.argsort(-fin, axis=1, kind="stable")[:, :5]
                hit2 = np.take_along_axis(top, o2, axis=1) == truth[:, None]
                out[f"e04v2|{c}|{b}"] = np.where(hit2.any(1), np.exp(-hit2.argmax(1).astype(float)), 0.0)
    # shuffled-label control for the baseline and first config
    rng = np.random.default_rng(S("shuf"))
    tr2 = rng.integers(0, len(pool), len(qs))
    out["ctl|m01"] = w_rank(rank_of_truth(C, tr2))
    return pd.DataFrame(out)


def run_set(which, cfgs, e04_cfgs):
    import lightgbm as lgb
    fro = json.loads((E05 / "frozen.json").read_text())
    assert hashlib.sha256((E05 / "model.txt").read_bytes()).hexdigest() == fro["model_sha256"]
    model = lgb.Booster(model_file=str(E05 / "model.txt"))
    st = sets()
    ds = get_ds(which)
    if which == "tune":
        players, lib, nd = st["tune"], 50, 10
    else:
        players, lib, nd = st["dev"], 100, 20
        assert not (set(players) & st["H"]) and not (set(players) & st["fit"])
    sims = build_player_pool_sim(ds.index, players, 400, nd, lib, NS, seed=S(which))
    frames = []
    t0 = time.perf_counter()
    for s in sims:
        df = eval_pool(ds, s, model, cfgs, set(e04_cfgs), set(players))
        df["draw"] = s.draw
        frames.append(df)
        print(which, "draw", s.draw, f"{time.perf_counter() - t0:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def stage_tune():
    cfgs = all_configs()
    df = run_set("tune", cfgs, cfgs)
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(SCR / "tune_scores.parquet")
    m = df.drop(columns=["qplayer", "n", "recall", "draw"]).mean()
    sel = {"base_m01": float(m["m01|base"]), "base_e04v2": float(m["e04v2|base"]), "m01": {}, "e04v2": {}}
    for fam in ("csls_cc", "csls_qq", "znorm", "krecip"):
        fc = [c for c in cfgs if c[0] == fam]
        best = max(fc, key=lambda c: (round(m[f"m01|{c}"], 12), tuple(-x for x in c[1:])))
        sel["m01"][fam] = {"cfg": list(best), "tune_score": float(m[f"m01|{best}"])}
        cand = [(c, b) for c in fc for b in BETAS]
        bc, bb = max(cand, key=lambda cb: (round(m[f"e04v2|{cb[0]}|{cb[1]}"], 12), tuple(-x for x in cb[0][1:]), -cb[1]))
        sel["e04v2"][fam] = {"cfg": list(bc), "beta": bb, "tune_score": float(m[f"e04v2|{bc}|{bb}"])}
    prim = max(sel["e04v2"], key=lambda f: sel["e04v2"][f]["tune_score"])
    sel["primary"] = prim
    sel["primary_tune_delta_e04v2"] = sel["e04v2"][prim]["tune_score"] - sel["base_e04v2"]
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


def stage_main():
    sel = json.loads((OUT / "selection.json").read_text())
    if (SCR / "main_scores.parquet").exists():
        raise SystemExit("main already evaluated; refusing to re-run")
    m01c = [(v["cfg"][0], int(v["cfg"][1]), float(v["cfg"][2])) for v in sel["m01"].values()]
    e04c = [(v["cfg"][0], int(v["cfg"][1]), float(v["cfg"][2])) for v in sel["e04v2"].values()]
    cfgs = list(dict.fromkeys(m01c + e04c))
    df = run_set("main", cfgs, e04c)
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(SCR / "main_scores.parquet")


def stage_report():
    sel = json.loads((OUT / "selection.json").read_text())
    df = pd.read_parquet(SCR / "main_scores.parquet")
    tdf = pd.read_parquet(SCR / "tune_scores.parquet")
    def key_m(v): return f"m01|{(v['cfg'][0], int(v['cfg'][1]), float(v['cfg'][2]))}"
    def key_e(v): return f"e04v2|{(v['cfg'][0], int(v['cfg'][1]), float(v['cfg'][2]))}|{v['beta']}"
    def bs(d, a, b, tag):
        r = paired_bootstrap(d[a].to_numpy(), d[b].to_numpy(), d["qplayer"].to_numpy(), n_boot=4000, seed=S("boot", tag))
        return {"new": float(d[a].mean()), "base": float(d[b].mean()), "delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high}
    res = {"selection": sel, "n_questions": int(len(df)), "n_players": int(df.qplayer.nunique()), "n_draws": int(df.draw.nunique()),
           "tune_questions": int(len(tdf)), "tune_players": int(tdf.qplayer.nunique()),
           "base_main": {"m01": float(df["m01|base"].mean()), "e04v2": float(df["e04v2|base"].mean()),
                         "recall_top20": float(df.recall.mean()), "shuffle_ctl_m01": float(df["ctl|m01"].mean())},
           "main": {}, "tune_delta_point": {}}
    for fam, v in sel["e04v2"].items():
        res["main"][f"e04v2+{fam}"] = {"cfg": v["cfg"], "beta": v["beta"], "all": bs(df, key_e(v), "e04v2|base", f"e{fam}"),
                                       "per_n": {str(n): bs(df[df.n == n], key_e(v), "e04v2|base", f"e{fam}{n}") for n in NS}}
        res["tune_delta_point"][f"e04v2+{fam}"] = float(tdf[key_e(v)].mean() - tdf["e04v2|base"].mean())
    for fam, v in sel["m01"].items():
        res["main"][f"m01+{fam}"] = {"cfg": v["cfg"], "all": bs(df, key_m(v), "m01|base", f"m{fam}"),
                                     "per_n": {str(n): bs(df[df.n == n], key_m(v), "m01|base", f"m{fam}{n}") for n in NS}}
        res["tune_delta_point"][f"m01+{fam}"] = float(tdf[key_m(v)].mean() - tdf["m01|base"].mean())
    p = res["main"][f"e04v2+{sel['primary']}"]["all"]
    res["primary"] = {"family": sel["primary"], "delta": p["delta"], "ci_low": p["ci_low"], "ci_high": p["ci_high"], "pass": bool(p["ci_low"] > 0),
                      "m01_same_family": res["main"][f"m01+{sel['primary']}"]["all"]}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["primary"], indent=1), json.dumps(res["base_main"]))


if __name__ == "__main__":
    {"prep": stage_prep, "tune": stage_tune, "main": stage_main, "report": stage_report}[sys.argv[1]]()
