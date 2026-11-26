"""E23b: CSLS-qq at contest density (400 queries/pool, N uniform 5-20, one query per pool player). CPU only.
Protocol frozen in experiments/E23_player_norm/PLAN_b.md. Usage: run_b.py main"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run as R  # noqa: E402  (E23 helpers; sets thread env)
from run import ROOT, E05, S, prr, paired_bootstrap, player_scores, norm_score, zrow, w_rank, rank_of_truth, unit, build_player_pool_sim  # noqa: E402

OUT = ROOT / "experiments" / "E23_player_norm" / "results_b"
K_, A_ = 5, 1.0
BETA = {"qq": 1.0, "cc": 0.25}
BINS = [(5, 8), (9, 12), (13, 16), (17, 20)]


def binid(ns):
    return np.array([next(i for i, (lo, hi) in enumerate(BINS) if lo <= n <= hi) for n in ns])


def thin(sim, rng):
    """One query per pool player, N uniform in 5..20 (from the 16-N full sim)."""
    by = {}
    for q in sim.queries:
        by.setdefault(q.player_id, []).append(q)
    qs = []
    for p in sim.players:
        n = int(rng.integers(5, 21))
        c = [q for q in by[p] if q.n == n]
        assert len(c) == 1, (p, n)
        qs.append(c[0])
    return dataclasses.replace(sim, queries=tuple(qs))


def eval_pool(ds, sim, model, st):
    pool = list(sim.players)
    qs = sim.queries
    assert len(qs) == 400 and all(p in st for p in pool)
    for q in qs:
        assert not (set(q.uids) & set(sim.library[q.player_id])) and len(sim.library[q.player_id]) == 100
    lib_z = [np.asarray(ds.Z[ds.rows(sim.library[p])]) for p in pool]
    q_z = [np.asarray(ds.Z[ds.rows(q.uids)]) for q in qs]
    pos = {p: i for i, p in enumerate(pool)}
    truth = np.array([pos[q.player_id] for q in qs])
    f = prr.candidate_features(lib_z, q_z, ds.sel, prr.K, prr.N_DENSE)
    C, top, ns = f["cos"], f["top"], f["ns"]
    G = unit(np.stack([z.mean(0) for z in lib_z])); G = G @ G.T
    e = model.predict(f["X"], num_threads=4).reshape(len(qs), prr.K)
    ze = zrow(e)
    out = {"qplayer": np.array([q.player_id for q in qs]), "n": ns, "recall": (top == truth[:, None]).any(1).astype(float)}

    def e04(Sm, beta):
        fin = ze + beta * zrow(np.take_along_axis(Sm, top, axis=1))
        o2 = np.argsort(-fin, axis=1, kind="stable")[:, :5]
        h = np.take_along_axis(top, o2, axis=1) == truth[:, None]
        return np.where(h.any(1), np.exp(-h.argmax(1).astype(float)), 0.0)

    out["m01|base"] = w_rank(rank_of_truth(C, truth))
    hit = np.take_along_axis(top, np.argsort(-e, axis=1, kind="stable"), axis=1) == truth[:, None]
    out["e04v2|base"] = np.where(hit.any(1), np.exp(-hit.argmax(1).astype(float)), 0.0) * (hit.argmax(1) < 5)
    ids = np.asarray(pool)
    assert np.allclose(out["m01|base"], player_scores(ids[prr.top5_indices(top, None)], ids[truth]))
    assert np.allclose(out["e04v2|base"], player_scores(ids[prr.top5_indices(top, e.reshape(-1))], ids[truth]))
    groups = {"a": np.zeros(len(ns), int), "b": binid(ns), "d": ns}
    for v, g in groups.items():
        Sm = norm_score("csls_qq", K_, A_, C, G, g)
        out[f"m01|{v}"] = w_rank(rank_of_truth(Sm, truth))
        out[f"e04v2|{v}"] = e04(Sm, BETA["qq"])
    Sc = norm_score("csls_cc", K_, A_, C, G, ns)
    out["m01|c"] = w_rank(rank_of_truth(Sc, truth))
    out["e04v2|c"] = e04(Sc, BETA["cc"])
    out["ctl|m01"] = w_rank(rank_of_truth(C, np.random.default_rng(S("b", "shuf")).integers(0, len(pool), len(qs))))
    return pd.DataFrame(out)


def stage_main():
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    fro = json.loads((E05 / "frozen.json").read_text())
    assert hashlib.sha256((E05 / "model.txt").read_bytes()).hexdigest() == fro["model_sha256"]
    model = lgb.Booster(model_file=str(E05 / "model.txt"))
    st = R.sets()
    assert not (set(st["dev"]) & st["H"]) and not (set(st["dev"]) & st["fit"])
    ds = R.get_ds("main")
    sims = build_player_pool_sim(ds.index, st["dev"], 400, 20, 100, tuple(range(5, 21)), seed=S("b"))
    rng = np.random.default_rng(S("b", "N"))
    frames, t0 = [], time.perf_counter()
    for s in sims:
        df = eval_pool(ds, thin(s, rng), model, set(st["dev"]))
        df["draw"] = s.draw
        frames.append(df)
        print("draw", s.draw, f"{time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT / "scores.parquet")

    def bs(a, b, tag, d=df):
        r = paired_bootstrap(d[a].to_numpy(), d[b].to_numpy(), d["qplayer"].to_numpy(), n_boot=4000, seed=S("b", "boot", tag))
        return {"new": float(d[a].mean()), "base": float(d[b].mean()), "delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high}
    res = {"n_questions": len(df), "n_draws": int(df.draw.nunique()), "n_players": int(df.qplayer.nunique()),
           "n_mean": float(df.n.mean()), "base": {"m01": float(df["m01|base"].mean()), "e04v2": float(df["e04v2|base"].mean()),
           "recall_top20": float(df.recall.mean()), "shuffle_ctl_m01": float(df["ctl|m01"].mean())}, "variants": {}}
    for v in "abcd":
        res["variants"][v] = {"e04v2": bs(f"e04v2|{v}", "e04v2|base", f"e{v}"), "m01": bs(f"m01|{v}", "m01|base", f"m{v}"),
                              "per_nbin": {f"{lo}-{hi}": bs(f"e04v2|{v}", "e04v2|base", f"e{v}{lo}", df[(df.n >= lo) & (df.n <= hi)]) for lo, hi in BINS}}
    res["primary"] = {**res["variants"]["a"]["e04v2"], "pass": bool(res["variants"]["a"]["e04v2"]["ci_low"] > 0)}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["primary"], indent=1), json.dumps(res["base"]))
    for v in "abcd":
        x = res["variants"][v]["e04v2"]
        print(v, f"{x['delta']:+.4f} [{x['ci_low']:+.4f},{x['ci_high']:+.4f}]  m01 {res['variants'][v]['m01']['delta']:+.4f}")


if __name__ == "__main__":
    {"main": stage_main}[sys.argv[1]]()
