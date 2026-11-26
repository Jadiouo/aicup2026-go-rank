"""E05: Player competition-like re-evaluation, M01 (L0 centroid cosine) vs E04v2 (pool-relative LightGBM reranker). CPU only.

Stages (see experiments/E05_player_comp/PLAN.md, frozen before any score):
  prep   standardise L0 on non-H games, cache black-game vectors of fit/dev/H players   -> .scratch/e05/
  train  fit E04v2 on fit pools, choose rounds on dev-ES 400 pools                       -> .scratch/e05/model.txt, frozen.json
  dev    report M01 / E04v2 on dev pools (400/200/120), per N                            -> .scratch/e05/dev_questions.parquet
  h      one-shot H confirmation (refuses to run twice)                                  -> .scratch/e05/h_questions.parquet
  report paired bootstrap + tables                                                       -> experiments/E05_player_comp/results.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from gogame.aggregate import L0Store, fit_sparse_selector  # noqa: E402
from gogame.gametable import load_table  # noqa: E402
from gogame.metrics import paired_bootstrap, player_scores  # noqa: E402
from gogame import player_rerank as prr  # noqa: E402
from gogame.questions import PoolSim, build_player_index, build_player_pool_sim  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402
from gogame.split import assert_no_player_leak, read_split  # noqa: E402

SCR = ROOT / ".scratch" / "e05"
OUT = ROOT / "experiments" / "E05_player_comp"
CACHE = ROOT / "data" / "cache"
NS = (5, 10, 15, 20)
K = prr.K
N_DENSE = prr.N_DENSE
N_SPARSE_SEL = prr.N_SPARSE_SEL
FIT_LIB = 95   # fit only: the pool builder needs lib+5 games, so lib 100 leaves only 393 eligible fit players (<400)
ROUNDS = (50, 100, 150, 200, 300, 400)
DRAWS = {"fit": 10, "dev_es": 5, "dev": 20, "H": 20}
LGB = prr.lgb_params(derive_seed(2026, "E05", "lgbm") % 2**31, num_threads=6)   # was 8 threads; deterministic, same model


def S(*keys):
    return derive_seed(2026, "E05", *keys)


# ------------------------------------------------------------------ sets / prep
def make_sets(split: pd.DataFrame) -> dict[str, list[str]]:
    H = sorted(split.loc[split.is_holdout, "player_id"])
    nh = split[~split.is_holdout]
    heavy = sorted(nh.loc[nh.n_black >= 120, "player_id"])
    mid = sorted(nh.loc[(nh.n_black >= 100) & (nh.n_black < 120), "player_id"])
    rng = np.random.default_rng(S("dev_players"))
    dev = sorted(heavy[i] for i in rng.permutation(len(heavy))[:450])
    fit = sorted((set(heavy) - set(dev)) | set(mid))
    assert_no_player_leak(set(fit), set(dev))
    assert_no_player_leak(set(fit) | set(dev), set(H))
    return {"H": H, "dev": dev, "fit": fit}


def stage_prep():
    SCR.mkdir(parents=True, exist_ok=True)
    split = read_split(ROOT / "data/splits/split.parquet")
    sets = make_sets(split)
    (SCR / "sets.json").write_text(json.dumps({k: len(v) for k, v in sets.items()}))
    (SCR / "sets_players.json").write_text(json.dumps(sets))
    meta = pd.read_parquet(CACHE / "games_meta.parquet")
    table = load_table(CACHE / "l0")
    assert list(table.uids) == meta["uid"].tolist()
    store = L0Store.from_table(table)
    Hset = set(sets["H"])
    nonH = ~meta["player_id"].isin(Hset).to_numpy()
    D = store.dense[nonH].astype(np.float64)
    mu, sd = np.nanmean(D, axis=0), np.nanstd(D, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    SP = store.sparse[np.flatnonzero(nonH)]
    cols = fit_sparse_selector(SP, 0.005)       # same as E01 stage_player (min_df 0.005, non-H only)
    Sp = SP[:, cols]
    smu = np.asarray(Sp.mean(axis=0)).ravel()
    ssd = np.sqrt(np.maximum(np.asarray(Sp.multiply(Sp).mean(axis=0)).ravel() - smu ** 2, 0))
    ssd = np.where(ssd > 0, ssd, 1.0)
    del D, SP, Sp
    use = set(sets["H"]) | set(sets["dev"]) | set(sets["fit"])
    rows = np.flatnonzero(meta["player_id"].isin(use).to_numpy() & (meta["color"].to_numpy() == 0))
    Z = np.empty((len(rows), N_DENSE + len(cols)), np.float32)
    for a in range(0, len(rows), 20000):
        r = rows[a:a + 20000]
        Zd = np.nan_to_num((store.dense[r] - mu) / sd, nan=0.0)
        Zs = (store.sparse[r][:, cols].toarray() - smu) / ssd
        Z[a:a + 20000] = np.hstack([Zd, Zs])
    np.save(SCR / "Z.npy", Z)
    (SCR / "uids.json").write_text(json.dumps(meta["uid"].to_numpy()[rows].tolist()))
    # F-ratio sparse column choice: fit players' games only
    pid = meta["player_id"].to_numpy()[rows]
    fm = np.isin(pid, sets["fit"])
    pf = pid[fm]
    sel = prr.fratio_sparse_select(Z[fm], pf, N_DENSE, N_SPARSE_SEL)
    np.save(SCR / "sel.npy", sel)
    print("prep done", Z.shape, {k: len(v) for k, v in sets.items()}, "sel", sel.shape, flush=True)


class Data:
    def __init__(self):
        self.Z = np.load(SCR / "Z.npy")
        uids = json.loads((SCR / "uids.json").read_text())
        self.rowof = {u: i for i, u in enumerate(uids)}
        self.sel = np.load(SCR / "sel.npy")
        self.sets = json.loads((SCR / "sets_players.json").read_text())
        meta = pd.read_parquet(CACHE / "games_meta.parquet")
        meta = meta[meta.uid.isin(self.rowof)]
        self.index = build_player_index(meta.assign(color_id=meta["color"]))

    def rows(self, uids):
        return np.fromiter((self.rowof[u] for u in uids), dtype=np.int64, count=len(uids))


# ------------------------------------------------------------------ pools
def draws(data: Data, name: str, players: list[str], pool_size=400, lib=100) -> list[PoolSim]:
    lib = FIT_LIB if name == "fit" else lib
    sims = build_player_pool_sim(data.index, players, pool_size, DRAWS[name], lib, NS, seed=S(name))
    return sims


def subsim(sim: PoolSim, m: int, tag: str) -> PoolSim:
    rng = np.random.default_rng(S("subpool", tag, str(sim.draw), str(m)))
    keep = sorted(sim.players[i] for i in rng.permutation(len(sim.players))[:m])
    ks = set(keep)
    return PoolSim(sim.draw, sim.lib_games, tuple(keep), {p: sim.library[p] for p in keep},
                   [q for q in sim.queries if q.player_id in ks])


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def check_sim(sim: PoolSim):
    for q in sim.queries:
        assert not (set(q.uids) & set(sim.library[q.player_id])), "query/library overlap"
        assert q.color == 0
    assert all(len(set(v)) == len(v) == sim.lib_games for v in sim.library.values())


def pool_features(data: Data, sim: PoolSim, with_features=True) -> dict:
    """Cosine ranks (M01) and, optionally, the E04v2 candidate feature matrix for one pool (shared code: gogame.player_rerank)."""
    Z, sel = data.Z, data.sel
    pool = list(sim.players)
    qs = sim.queries
    lib_z = [Z[data.rows(sim.library[p])] for p in pool]
    q_z = [Z[data.rows(q.uids)] for q in qs]
    pos = {p: i for i, p in enumerate(pool)}
    truth = np.array([pos[q.player_id] for q in qs])
    f = prr.candidate_features(lib_z, q_z, sel, K, N_DENSE)
    res = {"pool": pool, "truth": truth, "ns": f["ns"], "qplayer": np.array([q.player_id for q in qs]),
           "cos": f["cos"], "top": f["top"]}
    if with_features:
        res["X"] = f["X"]
        res["y"] = (f["top"] == truth[:, None]).astype(np.int8).reshape(-1)
    return res


def top5_players(pf: dict, score: np.ndarray | None) -> np.ndarray:
    return np.asarray(pf["pool"])[prr.top5_indices(pf["top"], score)]


def score_pool(pf: dict, model, iteration=None) -> pd.DataFrame:
    truth_ids = np.asarray(pf["pool"])[pf["truth"]]
    s_m01 = player_scores(top5_players(pf, None), truth_ids)
    out = {"qplayer": pf["qplayer"], "n": pf["ns"], "m01": s_m01,
           "recall": (pf["top"] == pf["truth"][:, None]).any(1).astype(float)}
    if model is not None:
        pred = model.predict(pf["X"], num_iteration=iteration, num_threads=6)
        out["e04v2"] = player_scores(top5_players(pf, pred), truth_ids)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ stages
def stage_train():
    import lightgbm as lgb
    t0 = time.perf_counter()
    data = Data()
    fit_sims = draws(data, "fit", data.sets["fit"])
    Xs, ys, gs = [], [], []
    for s in fit_sims:
        check_sim(s)
        assert set(s.players) <= set(data.sets["fit"])
        pf = pool_features(data, s)
        Xs.append(pf["X"]); ys.append(pf["y"]); gs.append(np.full(len(pf["truth"]), K))
        print(f"fit draw {s.draw}: {len(pf['truth'])} q, recall {(pf['top'] == pf['truth'][:, None]).any(1).mean():.3f}  {time.perf_counter() - t0:.0f}s", flush=True)
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    del Xs
    es = []
    for s in draws(data, "dev_es", data.sets["dev"]):
        check_sim(s)
        assert set(s.players) <= set(data.sets["dev"])
        es.append((s, pool_features(data, s)))
    ds = lgb.Dataset(X, y, group=g, free_raw_data=False)
    model = lgb.train(LGB, ds, num_boost_round=max(ROUNDS))
    curve = {}
    for r in ROUNDS:
        sc = pd.concat([score_pool(pf, model, r) for _, pf in es])
        curve[r] = float(sc["e04v2"].mean())
        print(f"ES rounds {r}: e04v2 {curve[r]:.4f} (m01 {sc['m01'].mean():.4f})", flush=True)
    best = min(r for r in ROUNDS if curve[r] == max(curve.values()))
    model.save_model(str(SCR / "model.txt"), num_iteration=best)
    h = hashlib.sha256((SCR / "model.txt").read_bytes()).hexdigest()
    fro = {"rounds": best, "es_curve": curve, "model_sha256": h, "n_train_groups": int(len(g)), "n_features": int(X.shape[1]),
           "lgb_params": {k: v for k, v in LGB.items()}, "h_evaluated": False, "seconds": time.perf_counter() - t0,
           "es_m01": float(pd.concat([score_pool(pf, None) for _, pf in es])["m01"].mean())}
    (SCR / "frozen.json").write_text(json.dumps(fro, indent=1))
    print("frozen", best, h[:12], f"{time.perf_counter() - t0:.0f}s", flush=True)


def eval_set(name: str, players: list[str], sizes: tuple[int, ...]) -> pd.DataFrame:
    import lightgbm as lgb
    data = Data()
    model = lgb.Booster(model_file=str(SCR / "model.txt"))
    fro = json.loads((SCR / "frozen.json").read_text())
    assert hashlib.sha256((SCR / "model.txt").read_bytes()).hexdigest() == fro["model_sha256"]
    frames = []
    for s in draws(data, name, players):
        check_sim(s)
        assert set(s.players) <= set(players)
        for m in sizes:
            sub = s if m == 400 else subsim(s, m, name)
            pf = pool_features(data, sub)
            df = score_pool(pf, model)
            df["draw"], df["pool"] = s.draw, m
            frames.append(df)
        print(name, "draw", s.draw, flush=True)
    return pd.concat(frames, ignore_index=True)


def stage_dev():
    df = eval_set("dev", json.loads((SCR / "sets_players.json").read_text())["dev"], (400, 200, 120))
    df.to_parquet(SCR / "dev_questions.parquet")


def stage_h():
    fro = json.loads((SCR / "frozen.json").read_text())
    if fro["h_evaluated"] or (SCR / "h_questions.parquet").exists():
        raise SystemExit("H was already evaluated once; refusing")
    assert (SCR / "dev_questions.parquet").exists(), "run dev first"
    df = eval_set("H", json.loads((SCR / "sets_players.json").read_text())["H"], (400,))
    df.to_parquet(SCR / "h_questions.parquet")
    fro["h_evaluated"] = True
    (SCR / "frozen.json").write_text(json.dumps(fro, indent=1))


def summarize(df: pd.DataFrame, seed_key: str) -> dict:
    out = {}
    for nlab, d in [("all", df)] + [(str(n), df[df.n == n]) for n in NS]:
        a, b, grp = d["e04v2"].to_numpy(), d["m01"].to_numpy(), d["qplayer"].to_numpy()
        bs = paired_bootstrap(a, b, grp, n_boot=4000, seed=S("boot", seed_key, nlab))
        out[nlab] = {"m01": float(b.mean()), "e04v2": float(a.mean()), "recall_top20": float(d["recall"].mean()),
                     "delta": bs.delta, "ci_low": bs.ci_low, "ci_high": bs.ci_high, "n_questions": int(len(d))}
    return out


def stage_report():
    res = {"frozen": json.loads((SCR / "frozen.json").read_text()), "dev": {}, "H": {}}
    dev = pd.read_parquet(SCR / "dev_questions.parquet")
    for m in (400, 200, 120):
        res["dev"][str(m)] = summarize(dev[dev.pool == m], f"dev{m}")
    h = pd.read_parquet(SCR / "h_questions.parquet")
    res["H"]["400"] = summarize(h, "H")
    res["H"]["m01_draw_std"] = float(h.groupby("draw").m01.mean().std(ddof=1))
    res["H"]["e04v2_draw_std"] = float(h.groupby("draw").e04v2.mean().std(ddof=1))
    d, hh = res["dev"]["400"]["all"], res["H"]["400"]["all"]
    res["decision"] = {"adopt": "E04v2" if (hh["ci_low"] > 0 and d["delta"] >= 0 and hh["delta"] >= 0) else "M01",
                       "H_ci_low": hh["ci_low"], "dev400_delta": d["delta"], "H_delta": hh["delta"]}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"]))


if __name__ == "__main__":
    {"prep": stage_prep, "train": stage_train, "dev": stage_dev, "h": stage_h, "report": stage_report}[sys.argv[1]]()
