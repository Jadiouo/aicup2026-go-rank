"""E10 pipeline stages: status / features / oof / report.  See experiments/E10_tier2_retrain/PLAN.md.

M04 (CORAL on the 522-column HSL bag aggregate) only: L0 is never loaded (PLAN 5.1 `needs_l0=False` by construction)."""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import lib
from .lib import CACHE, CANDS, CAND_ORDER, NS, OUT, ROOT, SCRATCH

from gogame.aggregate import L0Store, aggregate_many, make_spec  # noqa: E402
from gogame.constants import RANKS  # noqa: E402
from gogame.coral import fit_coral  # noqa: E402
from gogame.metrics import balanced_weights, rank_scores  # noqa: E402
from gogame.aggregate import fit_sparse_selector  # noqa: E402
from gogame.models import decide_argmax, final_num_rounds, fit_lgbm  # noqa: E402
from gogame.questions import Bag, bag_class_weights, sample_train_bags, to_questions_frame  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402
from gogame.constants import SEED  # noqa: E402
from gogame.split import read_split  # noqa: E402
import run_e01 as e01  # noqa: E402

FEAT_MANIFEST = "features_manifest.json"
HSL_FILE = "hsl_games.parquet"


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def default_cache(partial: bool) -> Path:
    return SCRATCH / "cache" if partial else CACHE


# ------------------------------------------------------------------------------------------------ features
def stage_features(allow_partial: bool = False, cache: Path | None = None) -> dict:
    hsl, seg, info = lib.read_hsl_segments()
    partial = not info["complete"]
    if partial and not allow_partial:
        raise RuntimeError(f"scan incomplete ({info['n_present']}/{info['n_shards_total']} shards; missing e.g. {info['missing_shards'][:5]}). "
                           f"Formal run refused; --allow-partial writes a PARTIAL cache under .scratch/e10/cache.")
    cache = Path(cache) if cache else default_cache(partial)
    if partial and cache.resolve().is_relative_to(CACHE.resolve()):
        raise SystemExit("partial features must not be written to data/cache/e10")
    cache.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet", columns=["uid"])
    missing = sorted(set(hsl["uid"]) - set(meta["uid"]))
    if missing:
        raise ValueError(f"{len(missing)} HSL uids without games_meta row, e.g. {missing[:3]}")
    if not partial:
        exp = info["n_games_expected"]
        if info["n_games"] + info["n_skipped_total"] != exp and info["n_games"] != exp:
            raise ValueError(f"rows {info['n_games']} (+skipped {info['n_skipped_total']}) != ledger games {exp}")
    path = cache / HSL_FILE
    tmp = path.with_suffix(".tmp")
    hsl.to_parquet(tmp, index=False)
    tmp.replace(path)
    seg.to_csv(cache / "segments.csv", index=False)
    info.update({"sha256": lib.sha256_file(path), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "partial": partial,
                 "dtype": "float32", "note": "tier-1+2 shards (status done) + pilot; includes H players' games (features only)"})
    (cache / FEAT_MANIFEST).write_text(json.dumps(info, indent=1))
    print(f"features: {info['n_present']}/{info['n_shards_total']} shards {info['n_present_by_tier']}, {info['n_games']} games "
          f"(ledger {info['n_games_expected']}), complete={info['complete']}, skipped={info['n_skipped_total']} -> {path}, "
          f"peak RSS {peak_rss_mb():.0f} MB")
    return info


class Ctx:
    """HSL features (float32), games_meta, split, indexes.  No L0."""

    def __init__(self, cache: Path, allow_partial: bool, max_players: int | None = None):
        self.cache = Path(cache)
        fm_path = self.cache / FEAT_MANIFEST
        if not fm_path.exists():
            raise FileNotFoundError(f"{fm_path} missing: run `features` first")
        self.fm = json.loads(fm_path.read_text())
        if lib.sha256_file(self.cache / HSL_FILE) != self.fm["sha256"]:
            raise ValueError("hsl_games.parquet differs from features_manifest.json; rerun `features`")
        self.partial = not self.fm["complete"]
        if self.partial and not allow_partial:
            raise RuntimeError(f"features are partial ({self.fm['n_present']}/{self.fm['n_shards_total']} shards); use --allow-partial for a smoke run")
        self.max_players = max_players
        self.split = read_split(ROOT / "data/splits/split.parquet")
        self.H = set(self.split.loc[self.split["is_holdout"], "player_id"])
        df = pd.read_parquet(self.cache / HSL_FILE)
        self.hnames = tuple(c for c in df.columns if c != "uid")
        uids = df["uid"].tolist()
        mat = df[list(self.hnames)].to_numpy(np.float32)
        del df
        assert mat.dtype == np.float32
        self.hs = L0Store(uids, mat, None, self.hnames)
        self.sp_h = make_spec(self.hnames, np.zeros(0, np.int32))
        meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
        self.meta = meta[meta["uid"].isin(set(uids))].reset_index(drop=True)
        self.meta_all = meta
        man = pd.read_parquet(ROOT / "data/scan/manifest.parquet", columns=["uid", "tier"])
        t1 = set(man.loc[man["tier"] == 1, "uid"])
        del man
        self.index_all = lib.hsl_index(self.meta, uids)
        self.index_t1 = lib.hsl_index(self.meta, [u for u in uids if u in t1])
        nonh = self.split[~self.split["is_holdout"]]
        if max_players:
            keep = np.random.default_rng(derive_seed(SEED, "e10", "smoke_players")).permutation(sorted(nonh["player_id"]))[:max_players]
            nonh = nonh[nonh["player_id"].isin(set(keep))]
        self.nonh = nonh.reset_index(drop=True)
        self._l0 = None

    @property
    def l0(self):
        """L0 store (dense + sparse) in the table's native row order, addressed by uid.  Loaded only when candidate D / the
        M03 repro asks for it (M04 candidates never touch L0).  No row re-indexing copy: the store may hold more uids than HSL has."""
        if self._l0 is None:
            from gogame.gametable import load_table
            table = load_table(ROOT / "data/cache/l0")
            self._l0 = L0Store(list(table.uids), table.dense, table.sparse, table.names)
            del table
        return self._l0


def stage_questions(ctx: Ctx, wd: Path) -> tuple[list[Bag], list[Bag]]:
    scope = set(ctx.nonh["player_id"])
    q1_frame = pd.read_parquet(lib.E06_OOF / "questions.parquet")
    q1 = [b for b in e01.frame_to_bags(q1_frame) if b.player_id in scope]
    q2 = lib.make_q2(ctx.index_all, scope)
    assert not ({b.player_id for b in q1 + q2} & ctx.H)
    assert not ({b.qid for b in q1} & {b.qid for b in q2})
    f2 = to_questions_frame(q2)
    f2["pool"] = lib.pool_sizes(ctx.index_all, q2)
    f2.to_parquet(wd / "questions_q2.parquet", index=False)
    cov = {"partial": ctx.partial, "max_players": ctx.max_players, "features_sha256": ctx.fm["sha256"],
           "q1_file_sha256": lib.sha256_file(lib.E06_OOF / "questions.parquet"), "n_q1": len(q1), "n_q2": len(q2),
           "q2_per_N": {n: int(sum(b.n == n for b in q2)) for n in NS}, "q1_per_N": {n: int(sum(b.n == n for b in q1)) for n in lib.NS_Q1},
           "n_players_scope": len(scope), "tier1_complete": bool(ctx.fm.get("tier1_complete", False)),
           "plan_expected_q2_per_N_full": {"5": 7150, "10": 6303, "15": 5279, "20": 4501}}
    (wd / "coverage.json").write_text(json.dumps(cov, indent=1))
    print("coverage:", {k: cov[k] for k in ("n_q1", "n_q2", "q2_per_N")})
    return q1, q2


# ------------------------------------------------------------------------------------------------ oof
def stage_oof(ctx: Ctx, wd: Path, folds, cands) -> None:
    wd.mkdir(parents=True, exist_ok=True)
    q1, q2 = stage_questions(ctx, wd)
    nonh = ctx.nonh
    reg = [c for c in CAND_ORDER if c in cands]
    for k in (folds if reg else ()):
        fs = lib.fold_sets(nonh, k)
        lib.assert_fold_disjoint(fs, ctx.H)
        val_bags = [b for b in q1 + q2 if b.player_id in fs["val"]]
        assert {b.player_id for b in val_bags} <= fs["val"]
        qset = np.array(["q2" if b.qid.startswith("q2_") else "q1" for b in val_bags])
        game_rep = lib.assert_no_game_leak(ctx.meta_all, [u for b in val_bags for u in b.uids], fs["train_all"])
        Xv = aggregate_many(val_bags, ctx.hs, ctx.sp_h)
        for name in reg:
            out = wd / f"oof_{name}_fold{k}.parquet"
            if out.exists():
                print(f"fold {k} {name} cached", flush=True)
                continue
            t0 = time.perf_counter()
            c = CANDS[name]
            index = ctx.index_t1 if c.tier1_only else ctx.index_all
            fit_in = sorted(p for p in fs["fit"] if p in index.games)
            tb = sample_train_bags(index, fit_in, c.bags_per_pc, c.n_range, seed=c.train_seed(k))
            tr_uids = {u for b in tb for u in b.uids}
            assert {b.player_id for b in tb} <= fs["fit"] and not ({b.player_id for b in tb} & (fs["val"] | fs["es"] | ctx.H))
            assert not (tr_uids & {u for b in val_bags for u in b.uids})
            ytr = np.array([b.rank_id for b in tb])
            wtr = bag_class_weights(tb, "inv_players")
            X = aggregate_many(tb, ctx.hs, ctx.sp_h)
            assert X.dtype == np.float32
            m = fit_coral(X, ytr, wtr, chunk=lib.CORAL_CHUNK)
            del X
            praw, p = m.predict_proba(Xv, raw=True), m.predict_proba(Xv)
            assert np.isfinite(praw).all() and np.isfinite(p).all()
            df = e01.probs_frame(val_bags, k, praw, p)
            df["qset"] = qset
            repro = _repro_check(ctx, c, k, df)
            tmp = out.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(out)
            (wd / f"meta_{name}_fold{k}.json").write_text(json.dumps({
                "fold": k, "cand": name, "partial": ctx.partial, "features_sha256": ctx.fm["sha256"], "n_train_bags": len(tb),
                "bags_per_pc": c.bags_per_pc, "n_range": list(c.n_range), "tier1_only": c.tier1_only, "seed_variant": c.seed_variant,
                "n_fit_players": len(fit_in), "n_val_players": len(fs["val"]), "n_es_players_excluded": len(fs["es"]),
                "n_val_q1": int((qset == "q1").sum()), "n_val_q2": int((qset == "q2").sum()), "game_leak_check": game_rep,
                "e06_repro": repro, "coral": m.info, "seconds": round(time.perf_counter() - t0, 1),
                "peak_rss_mb": round(peak_rss_mb(), 1), "max_players": ctx.max_players}, default=str))
            print(f"fold {k} {name}: bags {len(tb)}, iters {m.best_iteration}, repro {repro['status']}, {time.perf_counter() - t0:.0f}s, "
                  f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
            if repro["status"] == "MISMATCH":
                raise SystemExit(f"STOP (PLAN 12): {name} refit != E06 {c.e06_name} (max abs diff {repro['max_abs_diff']:.2e}); not running further candidates")
    # D runs strictly after every A/B/C fold of this invocation (PLAN revision 2), on its own so that a failure never touches W
    if lib.REPRO_NAME in cands:
        for k in folds:
            stage_m03_repro(ctx, wd, k, q1)
    if lib.D_NAME in cands:
        for k in folds:
            stage_d_fold(ctx, wd, k, q1 + q2)


# ------------------------------------------------------------------------------------------------ candidate D (M03 + M04_A, EV)
def _m03_matrix(ctx: Ctx, bags, spec_l0) -> np.ndarray:
    """E06 M03 feature layout: [L0 aggregate (dense + selected sparse), HSL aggregate without its duplicate bag_n / is_white]."""
    XL = aggregate_many(bags, ctx.l0, spec_l0)
    XH = aggregate_many(bags, ctx.hs, ctx.sp_h)
    X = np.hstack([XL, XH[:, :-2]])
    return X


def fit_m03(ctx: Ctx, tb, es_bags, val_sets) -> tuple[list, dict]:
    """M03 exactly as scripts/experiments/e06/pipeline.py: sparse columns chosen on the training games only (min_df),
    LightGBM lr 0.1 / max_bin 63 / 4 threads / seed derive_seed(SEED,'lgbm') / inv_players weights, early stopping on es bags.
    Returns ([(praw, p) per val set], info)."""
    ytr, yes = np.array([b.rank_id for b in tb]), np.array([b.rank_id for b in es_bags])
    wtr, wes = bag_class_weights(tb, "inv_players"), bag_class_weights(es_bags, "inv_players")
    used = sorted({u for b in tb for u in b.uids})
    cols = fit_sparse_selector(ctx.l0.sparse[ctx.l0.rows(used)], e01.CFG["min_df"])
    del used
    spec_l0 = make_spec(ctx.l0.names, cols)
    Xtr = _m03_matrix(ctx, tb, spec_l0)
    Xes = _m03_matrix(ctx, es_bags, spec_l0)
    prm = {"num_threads": lib.LGBM_THREADS, "learning_rate": 0.1, "max_bin": 63, "log_every": 0}
    m = fit_lgbm(Xtr, ytr, Xes, yes, prm, derive_seed(SEED, "lgbm"), wtr, wes)
    n_feat = int(Xtr.shape[1])
    del Xtr, Xes
    outs = []
    for vb in val_sets:
        Xv = _m03_matrix(ctx, vb, spec_l0)
        pr, p = m.predict_proba(Xv, raw=True), m.predict_proba(Xv)
        assert np.isfinite(pr).all() and np.isfinite(p).all()
        outs.append((pr, p))
        del Xv
    return outs, {"best_iteration": int(m.best_iteration), "n_features": n_feat, "n_sparse_cols": int(len(cols))}


def _atomic_parquet(df: pd.DataFrame, out: Path) -> None:
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)


def repro_state(wd: Path, folds=lib.M03_REPRO_FOLDS) -> dict:
    """Gate state from the repro_M03_fold<k>.json markers: status OK only if every gate fold is OK."""
    st = {}
    for k in folds:
        f = wd / f"repro_M03_fold{k}.json"
        st[k] = json.loads(f.read_text())["status"] if f.exists() else "absent"
    return {"folds": st, "ok": all(v == "OK" for v in st.values()), "mismatch": any(v == "MISMATCH" for v in st.values())}


def stage_m03_repro(ctx: Ctx, wd: Path, k: int, q1) -> dict:
    """Gate for D: refit M03 on TIER 1 only with E06's bags/seeds/es questions and compare with E06 oof_M03 on the Q1 questions
    (tolerance lib.M03_REPRO_TOL on praw_* and p_*, and equal best_iteration).  MISMATCH stops the run (D is cancelled)."""
    out = wd / f"repro_M03_fold{k}.json"
    if out.exists() and json.loads(out.read_text()).get("status") == "OK":
        print(f"M03 repro fold {k}: cached OK", flush=True)
        return json.loads(out.read_text())
    ref_f, ref_meta = lib.E06_OOF / f"oof_M03_fold{k}.parquet", lib.E06_OOF / f"meta_M03_fold{k}.json"
    if ctx.max_players or not ctx.fm.get("tier1_complete", False) or not ref_f.exists():
        rec = {"status": "skipped", "reason": "player-subset run, tier 1 incomplete or E06 file missing (not a verification)"}
        print(f"M03 repro fold {k}: {rec['status']} ({rec['reason']})", flush=True)
        return rec
    t0 = time.perf_counter()
    fs = lib.fold_sets(ctx.nonh, k)
    lib.assert_fold_disjoint(fs, ctx.H)
    index = ctx.index_t1
    fit_in = sorted(p for p in fs["fit"] if p in index.games)
    c = CANDS["C"]
    tb = sample_train_bags(index, fit_in, c.bags_per_pc, c.n_range, seed=c.train_seed(k))
    es_bags = lib.make_es_bags(index, fs["es"], k, True)
    val_bags = [b for b in q1 if b.player_id in fs["val"]]
    assert {b.player_id for b in tb} <= fs["fit"] and not ({b.player_id for b in tb} & (fs["val"] | fs["es"] | ctx.H))
    ((praw, p),), info = fit_m03(ctx, tb, es_bags, [val_bags])
    df = e01.probs_frame(val_bags, k, praw, p)
    ref = pd.read_parquet(ref_f).set_index("qid")
    mine = df.set_index("qid")
    cols = [f"praw_{i}" for i in range(10)] + lib.PCOLS
    ref_iter = json.loads(ref_meta.read_text()).get("best_iteration") if ref_meta.exists() else None
    if set(ref.index) != set(mine.index):
        d, status = float("inf"), "MISMATCH"
    else:
        d = float(np.abs(mine.loc[ref.index, cols].to_numpy() - ref[cols].to_numpy()).max())
        status = "OK" if (d <= lib.M03_REPRO_TOL and (ref_iter is None or ref_iter == info["best_iteration"])) else "MISMATCH"
    rec = {"status": status, "fold": k, "max_abs_diff": d, "tol": lib.M03_REPRO_TOL, "n_q1": int(len(ref)), "n_train_bags": len(tb),
           "best_iteration": info["best_iteration"], "e06_best_iteration": ref_iter, "e06_file": ref_f.name,
           "seconds": round(time.perf_counter() - t0, 1), "peak_rss_mb": round(peak_rss_mb(), 1), "features_sha256": ctx.fm["sha256"]}
    out.write_text(json.dumps(rec, indent=1))
    print(f"M03 repro fold {k}: {status}, max abs diff {d:.2e}, iters {info['best_iteration']} (E06 {ref_iter}), {rec['seconds']:.0f}s, "
          f"peak RSS {rec['peak_rss_mb']:.0f} MB", flush=True)
    if status == "MISMATCH":
        raise SystemExit(f"STOP (PLAN revision 2): M03 tier-1 refit != E06 oof_M03 fold {k} (max abs diff {d:.2e}); D is cancelled")
    return rec


def stage_d_fold(ctx: Ctx, wd: Path, k: int, all_q) -> bool:
    """Candidate D, fold k: M03 on A's data/bags/fold, 0.5/0.5 mean with the (already computed) M04_A probabilities.
    Writes oof_M03A_fold<k>.parquet, oof_D_fold<k>.parquet (p_* = ensemble probabilities; decode is EV, applied in report)."""
    out = wd / f"oof_{lib.D_NAME}_fold{k}.parquet"
    if out.exists():
        print(f"fold {k} {lib.D_NAME} cached", flush=True)
        return True
    a_file = wd / f"oof_A_fold{k}.parquet"
    if not a_file.exists():
        print(f"fold {k} D: oof_A_fold{k}.parquet missing - run A first; skipped", flush=True)
        return False
    formal = not (ctx.partial or ctx.max_players)
    if formal:
        q1 = [b for b in all_q if not b.qid.startswith("q2_")]
        gate = repro_state(wd)
        if gate["mismatch"]:
            raise SystemExit("STOP (PLAN revision 2): M03 reproduction gate is MISMATCH; D is cancelled")
        for gk in lib.M03_REPRO_FOLDS:
            if gate["folds"][gk] != "OK":
                stage_m03_repro(ctx, wd, gk, q1)
        if not repro_state(wd)["ok"]:
            raise SystemExit("STOP: the M03 reproduction gate did not pass (cannot verify against E06); D is not run")
    t0 = time.perf_counter()
    fs = lib.fold_sets(ctx.nonh, k)
    lib.assert_fold_disjoint(fs, ctx.H)
    val_bags = [b for b in all_q if b.player_id in fs["val"]]
    qset = np.array(["q2" if b.qid.startswith("q2_") else "q1" for b in val_bags])
    game_rep = lib.assert_no_game_leak(ctx.meta_all, [u for b in val_bags for u in b.uids], fs["train_all"])
    c = CANDS["A"]
    index = ctx.index_all
    fit_in = sorted(p for p in fs["fit"] if p in index.games)
    tb = sample_train_bags(index, fit_in, c.bags_per_pc, c.n_range, seed=c.train_seed(k))       # = A's bags
    es_bags = lib.make_es_bags(index, fs["es"], k, False)
    assert {b.player_id for b in tb} <= fs["fit"] and not ({b.player_id for b in tb} & (fs["val"] | fs["es"] | ctx.H))
    assert not ({u for b in tb for u in b.uids} & {u for b in val_bags for u in b.uids})
    ((praw, p),), info = fit_m03(ctx, tb, es_bags, [val_bags])
    df03 = e01.probs_frame(val_bags, k, praw, p)
    df03["qset"] = qset
    a = pd.read_parquet(a_file).set_index("qid")
    if set(a.index) != set(df03["qid"]):
        raise RuntimeError(f"fold {k}: M04_A question set differs from the D validation questions")
    a = a.loc[df03["qid"]]
    dd = df03.copy()
    rc = [f"praw_{j}" for j in range(10)]
    dd[lib.PCOLS] = lib.ensemble_probs(df03[lib.PCOLS].to_numpy(), a[lib.PCOLS].to_numpy())
    dd[rc] = lib.ensemble_probs(df03[rc].to_numpy(), a[rc].to_numpy())
    _atomic_parquet(df03, wd / f"oof_{lib.M03_NAME}_fold{k}.parquet")
    _atomic_parquet(dd, out)
    (wd / f"meta_{lib.D_NAME}_fold{k}.json").write_text(json.dumps({
        "fold": k, "cand": lib.D_NAME, "partial": ctx.partial, "features_sha256": ctx.fm["sha256"], "n_train_bags": len(tb),
        "bags_per_pc": c.bags_per_pc, "n_range": list(c.n_range), "w_m03": lib.D_WEIGHT_M03, "decode": lib.D_DECODE,
        "n_es_bags": len(es_bags), "n_fit_players": len(fit_in), "n_val_q1": int((qset == "q1").sum()), "n_val_q2": int((qset == "q2").sum()),
        "game_leak_check": game_rep, **info, "seconds": round(time.perf_counter() - t0, 1), "peak_rss_mb": round(peak_rss_mb(), 1),
        "max_players": ctx.max_players}, default=str))
    print(f"fold {k} D: bags {len(tb)}, M03 iters {info['best_iteration']}, {info['n_features']} feats, {time.perf_counter() - t0:.0f}s, "
          f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
    return True


def _repro_check(ctx: Ctx, c, k: int, df: pd.DataFrame) -> dict:
    """PLAN 3: the C refit must equal E06 oof_M04s0/M04s1 on the Q1 questions (tolerance 1e-6)."""
    if c.e06_name is None:
        return {"status": "n/a"}
    if ctx.max_players:
        return {"status": "skipped (player-subset smoke run: different training set)"}
    f = lib.E06_OOF / f"oof_{c.e06_name}_fold{k}.parquet"
    if not f.exists() or not ctx.fm.get("tier1_complete", False):
        return {"status": "skipped (E06 file missing or tier 1 incomplete)"}
    ref = pd.read_parquet(f).set_index("qid")
    mine = df[df["qset"] == "q1"].set_index("qid")
    if set(ref.index) != set(mine.index):
        return {"status": "MISMATCH", "max_abs_diff": float("inf"), "note": "question sets differ"}
    cols = [f"praw_{i}" for i in range(10)] + lib.PCOLS
    d = float(np.abs(mine.loc[ref.index, cols].to_numpy() - ref[cols].to_numpy()).max())
    return {"status": "OK" if d <= 1e-6 else "MISMATCH", "max_abs_diff": d, "n_q1": int(len(ref)), "e06_file": f.name}


# ------------------------------------------------------------------------------------------------ report
def _load(wd: Path, name: str, partial: bool) -> pd.DataFrame | None:
    files = sorted(wd.glob(f"oof_{name}_fold*.parquet"))
    if not files:
        return None
    if not partial and len(files) != 5:
        raise RuntimeError(f"{name}: {len(files)}/5 folds")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def stage_report(wd: Path, partial_ok: bool, n_boot: int = lib.N_BOOT) -> dict:
    cov = json.loads((wd / "coverage.json").read_text())
    partial = bool(cov["partial"])
    if partial and not partial_ok:
        raise RuntimeError("this OOF directory comes from a partial scan; pass --allow-partial (output is NOT a result)")
    P = {n: d for n in CAND_ORDER if (d := _load(wd, n, partial)) is not None}
    if "C" not in P:
        raise RuntimeError("candidate C (refit of E06 M04) is missing; nothing to compare against")
    pool = pd.read_parquet(wd / "questions_q2.parquet", columns=["qid", "pool"]).set_index("qid")["pool"]
    base = P["C"].sort_values("qid").reset_index(drop=True)
    for n in P:
        P[n] = P[n].set_index("qid").loc[base["qid"]].reset_index()
        if set(P[n]["qid"]) != set(base["qid"]) or P[n]["qid"].duplicated().any():
            raise RuntimeError(f"{n}: question set differs from C")
    truth, groups, ns, qset = base["rank_id"].to_numpy(), base["player_id"].to_numpy(), base["n"].to_numpy(), base["qset"].to_numpy()
    S = {n: rank_scores(decide_argmax(d[lib.PCOLS].to_numpy()), truth) for n, d in P.items()}
    q1, q2 = qset == "q1", qset == "q2"
    pool_q = np.array([pool.get(q, -1) for q in base["qid"]])

    def sub(m):
        return dict(groups=groups[m], truth=truth[m], ns=ns[m])

    R = {"partial": partial, "NOT_A_RESULT": partial, "coverage": cov, "features_sha256": cov["features_sha256"], "n_boot": n_boot,
         "cands_present": list(P), "scores": {}, "deltas": {}}
    for n in P:
        R["scores"][n] = {"q1_eq": lib.equal_n_point(S[n][q1], np.zeros(q1.sum()), truth[q1], ns[q1], lib.NS_Q1)["delta"] if q1.any() else None,
                          "q2_eq": lib.equal_n_point(S[n][q2], np.zeros(q2.sum()), truth[q2], ns[q2], NS)["delta"],
                          "q2_per_N": {int(a): float((S[n][q2 & (ns == a)] * balanced_weights(truth[q2 & (ns == a)])).mean())
                                       for a in NS if (q2 & (ns == a)).any()}}
    avail = {}
    for x in ("A", "B"):
        if x not in P:
            continue
        eq = lib.equal_n_delta(S[x][q2], S["C"][q2], n_boot=n_boot, level=0.975, **sub(q2))
        q1p = lib.equal_n_point(S[x][q1], S["C"][q1], truth[q1], ns[q1], lib.NS_Q1)["delta"] if q1.any() else 0.0
        R["deltas"][f"{x}-C|q2_eq"] = eq
        R["deltas"][f"{x}-C|q1_point"] = q1p
        R["deltas"][f"{x}-C|q2_by_pool"] = {
            lab: lib.equal_n_delta(S[x][q2 & mk], S["C"][q2 & mk], n_boot=n_boot, level=0.95, **sub(q2 & mk))
            for lab, mk in (("pool_lt60", (pool_q < lib.POOL_SPLIT) & (pool_q >= 0)), ("pool_ge60", pool_q >= lib.POOL_SPLIT)) if mk[q2].any()}
        avail[x] = {"q2_eq": eq, "q1_point": q1p, "q2_per_N": eq["per_N_delta"]}
    avail["BA"] = lib.equal_n_delta(S["B"][q2], S["A"][q2], n_boot=n_boot, level=0.95, **sub(q2)) if {"A", "B"} <= set(P) else None
    if avail["BA"]:
        R["deltas"]["B-A|q2_eq"] = avail["BA"]
    sel = lib.select_candidate(avail)
    noise = {}
    for base_n, s1 in (("C", "C_s1"), ("A", "A_s1")):
        if s1 in P:
            noise[base_n] = lib.equal_n_point(S[base_n][q2], S[s1][q2], truth[q2], ns[q2], NS)["delta"]
    stop = [k for k, v in noise.items() if k == "A" and abs(v) > lib.NOISE_STOP]
    R["noise_ruler_seed0_minus_seed1_q2_eq"] = noise
    R["per_rank_q2_argmax"] = {n: {RANKS[r]: float(S[n][q2 & (truth == r)].mean()) for r in range(10) if (q2 & (truth == r)).any()} for n in P}
    cand = sel["candidate"]
    c = CANDS[cand]
    d_info = _evaluate_d(wd, partial, cov, P, S, cand, base, pool, n_boot)
    R["deltas"].update(d_info.pop("_deltas", {}))
    R["scores"].update(d_info.pop("_scores", {}))
    sel.update({"main": "M04", "decode": "argmax", "bags_per_pc": c.bags_per_pc, "n_range": list(c.n_range), "seed_ns": "e10",
                "cache": str(CACHE.relative_to(ROOT)), "features_sha256": cov["features_sha256"], "partial": partial,
                "valid_for_confirm": (not partial) and not stop, "noise_ruler": noise, "noise_stop": bool(stop),
                "rule": "PLAN section 6 (a) q2 N-equal ci_low(97.5%)>0 (b) q1 point>=-0.005 (c) per-N q2 point>=-0.01; both qualified -> B only if B-A ci_low(95%)>0"})
    sel["w_candidate"] = cand
    sel["d"] = d_info
    if d_info.get("adopt"):
        a_cfg = CANDS["A"]
        sel.update({"candidate": lib.D_NAME, "main": "M03+M04", "decode": lib.D_DECODE, "bags_per_pc": a_cfg.bags_per_pc,
                    "n_range": list(a_cfg.n_range), "oof_best_iterations": {"M03": d_info["m03_best_iterations"]},
                    "ensemble": {"members": ["M03", "M04"], "weights": [lib.D_WEIGHT_M03, 1 - lib.D_WEIGHT_M03], "decode": lib.D_DECODE}})
    sel["final_candidate"] = sel["candidate"]
    R["selection"] = sel
    name = "results_PARTIAL_NOT_A_RESULT.json" if partial else "results.json"
    (wd / name).write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str))
    (wd / ("selection_PARTIAL.json" if partial else "selection.json")).write_text(json.dumps(sel, indent=1, default=str))
    tag = "[PARTIAL - NOT A RESULT] " if partial else ""
    for n in P:
        print(f"{tag}{n}: Q2 N-equal balanced {R['scores'][n]['q2_eq']:.4f}  per-N {R['scores'][n]['q2_per_N']}")
    for k, v in R["deltas"].items():
        if isinstance(v, dict) and "delta" in v:
            print(f"{tag}  {k}: {v['delta']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] (level {v['level']})")
        elif isinstance(v, float):
            print(f"{tag}  {k}: {v:+.4f}")
    print(tag + "selection:", json.dumps({k: sel[k] for k in ("w_candidate", "candidate", "qualified", "noise_stop")}), "| D:", d_info["status"],
          d_info.get("reason", ""))
    if stop:
        print(tag + f"STOP (PLAN 6.6): A seed0-seed1 = {noise['A']:+.4f} > {lib.NOISE_STOP}; ask the user")
    return R


def _d_late(paths, deadline: str = lib.D_DEADLINE) -> bool:
    """True if any D file was written after the PLAN deadline ('YYYY-MM-DD 24:00' = midnight at the end of that day)."""
    import datetime as dt
    day = dt.datetime.strptime(deadline.split()[0], "%Y-%m-%d") + dt.timedelta(days=1)
    return any(dt.datetime.fromtimestamp(Path(p).stat().st_mtime) > day for p in paths)


def _evaluate_d(wd: Path, partial: bool, cov: dict, P: dict, S: dict, w: str, base: pd.DataFrame, pool, n_boot: int) -> dict:
    """PLAN revision 2: test D (EV decode of the 0.5/0.5 M03+M04_A mean) against the section-6 winner `w` with (a)(b)(c).
    D never changes which of A/B/C is W; if D is not complete / not reproduced / late, status is 'abandoned' and W stays."""
    files = sorted(wd.glob(f"oof_{lib.D_NAME}_fold*.parquet"))
    gate = repro_state(wd)
    info = {"status": "abandoned", "adopt": False, "decode": lib.D_DECODE, "w_vs": w, "n_folds_present": len(files),
            "repro_gate": gate, "repro_verified": gate["ok"], "deadline": lib.D_DEADLINE}
    if not files:
        info["reason"] = "no oof_D_fold*.parquet (D was not run)"
        return info
    if not partial and len(files) != 5:
        info["reason"] = f"D has {len(files)}/5 folds -> D abandoned, W kept"
        return info
    if not partial and _d_late(files):
        info["reason"] = f"D finished after {lib.D_DEADLINE} (PLAN revision 2) -> abandoned"
        return info
    metas = [json.loads((wd / f"meta_{lib.D_NAME}_fold{int(f.stem.split('fold')[-1])}.json").read_text()) for f in files]
    if any(m.get("features_sha256") != cov["features_sha256"] for m in metas):
        info["reason"] = "D features_sha256 differs from this coverage.json"
        return info
    D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if D["qid"].duplicated().any() or set(D["qid"]) != set(base["qid"]):
        info["reason"] = "D question set differs from C's (folds missing) -> abandoned"
        return info
    D = D.set_index("qid").loc[base["qid"]].reset_index()
    gate_needed = not partial
    if gate_needed and not gate["ok"]:
        info["reason"] = f"M03 reproduction gate not OK: {gate['folds']} -> D cancelled"
        return info
    if partial and gate["mismatch"]:
        info["reason"] = f"M03 reproduction gate MISMATCH: {gate['folds']} -> D cancelled"
        return info
    truth, groups, ns, qset = base["rank_id"].to_numpy(), base["player_id"].to_numpy(), base["n"].to_numpy(), base["qset"].to_numpy()
    q1, q2 = qset == "q1", qset == "q2"
    pool_q = np.array([pool.get(q, -1) for q in base["qid"]])
    pd_ = D[lib.PCOLS].to_numpy()
    S_d = rank_scores(lib.decide_d(pd_), truth)
    S_d_arg = rank_scores(decide_argmax(pd_), truth)
    S_w = S[w]

    def sub(m):
        return dict(groups=groups[m], truth=truth[m], ns=ns[m])

    eq = lib.equal_n_delta(S_d[q2], S_w[q2], n_boot=n_boot, level=0.975, **sub(q2))
    q1p = lib.equal_n_point(S_d[q1], S_w[q1], truth[q1], ns[q1], lib.NS_Q1)["delta"] if q1.any() else 0.0
    avail = {"q2_eq": eq, "q1_point": q1p, "q2_per_N": eq["per_N_delta"]}
    sel = lib.select_d(avail, complete=True, repro_ok=(gate["ok"] or partial))
    info.update(sel)
    if partial:
        info["reason"] = "PARTIAL/smoke run: M03 reproduction gate " + ("verified" if gate["ok"] else "NOT verified") + "; not a result"
    info["m03_best_iterations"] = [int(m["best_iteration"]) for m in sorted(metas, key=lambda m: m["fold"])]
    info["independence_note"] = ("Not an independent confirmation: tier-2 players largely also appear in tier 1, and E11 found D's combination on the "
                                 "same players (PLAN revision 2).")
    info["_scores"] = {lib.D_NAME: {"q1_eq": lib.equal_n_point(S_d[q1], np.zeros(q1.sum()), truth[q1], ns[q1], lib.NS_Q1)["delta"] if q1.any() else None,
                                    "q2_eq": lib.equal_n_point(S_d[q2], np.zeros(q2.sum()), truth[q2], ns[q2], NS)["delta"],
                                    "q2_eq_if_argmax": lib.equal_n_point(S_d_arg[q2], np.zeros(q2.sum()), truth[q2], ns[q2], NS)["delta"],
                                    "q2_per_N": {int(a): float((S_d[q2 & (ns == a)] * balanced_weights(truth[q2 & (ns == a)])).mean())
                                                 for a in NS if (q2 & (ns == a)).any()}}}
    info["_deltas"] = {f"D-{w}|q2_eq": eq, f"D-{w}|q1_point": q1p,
                       f"D-{w}|q2_by_pool": {lab: lib.equal_n_delta(S_d[q2 & mk], S_w[q2 & mk], n_boot=n_boot, level=0.95, **sub(q2 & mk))
                                              for lab, mk in (("pool_lt60", (pool_q < lib.POOL_SPLIT) & (pool_q >= 0)), ("pool_ge60", pool_q >= lib.POOL_SPLIT))
                                              if mk[q2].any()}}
    return info


# ------------------------------------------------------------------------------------------------ status
def stage_status() -> None:
    led = pd.read_parquet(lib.SCAN / "ledger.parquet")
    print("ledger:", {int(t): g["status"].value_counts().to_dict() for t, g in led.groupby("tier")}, f"-> {int((led['status'] == 'done').sum())}/{len(led)} done")
    for tag, c in (("formal", CACHE), ("partial/smoke", SCRATCH / "cache")):
        f = c / FEAT_MANIFEST
        if f.exists():
            fm = json.loads(f.read_text())
            print(f"features {tag}: {fm['n_present']}/{fm['n_shards_total']} shards, {fm['n_games']} games, complete={fm['complete']}, built {fm['created']}")
        else:
            print(f"features {tag}: not built")
    for tag, wd in (("formal", OUT / "oof"), ("partial/smoke", SCRATCH / "partial_oof")):
        n = len(list(wd.glob("oof_*_fold*.parquet"))) if wd.exists() else 0
        print(f"oof {tag}: {n} cand-folds in {wd}")
