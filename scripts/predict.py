"""One-shot inference skeleton: official test CSVs -> submission_rank.csv + submission_player.csv, then format check.

  .venv/bin/python scripts/predict.py train   [--task rank|player|both] [--player-model e04v2|l0_cosine|all] [--include-holdout]
  .venv/bin/python scripts/predict.py predict --data-dir DIR --out-dir OUT [--rank-model l0_lgbm|hsl_lgbm] [--player-model e04v2]
                                              [--hsl-features data/test_scan/hsl_test_games.parquet]   # hsl_lgbm only
  .venv/bin/python scripts/predict.py report  --data-dir DIR [--out report.json] [--no-train-overlap]
  .venv/bin/python scripts/predict.py sim     [--out .scratch/predict_sim]    # fake test files from H players, full flow, smoke scores

DIR holds the four official files (names as in docs/contest{1,2}/data.md):
  rank_prediction_test_{public,private}.csv, player_identification_test_{public,private}.csv,
  player_identification_candidates_{public,private}.csv
Pluggable models: RANK_MODELS / PLAYER_MODELS registries below (RankModel.predict, PlayerModel.rank).  CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gogame.aggregate import L0Store, aggregate_many, fit_sparse_selector, make_spec  # noqa: E402
from gogame.constants import RANKS, SEED  # noqa: E402
from gogame.coral import CoralModel, fit_coral  # noqa: E402
from gogame.memory import MemoryBaseline  # noqa: E402
from gogame.errors import DataFormatError  # noqa: E402
from gogame.features_l0 import DENSE_NAMES, compute_l0_many  # noqa: E402
from gogame.io import (GameRecord, explode_question, iter_candidate_games, normalize_qid,  # noqa: E402
                       read_candidates_csv, read_questions_csv, validate_questions_frame)
from gogame import player_rerank as prr  # noqa: E402
from gogame.models import RankModel as _LgbmBundle, decide_argmax, decide_ev, final_num_rounds, fit_lgbm  # noqa: E402
from gogame.questions import Bag, bag_class_weights, build_player_index, sample_train_bags  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402
from gogame.submission import write_player_submission, write_rank_submission  # noqa: E402

MODEL_DIR = ROOT / "data" / "models"
CACHE = ROOT / "data" / "cache"
SPLITS = ROOT / "data" / "splits"
TRAIN_DIR = ROOT / "data" / "training"
STORE_NAMES = tuple("l0_" + n for n in DENSE_NAMES)
FILES = {
    "rank_pub": "rank_prediction_test_public.csv", "rank_priv": "rank_prediction_test_private.csv",
    "player_pub": "player_identification_test_public.csv", "player_priv": "player_identification_test_private.csv",
    "cand_pub": "player_identification_candidates_public.csv", "cand_priv": "player_identification_candidates_private.csv",
}
LGBM_PARAMS = {"learning_rate": 0.1, "max_bin": 63, "log_every": 25}   # as E01 (run_e01.PARAMS)
BAGS_PER_PLAYER_COLOR = 8
MIN_DF = 0.005
DEFAULT_ROUNDS = 191   # E01 final_meta.json: median of OOF best iterations


# ================================================================================ data containers
@dataclass
class Question:
    qid: str                    # normalised (pub_/priv_ prefixed) question id
    split: str                  # "pub" | "priv"
    color: int                  # 0=B, 1=W (colour of the first game; one colour per question)
    games: list[GameRecord]


def load_questions(path, split: str, task: str) -> list[Question]:
    df = read_questions_csv(path, split)
    bad = [i for i in validate_questions_frame(df, task) if not i.code.startswith("W_")]
    if bad:
        raise DataFormatError(f"{path}: {len(bad)} format issue(s), first: {bad[0].code} {bad[0].detail}")
    out = []
    for _, row in df.iterrows():
        games = explode_question(row, split)
        out.append(Question(normalize_qid(str(row["question_id"]), split), split, games[0].color, games))
    return out


def load_candidates(path, split: str) -> dict[str, list[GameRecord]]:
    df = read_candidates_csv(path, split)
    pool: dict[str, list[GameRecord]] = {}
    for g in iter_candidate_games(df, split):
        pool.setdefault(g.player_id, []).append(g)
    return pool


def l0_store(records: Sequence[GameRecord], n_jobs: int = 2) -> L0Store:
    """Per-game L0 features for the given records (uids must be unique)."""
    b = compute_l0_many(records, n_jobs=n_jobs)
    return L0Store(b.uids, b.dense, b.sparse, STORE_NAMES)


# ================================================================================ model interfaces
class RankModel:
    """Predict a rank id (0..9 = D..6D) for each question."""
    name = "base"

    def predict(self, questions: Sequence[Question]) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def load(cls, model_dir: Path) -> "RankModel":
        raise NotImplementedError


class PlayerModel:
    """Top-5 candidate player ids per query, best first.  `candidates`: player_id -> that player's black games."""
    name = "base"

    def rank(self, candidates: dict[str, list[GameRecord]], queries: Sequence[Question]) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def load(cls, model_dir: Path) -> "PlayerModel":
        raise NotImplementedError


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `scripts/predict.py train` first")
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------------------------------------- Rank: L0 LightGBM (E01 final)
class L0LgbmRank(RankModel):
    """E01 L0 hand-crafted features -> bag aggregation -> LightGBM; argmax of the prior-corrected probabilities."""
    name = "l0_lgbm"
    FILE = "rank_l0_lgbm.pkl"

    def __init__(self, bundle: _LgbmBundle, meta: dict, n_jobs: int = 2):
        self.bundle, self.meta, self.n_jobs = bundle, meta, n_jobs

    # ---- training: store = per-game features, meta = frame(uid, player_id, rank_id, color[0/1]) of the TRAIN games
    @classmethod
    def fit(cls, store: L0Store, meta: pd.DataFrame, rounds: int = DEFAULT_ROUNDS, threads: int = 2,
            bags_per_player_color: int = BAGS_PER_PLAYER_COLOR, min_df: float = MIN_DF, meta_extra: dict | None = None):
        index = build_player_index(meta.assign(color_id=meta["color"]))
        players = sorted(index.games)
        bags = sample_train_bags(index, players, bags_per_player_color, (5, 20), seed=derive_seed(SEED, "train_bags", "final"))
        w = bag_class_weights(bags, "inv_players")
        used = sorted({u for b in bags for u in b.uids})
        cols = fit_sparse_selector(store.sparse[store.rows(used)], min_df)
        spec = make_spec(store.names, cols)
        X = aggregate_many(bags, store, spec)
        params = {**LGBM_PARAMS, "num_threads": threads, "num_boost_round": rounds}
        m = fit_lgbm(X, np.array([b.rank_id for b in bags]), None, None, params, derive_seed(SEED, "lgbm"), w, None)
        m.spec = spec
        meta_out = {"model": cls.name, "rounds": rounds, "n_train_players": len(players), "n_train_bags": len(bags),
                    "n_sparse_cols": int(len(cols)), **(meta_extra or {})}
        return cls(m, meta_out)

    def save(self, model_dir: Path) -> Path:
        b = self.bundle
        p = model_dir / self.FILE
        _dump({"model_str": b.booster.model_to_string(), "spec": b.spec, "params": b.params,
               "train_prior": b.train_prior, "best_iteration": b.best_iteration, "meta": self.meta}, p)
        return p

    @classmethod
    def load(cls, model_dir: Path, n_jobs: int = 2):
        import lightgbm as lgb
        d = _load(Path(model_dir) / cls.FILE)
        booster = lgb.Booster(model_str=d["model_str"])
        bundle = _LgbmBundle(booster, d["spec"], d["params"], d["train_prior"], d["best_iteration"])
        return cls(bundle, d["meta"], n_jobs)

    def predict_proba(self, questions: Sequence[Question]) -> np.ndarray:
        games = [g for q in questions for g in q.games]
        store = l0_store(games, self.n_jobs)
        bags = [Bag(q.qid, "", -1, q.color, len(q.games), tuple(g.uid for g in q.games)) for q in questions]
        X = aggregate_many(bags, store, self.bundle.spec)
        return self.bundle.predict_proba(X)

    def predict(self, questions: Sequence[Question]) -> np.ndarray:
        return decide_argmax(self.predict_proba(questions))


# -------------------------------------------------------------------------------- Player: L0 centroid cosine (M01)
class L0CosinePlayer(PlayerModel):
    """z-scored L0 game vector (dense + sparse cols with df >= min_df), mean per bag/player, cosine, Top-5."""
    name = "l0_cosine"
    FILE = "player_l0_cosine.pkl"

    def __init__(self, state: dict, n_jobs: int = 2):
        self.s, self.n_jobs = state, n_jobs

    @classmethod
    def fit(cls, store: L0Store, rows: np.ndarray | None = None, min_df: float = MIN_DF, meta_extra: dict | None = None):
        """Standardiser + sparse-column selection from `rows` of the store (default all)."""
        rows = np.arange(len(store.uids)) if rows is None else np.asarray(rows)
        D = store.dense[rows].astype(np.float64)
        mu, sd = np.nanmean(D, axis=0), np.nanstd(D, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        SP = store.sparse[rows]
        cols = fit_sparse_selector(SP, min_df)
        S = SP[:, cols]
        smu = np.asarray(S.mean(axis=0)).ravel()
        ssd = np.sqrt(np.maximum(np.asarray(S.multiply(S).mean(axis=0)).ravel() - smu ** 2, 0))
        ssd = np.where(ssd > 0, ssd, 1.0)
        return cls({"mu": mu, "sd": sd, "cols": cols, "smu": smu, "ssd": ssd,
                    "meta": {"model": cls.name, "n_fit_games": int(len(rows)), "n_sparse_cols": int(len(cols)),
                             **(meta_extra or {})}})

    @property
    def meta(self) -> dict:
        return self.s["meta"]

    def save(self, model_dir: Path) -> Path:
        p = Path(model_dir) / self.FILE
        _dump(self.s, p)
        return p

    @classmethod
    def load(cls, model_dir: Path, n_jobs: int = 2):
        return cls(_load(Path(model_dir) / cls.FILE), n_jobs)

    def _z(self, store: L0Store, rows: np.ndarray) -> np.ndarray:
        s = self.s
        Zd = np.nan_to_num((store.dense[rows] - s["mu"]) / s["sd"], nan=0.0)
        Zs = (store.sparse[rows][:, s["cols"]].toarray() - s["smu"]) / s["ssd"]
        return np.hstack([Zd, Zs]).astype(np.float32)

    def _unit_means(self, store: L0Store, groups: list[list[str]], chunk: int = 4000) -> np.ndarray:
        """L2-normalised mean z-vector of every group of uids."""
        dim = len(self.s["mu"]) + len(self.s["cols"])
        out = np.zeros((len(groups), dim), dtype=np.float64)
        gid = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)]).astype(np.int64)
        rows = store.rows([u for g in groups for u in g])
        for a in range(0, len(rows), chunk):
            np.add.at(out, gid[a:a + chunk], self._z(store, rows[a:a + chunk]))
        out /= np.array([max(len(g), 1) for g in groups])[:, None]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(n > 0, n, 1.0)).astype(np.float32)

    def rank(self, candidates: dict[str, list[GameRecord]], queries: Sequence[Question]) -> np.ndarray:
        return self.rank_scored(candidates, queries)[0]

    def rank_scored(self, candidates: dict[str, list[GameRecord]], queries: Sequence[Question]):
        pool = sorted(candidates)
        if len(pool) < 5:
            raise DataFormatError(f"candidate pool has {len(pool)} players, need at least 5 for Top-5")
        games = [g for p in pool for g in candidates[p]] + [g for q in queries for g in q.games]
        store = l0_store(games, self.n_jobs)
        C = self._unit_means(store, [[g.uid for g in candidates[p]] for p in pool])
        Q = self._unit_means(store, [[g.uid for g in q.games] for q in queries])
        cos = Q @ C.T
        top = np.argsort(-cos, axis=1, kind="stable")[:, :5]          # ties -> alphabetical player id (pool is sorted)
        pool_arr = np.array(pool, dtype=object)
        return pool_arr[top], np.take_along_axis(cos, top, axis=1)


# -------------------------------------------------------------------------------- Player: E04v2 (cosine Top-20 -> LightGBM lambdarank)
class E04v2Player(L0CosinePlayer):
    """E05 frozen method (experiments/E05_player_comp/PLAN.md).  Same z-scored L0 space as M01; the candidate/feature
    computation is gogame.player_rerank (shared with scripts/experiments/e05/run.py).  Training: pools of 400 players
    (library 95 black games, queries N in 5/10/15/20 from the remaining games), 10 draws, 400 fixed rounds, no early
    stopping.  Players are never put into the model as ids."""
    name = "e04v2"
    FILE = "player_e04v2.pkl"
    POOL, DRAWS, FIT_LIB, NS = 400, 10, 95, (5, 10, 15, 20)

    def __init__(self, state: dict, n_jobs: int = 2):
        super().__init__(state, n_jobs)
        import lightgbm as lgb
        self.booster = lgb.Booster(model_str=state["model_str"])

    @classmethod
    def fit(cls, store: L0Store, meta: pd.DataFrame, train_players: set, rounds: int = prr.LGB_ROUNDS, threads: int = 6,
            pool_size: int = POOL, n_draws: int = DRAWS, lib_games: int = FIT_LIB, min_df: float = MIN_DF,
            meta_extra: dict | None = None, log=print):
        """store/meta: the L0 cache (aligned, meta has uid, player_id, rank_id, color 0/1).  Everything (z-score,
        sparse columns, column selection, pools, trees) is fitted on `train_players` only."""
        import lightgbm as lgb
        from gogame.questions import build_player_pool_sim
        t0 = time.perf_counter()
        is_tr = meta["player_id"].isin(train_players).to_numpy()
        base = L0CosinePlayer.fit(store, np.flatnonzero(is_tr), min_df)           # standardiser on ALL train games
        tmp = cls.__new__(cls)
        tmp.s, tmp.n_jobs = base.s, 1
        black = np.flatnonzero(is_tr & (meta["color"].to_numpy() == 0))
        bm = meta.iloc[black]
        nb = bm.groupby("player_id").size()
        eligible = sorted(nb.index[nb >= lib_games + 5])                          # can form library + >=5 query games
        bm = bm[bm["player_id"].isin(set(eligible))]
        black = black[meta["player_id"].iloc[black].isin(set(eligible)).to_numpy()]
        Z = np.empty((len(black), len(base.s["mu"]) + len(base.s["cols"])), np.float32)
        for a in range(0, len(black), 20000):
            Z[a:a + 20000] = tmp._z(store, black[a:a + 20000])
        rowof = {u: i for i, u in enumerate(meta["uid"].to_numpy()[black])}
        sel = prr.fratio_sparse_select(Z, bm["player_id"].to_numpy(), len(base.s["mu"]))
        index = build_player_index(bm.assign(color_id=bm["color"]))
        sims = build_player_pool_sim(index, eligible, pool_size, n_draws, lib_games, cls.NS,
                                     seed=derive_seed(2026, "E05", "fit"))
        Xs, ys, k = [], [], prr.K
        for sm in sims:
            pool = list(sm.players)
            pos = {p: i for i, p in enumerate(pool)}
            rows = lambda uids: np.fromiter((rowof[u] for u in uids), np.int64, len(uids))   # noqa: E731
            f = prr.candidate_features([Z[rows(sm.library[p])] for p in pool], [Z[rows(q.uids)] for q in sm.queries], sel,
                                       prr.K, len(base.s["mu"]))
            truth = np.array([pos[q.player_id] for q in sm.queries])
            k = f["k"]
            Xs.append(f["X"])
            ys.append((f["top"] == truth[:, None]).astype(np.int8).reshape(-1))
            log(f"  fit draw {sm.draw}: {len(truth)} queries, recall@{prr.K} {(f['top'] == truth[:, None]).any(1).mean():.3f}"
                f"  {time.perf_counter() - t0:.0f}s")
        X, y = np.concatenate(Xs), np.concatenate(ys)
        del Xs
        ds = lgb.Dataset(X, y, group=np.full(len(y) // k, k), free_raw_data=False)
        model = lgb.train(prr.lgb_params(derive_seed(2026, "E05", "lgbm") % 2**31, threads), ds, num_boost_round=rounds)
        state = dict(base.s)
        state.update(sel=sel, model_str=model.model_to_string(),
                     meta={**base.s["meta"], "model": cls.name, "rounds": rounds, "pool_size": pool_size,
                           "n_draws": n_draws, "fit_lib_games": lib_games, "n_pool_players": len(eligible),
                           "n_train_groups": int(len(y) // k), "n_features": int(X.shape[1]),
                           "fit_seconds": round(time.perf_counter() - t0), **(meta_extra or {})})
        return cls(state, threads)

    def save(self, model_dir: Path) -> Path:
        p = super().save(model_dir)
        (Path(model_dir) / "player_e04v2.txt").write_text(self.s["model_str"])     # LightGBM text model (inspection)
        return p

    def rank(self, candidates: dict[str, list[GameRecord]], queries: Sequence[Question]) -> np.ndarray:
        pool = sorted(candidates)
        if len(pool) < 5:
            raise DataFormatError(f"candidate pool has {len(pool)} players, need at least 5 for Top-5")
        games = [g for p in pool for g in candidates[p]] + [g for q in queries for g in q.games]
        store = l0_store(games, self.n_jobs)
        Z = np.empty((len(store.uids), len(self.s["mu"]) + len(self.s["cols"])), np.float32)
        for a in range(0, len(Z), 20000):
            Z[a:a + 20000] = self._z(store, np.arange(a, min(a + 20000, len(Z))))
        rowof = {u: i for i, u in enumerate(store.uids)}
        rows = lambda uids: np.fromiter((rowof[u] for u in uids), np.int64, len(uids))   # noqa: E731
        f = prr.candidate_features([Z[rows([g.uid for g in candidates[p]])] for p in pool],
                                   [Z[rows([g.uid for g in q.games])] for q in queries], self.s["sel"], prr.K, len(self.s["mu"]))
        score = self.booster.predict(f["X"], num_threads=self.n_jobs)
        return np.array(pool, dtype=object)[prr.top5_indices(f["top"], score)]


# -------------------------------------------------------------------------------- Rank: HSL (+L0) LightGBM (E06 formal, placeholder until E06 is done)
HSL_TEST_FEATURES = ROOT / "data" / "test_scan" / "hsl_test_games.parquet"      # written by scripts/predict_hsl.py features
E06_CACHE = CACHE / "e06"
E06_SELECTION = ROOT / "experiments" / "E06_rank_formal" / "oof" / "selection.json"


def hsl_store(df: pd.DataFrame) -> L0Store:
    """Per-game HSL frame (uid + 104 float columns, as data/cache/e06/hsl_games.parquet) -> store.  Same construction as
    scripts/experiments/e06/pipeline.py Ctx (hs / sp_h), so bag aggregates are identical to the E06 training features."""
    names = tuple(c for c in df.columns if c != "uid")
    return L0Store(df["uid"].tolist(), df[list(names)].to_numpy(np.float32), None, names)


def hsl_spec(names) -> "object":
    return make_spec(tuple(names), np.zeros(0, np.int32))


def check_hsl_color_semantics(hsl_path, log=print) -> str | None:
    """The HSL features were extracted under one color_semantics (recorded in features_manifest.json / prepare_meta.json next to
    the parquet); this run's L0 / bag handling uses another one -> DataFormatError.  Missing record (legacy files) -> warning."""
    from gogame.sgf import get_color_semantics
    mine = get_color_semantics()
    theirs = None
    for fn in ("features_manifest.json", "prepare_meta.json"):
        f = Path(hsl_path).parent / fn
        if f.exists():
            theirs = json.loads(f.read_text()).get("color_semantics")
            if theirs:
                break
    if theirs is None:
        log(f"WARNING: no color_semantics recorded next to {hsl_path}; cannot cross-check against {mine!r}")
        return None
    if theirs != mine:
        raise DataFormatError(f"color_semantics mismatch: HSL features were extracted with {theirs!r}, this run uses {mine!r} "
                              f"(pass --color-semantics {theirs} or re-run prepare/extraction)")
    return theirs


class HslLgbmRank(RankModel):
    """E06 Rank formal: bag aggregates (mean/std/q25/q50/q75) of the per-game KataGo HumanSL features, optionally plus L0
    (E06 M03).  The model file rank_hsl_lgbm.pkl is produced only after E06 selected a LightGBM main method:
    `predict.py train --task rank --rank-model hsl_lgbm` (reads experiments/E06_rank_formal/oof/selection.json).
    Per-game HSL features of the TEST games come from `scripts/predict_hsl.py features` (parquet keyed by the test uids
    '<pub|priv>_<qid>#<slot>').  Games without HSL (engine-skipped) are dropped from their bag; a question with no HSL game
    at all is an error (nothing sensible to predict from)."""
    name = "hsl_lgbm"
    FILE = "rank_hsl_lgbm.pkl"
    MAX_DROP_FRAC = 0.01        # more than this fraction of test games without HSL -> DataFormatError (else a warning)

    def __init__(self, bundle: _LgbmBundle, meta: dict, n_jobs: int = 2, spec_l0=None, hsl_path: Path | str | None = None):
        self.bundle, self.meta, self.n_jobs, self.spec_l0 = bundle, meta, n_jobs, spec_l0
        self.max_drop_frac = self.MAX_DROP_FRAC
        self.hsl_path = Path(hsl_path) if hsl_path else HSL_TEST_FEATURES

    def set_features(self, path) -> None:
        self.hsl_path = Path(path)

    # ---- training (after E06): hsl: per-game HSL frame; meta: uid, player_id, rank_id, color(0/1) of the TRAIN games;
    # l0_store: needed iff use_l0 (rows must cover the HSL uids)
    @classmethod
    def fit(cls, hsl: pd.DataFrame, meta: pd.DataFrame, rounds: int | None, threads: int = 4, use_l0: bool = False,
            l0: L0Store | None = None, seed_tag: str = "final", meta_extra: dict | None = None, min_df: float = MIN_DF,
            method: str = "lgbm", bags_per_pc: int = BAGS_PER_PLAYER_COLOR, n_range: tuple = (5, 15), seed_ns: str = "e06"):
        """method 'lgbm' (E06 M02 / M03) or 'coral' (E06 M04: CORAL ordinal logistic on the HSL aggregate, inv_players weights)."""
        if method not in ("lgbm", "coral", "ens"):
            raise ValueError(f"unknown method {method!r}")
        if method == "coral" and use_l0:
            raise ValueError("E06 M04 uses HSL features only")
        if method == "ens":
            use_l0 = True                  # E10 candidate D = M03 (L0+HSL LightGBM) + M04 (HSL CORAL), 0.5/0.5, EV decode
        sub = meta[meta["uid"].isin(set(hsl["uid"]))].reset_index(drop=True)
        index = build_player_index(sub.assign(color_id=sub["color"]))
        players = sorted(index.games)
        bags = sample_train_bags(index, players, bags_per_pc, tuple(n_range), seed=derive_seed(SEED, seed_ns, "train_bags", seed_tag))
        w = bag_class_weights(bags, "inv_players")
        hs = hsl_store(hsl)
        X = aggregate_many(bags, hs, hsl_spec(hs.names))
        spec_l0 = None
        Xh = X
        if use_l0:
            if l0 is None:
                raise DataFormatError("use_l0 needs the L0 store")
            cols = fit_sparse_selector(l0.sparse[l0.rows(sorted({u for b in bags for u in b.uids}))], min_df)
            spec_l0 = make_spec(l0.names, cols)
            X = np.hstack([aggregate_many(bags, l0, spec_l0), X[:, :-2]])      # E06 M03: drop the duplicate bag_n / is_white
        y = np.array([b.rank_id for b in bags])
        if method == "ens":
            params = {**LGBM_PARAMS, "num_threads": threads, "num_boost_round": rounds}
            m03 = fit_lgbm(X, y, None, None, params, derive_seed(SEED, "lgbm"), w, None)
            m03.spec = hsl_spec(hs.names)
            del X
            m = _EnsBundle(m03, fit_coral(Xh, y, w), ENS_W_M03)
        elif method == "coral":
            m = fit_coral(X, y, w)
        else:
            params = {**LGBM_PARAMS, "num_threads": threads, "num_boost_round": rounds}
            m = fit_lgbm(X, y, None, None, params, derive_seed(SEED, "lgbm"), w, None)
            m.spec = hsl_spec(hs.names)
        meta_out = {"model": cls.name, "method": method, "rounds": rounds, "features": "l0+hsl" if use_l0 else "hsl",
                    "decode": "ev" if method == "ens" else "argmax",
                    "n_train_players": len(players), "n_train_bags": len(bags), "bags_per_pc": int(bags_per_pc), "n_range": list(n_range),
                    "n_hsl_names": len(hs.names),
                    "hsl_names": list(hs.names), **(meta_extra or {})}
        if method == "ens":
            meta_out["decode"] = "ev"
        return (HslEnsembleRank if method == "ens" else cls)(m, meta_out, threads, spec_l0)

    def save(self, model_dir: Path) -> Path:
        b = self.bundle
        p = Path(model_dir) / self.FILE
        if isinstance(b, CoralModel):
            _dump({"coral": b, "spec_l0": None, "meta": self.meta}, p)
            return p
        if isinstance(b, _EnsBundle):
            g = b.m03
            _dump({"ens": {"m03": {"model_str": g.booster.model_to_string(), "spec": g.spec, "params": g.params, "train_prior": g.train_prior,
                                   "best_iteration": g.best_iteration}, "coral": b.m04, "w_m03": b.w_m03},
                   "spec_l0": self.spec_l0, "meta": self.meta}, p)
            return p
        _dump({"model_str": b.booster.model_to_string(), "spec": b.spec, "spec_l0": self.spec_l0, "params": b.params,
               "train_prior": b.train_prior, "best_iteration": b.best_iteration, "meta": self.meta}, p)
        return p

    @classmethod
    def load(cls, model_dir: Path, n_jobs: int = 2):
        import lightgbm as lgb
        p = Path(model_dir) / cls.FILE
        if not p.exists():
            raise FileNotFoundError(
                f"rank model 'hsl_lgbm' is not trained yet: {p} not found.  It is produced only after E06 (Rank formal) has "
                f"chosen its main LightGBM method (experiments/E06_rank_formal/oof/selection.json), by "
                f"`predict.py train --task rank --rank-model hsl_lgbm [--include-holdout]`.  Until then use --rank-model l0_lgbm.")
        d = _load(p)
        if "ens" in d:
            e = d["ens"]
            g = e["m03"]
            m03 = _LgbmBundle(lgb.Booster(model_str=g["model_str"]), g["spec"], g["params"], g["train_prior"], g["best_iteration"])
            return HslEnsembleRank(_EnsBundle(m03, e["coral"], e["w_m03"]), d["meta"], n_jobs, d.get("spec_l0"))
        if "coral" in d:
            return cls(d["coral"], d["meta"], n_jobs, None)
        booster = lgb.Booster(model_str=d["model_str"])
        return cls(_LgbmBundle(booster, d["spec"], d["params"], d["train_prior"], d["best_iteration"]), d["meta"], n_jobs, d.get("spec_l0"))

    def _hsl_frame(self) -> pd.DataFrame:
        if not self.hsl_path.exists():
            raise FileNotFoundError(f"test-set HSL features not found: {self.hsl_path}.  Run `scripts/predict_hsl.py prepare`, the "
                                    f"Colab extraction (`predict_hsl.py run`) and `predict_hsl.py features` first (docs/runbook_1104.md).")
        check_hsl_color_semantics(self.hsl_path)
        df = pd.read_parquet(self.hsl_path)
        want = list(self.meta.get("hsl_names") or [])
        if want and [c for c in df.columns if c != "uid"] != want:
            raise DataFormatError(f"{self.hsl_path}: HSL columns differ from the ones the model was trained on")
        return df

    def _hsl_features(self, questions: Sequence[Question]):
        """(HSL bag aggregate Xh, bags, uids that have HSL).  Drop / empty-question checks live here."""
        df = self._hsl_frame()
        have = set(df["uid"])
        bags, empty, dropped = [], [], 0
        for q in questions:
            us = tuple(g.uid for g in q.games if g.uid in have)
            dropped += len(q.games) - len(us)
            if not us:
                empty.append(q.qid)
            bags.append(Bag(q.qid, "", -1, q.color, len(us), us))
        if empty:
            raise DataFormatError(f"{len(empty)} question(s) have no HSL game at all (first: {empty[:3]}); "
                                  f"check `predict_hsl.py features` coverage or fall back to l0_lgbm for them")
        total = sum(len(q.games) for q in questions)
        if total and dropped / total > self.max_drop_frac:
            raise DataFormatError(f"hsl_lgbm: {dropped} of {total} test game(s) ({dropped / total:.2%}) have no HSL features, "
                                  f"more than the allowed {self.max_drop_frac:.2%} (--max-hsl-drop-frac); "
                                  f"check `predict_hsl.py features` coverage")
        if dropped:
            print(f"hsl_lgbm: {dropped} test game(s) without HSL features dropped from their bags", file=sys.stderr)
        hs = hsl_store(df)
        return aggregate_many(bags, hs, hsl_spec(hs.names)), bags, have

    def _m03_matrix(self, Xh, questions, bags, have) -> np.ndarray:
        games = [g for q in questions for g in q.games if g.uid in have]
        return np.hstack([aggregate_many(bags, l0_store(games, self.n_jobs), self.spec_l0), Xh[:, :-2]])

    def predict_proba(self, questions: Sequence[Question]) -> np.ndarray:
        X, bags, have = self._hsl_features(questions)
        if self.spec_l0 is not None:
            X = self._m03_matrix(X, questions, bags, have)
        return self.bundle.predict_proba(X)

    def predict(self, questions: Sequence[Question]) -> np.ndarray:
        p = self.predict_proba(questions)
        return decide_ev(p) if self.meta.get("decode") == "ev" else decide_argmax(p)


ENS_W_M03 = 0.5          # E10 candidate D: fixed weight of M03 in the probability mean (PLAN revision 2)


class _EnsBundle:
    """M03 (LightGBM bundle on L0+HSL features) + M04 (CoralModel on HSL features); probability mean."""

    def __init__(self, m03, m04, w_m03: float = ENS_W_M03):
        self.m03, self.m04, self.w_m03 = m03, m04, float(w_m03)

    def predict_proba_parts(self, X03: np.ndarray, Xh: np.ndarray) -> np.ndarray:
        p = self.w_m03 * self.m03.predict_proba(X03) + (1 - self.w_m03) * self.m04.predict_proba(Xh)
        return p / p.sum(axis=1, keepdims=True)


class HslEnsembleRank(HslLgbmRank):
    """E10 candidate D: 0.5 * P(M03) + 0.5 * P(M04) with EV decoding (meta['decode'] == 'ev').  Same file name / registry entry as
    hsl_lgbm (`rank_hsl_lgbm.pkl`), so `--rank-model hsl_lgbm` loads whichever was trained.  Needs spec_l0 (L0 of the test games)."""

    def predict_proba(self, questions: Sequence[Question]) -> np.ndarray:
        Xh, bags, have = self._hsl_features(questions)
        return self.bundle.predict_proba_parts(self._m03_matrix(Xh, questions, bags, have), Xh)


# -------------------------------------------------------------------------------- Rank: memory route (SDD 10.1 / rank_via_player)
class MemoryRank(RankModel):
    """Overlap route: test players are (probably) training players, so a Rank question = identify the most similar training
    player and take that player's rank.  Same bag space as l0_lgbm (aggregated L0 features, z-scored), one L2-normalised
    centroid per (player, colour) from sampled training bags, k=1 cosine nearest centroid (gogame.memory.MemoryBaseline).
    Use only after the overlap report says so (docs/runbook_1104.md); on unseen players it is far worse than l0_lgbm."""
    name = "memory"
    FILE = "rank_memory.pkl"

    def __init__(self, base: MemoryBaseline, spec, meta: dict, n_jobs: int = 2):
        self.base, self.spec, self.meta, self.n_jobs = base, spec, meta, n_jobs

    @classmethod
    def fit(cls, store: L0Store, meta: pd.DataFrame, bags_per_player_color: int = BAGS_PER_PLAYER_COLOR, min_df: float = MIN_DF,
            meta_extra: dict | None = None, n_range: tuple[int, int] = (5, 20)):
        index = build_player_index(meta.assign(color_id=meta["color"]))
        players = sorted(index.games)
        bags = sample_train_bags(index, players, bags_per_player_color, n_range, seed=derive_seed(SEED, "memory_bags", "final"))
        used = sorted({u for b in bags for u in b.uids})
        spec = make_spec(store.names, fit_sparse_selector(store.sparse[store.rows(used)], min_df))
        X = aggregate_many(bags, store, spec)
        base = MemoryBaseline().fit(X, np.array([b.player_id for b in bags]), np.array([b.rank_id for b in bags]),
                                    np.array([b.color for b in bags]))
        return cls(base, spec, {"model": cls.name, "n_train_players": len(players), "n_train_bags": len(bags),
                                "n_centroids": base.n_centroids, **(meta_extra or {})})

    def save(self, model_dir: Path) -> Path:
        p = Path(model_dir) / self.FILE
        _dump({"base": self.base, "spec": self.spec, "meta": self.meta}, p)
        return p

    @classmethod
    def load(cls, model_dir: Path, n_jobs: int = 2):
        p = Path(model_dir) / cls.FILE
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; train it with `predict.py train --task rank --rank-model memory "
                                    f"[--include-holdout] --model-dir ...` (the memory route is only for overlapping test players)")
        d = _load(p)
        return cls(d["base"], d["spec"], d["meta"], n_jobs)

    def bag_vectors(self, groups: Sequence[Sequence[GameRecord]], colors: Sequence[int]) -> np.ndarray:
        """Aggregated (not yet z-scored) bag vectors for the given game groups."""
        games = [g for gs in groups for g in gs]
        store = l0_store(games, self.n_jobs)
        bags = [Bag(f"b{i}", "", -1, int(c), len(gs), tuple(g.uid for g in gs)) for i, (gs, c) in enumerate(zip(groups, colors))]
        return aggregate_many(bags, store, self.spec)

    def predict_scored(self, questions: Sequence[Question]):
        X = self.bag_vectors([q.games for q in questions], [q.color for q in questions])
        return self.base.predict(X, k=1)

    def predict(self, questions: Sequence[Question]) -> np.ndarray:
        return np.asarray(self.predict_scored(questions)[0], dtype=np.int64)


RANK_MODELS = {L0LgbmRank.name: L0LgbmRank, HslLgbmRank.name: HslLgbmRank, MemoryRank.name: MemoryRank}
PLAYER_MODELS = {E04v2Player.name: E04v2Player, L0CosinePlayer.name: L0CosinePlayer}
DEFAULT_PLAYER_MODEL = E04v2Player.name


# ================================================================================ training from the E01 cache
def _cache_inputs():
    from gogame.gametable import load_table
    from gogame.split import read_split
    split = read_split(SPLITS / "split.parquet")
    meta = pd.read_parquet(CACHE / "games_meta.parquet")
    store = L0Store.from_table(load_table(CACHE / "l0"))
    assert list(store.uids) == meta["uid"].tolist()
    return split, meta, store


def train_hsl_rank(model_dir: Path, include_holdout: bool, threads: int, rounds: int | None,
                   e06_cache: Path = E06_CACHE, selection: Path = E06_SELECTION, rank_config: Path | None = None) -> Path:
    """Train the E06 formal Rank model (HSL or L0+HSL LightGBM) with the method/rounds E06 selected.  Refuses until E06 is
    complete (full tier-1 features, formal selection.json).

    rank_config (E10, `--rank-config`): path of an E10 selection.json (written by `scripts.experiments.e10 report`); it
    replaces the E06 cache / selection / bag settings (cache dir, bags_per_pc, n_range, features_sha256).  None = E06
    behaviour, unchanged."""
    from gogame.gametable import load_table
    from gogame.split import assert_no_player_leak, read_split
    bags_kw = {}
    if rank_config is not None:
        cfg = json.loads(Path(rank_config).read_text())
        if cfg.get("candidate") not in ("A", "B", "D"):
            raise RuntimeError(f"--rank-config {rank_config}: candidate {cfg.get('candidate')!r}; only an E10 selection that chose A, B or D "
                               f"needs it (C = E06 M04: omit --rank-config)")
        e06_cache, selection = ROOT / cfg["cache"], Path(rank_config)
        bags_kw = {"bags_per_pc": int(cfg["bags_per_pc"]), "n_range": tuple(cfg["n_range"]), "seed_ns": cfg.get("seed_ns", "e10")}
    fm_path = Path(e06_cache) / "features_manifest.json"
    if not fm_path.exists():
        raise FileNotFoundError(f"{fm_path} missing: E06 `features` has not been run (tier-1 scan not merged)")
    fm = json.loads(fm_path.read_text())
    if not fm.get("complete"):
        raise RuntimeError(f"features are partial ({fm.get('n_present')}/{fm.get('n_shards_total', fm.get('n_shards_tier1'))} shards); wait for the full scan")
    if not Path(selection).exists():
        raise FileNotFoundError(f"{selection} missing: E06 oof/report have not finished, so no Rank method is selected yet")
    sel = json.loads(Path(selection).read_text())
    if sel.get("partial") or not sel.get("valid_for_confirm"):
        raise RuntimeError("E06 selection.json comes from a partial scan; refusing")
    if sel.get("features_sha256") != fm.get("sha256"):
        raise RuntimeError("E06 features changed since selection.json was written; rerun E06 oof/report")
    main = sel["main"]
    if main not in ("M02", "M03", "M04") and not (rank_config is not None and main == "M03+M04" and sel.get("candidate") == "D"):
        raise NotImplementedError(f"E06 selected {main!r}: hsl_lgbm supports M02 (HSL LightGBM), M03 (L0+HSL LightGBM), M04 (HSL CORAL), "
                                  f"and (--rank-config, E10 candidate D) M03+M04")
    ens = main == "M03+M04"
    hsl_path = Path(e06_cache) / "hsl_games.parquet"
    hsl = pd.read_parquet(hsl_path)
    split = read_split(SPLITS / "split.parquet")
    H = set(split.loc[split["is_holdout"], "player_id"])
    train_players = set(split["player_id"]) if include_holdout else set(split.loc[split["fold"] >= 0, "player_id"])
    if not include_holdout:
        assert_no_player_leak(train_players, H)
    meta = pd.read_parquet(CACHE / "games_meta.parquet")
    meta = meta[meta["player_id"].isin(train_players)].reset_index(drop=True)
    r = None if main == "M04" else (rounds or final_num_rounds(sel["oof_best_iterations"]["M03" if ens else main]))
    l0 = None
    if main == "M03" or ens:
        table = load_table(CACHE / "l0")
        pos = pd.Series(np.arange(len(table.uids)), index=table.uids)
        rows = pos.loc[hsl["uid"].to_numpy()].to_numpy()
        l0 = L0Store(hsl["uid"].tolist(), table.dense[rows], table.sparse[rows], table.names)
    extra = {"include_holdout": bool(include_holdout), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "e06_main": main,
             "decode": sel.get("decode", "argmax"), "features_sha256": fm["sha256"]}
    t0 = time.perf_counter()
    m = HslLgbmRank.fit(hsl, meta, r, threads, use_l0=main == "M03", l0=l0, meta_extra=extra,
                          method="ens" if ens else ("coral" if main == "M04" else "lgbm"), **bags_kw)
    p = m.save(model_dir)
    print("rank saved:", p, {k: v for k, v in m.meta.items() if k != "hsl_names"}, f"{time.perf_counter() - t0:.0f}s", flush=True)
    return p


def train_models(task: str, model_dir: Path, include_holdout: bool, threads: int, rounds: int | None,
                 player_model: str = DEFAULT_PLAYER_MODEL, rank_model: str = "l0_lgbm", rank_config: Path | None = None) -> None:
    from gogame.split import assert_no_player_leak
    if rank_model == MemoryRank.name and task in ("rank", "both"):
        split, meta, store = _cache_inputs()
        tp = set(split["player_id"]) if include_holdout else set(split.loc[split["fold"] >= 0, "player_id"])
        t0 = time.perf_counter()
        m = MemoryRank.fit(store, meta[meta["player_id"].isin(tp)].reset_index(drop=True),
                           meta_extra={"include_holdout": bool(include_holdout), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        print("rank saved:", m.save(model_dir), m.meta, f"{time.perf_counter() - t0:.0f}s", flush=True)
        if task == "rank":
            return
        task = "player"
    if rank_model == HslLgbmRank.name and task in ("rank", "both"):
        train_hsl_rank(model_dir, include_holdout, threads, rounds, rank_config=rank_config)
        if task == "rank":
            return
        task = "player"
    split, meta, store = _cache_inputs()
    H = set(split.loc[split["is_holdout"], "player_id"])
    train_players = set(split["player_id"]) if include_holdout else set(split.loc[split["fold"] >= 0, "player_id"])
    if not include_holdout:
        assert_no_player_leak(train_players, H)
    extra = {"include_holdout": bool(include_holdout), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if task in ("rank", "both"):
        fm = ROOT / "experiments" / "E01_l0" / "final_meta.json"
        r = rounds or (json.loads(fm.read_text())["rounds"] if fm.exists() else DEFAULT_ROUNDS)
        sub = meta[meta["player_id"].isin(train_players)].reset_index(drop=True)
        t0 = time.perf_counter()
        m = L0LgbmRank.fit(store, sub, rounds=r, threads=threads, meta_extra=extra)
        print("rank saved:", m.save(model_dir), m.meta, f"{time.perf_counter() - t0:.0f}s", flush=True)
    if task in ("player", "both"):
        if player_model in ("l0_cosine", "all"):
            rows = np.flatnonzero(meta["player_id"].isin(train_players).to_numpy())
            m = L0CosinePlayer.fit(store, rows, meta_extra=extra)
            print("player saved:", m.save(model_dir), m.meta, flush=True)
        if player_model in ("e04v2", "all"):
            t0 = time.perf_counter()
            m = E04v2Player.fit(store, meta, train_players, threads=threads, rounds=rounds or prr.LGB_ROUNDS, meta_extra=extra)
            print("player saved:", m.save(model_dir), m.meta, f"{time.perf_counter() - t0:.0f}s", flush=True)


# ================================================================================ predict
def format_check(data_dir: Path, strict: bool = False, log=print) -> dict:
    """SGF format report (root properties, HD, skip reasons) of every official test file present, printed before any model runs.
    Raises DataFormatError on WARN-level findings only when strict (--strict-format).  Returns {file key: report}."""
    from gogame.sgf import format_gate, format_report
    data_dir = Path(data_dir)
    out, bad = {}, []
    for key, fn in FILES.items():
        p = data_dir / fn
        if not p.exists():
            continue
        split = key.split("_")[1]
        if key.startswith("cand"):
            recs = list(iter_candidate_games(read_candidates_csv(p, split), split))
        else:
            recs = [g for _, row in read_questions_csv(p, split).iterrows() for g in explode_question(row, split)]
        rep = format_report([g.sgf for g in recs], [g.color for g in recs])
        out[key] = rep
        format_gate(rep, False, log, title=f"SGF format report {fn}")
        bad += [f"{fn}: {m}" for lvl, m in rep["warnings"] if lvl == "WARN"]
    if strict and bad:
        raise DataFormatError("--strict-format: " + " | ".join(bad))
    return out


def run_predict(data_dir: Path, out_dir: Path, rank_model: RankModel, player_model: PlayerModel, check: bool = True,
                log=print, strict_format: bool = False) -> dict:
    """Write submission_rank.csv / submission_player.csv into out_dir; returns {"rank": path, "player": path, "ok": bool}."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    format_check(data_dir, strict_format, log)
    t0 = time.perf_counter()
    rq = load_questions(data_dir / FILES["rank_pub"], "pub", "rank") + load_questions(data_dir / FILES["rank_priv"], "priv", "rank")
    log(f"rank questions: {len(rq)}")
    pred = np.asarray(rank_model.predict(rq))
    rp = out_dir / "submission_rank.csv"
    write_rank_submission([q.qid for q in rq], pred, rp)
    log(f"wrote {rp} ({time.perf_counter() - t0:.0f}s)")
    qids, tops = [], []
    for sp in ("pub", "priv"):
        pq = load_questions(data_dir / FILES[f"player_{sp}"], sp, "player")
        pool = load_candidates(data_dir / FILES[f"cand_{sp}"], sp)
        log(f"player {sp}: {len(pq)} queries, {len(pool)} candidates")
        top = np.asarray(player_model.rank(pool, pq), dtype=object)
        qids += [q.qid for q in pq]
        tops.append(top)
    pp = out_dir / "submission_player.csv"
    write_player_submission(qids, np.vstack(tops), pp)
    log(f"wrote {pp} ({time.perf_counter() - t0:.0f}s)")
    ok = True
    if check:
        import check_submission
        for task, p in (("rank", rp), ("player", pp)):
            rc = check_submission.main([str(p), "--task", task, "--data-dir", str(data_dir)])
            ok &= rc == 0
    return {"rank": rp, "player": pp, "ok": bool(ok)}


# ================================================================================ report
def _train_overlap(test_games: dict[str, list[GameRecord]], train_dir: Path, log=print, exclude_players: set | None = None) -> dict:
    """Stream the training CSVs once; keep only the keys that occur in the test games (low memory).
    test_games: label -> games.  Returns per label exact / same-colour / prefix stats."""
    import hashlib
    from gogame.io import read_train_csv
    from gogame.overlap import _tokens, exact_overlap, game_key, prefix_overlap
    ks = (6, 10, 16, 24, 30)
    want_exact = {game_key(g.sgf) for gs in test_games.values() for g in gs}
    want_pref = {k: set() for k in ks}
    from gogame.overlap import prefix_key
    for gs in test_games.values():
        for g in gs:
            for k in ks:
                pk = prefix_key(g.sgf, k)
                if pk is not None:
                    want_pref[k].add(pk)
    idx: dict[str, list[int]] = {}
    colors: list[int] = []
    pref_hit: dict[int, set] = {k: set() for k in ks}
    for r in RANKS:
        path = train_dir / f"train_{r}.csv"
        if not path.exists():
            log(f"  (skip {path.name}: missing)")
            continue
        df = read_train_csv(path, usecols=["player_id", "game_id", "rank", "color", "sgf_content"])
        if exclude_players:
            df = df[~df["player_id"].isin(exclude_players)]
        for s, c in zip(df["sgf_content"], df["color_id"]):
            t = _tokens(s)
            key = hashlib.sha1(";".join(t).encode()).hexdigest()
            if key in want_exact:
                idx.setdefault(key, []).append(len(colors))
                colors.append(int(c))
            for k in ks:
                if len(t) >= k:
                    pk = hashlib.sha1(";".join(t[:k]).encode()).hexdigest()[:20]
                    if pk in want_pref[k]:
                        pref_hit[k].add(pk)
        log(f"  scanned {path.name}")
    carr = np.array(colors, dtype=np.int8)
    out = {}
    for label, gs in test_games.items():
        ex = exact_overlap(gs, idx, carr)
        pr = prefix_overlap(gs, pref_hit, k=30)
        out[label] = {"n_games": ex.n_test, "exact": ex.n_exact, "exact_same_color": ex.n_same_color,
                      "exact_other_side": ex.n_other_side, "exact_same_color_frac": ex.same_color_frac,
                      "prefix_match": {str(k): v for k, v in pr.n_match.items()},
                      "prefix30_frac": pr.frac[30]}
    return out


def overlap_shift(mem: "MemoryRank", rank_qs: list, player_qs: list, pool: dict, log=print) -> dict:
    """SDD 10.2 evidence (c) and (d).  Space = the memory model's aggregated-L0 bag vectors, z-scored; centroids = the
    training players' (the model MUST be trained without H: include_holdout False); null = hold-out (H) players, who the
    centroids have never seen.  (c) share of test bags (Rank + Player questions) whose nearest-centroid cosine exceeds the
    99th percentile of the H bags' (N 5-20, SDD sizes); (d) share of candidate libraries (centroid of all their games) above
    the 99th percentile of H players' 100-game centroids."""
    from gogame.overlap import bag_centroid_shift, library_centroid_check
    if mem.meta.get("include_holdout"):
        raise ValueError("memory model was trained with H; (c)/(d) need a non-H model (--memory-dir with a model trained without --include-holdout)")
    split, meta, store = _cache_inputs()
    H = sorted(split.loc[split["is_holdout"], "player_id"])
    z = mem.base._z
    # test bag vectors (separate stores per group: rank and player question ids collide)
    Xs = [mem.bag_vectors([q.games for q in qs], [q.color for q in qs]) for qs in (rank_qs, player_qs) if qs]
    Zt = z(np.vstack(Xs))
    # H null bags
    qh = pd.read_parquet(SPLITS / "questions_H.parquet")
    qh = qh[qh["player_id"].isin(set(H))]
    bags = [Bag(r.qid, r.player_id, int(r.rank_id), int(r.color), int(r.n), tuple(getattr(r, f"uid_{i}") for i in range(1, int(r.n) + 1)))
            for r in qh.itertuples()]
    Zh = z(aggregate_many(bags, store, mem.spec))
    out = {"n_test_bags": int(len(Zt)), "n_holdout_bags": int(len(Zh))}
    st = bag_centroid_shift(Zt, mem.base.centroids_, Zh)
    out["bag"] = {"p99_frac": st.p99_frac, "holdout_p99": st.holdout_p99, "ks": st.ks_stat, "ks_p": st.ks_pvalue,
                  "test_q": st.quantiles, "holdout_q": st.holdout_quantiles}
    if pool:
        ids = sorted(pool)
        Zc = z(mem.bag_vectors([pool[i] for i in ids], [0] * len(ids)))
        idx = build_player_index(meta.assign(color_id=meta["color"]))
        rng = np.random.default_rng(derive_seed(SEED, "overlap_shift", "H_library"))
        hb = []
        for pid in H:
            black = idx.games.get(pid, {}).get(0, [])
            if len(black) >= 100:
                hb.append(Bag(pid, pid, 0, 0, 100, tuple(sorted(rng.choice(black, 100, replace=False)))))
        Zhl = z(aggregate_many(hb, store, mem.spec))
        sl = library_centroid_check(Zc, mem.base.centroids_, Zhl)
        out["library"] = {"p99_frac": sl.p99_frac, "holdout_p99": sl.holdout_p99, "ks": sl.ks_stat, "ks_p": sl.ks_pvalue,
                          "n_candidates": len(ids), "n_holdout_libraries": len(hb), "test_q": sl.quantiles, "holdout_q": sl.holdout_quantiles}
    return out


def overlap_decision(rep: dict) -> dict:
    """OR rule over the four SDD 10.2 evidences (thresholds gogame.overlap.DEFAULT_THRESHOLDS; user still decides)."""
    from gogame.overlap import decide_overlap
    n = sum(r["train_overlap"]["n_games"] for r in rep.values() if isinstance(r, dict) and "train_overlap" in r)
    sc = sum(r["train_overlap"]["exact_same_color"] for r in rep.values() if isinstance(r, dict) and "train_overlap" in r)
    p30 = sum(r["train_overlap"]["prefix_match"]["30"] for r in rep.values() if isinstance(r, dict) and "train_overlap" in r)
    sh = rep.get("shift") or {}
    stats = {"exact_same_color_frac": sc / n if n else None, "prefix30_frac": p30 / n if n else None,
             "nn_p99_frac": (sh.get("bag") or {}).get("p99_frac"), "library_p99_frac": (sh.get("library") or {}).get("p99_frac")}
    d = decide_overlap(stats)
    return {"overlap": d.overlap, "label": d.label, "triggered": d.triggered, "evidence": d.evidence,
            "thresholds": d.thresholds, "not_evaluated": list(d.not_evaluated)}


def run_report(data_dir: Path, train_dir: Path | None, log=print, exclude_players: set | None = None,
               memory_dir: Path | None = None, threads: int = 2) -> dict:
    from gogame.overlap import build_train_index, distribution_report, exact_overlap
    data_dir = Path(data_dir)
    rep: dict = {"sgf_format": format_check(data_dir, False, log)}
    games: dict[str, list[GameRecord]] = {}
    for key in ("rank_pub", "rank_priv", "player_pub", "player_priv"):
        p = data_dir / FILES[key]
        if not p.exists():
            log(f"skip {p.name}: missing")
            continue
        split = key.split("_")[1]
        task = key.split("_")[0]
        df = read_questions_csv(p, split)
        cand = None
        cp = data_dir / FILES[f"cand_{split}"]
        if task == "player" and cp.exists():
            cand = pd.read_csv(cp, dtype=str, keep_default_na=False)
        rep[key] = distribution_report(df, task, cand)
        rep[key]["format_issues"] = [f"{i.code}:{i.detail}" for i in validate_questions_frame(df, task)][:20]
        games[key] = [g for _, row in df.iterrows() for g in explode_question(row, split)]
        rep[key]["color_dist_games"] = {c: int(sum(1 for g in games[key] if g.color == (0 if c == "B" else 1))) for c in "BW"}
        n = df["num_games"].astype(int)
        rep[key]["n_dist"] = {int(k): int(v) for k, v in n.value_counts().sort_index().items()}
    # player queries vs candidate library (same split)
    for sp in ("pub", "priv"):
        cp = data_dir / FILES[f"cand_{sp}"]
        if f"player_{sp}" in games and cp.exists():
            cdf = read_candidates_csv(cp, sp)
            idx, cc = build_train_index(cdf["sgf_content"], [0] * len(cdf))
            st = exact_overlap(games[f"player_{sp}"], idx, cc)
            rep[f"player_{sp}"]["exact_vs_candidates"] = {"n_exact": st.n_exact, "frac": st.exact_frac}
    if train_dir is not None and games:
        log("train overlap (streams 10 CSVs, a few minutes)...")
        ov = _train_overlap(games, Path(train_dir), log, exclude_players)
        for k, v in ov.items():
            rep[k]["train_overlap"] = v
    if memory_dir is not None and games:
        log("shift evidence (c)(d): L0 features of the test games...")
        mem = MemoryRank.load(Path(memory_dir), threads)
        rq = [q for sp in ("pub", "priv") if (data_dir / FILES[f"rank_{sp}"]).exists()
              for q in load_questions(data_dir / FILES[f"rank_{sp}"], sp, "rank")]
        pq_ = [q for sp in ("pub", "priv") if (data_dir / FILES[f"player_{sp}"]).exists()
               for q in load_questions(data_dir / FILES[f"player_{sp}"], sp, "player")]
        pool = {}
        for sp in ("pub", "priv"):
            if (data_dir / FILES[f"cand_{sp}"]).exists():
                pool.update({f"{sp}:{k}": v for k, v in load_candidates(data_dir / FILES[f"cand_{sp}"], sp).items()})
        rep["shift"] = overlap_shift(mem, rq, pq_, pool, log)
    rep["overlap_decision"] = overlap_decision(rep)
    return rep


def print_report(rep: dict) -> None:
    for k, r in rep.items():
        if k in ("shift", "overlap_decision", "sgf_format"):
            continue
        print(f"== {k}: {r['n_questions']} questions, {r['n_games']} games, N dist {r['n_dist']}, "
              f"first-colour {r['color1_dist']}, game colours {r['color_dist_games']}")
        print(f"   plies {r.get('ply')}  dup-in-question {r['dup_games']['n_extra_copies']}  qid prefix {r['qid_prefix']['state']}")
        if r["format_issues"]:
            print("   format issues:", r["format_issues"][:3])
        if "train_overlap" in r:
            o = r["train_overlap"]
            print(f"   vs train: exact {o['exact']} (same colour {o['exact_same_color']}), prefix>=30 {o['prefix30_frac']:.4f}")
        if "exact_vs_candidates" in r:
            print(f"   vs candidates: exact {r['exact_vs_candidates']}")
    if "shift" in rep:
        sh = rep["shift"]
        print(f"== shift: bag p99_frac {sh['bag']['p99_frac']:.3f} (KS {sh['bag']['ks']:.3f}, {sh['n_test_bags']} test / {sh['n_holdout_bags']} H bags)"
              + (f", library p99_frac {sh['library']['p99_frac']:.3f} ({sh['library']['n_candidates']} cand)" if "library" in sh else ""))
    d = rep.get("overlap_decision")
    if d:
        print("== OVERLAP DECISION:", d["label"], "| evidence", {k: (None if v is None else round(v, 4)) for k, v in d["evidence"].items()},
              "| triggered", [k for k, v in d["triggered"].items() if v], "| not evaluated", d["not_evaluated"])


# ================================================================================ simulated test files
def build_sim(out_dir: Path, seed: int = 2026, prefixed: bool = False, log=print, source: str = "H") -> dict:
    """Fake test set from H players (never in the non-H models). Returns truth dict and writes official-format CSVs.
    source="train": players come from the NON-H (training) players instead = a positive control for overlap detection and
    the memory route (every test player is a training player); such a file must not be used to score non-H models."""
    from gogame.io import read_train_csv
    from gogame.questions import build_player_pool_sim, to_questions_frame  # noqa: F401
    from gogame.split import read_split
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = read_split(SPLITS / "split.parquet")
    H = sorted(split.loc[split["is_holdout"], "player_id"])
    meta = pd.read_parquet(CACHE / "games_meta.parquet")
    index = build_player_index(meta.assign(color_id=meta["color"]))
    rng = np.random.default_rng(derive_seed(SEED, "predict_sim", str(seed)))
    rank_q = {"pub": [], "priv": []}      # list of (qid, color, uids, rank_id)
    if source == "H":
        qh = pd.read_parquet(SPLITS / "questions_H.parquet")
        qh = qh[qh["player_id"].isin(set(H))]
        for n in (5, 10, 15, 20):
            sub = qh[qh["n"] == n]
            pick = rng.permutation(len(sub))[:200]
            for j, i in enumerate(pick):
                r = sub.iloc[i]
                sp = "pub" if j < 100 else "priv"
                rank_q[sp].append((int(r["color"]), [r[f"uid_{k}"] for k in range(1, n + 1)], int(r["rank_id"])))
        sims = build_player_pool_sim(index, H, 400, 2, 100, (5, 10, 15, 20), seed=derive_seed(SEED, "predict_sim_pool", str(seed)),
                                     holdout_set=set(H))
    else:
        from gogame.questions import build_val_questions
        nonh = sorted(set(split.loc[~split["is_holdout"], "player_id"]) & set(index.games))
        pl = [nonh[i] for i in rng.permutation(len(nonh))[:1500]]
        vq = build_val_questions(index, pl, (5, 10, 15, 20), seed=derive_seed(SEED, "predict_sim_train_q", str(seed)))
        for n in (5, 10, 15, 20):
            sub = [b for b in vq if b.n == n]
            for j, i in enumerate(rng.permutation(len(sub))[:200]):
                sp = "pub" if j < 100 else "priv"
                rank_q[sp].append((sub[i].color, list(sub[i].uids), sub[i].rank_id))
        sims = build_player_pool_sim(index, nonh, 400, 2, 100, (5, 10, 15, 20), seed=derive_seed(SEED, "predict_sim_pool_tr", str(seed)))
    player_q = {}
    for sp, sim in zip(("pub", "priv"), sims):
        chosen = []
        for n in (5, 10, 15, 20):
            qs = [q for q in sim.queries if q.n == n]
            chosen += [qs[i] for i in rng.permutation(len(qs))[:100]]
        player_q[sp] = (sim, chosen)

    need = {u for sp in rank_q for (_, uids, _) in rank_q[sp] for u in uids}
    for sp, (sim, chosen) in player_q.items():
        need |= {u for p in sim.players for u in sim.library[p]} | {u for q in chosen for u in q.uids}
    text: dict[str, str] = {}
    for r in RANKS:
        df = read_train_csv(TRAIN_DIR / f"train_{r}.csv")
        m = df["uid"].isin(need)
        text.update(zip(df.loc[m, "uid"], df.loc[m, "sgf_content"]))
        del df
    log(f"sim: gathered {len(text)}/{len(need)} games")

    cols = ["question_id", "num_games"] + [f"sgf_{k}" for k in range(1, 21)] + [f"color_{k}" for k in range(1, 21)]

    def qrow(qid, uids, color):
        c = "BW"[color]
        row = {"question_id": qid, "num_games": str(len(uids))}
        for k in range(20):
            row[f"sgf_{k + 1}"] = text[uids[k]] if k < len(uids) else ""
        for k in range(20):
            row[f"color_{k + 1}"] = c if k < len(uids) else ""
        return row

    pre = lambda sp, s: f"{sp}_{s}" if prefixed else s   # noqa: E731
    truth = {"rank": {}, "player": {}}
    for sp, name in (("pub", "public"), ("priv", "private")):
        order = rng.permutation(len(rank_q[sp]))
        rows = []
        for i, j in enumerate(order, 1):
            color, uids, rid = rank_q[sp][j]
            q = pre(sp, f"q_{i:04d}")
            rows.append(qrow(q, uids, color))
            truth["rank"][f"{sp}_q_{i:04d}"] = rid
        pd.DataFrame(rows, columns=cols).to_csv(out_dir / FILES[f"rank_{sp}"], index=False)
        sim, chosen = player_q[sp]
        pid = {p: f"p_{i:04d}" for i, p in enumerate(rng.permutation(sim.players), 1)}
        crow = []
        for p in sim.players:
            for u in sim.library[p]:
                crow.append({"game_id": f"g_{sp}_{len(crow) + 1:06d}", "player_id": pre(sp, pid[p]),
                             "sgf_content": text[u], "color": "B"})
        cdf = pd.DataFrame(crow).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        cdf.to_csv(out_dir / FILES[f"cand_{sp}"], index=False)
        rows = []
        for i, j in enumerate(rng.permutation(len(chosen)), 1):
            q = chosen[j]
            qid = pre(sp, f"q_{i:04d}")
            rows.append(qrow(qid, list(q.uids), 0))
            truth["player"][f"{sp}_q_{i:04d}"] = f"{sp}_{pid[q.player_id]}"
        pd.DataFrame(rows, columns=cols).to_csv(out_dir / FILES[f"player_{sp}"], index=False)
    (out_dir / "truth.json").write_text(json.dumps(truth))
    return truth


def sim_scores(out_dir: Path, truth: dict) -> dict:
    from gogame.metrics import player_scores, rank_scores
    rk = pd.read_csv(Path(out_dir) / "submission_rank.csv", dtype=str)
    pl = pd.read_csv(Path(out_dir) / "submission_player.csv", dtype=str)
    t = np.array([truth["rank"][q] for q in rk["question_id"]])
    p = np.array([RANKS.index(x) for x in rk["pred_rank"]])
    rs = rank_scores(p, t)
    ps = player_scores(pl[[f"top_{i}" for i in range(1, 6)]].to_numpy(), np.array([truth["player"][q] for q in pl["question_id"]]))
    res = {"rank_score": float(rs.mean()), "rank_exact": float((p == t).mean()),
           "player_score": float(ps.mean()), "player_top1": float((ps == 1).mean()), "player_top5": float((ps > 0).mean())}
    for sp in ("pub", "priv"):
        mr = rk["question_id"].str.startswith(sp + "_").to_numpy()
        mp = pl["question_id"].str.startswith(sp + "_").to_numpy()
        res[f"rank_score_{sp}"], res[f"player_score_{sp}"] = float(rs[mr].mean()), float(ps[mp].mean())
    return res


# ================================================================================ CLI
def _limit_threads(n: int) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, str(n))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict", "report", "sim"):
        p = sub.add_parser(name)
        p.add_argument("--model-dir", default=str(MODEL_DIR))
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--color-semantics", choices=["as_annotated", "actual"], default=None,
                       help="color column of HD>=2 games: as_annotated (default) = swapped with the reversed B/W marks; actual = kept (also GOGAME_COLOR_SEMANTICS)")
        if name == "train":
            p.add_argument("--task", choices=["rank", "player", "both"], default="both")
            p.add_argument("--include-holdout", action="store_true",
                           help="train on ALL players incl. H (final submission model; invalidates the H simulation)")
            p.add_argument("--rounds", type=int, default=None, help="rank: LightGBM rounds; e04v2: boosting rounds (default 400, frozen)")
            p.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL, choices=sorted(PLAYER_MODELS) + ["all"])
            p.add_argument("--rank-model", default="l0_lgbm", choices=sorted(RANK_MODELS),
                           help="hsl_lgbm = E06 formal model (needs the finished E06 selection)")
            p.add_argument("--rank-config", default=None,
                           help="hsl_lgbm only: E10 selection.json (tier-1+2 cache, bags/N range); default = E06 behaviour")
        if name in ("predict", "sim"):
            p.add_argument("--rank-model", default="l0_lgbm", choices=sorted(RANK_MODELS))
            p.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL, choices=sorted(PLAYER_MODELS))
            p.add_argument("--no-check", action="store_true")
        if name in ("predict", "sim"):
            p.add_argument("--strict-format", action="store_true", help="stop on unknown SGF root properties or a skip rate > 1%% (default: warn and go on)")
            p.add_argument("--hsl-features", default=None, help="hsl_lgbm: per-game test HSL parquet (default data/test_scan/hsl_test_games.parquet)")
            p.add_argument("--max-hsl-drop-frac", type=float, default=HslLgbmRank.MAX_DROP_FRAC,
                           help="hsl_lgbm: error out if more than this fraction of test games lack HSL features (default 0.01)")
        if name == "predict":
            p.add_argument("--data-dir", required=True)
            p.add_argument("--out-dir", required=True)
        if name == "report":
            p.add_argument("--data-dir", required=True)
            p.add_argument("--out", default=None)
            p.add_argument("--no-train-overlap", action="store_true")
            p.add_argument("--train-dir", default=str(TRAIN_DIR))
            p.add_argument("--memory-dir", default=None,
                           help="dir with rank_memory.pkl trained WITHOUT --include-holdout: adds evidence (c)(d) (needs data/cache + splits)")
            p.add_argument("--exclude-holdout-train", action="store_true",
                           help="SIMULATION ONLY: drop H players' games from the train index (negative control with H-sourced fake tests)")
        if name == "sim":
            p.add_argument("--out", default=str(ROOT / ".scratch" / "predict_sim"))
            p.add_argument("--prefixed-ids", action="store_true", help="write ids with pub_/priv_ prefix in the fake test files")
            p.add_argument("--sim-source", choices=["H", "train"], default="H",
                           help="train = fake test from NON-H players (positive control for overlap/memory; scores of non-H models are meaningless)")
            p.add_argument("--reuse-files", action="store_true", help="do not rebuild the fake CSVs if truth.json exists")
    a = ap.parse_args(argv)
    _limit_threads(a.threads)
    if a.color_semantics:
        os.environ["GOGAME_COLOR_SEMANTICS"] = a.color_semantics
    from gogame.sgf import get_color_semantics
    get_color_semantics()
    md = Path(a.model_dir)
    if a.cmd == "train":
        train_models(a.task, md, a.include_holdout, a.threads, a.rounds, a.player_model, a.rank_model,
                     Path(a.rank_config) if a.rank_config else None)
        return 0
    if a.cmd == "report":
        excl = None
        if a.exclude_holdout_train:
            from gogame.split import read_split
            sp_ = read_split(SPLITS / "split.parquet")
            excl = set(sp_.loc[sp_["is_holdout"], "player_id"])
        rep = run_report(Path(a.data_dir), None if a.no_train_overlap else Path(a.train_dir), exclude_players=excl,
                         memory_dir=Path(a.memory_dir) if a.memory_dir else None, threads=a.threads)
        print_report(rep)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
        return 0
    rm = RANK_MODELS[a.rank_model].load(md, a.threads)
    pm = PLAYER_MODELS[a.player_model].load(md, a.threads)
    if a.hsl_features and hasattr(rm, "set_features"):
        rm.set_features(a.hsl_features)
    if hasattr(rm, "max_drop_frac"):
        rm.max_drop_frac = a.max_hsl_drop_frac
    if a.cmd == "sim" and a.rank_model == HslLgbmRank.name:
        print("sim does not support hsl_lgbm (the fake test games have no HSL features); use `scripts/predict_hsl.py sim-results`", file=sys.stderr)
        return 2
    if a.cmd == "predict":
        return 0 if run_predict(Path(a.data_dir), Path(a.out_dir), rm, pm, not a.no_check, strict_format=a.strict_format)["ok"] else 1
    # sim
    for m in (rm, pm):
        if m.meta.get("include_holdout", False):
            print("refusing: model was trained with --include-holdout, the H simulation would leak", file=sys.stderr)
            return 2
    out = Path(a.out)
    if a.reuse_files and (out / "truth.json").exists():
        truth = json.loads((out / "truth.json").read_text())
    else:
        truth = build_sim(out, prefixed=a.prefixed_ids, source=a.sim_source)
    res = run_predict(out, out / "submission", rm, pm, not a.no_check)
    sc = sim_scores(out / "submission", truth)
    print("SIM SCORES (smoke only, H players, not a formal result):", json.dumps(sc, indent=1))
    (out / "sim_scores.json").write_text(json.dumps(sc, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
