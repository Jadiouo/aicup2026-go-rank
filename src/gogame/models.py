"""Rank model: LightGBM wrapper, decision rules, prior correction, OOF driver, memory baseline (SDD 5.11)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .aggregate import BagFeatureSpec, L0Store, aggregate_many, fit_sparse_selector, make_spec
from .errors import DataFormatError
from .metrics import balanced_weights, rank_score_matrix, rank_scores

CLASSES = 10

DEFAULT_PARAMS: dict = {
    "objective": "multiclass", "num_class": CLASSES, "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 100, "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "num_boost_round": 2000, "early_stopping_rounds": 100,
    "deterministic": True, "force_row_wise": True, "num_threads": 8, "verbosity": -1,
}


# ---------------------------------------------------------------- probabilities / decisions ------------------------
def apply_prior_correction(p: np.ndarray, train_prior: np.ndarray, target_prior: np.ndarray | None = None) -> np.ndarray:
    """p * target / train, row-normalised.  Classes with train_prior == 0 keep probability 0."""
    p = np.asarray(p, dtype=np.float64)
    tr = np.asarray(train_prior, dtype=np.float64)
    tg = np.full(tr.shape, 1.0 / tr.size) if target_prior is None else np.asarray(target_prior, dtype=np.float64)
    ratio = np.divide(tg, tr, out=np.zeros_like(tr), where=tr > 0)
    q = p * ratio
    s = q.sum(axis=1, keepdims=True)
    return np.divide(q, s, out=np.zeros_like(q), where=s > 0)


def decide_argmax(p: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(p), axis=1).astype(np.int64)


def decide_ev(p: np.ndarray) -> np.ndarray:
    """argmax_j sum_k p_k s(j,k); ties -> lowest j.  Values are rounded to 12 digits so that mathematically
    equal expectations tie."""
    ev = np.round(np.asarray(p, dtype=np.float64) @ rank_score_matrix(), 12)
    return np.argmax(ev, axis=1).astype(np.int64)


# ---------------------------------------------------------------- model ------------------------------------------
@dataclass
class RankModel:
    booster: object
    spec: BagFeatureSpec | None
    params: dict
    train_prior: np.ndarray
    best_iteration: int = 0
    meta: dict = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray, target_prior: np.ndarray | None = None, raw: bool = False) -> np.ndarray:
        p = np.asarray(self.booster.predict(np.asarray(X, dtype=np.float32)), dtype=np.float64).reshape(len(X), CLASSES)
        p = p * (self.train_prior > 0)                       # classes never seen in training get probability 0
        s = p.sum(axis=1, keepdims=True)
        p = np.divide(p, s, out=np.full_like(p, 1.0 / CLASSES), where=s > 0)
        if not raw:
            p = apply_prior_correction(p, self.train_prior, target_prior)
        return p.astype(np.float32)


def fit_lgbm(Xtr, ytr, Xva=None, yva=None, params: dict | None = None, seed: int = 0, w_tr=None, w_va=None) -> RankModel:
    import lightgbm as lgb
    prm = dict(DEFAULT_PARAMS)
    prm.update(params or {})
    prm["seed"] = int(seed)
    rounds = int(prm.pop("num_boost_round"))
    es = prm.pop("early_stopping_rounds", None)
    log_every = int(prm.pop("log_every", 0))
    ytr = np.asarray(ytr, dtype=np.int64)
    w = None if w_tr is None else np.asarray(w_tr, dtype=np.float64)
    train_prior = np.bincount(ytr, weights=w, minlength=CLASSES).astype(np.float64)
    train_prior = train_prior / train_prior.sum()
    dtr = lgb.Dataset(np.asarray(Xtr, dtype=np.float32), label=ytr, weight=w, free_raw_data=False)
    callbacks = [lgb.log_evaluation(log_every)]
    valid = []
    if Xva is not None and yva is not None and len(yva):
        valid = [lgb.Dataset(np.asarray(Xva, dtype=np.float32), label=np.asarray(yva, dtype=np.int64),
                             weight=None if w_va is None else np.asarray(w_va, dtype=np.float64), reference=dtr)]
        if es:
            callbacks.append(lgb.early_stopping(int(es), verbose=False))
    booster = lgb.train(prm, dtr, num_boost_round=rounds, valid_sets=valid, callbacks=callbacks)
    best = int(booster.best_iteration) if valid and es and booster.best_iteration > 0 else int(booster.current_iteration())
    if best != booster.current_iteration():
        booster.model_from_string(booster.model_to_string(num_iteration=best))
    return RankModel(booster=booster, spec=None, params={**prm, "num_boost_round": rounds}, train_prior=train_prior,
                     best_iteration=best)


# ---------------------------------------------------------------- OOF -------------------------------------------
@dataclass
class OofFold:
    fold: int
    train_bags: Sequence
    es_bags: Sequence
    val_bags: Sequence
    w_train: np.ndarray | None = None
    w_es: np.ndarray | None = None


@dataclass
class OofResult:
    frame: pd.DataFrame
    best_iterations: list[int]
    train_players: dict[int, set]
    val_players: dict[int, set]


def _players(bags) -> set:
    return {b.player_id for b in bags}


def _labels(bags) -> np.ndarray:
    return np.array([b.rank_id for b in bags], dtype=np.int64)


def run_oof(cfg: dict, store: L0Store, folds: Sequence[OofFold], trainer: Callable | None = None) -> OofResult:
    """K-fold OOF.  Bag sampling / weights are prepared by the caller (questions module); this driver enforces the
    leakage rules, fits the sparse selector on the training fold's games only and predicts each validation bag once."""
    trainer = trainer or fit_lgbm
    cfg = cfg or {}
    params, seed = cfg.get("params", {}), int(cfg.get("seed", 0))
    target = cfg.get("target_prior")
    holdout = set(cfg.get("holdout", ()))
    min_df = cfg.get("min_df", 0.005)
    for t in cfg.get("tables", ()):
        from .gametable import require_oof_table
        require_oof_table(t)
    rows, best, tps, vps = [], [], {}, {}
    seen_q: set = set()
    for f in folds:
        tp, ep, vp = _players(f.train_bags), _players(f.es_bags), _players(f.val_bags)
        if (tp | ep) & vp:
            raise DataFormatError(f"fold {f.fold}: training/early-stopping players overlap the validation players")
        if tp & ep:
            raise DataFormatError(f"fold {f.fold}: early-stopping players overlap the training players")
        if (tp | ep | vp) & holdout:
            raise DataFormatError(f"fold {f.fold}: hold-out players present")
        used = sorted({u for b in f.train_bags for u in b.uids})
        rows_idx = store.rows(used)
        sp = store.sparse[rows_idx] if store.sparse is not None else None
        cols = fit_sparse_selector(sp, min_df) if sp is not None else np.zeros(0, np.int32)
        spec = make_spec(store.names, cols)
        Xtr = aggregate_many(f.train_bags, store, spec)
        Xes = aggregate_many(f.es_bags, store, spec) if len(f.es_bags) else None
        model = trainer(Xtr, _labels(f.train_bags), Xes, _labels(f.es_bags) if len(f.es_bags) else None,
                        params, seed, f.w_train, f.w_es)
        model.spec = spec
        Xva = aggregate_many(f.val_bags, store, spec)
        praw = model.predict_proba(Xva, raw=True)
        pcor = model.predict_proba(Xva, target_prior=target)
        d = {"qid": [b.qid for b in f.val_bags], "player_id": [b.player_id for b in f.val_bags],
             "rank_id": [b.rank_id for b in f.val_bags], "n": [b.n for b in f.val_bags],
             "color": [b.color for b in f.val_bags], "fold": f.fold}
        for i in range(CLASSES):
            d[f"praw_{i}"] = praw[:, i]
        for i in range(CLASSES):
            d[f"p_{i}"] = pcor[:, i]
        rows.append(pd.DataFrame(d))
        best.append(int(model.best_iteration))
        tps[f.fold], vps[f.fold] = tp | ep, vp
        dup = seen_q & set(d["qid"])
        if dup:
            raise DataFormatError(f"question predicted more than once: {sorted(dup)[:3]}")
        seen_q |= set(d["qid"])
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return OofResult(frame, best, tps, vps)


def oof_scores(frame: pd.DataFrame, decision: str = "argmax") -> dict[str, float]:
    """Rank score under raw / balanced weighting x uncorrected (praw) / corrected (p) probabilities."""
    dec = {"argmax": decide_argmax, "ev": decide_ev}[decision]
    truth = frame["rank_id"].to_numpy()
    w = balanced_weights(truth)
    out = {}
    for pk in ("praw", "p"):
        P = frame[[f"{pk}_{i}" for i in range(CLASSES)]].to_numpy()
        s = rank_scores(dec(P), truth)
        out[f"raw_{pk}"] = float(s.mean())
        out[f"balanced_{pk}"] = float((s * w).mean())
    return out


def constant_prediction(y: np.ndarray) -> int:
    """E00: the modal class (lowest id on ties)."""
    return int(np.argmax(np.bincount(np.asarray(y, dtype=np.int64), minlength=CLASSES)))


# ---------------------------------------------------------------- final model ------------------------------------
def final_num_rounds(best_iterations: Sequence[int]) -> int:
    """round-half-up of the median of the per-fold best iterations."""
    return int(math.floor(float(np.median(np.asarray(best_iterations, dtype=np.float64))) + 0.5))


def train_final(store: L0Store, bags: Sequence, holdout: set, best_iterations: Sequence[int], params: dict, seed: int,
                w: np.ndarray | None = None, min_df: float = 0.005, trainer: Callable | None = None) -> RankModel:
    """SDD 13.2.1: all non-H players, no early stopping, rounds = median OOF best_iteration."""
    trainer = trainer or fit_lgbm
    players = _players(bags)
    if players & set(holdout):
        raise DataFormatError(f"hold-out players in the final training set: {sorted(players & set(holdout))[:3]}")
    rounds = final_num_rounds(best_iterations)
    used = sorted({u for b in bags for u in b.uids})
    sp = store.sparse[store.rows(used)] if store.sparse is not None else None
    cols = fit_sparse_selector(sp, min_df) if sp is not None else np.zeros(0, np.int32)
    spec = make_spec(store.names, cols)
    X = aggregate_many(bags, store, spec)
    model = trainer(X, _labels(bags), None, None, {**params, "num_boost_round": rounds}, seed, w, None)
    model.spec = spec
    model.meta = {"train_players": sorted(players), "rounds": rounds, "n_bags": len(bags)}
    return model


def predict_bags(model: RankModel, bags: Sequence, store: L0Store, target_prior=None, raw: bool = False) -> np.ndarray:
    return model.predict_proba(aggregate_many(bags, store, model.spec), target_prior=target_prior, raw=raw)


# ---------------------------------------------------------------- memory baseline --------------------------------
# Implementation lives in memory.py (single copy); re-exported here for the SDD 5.11 import location.
from .memory import MemoryBaseline  # noqa: E402,F401
