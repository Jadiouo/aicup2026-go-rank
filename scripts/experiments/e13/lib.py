"""E13 (HumanSL fine-tune, Rank) library.  numpy / pandas only: no torch, no KataGo import here.

Plan: experiments/E13_humansl_ft/PLAN.md.  H players are never read: every player set used here is checked against
the split's is_holdout flag, and the training/eval game lists are built from those sets only.
Fold identity with E10 candidate D comes from e10.lib.fold_sets (same code path as E06/E10), the validation questions from
E10's saved questions_q2.parquet (identical to what oof_D was scored on).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gogame.constants import SEED
from gogame.metrics import balanced_weights, rank_scores
from gogame.models import decide_ev
from gogame.seeds import derive_seed
from scripts.experiments.e10 import lib as e10lib

ROOT = e10lib.ROOT
OUT = ROOT / "experiments" / "E13_humansl_ft"
E10_OOF = e10lib.OUT / "oof"          # oof_D_fold*.parquet, questions_q2.parquet (NOT the ledger)
SPLIT_PATH = ROOT / "data" / "splits" / "split.parquet"
META_PATH = ROOT / "data" / "cache" / "games_meta.parquet"
TRAIN_CSV_DIR = ROOT / "data" / "training"

NS = e10lib.NS
PCOLS = e10lib.PCOLS
N_CLASSES = 10
PROFILE = "rank_1d"                    # fixed KataGo metadata profile for every position (PLAN 4)
POS_LO, POS_HI = 2, 60                 # own-move index range [lo, hi): the scan's "first 60 own moves", minus the 2 identical openers
POS_PER_GAME = 16
TARGET_TRAIN_GAMES = 300_000
GAMES_PER_SHARD = 2000
N_BOOT = 4000


# ------------------------------------------------------------------------------------------------ split / leakage
def split_sets(fold: int, split_path: Path = SPLIT_PATH) -> dict:
    """fit / es / val player sets of `fold` (E06/E10 logic) + H.  Asserts pairwise disjointness and that H is untouched."""
    sp = pd.read_parquet(split_path)
    nonh = sp[~sp["is_holdout"]]
    fs = e10lib.fold_sets(nonh, fold)
    H = set(sp.loc[sp["is_holdout"], "player_id"])
    e10lib.assert_fold_disjoint(fs, H)
    assert not ((fs["fit"] | fs["es"] | fs["val"]) & H)
    rank = dict(zip(sp["player_id"], sp["rank_id"].astype(int)))
    return {**fs, "H": H, "rank": rank}


def assert_games_clean(games: pd.DataFrame, allowed: set, forbidden: set, name: str) -> None:
    """Every game's player is in `allowed` and in none of `forbidden` (H, other roles)."""
    pl = set(games["player_id"])
    if not pl <= allowed:
        raise AssertionError(f"{name}: players outside the allowed set: {sorted(pl - allowed)[:3]}")
    if pl & forbidden:
        raise AssertionError(f"{name}: players also in a forbidden set (leak): {sorted(pl & forbidden)[:3]}")


# ------------------------------------------------------------------------------------------------ training game list
def water_fill_cap(pool: np.ndarray, target: int) -> int:
    """Smallest per-player cap c with sum(min(pool, c)) >= target (all games if the total is below target)."""
    pool = np.asarray(pool)
    if pool.sum() <= target:
        return int(pool.max())
    lo, hi = 1, int(pool.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if np.minimum(pool, mid).sum() >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def select_train_games(meta: pd.DataFrame, fit_players: set, target: int = TARGET_TRAIN_GAMES) -> pd.DataFrame:
    """Per-player capped (water-filling) uniform sample of games, both colours pooled; deterministic per player.
    Total is exactly min(target, all games): players at the cap level get one extra game in a fixed hash order."""
    m = meta[meta["player_id"].isin(fit_players)]
    pool = m.groupby("player_id").size()
    cap = water_fill_cap(pool.to_numpy(), target)
    base = np.minimum(pool.to_numpy(), cap - 1)
    need = max(0, min(target, int(pool.sum())) - int(base.sum()))
    tie = [p for p, n in pool.items() if n >= cap]
    tie.sort(key=lambda p: derive_seed(SEED, "e13", "cap_tiebreak", p))
    extra = set(tie[:need])
    parts = []
    for p, g in m.groupby("player_id", sort=True):
        k = min(len(g), cap - 1) + (1 if p in extra else 0)
        uids = np.array(sorted(g["uid"]))
        if len(uids) > k:
            rng = np.random.default_rng(derive_seed(SEED, "e13", "train_games", p))
            uids = np.sort(uids[rng.permutation(len(uids))[:k]])
        parts.append(g.set_index("uid").loc[uids].reset_index())
    out = pd.concat(parts, ignore_index=True)
    out.attrs["cap"] = cap
    return out[["uid", "player_id", "rank_id", "color"]].reset_index(drop=True)


def games_for_questions(meta: pd.DataFrame, bags_frame: pd.DataFrame) -> pd.DataFrame:
    """Unique games used by a question frame (uid_1..uid_20), with player/label/colour from games_meta."""
    ucols = [c for c in bags_frame.columns if c.startswith("uid_")]
    u = pd.unique(bags_frame[ucols].to_numpy().ravel())
    u = sorted(x for x in u if isinstance(x, str) and x)
    m = meta.set_index("uid").loc[u].reset_index()
    return m[["uid", "player_id", "rank_id", "color"]]


def games_for_es(meta: pd.DataFrame, es_players: set, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D's early-stopping questions (e10.make_es_bags, tier 1+2 tag) -> (games, questions frame)."""
    from gogame.questions import to_questions_frame
    index = e10lib.hsl_index(meta, meta["uid"])
    bags = e10lib.make_es_bags(index, es_players, fold, tier1_only=False)
    qf = to_questions_frame(bags)
    return games_for_questions(meta, qf), qf


# ------------------------------------------------------------------------------------------------ position selection
def own_move_idx_train(n_own: int, rng: np.random.Generator, k: int = POS_PER_GAME, lo: int = POS_LO, hi: int = POS_HI) -> np.ndarray:
    """Random k distinct own-move indices in [lo, min(hi, n_own)) (all of them if fewer), sorted."""
    pool = np.arange(lo, min(hi, n_own))
    if len(pool) <= k:
        return pool
    return np.sort(rng.choice(pool, size=k, replace=False))


def own_move_idx_eval(n_own: int, k: int = POS_PER_GAME, lo: int = POS_LO, hi: int = POS_HI) -> np.ndarray:
    """Deterministic, evenly spaced k own-move indices in [lo, min(hi, n_own))."""
    top = min(hi, n_own)
    if top <= lo:
        return np.zeros(0, np.int64)
    if top - lo <= k:
        return np.arange(lo, top)
    return np.unique(np.round(np.linspace(lo, top - 1, k)).astype(np.int64))


def own_move_idx_all(n_own: int, lo: int = POS_LO, hi: int = POS_HI) -> np.ndarray:
    """Every own-move index in [lo, min(hi, n_own)) -- what val / es games are encoded with (<= hi-lo = 58 positions)."""
    return np.arange(lo, max(lo, min(hi, n_own)), dtype=np.int64)


def select_positions(pos: pd.DataFrame, k: int | None) -> np.ndarray:
    """Boolean mask over the rows of `pos` (uid, own_idx; rows of a game contiguous and ascending) picking, per game, k evenly
    spaced positions of those available (all if k is None / 0 / the game has <= k).  Deterministic; for a game with the full
    range this equals own_move_idx_eval."""
    if not k or len(pos) == 0:
        return np.ones(len(pos), bool)
    g = pos.groupby("uid", sort=False)
    rank, m = g.cumcount().to_numpy(), g["uid"].transform("size").to_numpy()
    keep = {}
    for mm in np.unique(m):
        sel = np.zeros(mm, bool)
        sel[np.unique(np.round(np.linspace(0, mm - 1, k)).astype(np.int64)) if mm > k else np.arange(mm)] = True
        keep[mm] = sel
    return np.fromiter((keep[a][b] for a, b in zip(m.tolist(), rank.tolist())), bool, len(pos))


def k_tag(k: int | None) -> str:
    return "kall" if not k else f"k{int(k)}"


# ------------------------------------------------------------------------------------------------ labels / loss helpers
def class_weights(labels: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Inverse-frequency weights, mean 1 over the classes present (the metric is class-balanced)."""
    cnt = np.bincount(np.asarray(labels, np.int64), minlength=n_classes).astype(np.float64)
    w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)
    return w / w[cnt > 0].mean()


def soft_targets(y: np.ndarray, sigma: float, n_classes: int = N_CLASSES) -> np.ndarray:
    """One-hot (sigma<=0) or discretised-Gaussian ordinal soft labels over rank distance."""
    y = np.asarray(y, np.int64)
    if sigma <= 0:
        return np.eye(n_classes)[y]
    k = np.arange(n_classes)[None, :]
    t = np.exp(-0.5 * ((k - y[:, None]) / sigma) ** 2)
    return t / t.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------ position -> game -> question
def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def game_logits(pos_uid: np.ndarray, pos_logits: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Mean of the position logits of each game (every position weighs the same inside a game)."""
    df = pd.DataFrame(np.asarray(pos_logits, np.float64))
    df["uid"] = np.asarray(pos_uid)
    g = df.groupby("uid", sort=True).mean()
    return g.index.tolist(), g.to_numpy()


UCOLS = [f"uid_{i}" for i in range(1, 21)]


def question_coverage(qframe: pd.DataFrame, available, name: str, max_missing_frac: float = 0.01) -> dict:
    """Which games of the questions have no position?  Policy (PLAN 4a): a question is aggregated over its games that have
    positions; this raises (listing the uids) when a question has no such game at all, or when more than `max_missing_frac`
    of all question-game slots are missing.  Otherwise the counts are returned for the record."""
    avail = set(available)
    miss, empty, affected, slots = [], [], 0, 0
    for qid, n, row in zip(qframe["qid"], qframe["n"].to_numpy(), qframe[UCOLS].to_numpy()):
        us = list(row[: int(n)])
        bad = [u for u in us if u not in avail]
        slots += len(us)
        miss += bad
        affected += bool(bad)
        if len(bad) == len(us):
            empty.append(qid)
    rep = {"name": name, "n_questions": int(len(qframe)), "n_slots": int(slots), "n_missing_slots": len(miss),
           "missing_frac": len(miss) / max(1, slots), "n_questions_affected": int(affected), "n_empty_questions": len(empty),
           "missing_uids": sorted(set(miss)), "empty_qids": empty[:20]}
    if empty or rep["missing_frac"] > max_missing_frac:
        raise RuntimeError(f"{name}: games without positions -- {rep['n_empty_questions']} questions have none at all, "
                           f"{len(rep['missing_uids'])} missing uids ({rep['missing_frac']:.2%} of slots, limit {max_missing_frac:.2%}): "
                           f"{rep['missing_uids'][:50]}")
    return rep


def bag_logits(qframe: pd.DataFrame, game_uids: list[str], game_z: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """Question logit = mean of the logits of its games that have positions (equal weight per game).  A game without positions
    is dropped from that question's mean and counted in `stats`; a question with no usable game raises ValueError."""
    pos = {u: i for i, u in enumerate(game_uids)}
    out = np.zeros((len(qframe), game_z.shape[1]))
    n_miss, affected, miss = 0, 0, set()
    for r, (qid, n, row) in enumerate(zip(qframe["qid"], qframe["n"].to_numpy(), qframe[UCOLS].to_numpy())):
        us = list(row[: int(n)])
        ids = [pos[u] for u in us if u in pos]
        if not ids:
            raise ValueError(f"question {qid}: none of its {len(us)} games has positions")
        if len(ids) < len(us):
            n_miss, affected = n_miss + len(us) - len(ids), affected + 1
            miss.update(u for u in us if u not in pos)
        out[r] = game_z[ids].mean(axis=0)
    if stats is not None:
        stats.update({"n_missing_slots": n_miss, "n_questions_affected": affected, "missing_uids": sorted(miss)})
    return out


def question_probs(z: np.ndarray, T: float = 1.0, bias: np.ndarray | None = None) -> np.ndarray:
    return softmax(np.asarray(z, np.float64) / T + (0.0 if bias is None else np.asarray(bias)))


def fit_calibration(z: np.ndarray, y: np.ndarray, mode: str = "t") -> dict:
    """Fit on early-stopping questions only.  mode: none | t (temperature) | tb (temperature + 10 class biases, ridge 1e-2)."""
    if mode == "none":
        return {"T": 1.0, "bias": [0.0] * N_CLASSES, "mode": mode}
    from scipy.optimize import minimize
    y = np.asarray(y, np.int64)
    w = balanced_weights(y)

    def nll(theta):
        T = np.exp(theta[0])
        b = theta[1:] if mode == "tb" else None
        p = question_probs(z, T, b)
        v = -(w * np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))).mean()
        return v + (1e-2 * float((theta[1:] ** 2).sum()) if mode == "tb" else 0.0)

    x0 = np.zeros(1 + (N_CLASSES if mode == "tb" else 0))
    r = minimize(nll, x0, method="L-BFGS-B")
    return {"T": float(np.exp(r.x[0])), "bias": (r.x[1:].tolist() if mode == "tb" else [0.0] * N_CLASSES), "mode": mode,
            "nll": float(r.fun), "nll_uncalibrated": float(nll(x0))}


def apply_calibration(z: np.ndarray, cal: dict) -> np.ndarray:
    return question_probs(z, cal["T"], np.asarray(cal["bias"]))


def es_cv_nll(z: np.ndarray, y: np.ndarray, groups: np.ndarray, mode: str, folds: int = 5) -> float:
    """Cross-fitted class-balanced bag NLL on the early-stopping questions: calibration is fitted on the other player-folds and
    scored on the held-out one, so a richer calibration (tb) cannot win just by fitting es in-sample.  Selection uses ONLY this."""
    y, groups = np.asarray(y, np.int64), np.asarray(groups)
    pl = sorted(set(groups.tolist()))
    perm = np.random.default_rng(derive_seed(SEED, "e13", "es_cv")).permutation(len(pl))
    fold_of = {p: int(perm[i]) % folds for i, p in enumerate(pl)}
    f = np.array([fold_of[g] for g in groups.tolist()])
    w = balanced_weights(y)
    nll = np.zeros(len(y))
    for k in range(folds):
        te = f == k
        if not te.any() or te.all():
            continue
        cal = fit_calibration(z[~te], y[~te], mode)
        p = apply_calibration(z[te], cal)
        nll[te] = -np.log(np.clip(p[np.arange(te.sum()), y[te]], 1e-12, 1))
    return float((w * nll).mean())


def select_primary(cands: list[dict], q_es: pd.DataFrame, calibs=("t", "tb")) -> dict:
    """cands: [{"run", "scope", "k", "z"(es question logits)}].  Picks the (candidate, calib) with the lowest cross-fitted es NLL.
    Ties: the earlier candidate / calib in the given order.  Never sees val."""
    y = q_es["rank_id"].to_numpy()
    g = q_es["player_id"].to_numpy() if "player_id" in q_es.columns else q_es["qid"].to_numpy()
    table = [{"run": c["run"], "scope": c["scope"], "k": c["k"], "calib": cal, "es_cv_nll": es_cv_nll(c["z"], y, g, cal)}
             for c in cands for cal in calibs]
    return {"best": min(table, key=lambda r: r["es_cv_nll"]), "table": table}


def is_primary(primary: dict | None, scope: str, calib: str, k, ckpt_sha256: str) -> bool:
    """True iff (scope, calib, pos_per_game, checkpoint bytes) are exactly what primary.json fixed."""
    return bool(primary) and primary["scope"] == scope and primary["calib"] == calib and int(primary["pos_per_game"] or 0) == int(k or 0) \
        and primary["ckpt_sha256"] == ckpt_sha256


def append_ledger(path: Path, entry: dict) -> None:
    import datetime
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **entry}, default=str) + "\n")


def ensemble(p_a: np.ndarray, p_b: np.ndarray, w_a: float = 0.5) -> np.ndarray:
    return e10lib.ensemble_probs(p_a, p_b, w_a)


# ------------------------------------------------------------------------------------------------ evaluation
def score_ev(p: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-question official score with EV decoding (the same decode as D)."""
    return rank_scores(decide_ev(np.asarray(p)), np.asarray(truth))


def paired_eval(d: pd.DataFrame, p_e13: np.ndarray, n_boot: int = N_BOOT, level: float = 0.975) -> dict:
    """d = oof_D rows (Q2 only, any order) with qid/player_id/rank_id/n and p_0..p_9.  p_e13 is aligned to d row order.
    Returns point scores and player-level paired bootstrap deltas of E13-alone and E13+D (equal weight) against D."""
    truth, groups, ns = d["rank_id"].to_numpy(), d["player_id"].to_numpy(), d["n"].to_numpy()
    nv = [n for n in NS if (ns == n).any()]
    pD = d[PCOLS].to_numpy()
    cands = {"D": pD, "E13": p_e13, "ENS": ensemble(pD, p_e13, 0.5)}
    S = {k: score_ev(v, truth) for k, v in cands.items()}
    res = {"n_questions": int(len(d)), "n_players": int(len(set(groups))), "scores": {}, "deltas": {}}
    for k, s in S.items():
        pt = e10lib.equal_n_point(s, np.zeros(len(s)), truth, ns, nv)
        res["scores"][k] = {"q2_eq": pt["delta"], "per_N": pt["per_N_delta"]}
    for k in ("E13", "ENS"):
        res["deltas"][f"{k}-D"] = e10lib.equal_n_delta(S[k], S["D"], groups, truth, ns, nv, n_boot=n_boot, level=level)
    return res


# Pre-registered gate (PLAN 8): expand to 5 folds only if fold-0 Q2 N-equal balanced beats D by this much.
GATE_MIN_DELTA = 0.010
GATE_PER_N_FLOOR = -0.010


def gate(res: dict) -> dict:
    """ENS-D or E13-D: point >= GATE_MIN_DELTA and the 97.5% player-bootstrap ci_low > 0 and no per-N delta below the floor."""
    out = {"min_delta": GATE_MIN_DELTA, "per_N_floor": GATE_PER_N_FLOOR, "checks": {}}
    for k in ("ENS-D", "E13-D"):
        d = res["deltas"][k]
        ok = d["delta"] >= GATE_MIN_DELTA and d["ci_low"] > 0 and min(d["per_N_delta"].values()) >= GATE_PER_N_FLOOR
        out["checks"][k] = {"pass": bool(ok), "delta": d["delta"], "ci_low": d["ci_low"], "min_per_N": min(d["per_N_delta"].values())}
    out["expand_to_5_folds"] = any(v["pass"] for v in out["checks"].values())
    return out


def sha256_file(path: Path) -> str:
    return e10lib.sha256_file(Path(path))


def write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)
