"""E10 (tier-1+2 retrain of Rank M04 / CORAL) library.  Pure functions where possible (tests/test_e10.py).

Plan: experiments/E10_tier2_retrain/PLAN.md (+ revision 1).  Read-only on data/scan.  E06 files are only read.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gogame.constants import SEED  # noqa: E402
from gogame.metrics import balanced_weights  # noqa: E402
from gogame.models import decide_ev  # noqa: E402
from gogame.questions import Bag, PlayerIndex, build_player_index, build_val_questions  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402
from gogame.split import assert_no_player_leak  # noqa: E402
import run_e01 as e01  # noqa: E402

OUT = ROOT / "experiments" / "E10_tier2_retrain"
CACHE = ROOT / "data" / "cache" / "e10"
SCAN = ROOT / "data" / "scan"
SCRATCH = ROOT / ".scratch" / "e10"
E06_OOF = ROOT / "experiments" / "E06_rank_formal" / "oof"
PILOT_HSL = ROOT / "data" / "cache" / "katago" / "pilot_hsl_games.parquet"

NS = (5, 10, 15, 20)
NS_Q1 = (5, 10, 15)
N_BOOT = 4000
THREADS = 4
CORAL_CHUNK = None            # smoke 2026-10-20: chunked standardisation (50_000) is NOT bit-identical -> L-BFGS stops at 366 instead of 343
                              # iterations and C differs from E06 by 7e-4; the default (float64-copy) path reproduces E06 exactly (diff 0.0).
                              # Memory is not a constraint (PLAN 5.2, measured), so every candidate uses the default path.
NOISE_STOP = 0.003            # PLAN 6.6
Q1_GUARD = -0.005             # PLAN 6.1(b)
PER_N_GUARD = -0.01           # PLAN 6.1(c)
POOL_SPLIT = 60               # PLAN risk 5: stratify Q2 by games in the question's (player, colour) pool
PCOLS = [f"p_{i}" for i in range(10)]

# ---- candidate D (PLAN revision 2): M03 (L0+HSL LightGBM, E06 settings) + M04_A, 0.5/0.5 probability mean, EV decode
D_NAME = "D"
M03_NAME = "M03A"             # M03 fitted on A's data / bags / folds (probabilities kept for transparency)
REPRO_NAME = "M03_REPRO"      # pseudo candidate: tier-1 M03 refit vs E06 oof_M03 (fold 0 gate)
D_WEIGHT_M03 = 0.5            # fixed by the PLAN, not tuned
D_DECODE = "ev"
D_DEADLINE = "2026-10-27 24:00"   # PLAN revision 2: no complete 5-fold D by then -> abandon D, ship W
M03_REPRO_TOL = 1e-6          # same tolerance as the C check (PLAN 3).  Same code path, seeds and thread count -> expected 0.0;
                              # 1e-6 is below any real change (one different tree moves a probability by >=1e-4) and above float32 noise (~1e-7).
M03_REPRO_FOLDS = (0,)        # the gate is fold 0 only (a full 5-fold tier-1 M03 refit would cost ~1 h); override with --folds
LGBM_THREADS = 4


@dataclasses.dataclass(frozen=True)
class Cand:
    name: str
    base: str                 # C / A / B
    tier1_only: bool
    bags_per_pc: int
    n_range: tuple[int, int]
    seed_variant: int         # 0 / 1  (1 = noise ruler: different bag sampling seed)

    def train_seed(self, fold: int) -> int:
        """C / C_s1 reuse E06's exact seed names so that the refit reproduces E06 M04s0 / M04s1."""
        if self.base == "C":
            return derive_seed(SEED, "e06", "train_bags", f"fold{fold}", f"s{self.seed_variant}")
        return derive_seed(SEED, "e10", "train_bags", self.base, f"fold{fold}", f"s{self.seed_variant}")

    @property
    def e06_name(self) -> str | None:
        return {"C": "M04s0", "C_s1": "M04s1"}.get(self.name)


CANDS = {c.name: c for c in (
    Cand("C", "C", True, 8, (5, 15), 0),
    Cand("A", "A", False, 8, (5, 15), 0),
    Cand("B", "B", False, 16, (5, 20), 0),
    Cand("C_s1", "C", True, 8, (5, 15), 1),
    Cand("A_s1", "A", False, 8, (5, 15), 1),
)}
CAND_ORDER = ("C", "A", "B", "C_s1", "A_s1")      # C first: its E06 reproduction check gates the rest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ HSL ingest
def read_hsl_segments(scan_dir: Path = SCAN, pilot_path: Path = PILOT_HSL, tiers=(1, 2)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(hsl_games float32, segment_table, info).  Read-only on scan_dir.  Shards are read one at a time and cast to
    float32 immediately (no float64 frame of the whole table).  A shard counts as present only if the ledger says done,
    both files exist and the content checks pass; everything else is listed with a reason.  info['complete'] is True
    only if every ledger shard of `tiers` is present."""
    ledger = pd.read_parquet(scan_dir / "ledger.parquet")
    manifest = pd.read_parquet(scan_dir / "manifest.parquet", columns=["uid", "shard_id"])
    by_shard = manifest.groupby("shard_id").indices
    man_uid = manifest["uid"].to_numpy()
    led = ledger[ledger["tier"].isin(tiers)].sort_values("shard_id")
    pilot = pd.read_parquet(pilot_path)
    ref_cols = list(pilot.columns)
    fcols = ref_cols[1:]
    frames, rows = [], []
    for r in led.itertuples(index=False):
        sid = r.shard_id
        rec = {"shard_id": sid, "tier": int(r.tier), "ledger_status": r.status, "n_games_manifest": int(r.n_games),
               "n_rows": 0, "n_skipped": 0, "present": False, "reason": ""}
        if sid == "P0000":
            g = pilot
            rec.update(n_rows=len(g), present=True, reason="pilot file")
            frames.append(g.astype({c: np.float32 for c in fcols}))
            rows.append(rec)
            continue
        if r.status != "done":
            rec["reason"] = f"ledger status {r.status}"
            rows.append(rec)
            continue
        gp, mp = scan_dir / "results" / f"{sid}_games.parquet", scan_dir / "results" / f"{sid}_meta.json"
        if not gp.exists() or not mp.exists():
            rec["reason"] = "ledger done but files missing"
            rows.append(rec)
            continue
        g = pd.read_parquet(gp)
        meta = json.loads(mp.read_text())
        problems = []
        if list(g.columns) != ref_cols:
            problems.append("columns differ from pilot")
        if len(g) != meta.get("n_games_out"):
            problems.append(f"rows {len(g)} != meta n_games_out {meta.get('n_games_out')}")
        if g["uid"].duplicated().any():
            problems.append("duplicate uid")
        if not problems and g[fcols].isna().any().any():
            problems.append("NaN in features")
        if not set(g["uid"]) <= set(man_uid[by_shard.get(sid, [])]):
            problems.append("uid outside shard manifest")
        rec["n_skipped"] = int(meta.get("n_skipped", 0))
        if problems:
            rec["reason"] = "; ".join(problems)
            rows.append(rec)
            continue
        rec.update(n_rows=len(g), present=True)
        frames.append(g.astype({c: np.float32 for c in fcols}))
        rows.append(rec)
        del g
    hsl = pd.concat(frames, ignore_index=True)
    del frames
    if hsl["uid"].duplicated().any():
        raise ValueError("duplicate uid across pilot/shards")
    seg = pd.DataFrame(rows)
    hsl = hsl.sort_values("uid", kind="stable").reset_index(drop=True)
    info = {"complete": bool(seg["present"].all()), "n_shards_total": int(len(seg)), "n_present": int(seg["present"].sum()),
            "n_present_by_tier": {int(t): int(g["present"].sum()) for t, g in seg.groupby("tier")},
            "n_shards_by_tier": {int(t): int(len(g)) for t, g in seg.groupby("tier")},
            "tier1_complete": bool(seg.loc[seg["tier"] == 1, "present"].all()),
            "n_games": int(len(hsl)), "n_skipped_total": int(seg["n_skipped"].sum()),
            "missing_shards": seg.loc[~seg["present"], "shard_id"].tolist(),
            "n_games_expected": int(led["n_games"].sum()), "tiers": list(tiers), "hsl_columns": fcols}
    return hsl, seg, info


# ------------------------------------------------------------------------------------------------ folds / questions
def fold_sets(nonh: pd.DataFrame, k: int) -> dict:
    """Exactly E06's fold logic (stage_oof): train_all = non-H players not in fold k; es = e01.es_players of those (10% per
    rank, named seed); fit = train_all - es."""
    train_all = set(nonh.loc[nonh["fold"] != k, "player_id"])
    val_pl = set(nonh.loc[nonh["fold"] == k, "player_id"])
    es_pl = e01.es_players(nonh[nonh["player_id"].isin(train_all)], e01.CFG["es_frac"])
    return {"train_all": train_all, "val": val_pl, "es": es_pl, "fit": train_all - es_pl}


def assert_fold_disjoint(fs: dict, H) -> None:
    for a, b in ((fs["fit"], fs["val"]), (fs["es"], fs["val"]), (fs["fit"], fs["es"]), (fs["train_all"] | fs["val"], set(H))):
        assert_no_player_leak(a, b)


def hsl_index(meta: pd.DataFrame, hsl_uids) -> PlayerIndex:
    sub = meta[meta["uid"].isin(set(hsl_uids))]
    return build_player_index(sub.assign(color_id=sub["color"]))


def make_q2(index: PlayerIndex, players, ns=NS) -> list[Bag]:
    """Q2: at most 1 question per (player, N), single colour, N games all with HSL.  qid prefixed `q2_` (Q1 uses E06's
    `val_<player>_<N>` names; the two sets must never collide)."""
    pl = sorted(p for p in set(players) if p in index.games)
    qs = build_val_questions(index, pl, ns, seed=derive_seed(SEED, "e10", "questions_oof"))
    return [dataclasses.replace(b, qid="q2_" + b.qid) for b in qs]


def pool_sizes(index: PlayerIndex, bags) -> np.ndarray:
    return np.array([len(index.games[b.player_id][b.color]) for b in bags])


# ------------------------------------------------------------------------------------------------ game_id leak check
class GameLeakError(AssertionError):
    pass


def game_key_overlap(meta: pd.DataFrame, query_uids, train_players, key: str = "rank_game") -> dict:
    """PLAN 4: query games of the validation fold must not appear among the games of training-fold players.
    key='rank_game' (hard gate): (rank_id, game_id).  The raw `game_id` is renumbered from g_0000001 in EACH train file
    (docs/sdd.md:49), i.e. one game_id value exists once per rank, so raw-game_id equality is not game identity;
    key='game_id' is reported for information only.  Returns {'n_query_games','n_overlap','examples'}."""
    q = meta[meta["uid"].isin(set(query_uids))]
    t = meta[meta["player_id"].isin(set(train_players))]
    if key == "rank_game":
        qk = set(zip(q["rank_id"].tolist(), q["game_id"].tolist()))
        tk = set(zip(t["rank_id"].tolist(), t["game_id"].tolist()))
    elif key == "game_id":
        qk, tk = set(q["game_id"]), set(t["game_id"])
    else:
        raise ValueError(key)
    inter = qk & tk
    return {"key": key, "n_query_games": len(qk), "n_overlap": len(inter), "examples": sorted(map(str, inter))[:3]}


def assert_no_game_leak(meta: pd.DataFrame, query_uids, train_players) -> dict:
    rep = game_key_overlap(meta, query_uids, train_players, "rank_game")
    if rep["n_overlap"]:
        raise GameLeakError(f"{rep['n_overlap']} validation-query game(s) also appear under training-fold players, e.g. {rep['examples']}; "
                            f"PLAN section 12: stop and ask the user")
    rep["info_raw_game_id"] = game_key_overlap(meta, query_uids, train_players, "game_id")["n_overlap"]
    return rep


# ------------------------------------------------------------------------------------------------ statistics
def _n_groups(sa, sb, groups, truth, ns, n_vals):
    sa, sb = np.asarray(sa, np.float64), np.asarray(sb, np.float64)
    groups, truth, ns = np.asarray(groups), np.asarray(truth), np.asarray(ns)
    _, gidx = np.unique(groups, return_inverse=True)
    P = int(gidx.max()) + 1
    dg, wg = [], []
    for n in n_vals:
        m = ns == n
        w = np.zeros(len(sa))
        w[m] = balanced_weights(truth[m])
        dg.append(np.bincount(gidx, weights=w * (sa - sb), minlength=P))
        wg.append(np.bincount(gidx, weights=w, minlength=P))
    return np.array(dg), np.array(wg), P


def equal_n_delta(sa, sb, groups, truth, ns, n_vals=None, n_boot: int = N_BOOT, level: float = 0.95, seed: int | None = None) -> dict:
    """N-equal-weight balanced delta (mean over N of the class-balanced per-N delta), player-level paired bootstrap, one
    resample shared by all N.  `level` sets the central interval (0.975 for the Bonferroni m=2 candidate tests)."""
    ns = np.asarray(ns)
    n_vals = sorted(set(ns.tolist())) if n_vals is None else list(n_vals)
    dg, wg, P = _n_groups(sa, sb, groups, truth, ns, n_vals)
    per_n = {int(n): float(dg[i].sum() / wg[i].sum()) for i, n in enumerate(n_vals)}
    rng = np.random.default_rng(derive_seed(SEED, "e10", "bootstrap") if seed is None else seed)
    c = rng.multinomial(P, np.full(P, 1.0 / P), size=n_boot).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        db = np.mean((c @ dg.T) / (c @ wg.T), axis=1)
    db = db[np.isfinite(db)]
    a = (1 - level) / 2
    return {"delta": float(np.mean(list(per_n.values()))), "ci_low": float(np.quantile(db, a)), "ci_high": float(np.quantile(db, 1 - a)),
            "level": level, "per_N_delta": per_n, "ns": [int(n) for n in n_vals]}


def equal_n_point(sa, sb, truth, ns, n_vals=None) -> dict:
    """Point estimate only (no bootstrap): per-N balanced delta and their mean."""
    ns, truth = np.asarray(ns), np.asarray(truth)
    n_vals = sorted(set(ns.tolist())) if n_vals is None else list(n_vals)
    per = {}
    for n in n_vals:
        m = ns == n
        if m.any():
            per[int(n)] = float(((np.asarray(sa, np.float64)[m] - np.asarray(sb, np.float64)[m]) * balanced_weights(truth[m])).mean())
    return {"delta": float(np.mean(list(per.values()))) if per else float("nan"), "per_N_delta": per}


def guards(a: dict) -> dict:
    """PLAN 6.1 (a)(b)(c) for one candidate against its reference.  a = {'q2_eq': equal_n_delta dict (97.5%), 'q1_point': float,
    'q2_per_N': {N: delta}}.  Returns the three booleans and 'ok'."""
    a_ok = a["q2_eq"]["ci_low"] > 0
    b_ok = a["q1_point"] >= Q1_GUARD
    c_ok = all(v >= PER_N_GUARD for v in a["q2_per_N"].values())
    return {"a_ci_low_gt0": bool(a_ok), "b_q1_ge_-0.005": bool(b_ok), "c_perN_ge_-0.01": bool(c_ok), "ok": bool(a_ok and b_ok and c_ok)}


def select_candidate(avail: dict) -> dict:
    """PLAN 6.1-6.2.  avail[X] = {'q2_eq': equal_n_delta dict (97.5%), 'q1_point': float, 'q2_per_N': {N: delta}} for X in
    {A, B} that were run; avail['BA'] = equal_n_delta dict (95%) of B-A or None."""
    qual, why = [], {}
    for x in ("A", "B"):
        if x not in avail:
            why[x] = "not run"
            continue
        g = guards(avail[x])
        why[x] = {k: v for k, v in g.items() if k != "ok"}
        if g["ok"]:
            qual.append(x)
    if not qual:
        chosen = "C"
    elif len(qual) == 1:
        chosen = qual[0]
    else:
        chosen = "B" if avail.get("BA") and avail["BA"]["ci_low"] > 0 else "A"
    return {"candidate": chosen, "qualified": qual, "checks": why}


# ------------------------------------------------------------------------------------------------ candidate D
def ensemble_probs(p_m03, p_m04, w_m03: float = D_WEIGHT_M03) -> np.ndarray:
    """w*P(M03) + (1-w)*P(M04), rows renormalised (same as scripts/experiments/e11 `ensemble`)."""
    p = w_m03 * np.asarray(p_m03, np.float64) + (1 - w_m03) * np.asarray(p_m04, np.float64)
    return p / p.sum(axis=1, keepdims=True)


def decide_d(p: np.ndarray) -> np.ndarray:
    """EV decode of the D probabilities (sum_k k p_k, nearest rank via gogame.models.decide_ev)."""
    return decide_ev(p)


def select_d(avail_d: dict | None, *, complete: bool, repro_ok: bool, reason: str = "") -> dict:
    """PLAN revision 2: D replaces the §6 winner W only if (a)(b)(c) all hold for D-vs-W; otherwise W stays.
    D is ABANDONED (W stays, nothing is tested) if it has no complete 5-fold OOF or the M03 reproduction gate is not OK.
    avail_d has the same layout as one entry of select_candidate's `avail`."""
    if not repro_ok:
        return {"status": "abandoned", "adopt": False, "reason": reason or "M03 refit does not reproduce E06 oof_M03 (or gate not run)"}
    if not complete or avail_d is None:
        return {"status": "abandoned", "adopt": False, "reason": reason or "no complete 5-fold OOF for D"}
    g = guards(avail_d)
    return {"status": "adopted" if g["ok"] else "rejected", "adopt": bool(g["ok"]), "checks": {k: v for k, v in g.items() if k != "ok"}}


def make_es_bags(index: PlayerIndex, es_players, fold: int, tier1_only: bool) -> list[Bag]:
    """Early-stopping questions for the M03 LightGBM.  tier-1 (repro): E06's exact tag, so the bags equal E06's
    `make_questions(index, es_pl, 'es_fold<k>')`.  Tier 1+2: own tag."""
    pl = sorted(p for p in set(es_players) if p in index.games)
    tag = ("e06", f"es_fold{fold}") if tier1_only else ("e10", f"es_fold{fold}")
    return build_val_questions(index, pl, NS, seed=derive_seed(SEED, *tag))
