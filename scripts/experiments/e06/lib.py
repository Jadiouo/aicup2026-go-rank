"""E06 (Rank formal) library: HSL ingest (read-only on data/scan), questions, coverage, selection rule, H ledger.

Pure functions where possible so they can be unit-tested on synthetic frames (tests/test_e06_lib.py).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gogame.constants import SEED  # noqa: E402
from gogame.metrics import balanced_weights  # noqa: E402
from gogame.questions import Bag, PlayerIndex, build_player_index, build_val_questions  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402

OUT = ROOT / "experiments" / "E06_rank_formal"
CACHE = ROOT / "data" / "cache" / "e06"
SCAN = ROOT / "data" / "scan"
PILOT_HSL = ROOT / "data" / "cache" / "katago" / "pilot_hsl_games.parquet"
SCRATCH = ROOT / ".scratch" / "e06"

MODELS = ("M01", "M02", "M03", "M04s0", "M04s1")
NS = (5, 10, 15, 20)            # N=20 can never occur (tier 1 has <= 15 games per player x colour); kept so that the
                                # skip is visible in the coverage report instead of silent.
TRAIN_N_RANGE = (5, 15)
BAGS_PER_PLAYER_COLOR = 8
N_BOOT = 4000
THREADS = 4
SUCCESS_MARGIN = 0.015
MEMO_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ HSL ingest
def read_hsl_segments(scan_dir: Path = SCAN, pilot_path: Path = PILOT_HSL) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (hsl_games, segment_table, info).  Only tier-1 shards (+ the pilot) are used.

    Read-only on scan_dir.  A shard counts as present only if its ledger status is `done` AND its files exist AND its
    meta says n_games_out == rows.  Anything else is listed in the segment table with the reason (never dropped
    silently).  `info['complete']` is True only when every tier-1 shard is present.
    """
    ledger = pd.read_parquet(scan_dir / "ledger.parquet")
    shards = pd.read_parquet(scan_dir / "shards.parquet")
    manifest = pd.read_parquet(scan_dir / "manifest.parquet", columns=["uid", "shard_id", "tier"])
    t1 = ledger[ledger["tier"] == 1].copy()
    pilot = pd.read_parquet(pilot_path)
    frames, rows = [pilot], []
    ref_cols = list(pilot.columns)
    for r in t1.itertuples(index=False):
        sid = r.shard_id
        rec = {"shard_id": sid, "ledger_status": r.status, "n_games_manifest": int(r.n_games), "n_rows": 0,
               "n_skipped": 0, "present": False, "reason": ""}
        if sid == "P0000":
            rec.update(n_rows=len(pilot), present=True, reason="pilot file")
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
        if g.drop(columns="uid").isna().any().any():
            problems.append("NaN in features")
        if not set(g["uid"]) <= set(manifest.loc[manifest["shard_id"] == sid, "uid"]):
            problems.append("uid outside shard manifest")
        rec["n_skipped"] = int(meta.get("n_skipped", 0))
        if problems:
            rec["reason"] = "; ".join(problems)
            rows.append(rec)
            continue
        rec.update(n_rows=len(g), present=True, reason="")
        frames.append(g)
        rows.append(rec)
    hsl = pd.concat(frames, ignore_index=True)
    if hsl["uid"].duplicated().any():
        raise ValueError("duplicate uid across pilot/shards")
    if hsl.drop(columns="uid").isna().any().any():
        raise ValueError("NaN in HSL features")
    seg = pd.DataFrame(rows)
    hsl = hsl.sort_values("uid", kind="stable").reset_index(drop=True)
    hsl[ref_cols[1:]] = hsl[ref_cols[1:]].astype(np.float32)
    info = {"complete": bool(seg["present"].all()), "n_shards_tier1": int(len(seg)), "n_present": int(seg["present"].sum()),
            "n_games": int(len(hsl)), "n_skipped_total": int(seg["n_skipped"].sum()),
            "missing_shards": seg.loc[~seg["present"], "shard_id"].tolist(),
            "n_games_expected_tier1": int(t1["n_games"].sum()), "hsl_columns": ref_cols[1:]}
    del shards
    return hsl, seg, info


# ------------------------------------------------------------------------------------------------ index / questions
def hsl_index(meta: pd.DataFrame, hsl_uids) -> PlayerIndex:
    """PlayerIndex restricted to games that have HSL features (questions and bags may only use these)."""
    sub = meta[meta["uid"].isin(set(hsl_uids))]
    return build_player_index(sub.assign(color_id=sub["color"]))


def make_questions(index: PlayerIndex, players, tag: str, ns=NS) -> list[Bag]:
    """At most one question per (player, N), single colour, N games all with HSL.  Players absent from the index
    (no HSL game) and (player, N) without N HSL games in one colour simply produce no question; coverage_table
    records them."""
    pl = sorted(p for p in set(players) if p in index.games)
    return build_val_questions(index, pl, ns, seed=derive_seed(SEED, "e06", tag))


def coverage_table(split_players: pd.DataFrame, index: PlayerIndex, manifest_t1: pd.DataFrame, questions: list[Bag],
                   ns=NS) -> pd.DataFrame:
    """One row per player of `split_players` (player_id, rank_id, fold, is_holdout): scanned/target games per colour,
    number of questions per N and a status.  status in {ok, not_scanned, partial_lt5, structural_lt5}.
    structural_lt5 = the tier-1 manifest itself has < min(ns) games in every colour (can never get a question);
    partial_lt5 = manifest has enough but fewer than min(ns) are scanned yet (pending data)."""
    tgt = manifest_t1.groupby(["player_id", "color"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    tgt.columns = ["target_b", "target_w"]
    qn = pd.DataFrame([(b.player_id, b.n) for b in questions], columns=["player_id", "n"])
    qcount = qn.groupby(["player_id", "n"]).size().unstack(fill_value=0).reindex(columns=list(ns), fill_value=0) \
        if len(qn) else pd.DataFrame(columns=list(ns))
    qcount.columns = [f"q_n{n}" for n in qcount.columns]
    out = split_players[["player_id", "rank_id", "fold", "is_holdout"]].set_index("player_id")
    scan_b = pd.Series({p: len(index.games[p][0]) for p in index.games}, name="scanned_b")
    scan_w = pd.Series({p: len(index.games[p][1]) for p in index.games}, name="scanned_w")
    out = out.join(tgt).join(scan_b).join(scan_w).join(qcount)
    for c in ("target_b", "target_w", "scanned_b", "scanned_w", *[f"q_n{n}" for n in ns]):
        out[c] = out[c].fillna(0).astype(int)
    lo = min(ns)
    nq = out[[f"q_n{n}" for n in ns]].sum(axis=1)
    mx_scan = out[["scanned_b", "scanned_w"]].max(axis=1)
    mx_tgt = out[["target_b", "target_w"]].max(axis=1)
    out["status"] = np.where(nq > 0, "ok", np.where(mx_tgt < lo, "structural_lt5",
                    np.where(mx_scan == 0, "not_scanned", "partial_lt5")))
    return out.reset_index()


def coverage_summary(cov: pd.DataFrame) -> dict:
    nonh = cov[~cov["is_holdout"]]
    return {"n_players": int(len(cov)), "n_nonH": int(len(nonh)),
            "status_counts_nonH": nonh["status"].value_counts().to_dict(),
            "status_counts_per_fold": {int(k): g["status"].value_counts().to_dict() for k, g in nonh.groupby("fold")},
            "excluded_rank_counts_nonH": nonh.loc[nonh["status"] != "ok", "rank_id"].value_counts().sort_index().to_dict(),
            "questions_per_N_nonH": {n: int(nonh[f"q_n{n}"].sum()) for n in NS if f"q_n{n}" in nonh}}


# ------------------------------------------------------------------------------------------------ selection rule
def select_models(deltas: dict, scores: dict, alpha_key: str = "ci_low") -> dict:
    """Pre-registered rule (PLAN.md section 6).  `deltas[name]` = balanced/argmax paired-bootstrap dict for
    the named comparison; `scores[model]` = balanced argmax mean.

    main = M02 unless a challenger beats M02 with ci_low > 0:
      M04 needs BOTH training seeds (M04s0-M02 and M04s1-M02) ci_low > 0; M03 needs M03-M02 ci_low > 0.
    If several qualify the larger point-estimate delta (M04: mean of the two seeds) wins.
    backup = highest-scoring model among {M02, M03, M04} that is not main (M04 stands for seed 0).
    """
    gains = {}
    if deltas["M03-M02"][alpha_key] > 0:
        gains["M03"] = deltas["M03-M02"]["delta"]
    if deltas["M04s0-M02"][alpha_key] > 0 and deltas["M04s1-M02"][alpha_key] > 0:
        gains["M04"] = 0.5 * (deltas["M04s0-M02"]["delta"] + deltas["M04s1-M02"]["delta"])
    main = max(gains, key=gains.get) if gains else "M02"
    pool = {"M02": scores["M02"], "M03": scores["M03"], "M04": scores["M04s0"]}
    backup = max((m for m in pool if m != main), key=lambda m: pool[m])
    return {"main": main, "backup": backup, "challengers_qualified": gains,
            "rule": "main=M02 unless M03/M04 ci_low(delta vs M02)>0 (M04 both seeds); backup=best remaining by OOF balanced"}


# ------------------------------------------------------------------------------------------------ N-equal-weight (report only)
def equal_n_delta(sa, sb, groups, truth, ns, n_boot: int = N_BOOT, seed: int | None = None) -> dict:
    """REPORT-ONLY (PLAN revision 1): N-equal-weight balanced delta = arithmetic mean over the N values that have
    questions of the per-N balanced delta (class-balanced within each N).  Group (player) bootstrap; one resample is
    shared by all N so the per-N deltas are computed on the same resampled players.  Never used for selection."""
    sa, sb = np.asarray(sa, np.float64), np.asarray(sb, np.float64)
    groups, truth, ns = np.asarray(groups), np.asarray(truth), np.asarray(ns)
    n_vals = sorted(set(ns.tolist()))
    _, gidx = np.unique(groups, return_inverse=True)
    P = int(gidx.max()) + 1
    dg, wg = [], []
    for n in n_vals:
        m = ns == n
        w = np.zeros(len(sa))
        w[m] = balanced_weights(truth[m])
        dg.append(np.bincount(gidx, weights=w * (sa - sb), minlength=P))
        wg.append(np.bincount(gidx, weights=w, minlength=P))
    dg, wg = np.array(dg), np.array(wg)                                   # [K,P]
    per_n = {int(n): float(dg[i].sum() / wg[i].sum()) for i, n in enumerate(n_vals)}
    rng = np.random.default_rng(derive_seed(SEED, "bootstrap") if seed is None else seed)
    c = rng.multinomial(P, np.full(P, 1.0 / P), size=n_boot).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        db = np.mean((c @ dg.T) / (c @ wg.T), axis=1)
    db = db[np.isfinite(db)]
    return {"delta": float(np.mean(list(per_n.values()))), "ci_low": float(np.quantile(db, 0.025)),
            "ci_high": float(np.quantile(db, 0.975)), "p_gt0": float(np.mean(db > 0)), "per_N_delta": per_n,
            "ns": [int(n) for n in n_vals]}


def equal_n_score(s, truth, ns) -> float:
    """N-equal-weight balanced mean score = mean over N of the class-balanced mean score within that N."""
    s, truth, ns = np.asarray(s, np.float64), np.asarray(truth), np.asarray(ns)
    return float(np.mean([(s[ns == n] * balanced_weights(truth[ns == n])).mean() for n in sorted(set(ns.tolist()))]))


def equal_n_rank_info(main: str, deltas_eq: dict, scores_eq: dict) -> dict:
    """REPORT-ONLY: re-apply the unchanged section-6 rule (select_models) to the N-equal-weight argmax deltas
    (`deltas_eq["M03-M02"]` etc., dicts with delta/ci_low) and scores (`scores_eq[model]`, N-equal balanced argmax).
    equal_n_main = that result; equal_n_rank_agrees = (equal_n_main == section-6 main).  Never feeds back into selection."""
    eq = select_models(deltas_eq, scores_eq)
    pool = {"M02": scores_eq["M02"], "M03": scores_eq["M03"], "M04": 0.5 * (scores_eq["M04s0"] + scores_eq["M04s1"])}
    return {"equal_n_main": eq["main"], "equal_n_rank_agrees": bool(eq["main"] == main),
            "equal_n_ranking": sorted(pool, key=lambda m: -pool[m]), "equal_n_scores": {m: float(v) for m, v in pool.items()}}


def model_key(method: str) -> str:
    return "M04s0" if method == "M04" else method


# ------------------------------------------------------------------------------------------------ H ledger
LEDGER = OUT / "h_eval_ledger.jsonl"


def read_ledger(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def assert_not_evaluated(methods, path: Path = LEDGER) -> None:
    seen = {e["method"] for e in read_ledger(path)}
    dup = sorted(set(methods) & seen)
    if dup:
        raise PermissionError(f"H was already used for {dup} (see {path}); H is evaluated once per method. "
                              f"Nothing was run.")


def append_ledger(entries: list[dict], path: Path = LEDGER) -> None:
    assert_not_evaluated([e["method"] for e in entries], path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for e in entries:
            f.write(json.dumps({**e, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")
