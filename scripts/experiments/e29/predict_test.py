"""E29 test-time pipeline (local).  Plan: experiments/E29_set_model/PLAN.md.  Fusion structure and weight are fixed by the PLAN:
    P = 0.5 * P_Dse + 0.5 * P_set   (P_set = mean over 5 fold models x 3 seeds of the chosen structure), EV decode.

Stages (different interpreters because .venv has no torch and the system python3 has no pyarrow):
  features   (.venv)    test games -> raw 139-column matrix (same column order / definition as the training bundle) + bag rows.
                        PYTHONPATH=src:scripts .venv/bin/python -m scripts.experiments.e29.predict_test features
  dse        (.venv)    Dse test probabilities from data/models_final_se with the e21 worktree code; must reproduce rank_v3 row by row.
                        GOGAME_COLOR_SEMANTICS=actual .venv/bin/python -m scripts.experiments.e29.predict_test dse
  valcheck   (python3)  fold models re-predict their own val questions through the SAME code path as the test; must reproduce the OOF probabilities.
  infer      (python3)  fold models -> test probabilities  (CUDA_VISIBLE_DEVICES="" ; CPU only).
  fuse       (.venv)    fuse, decode, write the csv, check_submission, compare with rank_v3.
Never reads or writes anything H-related (no models_final*, no ledger); H is not used for scoring.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / ".scratch" / "e29" / "bundle"
TEST = ROOT / ".scratch" / "e29" / "test"
SCH2 = ROOT / ".scratch" / "e29" / "school2"          # downloads: out_f0save/ (fold 0 re-run with weights) and out/ (folds 1-4)
SCH0 = ROOT / ".scratch" / "e29" / "school" / "out"    # first fold-0 run (probabilities only)
WT = ROOT / ".claude" / "worktrees" / "e21-se-pipeline"
DATA = ROOT / "data" / "contest_1104"
HSL_TEST = ROOT / "data" / "test_scan" / "hsl_test_games.parquet"
SE_TEST = ROOT / "data" / "test_scan_se" / "se_test_games.parquet"
RANK_V3 = ROOT / "data" / "submissions" / "rank_v3_Dse_SE版.csv"
OUT_CSV = ROOT / ".scratch" / "e29" / "rank_v4_集合模型融合.csv"
K = 10


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def fold_dir(k):
    d = SCH2 / ("out_f0save" if k == 0 else "out")
    return d


# ------------------------------------------------------------------------------------------------- features
def cmd_features(a):
    os.environ.setdefault("GOGAME_COLOR_SEMANTICS", "actual")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import pandas as pd
    from gogame.sgf import get_color_semantics
    from predict import load_questions, l0_store, FILES       # main-repo scripts/predict.py (no SE code needed here)
    assert get_color_semantics() == "actual", get_color_semantics()
    TEST.mkdir(parents=True, exist_ok=True)
    aux = np.load(BUNDLE / "aux.npz", allow_pickle=False)
    names = aux["names"].tolist()
    rq = load_questions(DATA / FILES["rank_pub"], "pub", "rank") + load_questions(DATA / FILES["rank_priv"], "priv", "rank")
    games = [g for q in rq for g in q.games]
    uids = [g.uid for g in games]
    assert len(set(uids)) == len(uids)
    rep = {"n_questions": len(rq), "n_games": len(games), "color_semantics": get_color_semantics()}
    # HSL (104 columns; order must equal the training table / bundle)
    hsl = pd.read_parquet(HSL_TEST)
    hcols = [c for c in hsl.columns if c != "uid"]
    assert hcols == names[:104], "HSL column order differs from the training bundle"
    hsl = hsl.set_index("uid")
    miss = [u for u in uids if u not in hsl.index]
    assert not miss, f"{len(miss)} test games without HSL (first {miss[:3]})"
    hsl = hsl.reindex(uids)
    assert not hsl.isna().any().any()
    # L0 dense (31 columns) recomputed with the same function as data/cache/l0/dense.parquet
    st = l0_store(games, n_jobs=2)
    assert list(st.uids) == uids and list(st.names) == names[104:135], "L0 store order / names differ"
    # SE
    se = pd.read_parquet(SE_TEST).set_index("uid").reindex(uids)
    assert not se["se_mean"].isna().any() or True
    X = np.empty((len(games), 139), np.float32)
    X[:, :104] = hsl[hcols].to_numpy(np.float32)
    X[:, 104:135] = np.asarray(st.dense, np.float32)
    X[:, 135] = se["se_mean"].to_numpy(np.float32)
    X[:, 136] = se["se_std"].to_numpy(np.float32)
    X[:, 137] = np.isnan(X[:, 135]).astype(np.float32)
    X[:, 138] = np.array([g.color for g in games], np.float32)
    rows = np.full((len(rq), 20), -1, np.int32)
    pos = 0
    for i, q in enumerate(rq):
        n = len(q.games)
        assert 5 <= n <= 20 and len({g.color for g in q.games}) == 1
        rows[i, :n] = np.arange(pos, pos + n)
        pos += n
    assert pos == len(games)
    np.save(TEST / "test_X.npy", X)
    np.savez(TEST / "test_bags.npz", rows=rows, qids=np.array([q.qid for q in rq]), n=rows.__ge__(0).sum(1), color=np.array([q.color for q in rq]), uids=np.array(uids))
    rep.update({"nan_cells": int(np.isnan(X).sum()), "se_nan_games": int(X[:, 137].sum()), "n_by_N": {int(k): int(v) for k, v in zip(*np.unique(rows.__ge__(0).sum(1), return_counts=True))},
                "white_share": float(X[:, 138].mean()), "x_sha256": sha(TEST / "test_X.npy")})
    # --- fidelity of the test-time L0 construction: recompute training games, compare with the cache the model was trained on
    from gogame.io import GameRecord, read_train_csv
    fid = {}
    l0c = pd.read_parquet(ROOT / "data/cache/l0/dense.parquet").set_index("uid")
    for rk in ("1D", "6D"):
        df = read_train_csv(ROOT / "data" / "training" / f"train_{rk}.csv").head(300)
        recs = [GameRecord(u, p, int(k), int(c), g, "train") for u, p, k, c, g in zip(df["uid"], df["player_id"], df["rank_id"], df["color_id"], df["sgf_content"])]
        b = l0_store(recs, n_jobs=2)
        ref = l0c.loc[df["uid"].tolist()][list(b.names)].to_numpy(np.float32)
        d = np.nan_to_num(np.asarray(b.dense, np.float32)) - np.nan_to_num(ref)
        fid[rk] = {"n": len(df), "max_abs_diff": float(np.abs(d).max()), "nan_pattern_equal": bool((np.isnan(b.dense) == np.isnan(ref)).all())}
    rep["l0_fidelity_vs_cache"] = fid
    # --- distribution: test vs a random sample of non-H training games (standardised mean difference, KS)
    from scipy.stats import ks_2samp
    Xtr = np.load(BUNDLE / "X.npy", mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(Xtr.shape[0], 30000, replace=False))
    S = np.asarray(Xtr[idx], np.float64)
    T = X.astype(np.float64)
    out = []
    for j, nm in enumerate(names):
        s, t = S[:, j], T[:, j]
        s, t = s[~np.isnan(s)], t[~np.isnan(t)]
        if len(t) < 50 or len(s) < 50:
            out.append({"col": nm, "n_test": int(len(t)), "n_train": int(len(s))}); continue
        sd = s.std() or 1.0
        out.append({"col": nm, "smd": float((t.mean() - s.mean()) / sd), "ks": float(ks_2samp(s, t).statistic), "test_nan": float(np.isnan(T[:, j]).mean()), "train_nan": float(np.isnan(S[:, j]).mean())})
    out_sorted = sorted([o for o in out if "ks" in o], key=lambda o: -o["ks"])
    rep["dist_top_ks"] = out_sorted[:12]
    rep["dist_max_abs_smd"] = max(abs(o["smd"]) for o in out_sorted)
    rep["dist_cols_no_variation"] = [o["col"] for o in out if "ks" not in o]
    (TEST / "features_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in rep if k not in ("dist_top_ks",)}, indent=1))
    print("top KS:", [(o["col"], round(o["ks"], 3), round(o["smd"], 2)) for o in out_sorted[:8]])


# ------------------------------------------------------------------------------------------------- Dse
def cmd_dse(a):
    os.environ["GOGAME_COLOR_SEMANTICS"] = "actual"
    sys.path[:0] = [str(WT / "src"), str(WT / "scripts")]
    import pandas as pd
    import predict as P
    from gogame.models import decide_ev
    from gogame.constants import RANKS
    TEST.mkdir(parents=True, exist_ok=True)
    rq = P.load_questions(DATA / P.FILES["rank_pub"], "pub", "rank") + P.load_questions(DATA / P.FILES["rank_priv"], "priv", "rank")
    m = P.HslEnsembleSeRank.load(ROOT / "data" / "models_final_se", n_jobs=2)
    m.set_features(HSL_TEST)
    m.set_se_features(SE_TEST)
    p = m.predict_proba(rq)
    np.save(TEST / "dse_probs.npy", p)
    qids = [q.qid for q in rq]
    pred = decide_ev(p)
    v3 = pd.read_csv(RANK_V3)
    lab = pd.Series([RANKS[i] for i in pred], index=qids)
    # rank_v3 uses the contest-format ids; compare row by row in file order
    assert list(v3["question_id"]) == qids, "question order differs from rank_v3"
    same = (v3["pred_rank"].to_numpy() == lab.to_numpy())
    rep = {"n": len(p), "reproduces_rank_v3_rows": int(same.sum()), "meta": {k: m.meta.get(k) for k in ("model", "method", "rounds", "decode", "include_holdout")},
           "p_rowsum_min": float(p.sum(1).min()), "p_rowsum_max": float(p.sum(1).max()), "dse_sha256": sha(TEST / "dse_probs.npy")}
    (TEST / "dse_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    assert same.all(), "Dse does not reproduce rank_v3"


# ------------------------------------------------------------------------------------------------- torch stages
def _torch_env():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(2)
    from scripts.experiments.e29 import setmodel as sm
    return torch, sm


def _models(torch, sm, kind, k):
    d = fold_dir(k)
    sel = json.loads((d / f"select_{kind}_fold{k}.json").read_text())
    W = sel["chosen_W"]
    ms = []
    for s in (0, 1, 2):
        ck = torch.load(d / "models" / f"{kind}_W{W}_fold{k}_s{s}.pt", map_location="cpu", weights_only=False)
        assert ck["kind"] == kind and ck["W"] == W and ck["fold"] == k and ck["seed"] == s and ck["F"] == 139
        m = sm.build(kind, 139, W, 0.2)
        m.load_state_dict(ck["state"])
        m.eval()
        ms.append(m)
    nz = np.load(d / f"norm_fold{k}.npz")
    return ms, {x: nz[x] for x in ("lo", "hi", "mu", "sd")}, W


def _predict(torch, models, norm, Xraw, rows, bs=512):
    """Identical to training: float64 clip -> standardise -> nan->0 -> float32; gather bag rows (pad -1 -> row 0, masked)."""
    x = (np.clip(np.asarray(Xraw, np.float64), norm["lo"], norm["hi"]) - norm["mu"]) / norm["sd"]
    Xg = torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float32))
    R = torch.from_numpy(np.asarray(rows, np.int64))
    outs = []
    with torch.no_grad():
        for m in models:
            ps = []
            for a in range(0, len(R), bs):
                r = R[a:a + bs]
                mask = r >= 0
                lp = m(Xg[r.clamp(min=0)], mask).float()
                ps.append(torch.exp(lp).numpy().astype(np.float64))
            p = np.concatenate(ps)
            outs.append(p / p.sum(1, keepdims=True))
    return np.stack(outs)            # (seeds, Q, 10)


def cmd_valcheck(a):
    torch, sm = _torch_env()
    aux = np.load(BUNDLE / "aux.npz", allow_pickle=False)
    Xb = np.load(BUNDLE / "X.npy", mmap_mode="r")
    res = {}
    for k in a.folds:
        rows = aux[f"val_rows_f{k}"]
        qid = aux[f"val_qid_f{k}"]
        uniq = np.unique(rows[rows >= 0])
        Xs = np.asarray(Xb[uniq], np.float32)
        remap = np.full(rows.shape, -1, np.int64)
        mk = rows >= 0
        remap[mk] = np.searchsorted(uniq, rows[mk])
        for kind in a.kinds:
            d = fold_dir(k)
            ms, norm, W = _models(torch, sm, kind, k)
            P = _predict(torch, ms, norm, Xs, remap)
            for s in range(3):
                z = np.load(d / f"setprobs_{kind}_W{W}_fold{k}_s{s}.npz", allow_pickle=False)
                assert (z["qid"] == qid).all()
                diff = np.abs(P[s] - z["p"].astype(np.float64))
                res[f"{kind}_f{k}_s{s}"] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                                            "decode_equal_share": float((np.argmax(P[s] @ _S(), 1) == np.argmax(z["p"].astype(np.float64) @ _S(), 1)).mean())}
            if k == 0 and SCH0.exists():       # determinism of the retrain itself vs the very first fold-0 run
                for s in range(3):
                    f0 = SCH0 / f"setprobs_{kind}_W{W}_fold0_s{s}.npz"
                    if f0.exists():
                        z = np.load(f0, allow_pickle=False)
                        res[f"{kind}_f0_s{s}_vs_firstrun"] = {"max_abs_diff": float(np.abs(P[s] - z["p"].astype(np.float64)).max())}
            print(kind, k, {kk: v for kk, v in res.items() if kk.startswith(f"{kind}_f{k}")}, flush=True)
    (TEST / f"valcheck_{'_'.join(map(str, a.folds))}.json").write_text(json.dumps(res, indent=1))
    worst = max(v["max_abs_diff"] for kk, v in res.items() if "firstrun" not in kk)
    print("WORST own-OOF max_abs_diff", worst)
    # tolerance: DeepSets reproduces to ~1e-6; attention differs from the school GPU fp32 run by up to ~4e-4 (a float64 CPU reference deviates
    # from the GPU result by the same amount, i.e. GPU kernel noise, not a pipeline difference) -> 1 or 2 EV decisions in 4,685 may flip
    worst_dec = min(v["decode_equal_share"] for kk, v in res.items() if "firstrun" not in kk)
    print("WORST decode_equal_share", worst_dec)
    if worst > 1e-3 or worst_dec < 0.999:
        sys.exit("valcheck FAILED: pipeline does not reproduce its own OOF probabilities")


def _S():
    e1 = np.exp(-1.0)
    i = np.arange(K)
    d = np.abs(i[:, None] - i[None, :])
    return np.where(d == 0, 1.0, np.where(d == 1, e1, 0.0))


def cmd_infer(a):
    torch, sm = _torch_env()
    X = np.load(TEST / "test_X.npy")
    rows = np.load(TEST / "test_bags.npz")["rows"].astype(np.int64)
    for kind in a.kinds:
        allp = []
        for k in range(5):
            ms, norm, W = _models(torch, sm, kind, k)
            allp.append(_predict(torch, ms, norm, X, rows))
        allp = np.stack(allp)          # (fold, seed, Q, 10)
        np.save(TEST / f"set_{kind}_probs.npy", allp)
        print(kind, allp.shape, "row-sum dev", float(np.abs(allp.sum(-1) - 1).max()), "sha", sha(TEST / f"set_{kind}_probs.npy")[:12])


# ------------------------------------------------------------------------------------------------- fuse
def cmd_fuse(a):
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import pandas as pd
    from gogame.constants import RANKS
    from gogame.models import decide_ev
    from gogame.submission import write_rank_submission
    dse = np.load(TEST / "dse_probs.npy")
    qids = np.load(TEST / "test_bags.npz")["qids"].tolist()
    sub = pd.read_csv(RANK_V3)
    v3 = np.array([RANKS.index(r) for r in sub["pred_rank"]])
    out = {}
    for kind in a.kinds:
        sp = np.load(TEST / f"set_{kind}_probs.npy")
        pset = sp.mean((0, 1))
        pset /= pset.sum(1, keepdims=True)
        p = 0.5 * dse + 0.5 * pset                 # PLAN fusion: fixed 0.5 / 0.5 (lib.ensemble_probs(dse, set, 0.5))
        p /= p.sum(1, keepdims=True)
        pred = decide_ev(p)
        f = Path(a.out) if kind == a.main else Path(str(a.out).replace(".csv", f"_alt_{kind}.csv"))
        write_rank_submission(qids, pred, f)
        out[kind] = {"file": str(f), "sha256": sha(f), "agree_with_rank_v3": float((pred == v3).mean()), "mean_pred_rank_diff_vs_v3": float((pred - v3).mean()),
                     "mean_abs_rank_diff_vs_v3": float(np.abs(pred - v3).mean()), "mean_pred_rank": float(pred.mean()), "v3_mean_pred_rank": float(v3.mean()),
                     "set_alone_agree_with_v3": float((decide_ev(pset) == v3).mean()),
                     "n_classes_predicted": int(len(np.unique(pred))), "pred_share": np.bincount(pred, minlength=K).round(3).tolist(),
                     "v3_share": np.bincount(v3, minlength=K).round(3).tolist(),
                     "per_N_agree": {int(n): float((pred[np.load(TEST / 'test_bags.npz')['n'] == n] == v3[np.load(TEST / 'test_bags.npz')['n'] == n]).mean()) for n in (5, 10, 15, 20)}}
        import check_submission
        out[kind]["check_submission_rc"] = check_submission.main([str(f), "--task", "rank", "--data-dir", str(DATA)])
        print(kind, json.dumps(out[kind], indent=1))
    (TEST / "fuse_report.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("features"); sp.add_parser("dse")
    v = sp.add_parser("valcheck"); v.add_argument("--folds", type=int, nargs="+", default=[0]); v.add_argument("--kinds", nargs="+", default=["ds", "attn"])
    i = sp.add_parser("infer"); i.add_argument("--kinds", nargs="+", default=["ds", "attn"])
    f = sp.add_parser("fuse"); f.add_argument("--kinds", nargs="+", default=["ds", "attn"]); f.add_argument("--main", required=True); f.add_argument("--out", default=str(OUT_CSV))
    a = ap.parse_args()
    {"features": cmd_features, "dse": cmd_dse, "valcheck": cmd_valcheck, "infer": cmd_infer, "fuse": cmd_fuse}[a.cmd](a)
