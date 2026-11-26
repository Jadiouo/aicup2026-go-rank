"""E37 rank_v6: test-side Dse leg (final models trained on all training games, then applied to the Rank test set).  Same code path as rank_v3/v5 (`predict.py train --task rank --rank-model hsl_lgbm_se`,
branch e21-se-pipeline worktree = `.claude/worktrees/e21-se-pipeline`): `HslLgbmRank.fit(method="ens", se=...)` = M03 LightGBM (L0 + HSL aggregate + 7 SE columns, fixed rounds, no early stopping)
+ M04 CORAL (HSL aggregate), 0.5/0.5, EV decode; E10 candidate-D bag settings from experiments/E10_tier2_retrain/oof/selection.json; per-game HSL from data/cache/e10/hsl_games.parquet.
This file reproduces `train_hsl_rank` line by line (the original hard-wires the HSL parquet path); only the per-game HSL SOURCE and the player set differ between modes:
  repro60 : HSL = hsl_games.parquet (W60), all 1,000,000 games incl. H players, rounds = default_se_rounds (60), SE tables se_uid+se_h  -> must reproduce data/models_final_se (v3/v5 Dse) bit for bit on the test set
  w120    : HSL = first 104 columns of .scratch/e36/bundle/X120_full.npy (W120; aux uid order = the 712,945 non-H games), non-H players only (H has no W120 features), rounds = median of the 5 W120 OOF
            best iterations (E37 dse_w120 meta_fold*.json `best_iteration_m03`; same rule as E21's 60 = median of 75,60,55,68,58)
Commands (.venv; PYTHONPATH not needed):
  train   --mode repro60|w120 --out DIR [--threads 4]        -> DIR/rank_hsl_lgbm_se.pkl, DIR/train_info.json
  mktest  --out DIR                                          -> DIR/hsl_test120_games.parquet + features_manifest.json (test_X120[:, :104] with test_bags uids), bit checks vs the stored W60 test table
  predict --model-dir DIR --hsl PARQUET --out NPY            -> probabilities (800 x 10) through HslEnsembleSeRank (HSL + SE of the test games), color_semantics=actual
Never touches H labels for anything but the repro60 control (include_holdout is the v3 setting); never reads h_eval_ledger.jsonl; writes only under the given --out."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WT = ROOT / ".claude/worktrees/e21-se-pipeline"
DATA = ROOT / "data/contest_1104"
SELECTION = ROOT / "experiments/E10_tier2_retrain/oof/selection.json"
X120_FULL = ROOT / ".scratch/e36/bundle/X120_full.npy"
AUX = ROOT / ".scratch/e29/bundle/aux.npz"
OOF_W120 = ROOT / ".scratch/e37/runs/v6_dse/out/W120"
SE_TABLES = [ROOT / ".scratch/e21/se_uid.parquet", ROOT / ".scratch/e21/se_h.parquet"]
TEST_X120 = ROOT / ".scratch/e35/test/test_X120.npy"
TEST_BAGS = ROOT / ".scratch/e29/test/test_bags.npz"
HSL_TEST = ROOT / "data/test_scan/hsl_test_games.parquet"
SE_TEST = ROOT / "data/test_scan_se/se_test_games.parquet"
N_HSL = 104


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _import_predict():
    sys.path[:0] = [str(WT / "src"), str(WT / "scripts")]
    import predict as P
    return P


def cmd_train(a):
    import numpy as np, pandas as pd
    P = _import_predict()
    from gogame.aggregate import L0Store
    from gogame.gametable import load_table
    from gogame.split import assert_no_player_leak, read_split
    assert P.ROOT == WT and (WT / "data").resolve() == (ROOT / "data").resolve()
    cfg = json.loads(SELECTION.read_text())
    assert cfg["candidate"] == "D" and cfg["main"] == "M03+M04" and cfg["decode"] == "ev" and cfg["valid_for_confirm"] and not cfg["partial"]
    bags_kw = {"bags_per_pc": int(cfg["bags_per_pc"]), "n_range": tuple(cfg["n_range"]), "seed_ns": cfg.get("seed_ns", "e10")}
    e06_cache, selection = ROOT / cfg["cache"], SELECTION
    fm = json.loads((e06_cache / "features_manifest.json").read_text())
    assert fm.get("complete") and cfg.get("features_sha256") == fm.get("sha256"), "E10 features changed since selection.json"
    hsl = pd.read_parquet(e06_cache / "hsl_games.parquet")
    split = read_split(P.SPLITS / "split.parquet")
    H = set(split.loc[split["is_holdout"], "player_id"])
    include_holdout = a.mode == "repro60"
    train_players = set(split["player_id"]) if include_holdout else set(split.loc[split["fold"] >= 0, "player_id"])
    if not include_holdout:
        assert_no_player_leak(train_players, H)
    meta = pd.read_parquet(P.CACHE / "games_meta.parquet")
    meta = meta[meta["player_id"].isin(train_players)].reset_index(drop=True)
    info = {"mode": a.mode, "include_holdout": include_holdout, "threads": a.threads}
    if a.mode == "repro60":
        r, rounds_meta = P.default_se_rounds(P.E21_RESULTS)
        info["hsl_source"] = str(e06_cache / "hsl_games.parquet")
    else:
        its = [int(json.loads((OOF_W120 / f"meta_fold{k}.json").read_text())["best_iteration_m03"]) for k in range(5)]
        r = P.final_num_rounds(its)
        rounds_meta = {"rounds_source": "median of E37 W120 OOF best_iteration_m03", "best_iterations": its}
        aux = np.load(AUX, allow_pickle=False); uid = aux["uid"].astype(str); names = aux["names"].astype(str).tolist()
        X = np.load(X120_FULL, mmap_mode="r")
        assert X.shape == (len(uid), 139) and X.dtype == np.float32, X.shape
        assert list(hsl.columns[1:]) == names[:N_HSL], "hsl_games column order != bundle X column order"
        pos = pd.Series(np.arange(len(uid)), index=uid)
        hsl = hsl[hsl["uid"].isin(set(uid))].reset_index(drop=True)              # file order kept (as the original)
        assert len(hsl) == len(uid)
        rows = pos.loc[hsl["uid"].to_numpy()].to_numpy()
        old = hsl.iloc[:, 1:].to_numpy(np.float32)
        new = np.asarray(X[rows, :N_HSL], np.float32)
        assert np.isfinite(new).all()
        info["w60_rows_equal_w120_bitwise_share"] = float((old.view(np.uint32) == new.view(np.uint32)).all(1).mean())
        hsl = pd.concat([hsl[["uid"]], pd.DataFrame(new, columns=hsl.columns[1:])], axis=1)
        info["hsl_source"] = f"{X120_FULL} [:, :104] (sha256 {sha(X120_FULL)[:16]})"
        assert not (set(hsl["uid"]) & set(meta.loc[meta["player_id"].isin(H), "uid"])), "H game in the W120 training frame"
    info["rounds"] = int(r); info["rounds_meta"] = rounds_meta
    table = load_table(P.CACHE / "l0")
    pos_l0 = pd.Series(np.arange(len(table.uids)), index=table.uids)
    rows = pos_l0.loc[hsl["uid"].to_numpy()].to_numpy()
    l0 = L0Store(hsl["uid"].tolist(), table.dense[rows], table.sparse[rows], table.names)
    extra = {"include_holdout": bool(include_holdout), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "e06_main": "M03+M04", "decode": cfg.get("decode", "argmax"),
             "features_sha256": fm["sha256"], **rounds_meta, "e37_mode": a.mode}
    sm, ss, sinfo = P.load_se_tables(SE_TABLES)
    cov = P.se_coverage(meta, hsl["uid"], sm, H)
    print("SE tables:", sinfo, "coverage:", cov, flush=True)
    extra["se_tables"] = [{"path": t["path"], "rows": t["rows"], "sha256": sha(t["path"])} for t in sinfo["tables"]]; extra["se_coverage"] = cov
    se = P.se_maps(sm, ss)
    t0 = time.perf_counter()
    m = P.HslLgbmRank.fit(hsl, meta, r, a.threads, use_l0=False, l0=l0, meta_extra=extra, method="ens", se=se, **bags_kw)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = m.save(out)
    info.update({"meta": {k: v for k, v in m.meta.items() if k != "hsl_names"}, "seconds": round(time.perf_counter() - t0), "model": str(p), "model_sha256": sha(p)})
    (out / "train_info.json").write_text(json.dumps(info, indent=1, default=str)); print(json.dumps(info, indent=1, default=str))


def cmd_mktest(a):
    import numpy as np, pandas as pd
    X = np.load(TEST_X120, mmap_mode="r"); bg = np.load(TEST_BAGS); uids = bg["uids"].astype(str)
    assert X.shape == (len(uids), 139) and len(set(uids)) == len(uids)
    old = pd.read_parquet(HSL_TEST)
    cols = [c for c in old.columns if c != "uid"]; assert len(cols) == N_HSL
    o = old.set_index("uid").loc[uids]
    new = np.asarray(X[:, :N_HSL], np.float32)
    assert np.isfinite(new).all()
    eq = (o.to_numpy(np.float32).view(np.uint32) == new.view(np.uint32)).all(1)
    man = json.loads((ROOT / ".scratch/e35/test/test_x120_manifest.json").read_text())
    assert int(eq.sum()) >= man["games_without_new_rows"], (int(eq.sum()), man["games_without_new_rows"])   # games without new rows keep their W60 values bitwise
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.Series(uids, name="uid"), pd.DataFrame(new, columns=cols)], axis=1)
    df.to_parquet(out / "hsl_test120_games.parquet", index=False)
    oldm = json.loads((HSL_TEST.parent / "features_manifest.json").read_text())
    rep = {"n_games": len(df), "rows_bitwise_equal_to_stored_w60": int(eq.sum()), "games_without_new_rows_manifest": man["games_without_new_rows"], "mean_n_moves_w60_cols": float(o["n_moves"].mean()) if "n_moves" in o else None,
           "mean_n_moves_w120_cols": float(df["n_moves"].mean()) if "n_moves" in df else None, "test_x120_sha256": sha(TEST_X120), "columns_equal_stored": cols == list(old.columns[1:])}
    (out / "features_manifest.json").write_text(json.dumps({"color_semantics": oldm["color_semantics"], "complete": True, "n_games": len(df), "source": "test_X120.npy[:, :104]", "report": rep}, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_predict(a):
    os.environ["GOGAME_COLOR_SEMANTICS"] = "actual"
    import numpy as np
    P = _import_predict()
    TEST_Q = P.load_questions(DATA / P.FILES["rank_pub"], "pub", "rank") + P.load_questions(DATA / P.FILES["rank_priv"], "priv", "rank")
    m = P.HslEnsembleSeRank.load(Path(a.model_dir), n_jobs=2)
    assert m.meta.get("decode") == "ev" and m.meta.get("method") == "ens" and m.meta.get("features") == "l0+hsl+se", m.meta
    m.set_features(Path(a.hsl)); m.set_se_features(SE_TEST)
    p = m.predict_proba(TEST_Q)
    assert p.shape == (800, 10) and np.isfinite(p).all()
    np.save(a.out, p)
    print(json.dumps({"out": a.out, "sha256": sha(a.out), "n": len(p), "meta": {k: m.meta.get(k) for k in ("model", "method", "rounds", "decode", "include_holdout", "n_train_players", "n_train_bags", "e37_mode")}}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--mode", required=True, choices=["repro60", "w120"]); t.add_argument("--out", required=True); t.add_argument("--threads", type=int, default=4)
    k = sp.add_parser("mktest"); k.add_argument("--out", required=True)
    q = sp.add_parser("predict"); q.add_argument("--model-dir", required=True); q.add_argument("--hsl", required=True); q.add_argument("--out", required=True)
    a = ap.parse_args(); {"train": cmd_train, "mktest": cmd_mktest, "predict": cmd_predict}[a.cmd](a)
