"""scripts/predict_hsl.py (test-day HSL extraction wiring) and predict.py HslLgbmRank, on a tiny synthetic contest.

The KataGo engine is replaced by the runner's own SCAN_FAKE_ENGINE; everything else (runner, scan_merge.validate, the
controller's collect path) is the real code.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helpers import make_contest, make_corpus, write_contest


@pytest.fixture(scope="module")
def mods(project_root):
    sys.path.insert(0, str(project_root / "scripts"))
    sys.path.insert(0, str(project_root / "scripts" / "katago"))
    import predict
    import predict_hsl
    return predict, predict_hsl


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(3, players_per_rank=3, games_per_player=40)


@pytest.fixture(scope="module")
def contest_dir(corpus, tmp_path_factory):
    d = tmp_path_factory.mktemp("contest")
    write_contest(make_contest(corpus, n_per_n=1, pool_size=6, lib_games=5, with_prefix=False, seed=3), d)
    return d


@pytest.fixture(scope="module")
def prepared(mods, contest_dir, tmp_path_factory):
    _, ph = mods
    if not (ph.ROOT / "data" / "scan" / "shards.parquet").exists() or not (ph.ROOT / "data" / "cache" / "katago" / "pilot_hsl_games.parquet").exists():
        pytest.skip("needs the HSL scan tables built from the official training data (data/scan, data/cache/katago)")
    root = tmp_path_factory.mktemp("test_scan")
    info = ph.prepare(contest_dir, root, shard_target=60, log=lambda *_: None)
    return root, info


def test_prepare_formats_match_scan(mods, prepared):
    _, ph = mods
    import scan_manifest
    root, info = prepared
    assert info["n_questions"] == 8 and info["n_games"] == 2 * (5 + 10 + 15 + 20)
    sh = pd.read_parquet(root / "shards.parquet")
    ref = pd.read_parquet(ph.ROOT / "data" / "scan" / "shards.parquet")
    assert list(sh.columns) == list(ref.columns)
    assert sh.shard_id.str.fullmatch(r"S9\d{3}").all()
    ref_sgf = pd.read_parquet(ph.ROOT / "data" / "scan" / "sgf" / "S0001.parquet")
    uids = []
    for r in sh.itertuples():
        d = pd.read_parquet(root / "sgf" / f"{r.shard_id}.parquet")
        assert list(d.columns) == list(ref_sgf.columns) and (d.dtypes == ref_sgf.dtypes).all()
        assert scan_manifest.content_hash(d.uid, d.color, d.sgf) == r.content_hash and len(d) == r.n_games
        uids += d.uid.tolist()
    assert len(set(uids)) == len(uids) and uids[0].startswith("pub_q_") and "#" in uids[0]
    led = pd.read_parquet(root / "ledger.parquet")
    assert (led.status == "todo").all() and (led.tier == 1).all()
    assert info["config_hash"] == ph.current_config_hash()


def test_prepare_refuses_data_scan_and_overwrite(mods, prepared, contest_dir, tmp_path):
    _, ph = mods
    with pytest.raises(ValueError):
        ph.prepare(contest_dir, ph.ROOT / "data" / "scan", log=lambda *_: None)
    root, _ = prepared
    led = pd.read_parquet(root / "ledger.parquet")
    led.loc[0, "status"] = "done"
    led.to_parquet(tmp_path / "x.parquet")
    (tmp_path / "ledger.parquet").write_bytes((tmp_path / "x.parquet").read_bytes())
    with pytest.raises(RuntimeError):
        ph.prepare(contest_dir, tmp_path, log=lambda *_: None)


def _fake_run(ph, root, out):
    """Real scan_runner with the fake engine on every prepared shard, then make the shards look like real-engine output."""
    env = {**os.environ, "SCAN_FAKE_ENGINE": "1"}
    sh = pd.read_parquet(root / "shards.parquet")
    subprocess.run([sys.executable, str(ph.KATAGO / "scan_runner.py"), "--shards", ",".join(sh.shard_id), "--sgf-dir", str(root / "sgf"),
                    "--shards-table", str(root / "shards.parquet"), "--katago-dir", str(out / "k"), "--out", str(out / "o"),
                    "--work", str(out / "w"), "--platform", "colab", "--gpus", ""], check=True, env=env, capture_output=True)
    for sid in sh.shard_id:
        meta = json.loads((out / "o" / f"{sid}_meta.json").read_text())
        assert meta["fake_engine"] is True
        meta["fake_engine"] = False                  # only so that the validator accepts the rehearsal; hashes are the pinned ones
        md = json.dumps(meta)
        (out / "o" / f"{sid}_meta.json").write_text(md)
        for nm in ("moves", "games"):
            f = out / "o" / f"{sid}_{nm}.parquet"
            tb = pq.read_table(f).replace_schema_metadata({b"scan_meta": md.encode()})
            pq.write_table(tb, f)


@pytest.fixture(scope="module")
def accepted(mods, prepared, tmp_path_factory):
    _, ph = mods
    root, _ = prepared
    out = tmp_path_factory.mktemp("fake_out")
    _fake_run(ph, root, out)
    return root, out / "o"


def test_controller_collect_ingests_and_features(mods, prepared, accepted, tmp_path):
    """controller.collect (original code) with a fake VM: download -> sha check -> our ingest -> ledger done."""
    import shutil
    _, ph = mods
    root, outdir = accepted
    work = tmp_path / "root"
    shutil.copytree(root, work)
    m = ph.controller_for(work)
    assert m.LEDGER == str(work / "ledger.parquet") and "test_scan" not in m.LEDGER.replace(str(work), "")
    assert str(m.D).startswith(str(work))

    def fake_rexec(sess, code, timeout=90):
        sids = json.loads(code.split("for sid in ")[1].split(":")[0].replace("'", '"'))
        return 0, "SHA:" + json.dumps({f"{s}_{nm}": ph.sha256_file(outdir / f"{s}_{nm}.{'json' if nm == 'meta' else 'parquet'}")
                                       for s in sids for nm in ("moves", "games", "meta")})

    def fake_colab(*args, timeout=120):
        if args[0] == "download":
            shutil.copy(outdir / os.path.basename(args[3]), args[4])
        return 0, ""
    m.rexec, m.colab = fake_rexec, fake_colab
    metas = sorted(p.name.split("_")[0] for p in outdir.glob("S*_meta.json"))
    st = {}
    got = m.collect("s", st, metas)
    assert got == metas and not st.get("halt")
    led = pd.read_parquet(work / "ledger.parquet")
    assert (led.status == "done").all(), led.note.tolist()
    assert sorted(p.name for p in (work / "results").glob("*_meta.json")) == [f"{s}_meta.json" for s in metas]
    # features: per-game HSL, same columns as the E06 file, complete
    info = ph.build_features(work, train_scan=None, log=lambda *_: None)
    assert info["complete"] and info["n_games_without_hsl"] == 0 and not info["questions_without_any_hsl"]
    df = pd.read_parquet(work / "hsl_test_games.parquet")
    assert list(df.columns) == ph._ref_columns() and df.uid.is_monotonic_increasing and (df.dtypes[1:] == np.float32).all()
    assert len(df) == info["n_games_manifest"]


def test_ingest_rejects_wrong_config(mods, prepared, accepted, tmp_path):
    import shutil
    _, ph = mods
    root, outdir = accepted
    work, bad = tmp_path / "root", tmp_path / "bad"
    shutil.copytree(root, work)
    shutil.copytree(outdir, bad)
    sid = sorted(p.name.split("_")[0] for p in bad.glob("S*_meta.json"))[0]
    meta = json.loads((bad / f"{sid}_meta.json").read_text())
    meta["config_hash"] = "0" * 64
    md = json.dumps(meta)
    (bad / f"{sid}_meta.json").write_text(md)
    for nm in ("moves", "games"):
        f = bad / f"{sid}_{nm}.parquet"
        pq.write_table(pq.read_table(f).replace_schema_metadata({b"scan_meta": md.encode()}), f)
    ph.cmd_ingest(work, bad)
    led = pd.read_parquet(work / "ledger.parquet").set_index("shard_id")
    assert led.loc[sid, "status"] == "failed" and "config_hash" in led.loc[sid, "note"]
    with pytest.raises(RuntimeError):           # incomplete scan must not silently produce features
        ph.build_features(work, train_scan=None, log=lambda *_: None)


def test_features_catch_move_count_mismatch(mods, prepared, accepted, tmp_path):
    import shutil
    _, ph = mods
    root, outdir = accepted
    work = tmp_path / "root"
    shutil.copytree(root, work)
    ph.cmd_ingest(work, outdir)
    g = pd.read_parquet(work / "results" / "S9001_games.parquet")
    g.loc[0, "n_moves"] = g.loc[0, "n_moves"] - 1
    f = work / "results" / "S9001_games.parquet"
    pq.write_table(pa.Table.from_pandas(g, preserve_index=False).replace_schema_metadata(pq.read_schema(f).metadata), f)
    with pytest.raises(RuntimeError, match="n_moves"):
        ph.build_features(work, train_scan=None, log=lambda *_: None)


# ------------------------------------------------------------------------------------------------ hsl_lgbm
@pytest.fixture(scope="module")
def hsl_model(mods, corpus):
    pr, _ = mods
    from gogame.constants import RANK_TO_ID
    meta = pd.DataFrame({"uid": [f"{r}_{g}" for r, g in zip(corpus["rank"], corpus.game_id)], "player_id": corpus.player_id,
                         "rank_id": [RANK_TO_ID[r] for r in corpus["rank"]], "color": (corpus.color == "W").astype(int)})
    rng = np.random.default_rng(0)
    names = [f"hsl_{i}" for i in range(6)]
    hsl = pd.DataFrame(rng.normal(size=(len(meta), 6)).astype(np.float32), columns=names)
    hsl.insert(0, "uid", meta.uid.to_numpy())
    hsl.iloc[:, 1:] += meta.rank_id.to_numpy()[:, None] * 0.5          # learnable signal
    return pr.HslLgbmRank.fit(hsl, meta, rounds=5, threads=1), names


def test_hsl_lgbm_missing_model_message(mods, tmp_path):
    pr, _ = mods
    with pytest.raises(FileNotFoundError, match="E06"):
        pr.RANK_MODELS["hsl_lgbm"].load(tmp_path, 1)


def test_hsl_lgbm_predicts_from_features(mods, hsl_model, contest_dir, tmp_path):
    pr, _ = mods
    m, names = hsl_model
    qs = pr.load_questions(contest_dir / "rank_prediction_test_public.csv", "pub", "rank")
    rng = np.random.default_rng(1)
    rows = [(g.uid, *rng.normal(size=6) + 1.0) for q in qs for g in q.games]
    path = tmp_path / "hsl_test_games.parquet"
    pd.DataFrame(rows, columns=["uid", *names]).astype({n: np.float32 for n in names}).to_parquet(path, index=False)
    m.save(tmp_path)
    m2 = pr.HslLgbmRank.load(tmp_path, 1)
    m2.set_features(path)
    p = m2.predict_proba(qs)
    assert p.shape == (len(qs), 10) and np.allclose(p.sum(1), 1, atol=1e-4)
    assert m2.predict(qs).shape == (len(qs),)
    # games without HSL are dropped from their bag; a question with none at all is a clear error
    df = pd.read_parquet(path)
    m2.max_drop_frac = 1.0                      # drop-fraction guard has its own test below
    df[~df.uid.str.endswith("#1")].to_parquet(path, index=False)
    assert m2.predict_proba(qs).shape == (len(qs), 10)
    from gogame.errors import DataFormatError
    df[~df.uid.str.startswith(qs[0].qid)].to_parquet(path, index=False)
    with pytest.raises(DataFormatError, match="no HSL game"):
        m2.predict_proba(qs)
    m2.set_features(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError, match="predict_hsl.py"):
        m2.predict_proba(qs)


def test_hsl_lgbm_bag_matches_e06_construction(mods, project_root):
    """hsl_store/hsl_spec == pipeline.Ctx.hs / sp_h : identical aggregates for the same bag."""
    pr, _ = mods
    from gogame.aggregate import L0Store, aggregate_many, make_spec
    from gogame.questions import Bag
    rng = np.random.default_rng(2)
    names = tuple(f"c{i}" for i in range(5))
    df = pd.DataFrame(rng.normal(size=(30, 5)).astype(np.float32), columns=names)
    df.insert(0, "uid", [f"u{i}" for i in range(30)])
    bag = Bag("q", "", -1, 1, 7, tuple(df.uid[:7]))
    ref = aggregate_many([bag], L0Store(df["uid"].tolist(), df[list(names)].to_numpy(np.float32), None, names),
                         make_spec(names, np.zeros(0, np.int32)))
    assert np.array_equal(ref, aggregate_many([bag], pr.hsl_store(df), pr.hsl_spec(pr.hsl_store(df).names)))


def test_train_hsl_refuses_until_e06_done(mods, tmp_path):
    pr, _ = mods
    with pytest.raises(FileNotFoundError, match="features"):
        pr.train_hsl_rank(tmp_path, False, 1, None, e06_cache=tmp_path / "none", selection=tmp_path / "sel.json")
    (tmp_path / "features_manifest.json").write_text(json.dumps({"complete": False, "n_present": 3, "n_shards_tier1": 48}))
    with pytest.raises(RuntimeError, match="partial"):
        pr.train_hsl_rank(tmp_path, False, 1, None, e06_cache=tmp_path, selection=tmp_path / "sel.json")
    (tmp_path / "features_manifest.json").write_text(json.dumps({"complete": True, "sha256": "x"}))
    with pytest.raises(FileNotFoundError, match="no Rank method"):
        pr.train_hsl_rank(tmp_path, False, 1, None, e06_cache=tmp_path, selection=tmp_path / "sel.json")
    (tmp_path / "sel.json").write_text(json.dumps({"main": "M04", "valid_for_confirm": True, "features_sha256": "x"}))
    (tmp_path / "sel.json").write_text(json.dumps({"main": "M09", "valid_for_confirm": True, "features_sha256": "x"}))
    with pytest.raises(NotImplementedError, match="supports"):
        pr.train_hsl_rank(tmp_path, False, 1, None, e06_cache=tmp_path, selection=tmp_path / "sel.json")


def test_hsl_coral_fit_save_predict(mods, hsl_model, contest_dir, tmp_path, corpus):
    """E06 M04 through the same class: fit -> save -> load -> predict; probabilities are a valid distribution."""
    pr, _ = mods
    from gogame.constants import RANK_TO_ID
    meta = pd.DataFrame({"uid": [f"{r}_{g}" for r, g in zip(corpus["rank"], corpus.game_id)], "player_id": corpus.player_id,
                         "rank_id": [RANK_TO_ID[r] for r in corpus["rank"]], "color": (corpus.color == "W").astype(int)})
    rng = np.random.default_rng(0)
    names = [f"hsl_{i}" for i in range(6)]
    hsl = pd.DataFrame(rng.normal(size=(len(meta), 6)).astype(np.float32) + meta.rank_id.to_numpy()[:, None] * 0.5, columns=names)
    hsl.insert(0, "uid", meta.uid.to_numpy())
    m = pr.HslLgbmRank.fit(hsl, meta, None, method="coral", meta_extra={"decode": "ev"})
    assert m.meta["method"] == "coral"
    m.save(tmp_path)
    m2 = pr.HslLgbmRank.load(tmp_path, 1)
    qs = pr.load_questions(contest_dir / "rank_prediction_test_public.csv", "pub", "rank")
    rows = [(g.uid, *(rng.normal(size=6) + 1.0)) for q in qs for g in q.games]
    path = tmp_path / "f.parquet"
    pd.DataFrame(rows, columns=["uid", *names]).astype({n: np.float32 for n in names}).to_parquet(path, index=False)
    m2.set_features(path)
    p = m2.predict_proba(qs)
    assert p.shape == (len(qs), 10) and np.allclose(p.sum(1), 1, atol=1e-4)
    assert m2.predict(qs).shape == (len(qs),)
    with pytest.raises(ValueError):
        pr.HslLgbmRank.fit(hsl, meta, None, method="coral", use_l0=True)


# ------------------------------------------------------------------------------------------------ memory route / overlap decision
def test_memory_rank_identifies_seen_players(mods, corpus, tmp_path):
    """Fit on the corpus, then ask questions built from the same players' games: nearest centroid -> that player's rank."""
    pr, _ = mods
    from gogame.constants import RANK_TO_ID
    from gogame.io import GameRecord
    recs = [GameRecord(f"{r}_{g}", p, RANK_TO_ID[r], 0 if c == "B" else 1, s, "train")
            for p, g, r, c, s in zip(corpus.player_id, corpus.game_id, corpus["rank"], corpus.color, corpus.sgf_content)]
    store = pr.l0_store(recs, 1)
    meta = pd.DataFrame({"uid": [r.uid for r in recs], "player_id": [r.player_id for r in recs],
                         "rank_id": [r.rank_id for r in recs], "color": [r.color for r in recs]})
    m = pr.MemoryRank.fit(store, meta, bags_per_player_color=4, min_df=0.05)
    m.save(tmp_path)
    m = pr.MemoryRank.load(tmp_path, 1)
    qs, truth = [], []
    for pid, d in list(corpus.groupby("player_id"))[:9]:
        b = d[d.color == "B"].head(10)
        games = [GameRecord(f"{pid}#{i}", "", None, 0, s, "test_pub") for i, s in enumerate(b.sgf_content)]
        qs.append(pr.Question(f"pub_{pid}", "pub", 0, games))
        truth.append(RANK_TO_ID[d["rank"].iloc[0]])
    pred = m.predict(qs)
    assert pred.shape == (len(qs),) and (pred == np.array(truth)).mean() > 0.6
    with pytest.raises(FileNotFoundError, match="overlap"):
        pr.MemoryRank.load(tmp_path / "none", 1)


def test_overlap_decision_or_rule(mods):
    pr, _ = mods
    ov = lambda n, sc, p30: {"train_overlap": {"n_games": n, "exact_same_color": sc, "prefix_match": {"30": p30}}}   # noqa: E731
    d = pr.overlap_decision({"a": ov(1000, 0, 0), "b": ov(1000, 0, 5)})
    assert not d["overlap"] and d["label"] == "否" and set(d["not_evaluated"]) == {"nn_p99_frac", "library_p99_frac"}
    d = pr.overlap_decision({"a": ov(1000, 0, 0), "b": ov(1000, 0, 30)})        # 1.5% share of 30-ply prefixes
    assert d["overlap"] and d["triggered"]["prefix30_frac"]
    d = pr.overlap_decision({"a": ov(100, 0, 0), "shift": {"bag": {"p99_frac": 0.2}, "library": {"p99_frac": 0.01}}})
    assert d["overlap"] and d["triggered"]["nn_p99_frac"] and not d["triggered"]["library_p99_frac"]


def test_hsl_lgbm_drop_fraction_threshold(mods, hsl_model, contest_dir, tmp_path, capsys):
    pr, _ = mods
    from gogame.errors import DataFormatError
    m, names = hsl_model
    qs = pr.load_questions(contest_dir / "rank_prediction_test_public.csv", "pub", "rank")
    uids = [g.uid for q in qs for g in q.games]
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(len(uids), 6)).astype(np.float32), columns=names)
    df.insert(0, "uid", uids)
    path = tmp_path / "hsl_drop.parquet"
    m.set_features(path)
    assert m.max_drop_frac == 0.01
    # well over 1% dropped -> error naming count and share
    df[~df.uid.str.endswith("#1")].to_parquet(path, index=False)
    with pytest.raises(DataFormatError, match=r"have no HSL features"):
        m.predict_proba(qs)
    # raising the threshold turns it back into a warning
    m.max_drop_frac = 1.0
    assert m.predict_proba(qs).shape == (len(qs), 10)
    assert "without HSL features dropped" in capsys.readouterr().err
    # nothing dropped -> silent
    df.to_parquet(path, index=False)
    m.max_drop_frac = 0.01
    m.predict_proba(qs)
    assert "dropped" not in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ --rank-config (E10)
def _coral_inputs(corpus):
    from gogame.constants import RANK_TO_ID
    meta = pd.DataFrame({"uid": [f"{r}_{g}" for r, g in zip(corpus["rank"], corpus.game_id)], "player_id": corpus.player_id,
                         "rank_id": [RANK_TO_ID[r] for r in corpus["rank"]], "color": (corpus.color == "W").astype(int)})
    rng = np.random.default_rng(0)
    hsl = pd.DataFrame(rng.normal(size=(len(meta), 6)).astype(np.float32) + meta.rank_id.to_numpy()[:, None] * 0.5,
                       columns=[f"hsl_{i}" for i in range(6)])
    hsl.insert(0, "uid", meta.uid.to_numpy())
    return hsl, meta


def test_rank_config_default_is_e06_behaviour(mods, corpus):
    """fit() without the new kwargs == fit() with the E06 values spelled out (default path unchanged)."""
    pr, _ = mods
    hsl, meta = _coral_inputs(corpus)
    a = pr.HslLgbmRank.fit(hsl, meta, None, method="coral")
    b = pr.HslLgbmRank.fit(hsl, meta, None, method="coral", bags_per_pc=8, n_range=(5, 15), seed_ns="e06")
    assert np.array_equal(a.bundle.slope, b.bundle.slope) and a.meta["n_train_bags"] == b.meta["n_train_bags"]
    assert a.meta["bags_per_pc"] == 8 and a.meta["n_range"] == [5, 15]
    c = pr.HslLgbmRank.fit(hsl, meta, None, method="coral", bags_per_pc=3, n_range=(5, 20), seed_ns="e10")
    assert c.meta["bags_per_pc"] == 3 and c.meta["n_range"] == [5, 20] and c.meta["n_train_bags"] != a.meta["n_train_bags"]


def test_rank_config_cli_default_and_refusals(mods, tmp_path, monkeypatch):
    pr, _ = mods
    seen = {}
    monkeypatch.setattr(pr, "train_models", lambda *a, **k: seen.setdefault("args", a))
    pr.main(["train", "--task", "rank", "--rank-model", "hsl_lgbm", "--model-dir", str(tmp_path)])
    assert seen["args"][-1] is None                                    # default: no rank_config -> E06 path
    seen.clear()
    pr.main(["train", "--task", "rank", "--rank-model", "hsl_lgbm", "--model-dir", str(tmp_path), "--rank-config", "x.json"])
    assert str(seen["args"][-1]) == "x.json"
    cfg = tmp_path / "sel.json"
    cfg.write_text(json.dumps({"candidate": "C", "main": "M04"}))
    with pytest.raises(RuntimeError, match="omit --rank-config"):
        pr.train_hsl_rank(tmp_path, False, 1, None, rank_config=cfg)
    cache = tmp_path / "cache"
    cache.mkdir()
    cfg.write_text(json.dumps({"candidate": "A", "main": "M04", "cache": str(cache), "bags_per_pc": 8, "n_range": [5, 15],
                               "features_sha256": "x", "valid_for_confirm": True}))
    with pytest.raises(FileNotFoundError, match="features_manifest"):
        pr.train_hsl_rank(tmp_path, False, 1, None, rank_config=cfg)
    (cache / "features_manifest.json").write_text(json.dumps({"complete": False, "n_present": 3, "n_shards_total": 248}))
    with pytest.raises(RuntimeError, match="partial"):
        pr.train_hsl_rank(tmp_path, False, 1, None, rank_config=cfg)
    (cache / "features_manifest.json").write_text(json.dumps({"complete": True, "sha256": "y"}))
    with pytest.raises(RuntimeError, match="features changed"):
        pr.train_hsl_rank(tmp_path, False, 1, None, rank_config=cfg)


# ------------------------------------------------------------------------------------------------ E10 candidate D (M03 + M04, EV)
def _ens_inputs(corpus, pr):
    from gogame.constants import RANK_TO_ID
    from gogame.io import GameRecord
    recs = [GameRecord(f"{r}_{g}", p, RANK_TO_ID[r], 0 if c == "B" else 1, s, "train")
            for p, g, r, c, s in zip(corpus.player_id, corpus.game_id, corpus["rank"], corpus.color, corpus.sgf_content)]
    store = pr.l0_store(recs, 1)
    meta = pd.DataFrame({"uid": [r.uid for r in recs], "player_id": [r.player_id for r in recs],
                         "rank_id": [r.rank_id for r in recs], "color": [r.color for r in recs]})
    rng = np.random.default_rng(0)
    hsl = pd.DataFrame(rng.normal(size=(len(meta), 6)).astype(np.float32) + meta.rank_id.to_numpy()[:, None] * 0.5,
                       columns=[f"hsl_{i}" for i in range(6)])
    hsl.insert(0, "uid", meta.uid.to_numpy())
    return store, meta, hsl


def test_hsl_ensemble_fit_save_predict_is_half_half_ev(mods, corpus, contest_dir, tmp_path):
    from gogame.models import decide_ev
    pr, _ = mods
    store, meta, hsl = _ens_inputs(corpus, pr)
    with pytest.raises(Exception, match="L0"):
        pr.HslLgbmRank.fit(hsl, meta, 4, 1, method="ens")                      # ens needs the L0 store
    m = pr.HslLgbmRank.fit(hsl, meta, 4, 1, method="ens", l0=store, min_df=0.05, bags_per_pc=2, n_range=(5, 10), seed_ns="e10")
    assert isinstance(m, pr.HslEnsembleRank) and m.meta["method"] == "ens" and m.meta["decode"] == "ev"
    assert m.meta["bags_per_pc"] == 2 and m.meta["features"] == "l0+hsl" and m.bundle.w_m03 == 0.5
    m.save(tmp_path)
    m2 = pr.RANK_MODELS["hsl_lgbm"].load(tmp_path, 1)                          # same registry entry / file name as the plain model
    assert isinstance(m2, pr.HslEnsembleRank) and m2.spec_l0 is not None
    qs = pr.load_questions(contest_dir / "rank_prediction_test_public.csv", "pub", "rank")
    rng = np.random.default_rng(1)
    rows = [(g.uid, *rng.normal(size=6) + 1.0) for q in qs for g in q.games]
    path = tmp_path / "f.parquet"
    pd.DataFrame(rows, columns=["uid", *[f"hsl_{i}" for i in range(6)]]).astype(
        {f"hsl_{i}": np.float32 for i in range(6)}).to_parquet(path, index=False)
    m2.set_features(path)
    p = m2.predict_proba(qs)
    Xh, bags, have = m2._hsl_features(qs)
    p03 = m2.bundle.m03.predict_proba(m2._m03_matrix(Xh, qs, bags, have))
    p04 = m2.bundle.m04.predict_proba(Xh)
    want = 0.5 * p03 + 0.5 * p04
    assert np.allclose(p, want / want.sum(1, keepdims=True), atol=1e-6) and np.allclose(p.sum(1), 1, atol=1e-6)
    assert np.array_equal(m2.predict(qs), decide_ev(p))                         # EV decode, not argmax
    assert not np.allclose(p03, p04)                                            # two genuinely different members


def test_rank_config_candidate_d_parsing_and_training(mods, corpus, tmp_path, monkeypatch):
    """--rank-config with candidate D: accepted; trains M03+M04 with the config's bags and the M03 rounds from the D OOF."""
    from gogame.gametable import GameTable
    from gogame.split import SPLIT_COLUMNS
    pr, _ = mods
    store, meta, hsl = _ens_inputs(corpus, pr)
    cache = tmp_path / "cache"
    cache.mkdir()
    hsl.to_parquet(cache / "hsl_games.parquet", index=False)
    (cache / "features_manifest.json").write_text(json.dumps({"complete": True, "sha256": "x"}))
    cfg = tmp_path / "sel.json"
    base = {"candidate": "D", "main": "M03+M04", "decode": "ev", "cache": str(cache), "bags_per_pc": 2, "n_range": [5, 10], "seed_ns": "e10",
            "features_sha256": "x", "valid_for_confirm": True, "partial": False, "oof_best_iterations": {"M03": [3, 5, 4, 6, 4]}}
    cfg.write_text(json.dumps(base))
    cdir, sdir = tmp_path / "gcache", tmp_path / "splits"
    cdir.mkdir(); sdir.mkdir()
    meta.assign(game_id="g").to_parquet(cdir / "games_meta.parquet")
    pl = meta.groupby("player_id")["rank_id"].first().reset_index()
    pl["fold"] = np.arange(len(pl)) % 5
    pl["is_holdout"] = False
    pl["n_black"] = 10
    pl[SPLIT_COLUMNS].to_parquet(sdir / "split.parquet")
    monkeypatch.setattr(pr, "CACHE", cdir)
    monkeypatch.setattr(pr, "SPLITS", sdir)
    monkeypatch.setattr("gogame.gametable.load_table", lambda path: GameTable("l0", store.uids, store.dense, store.names, store.sparse))
    # E06 path (no --rank-config) must not accept an ensemble selection
    with pytest.raises(NotImplementedError, match="supports"):
        pr.train_hsl_rank(tmp_path, False, 1, None, e06_cache=cache, selection=cfg)
    out = tmp_path / "models"
    out.mkdir()
    pr.train_hsl_rank(out, False, 1, None, rank_config=cfg)
    m = pr.RANK_MODELS["hsl_lgbm"].load(out, 1)
    assert isinstance(m, pr.HslEnsembleRank) and m.meta["decode"] == "ev" and m.meta["rounds"] == 4        # median of [3,5,4,6,4]
    assert m.meta["bags_per_pc"] == 2 and m.meta["n_range"] == [5, 10] and m.meta["e06_main"] == "M03+M04"
    # missing iteration list for the M03 member -> clear failure, not a silent default
    cfg.write_text(json.dumps({**base, "oof_best_iterations": {}}))
    with pytest.raises(KeyError):
        pr.train_hsl_rank(out, False, 1, None, rank_config=cfg)
