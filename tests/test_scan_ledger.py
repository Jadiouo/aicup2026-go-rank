"""scan_ledger: locked/atomic ledger, claims in fixed random order, leases, validated mark_done, reconcile; scan_merge
ingest + scan_runner queue-file glue.  Fake ledgers in tmp dirs only (never data/scan)."""
import json, multiprocessing as mp, os, signal, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KATAGO = Path(__file__).resolve().parents[1] / "scripts" / "katago"
sys.path.insert(0, str(KATAGO))
import scan_ledger as sl  # noqa: E402


def make_ledger(path, n2=40, n1=3):
    rows = [dict(shard_id="P0000", tier=1, n_games=16000, content_hash="p", status="done", platform="colab", gpu="NVIDIA L4",
                 config_hash="", note="pilot", started=np.nan, finished=np.nan, wall_sec=np.nan, games_per_sec=np.nan)]
    for i in range(1, n1 + 1):
        rows.append(dict(shard_id=f"S{i:04d}", tier=1, n_games=4000, content_hash=f"h{i}", status="done", platform="colab", gpu="NVIDIA L4 x1",
                         config_hash="c", note="", started=1.0 + i, finished=2.0 + i, wall_sec=1000.0 + i, games_per_sec=2.3))
    for i in range(n1 + 1, n1 + 1 + n2):
        rows.append(dict(shard_id=f"S{i:04d}", tier=2, n_games=4000, content_hash=f"h{i}", status="todo", platform="", gpu="",
                         config_hash="", note="", started=np.nan, finished=np.nan, wall_sec=np.nan, games_per_sec=np.nan))
    df = pd.DataFrame(rows); df.to_parquet(path, index=False); return df


@pytest.fixture
def led(tmp_path):
    p = str(tmp_path / "ledger.parquet"); make_ledger(p); return p


def _claimer(path, owner, k, q):
    got = []
    for _ in range(k):
        got += sl.claim(path, owner, 1, 3600)
    q.put((owner, got))


def test_concurrent_claims_never_duplicate(tmp_path):
    p = str(tmp_path / "ledger.parquet"); make_ledger(p, n2=1000)
    sl.ensure_schema(p)
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=_claimer, args=(p, f"o{i}", 250, q)) for i in range(4)]
    [x.start() for x in ps]
    res = [q.get(timeout=300) for _ in ps]; [x.join(60) for x in ps]
    allg = [s for _, g in res for s in g]
    assert len(allg) == 1000 and len(set(allg)) == 1000
    df = sl.read(p)
    assert (df[df.tier == 2].status == "running").all() and not df.shard_id.duplicated().any()
    assert all(len(g) == 250 for _, g in res) or sum(len(g) for _, g in res) == 1000


def test_claim_order_is_fixed_random_and_reproducible(tmp_path):
    a, b = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    make_ledger(a); make_ledger(b)
    ga, gb = sl.claim(a, "x", 40, 100), sl.claim(b, "y", 40, 100)
    assert ga == gb and ga != sorted(ga)            # same order every time, and not S-order
    assert sl.prio_order(["S3", "S1", "S2"]) == sl.prio_order(["S1", "S2", "S3"])
    assert sl.prio_order([f"S{i}" for i in range(30)], seed=1) != sl.prio_order([f"S{i}" for i in range(30)], seed=2)
    assert sorted(sl.prio_order([f"S{i}" for i in range(30)]).values()) == list(range(30))


def test_tier1_untouched_and_migrate_idempotent(led):
    before = sl.read(led)
    assert sl.ensure_schema(led) is True and sl.ensure_schema(led) is False
    sl.claim(led, "a", 5, 100); sl.release(led, "a"); sl.expire_leases(led, now=time.time() + 1e6)
    after = sl.read(led)
    t1b, t1a = before[before.tier == 1].reset_index(drop=True), after[after.tier == 1][before.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(t1b, t1a)
    assert after[after.tier == 1].owner.eq("").all() and after[after.tier == 1].prio.isna().all()
    assert os.path.exists(led + ".premigrate")


def test_lease_expiry_todo_first_then_expired(led):
    g1 = sl.claim(led, "a", 3, 10, now=1000.0)
    assert sl.claim(led, "b", 3, 10, now=1005.0) != g1            # lease still valid: b gets other (todo) shards
    g3 = sl.claim(led, "c", 100, 10, now=1006.0)                  # takes all remaining todo, nothing expired yet
    assert not set(g3) & set(g1)
    g4 = sl.claim(led, "d", 2, 10, now=2000.0)                    # everything expired -> taken in prio order
    assert len(g4) == 2
    df = sl.read(led).set_index("shard_id")
    assert (df.loc[g4, "owner"] == "d").all() and (df.loc[g4, "attempts"] == 2).all()
    kept = sl.renew(led, "a", g1, 10, now=2001.0)
    assert set(kept) == set(g1) - set(g4)                          # the gap is what d took


def test_todo_before_expired_running(led):
    e = sl.claim(led, "a", 1, 10, now=0.0)
    nxt = sl.claim(led, "b", 1, 10, now=1000.0)
    assert nxt != e                                                # todo shard preferred over the expired one


def test_attempts_exhausted_becomes_failed(led):
    first = None
    for k in range(3):
        g = sl.claim(led, f"o{k}", 40, 1, now=100.0 * (k + 1))
        first = first or g[0]
    g = sl.claim(led, "o9", 1, 1, now=1000.0)
    df = sl.read(led).set_index("shard_id")
    assert df.loc[first, "status"] == "failed" and "attempts" in df.loc[first, "note"] and g == []


def test_release_only_own_running_and_gives_attempt_back(led):
    a = sl.claim(led, "a", 3, 100); b = sl.claim(led, "b", 3, 100)
    assert sorted(sl.release(led, "a")) == sorted(a)
    df = sl.read(led).set_index("shard_id")
    assert (df.loc[a, "status"] == "todo").all() and (df.loc[a, "attempts"] == 0).all()
    assert (df.loc[b, "owner"] == "b").all() and (df.loc[["S0001", "P0000"], "status"] == "done").all()
    assert sl.release(led, "a") == []


def test_mark_done_requires_validation(led):
    meta = dict(platform="colab", gpu="X", n_gpus_used=1, config_hash="c", t_start=1.0, t_end=2.0, wall_sec=1.0, games_per_sec_wall=2.0)
    sid = sl.claim(led, "a", 1, 100)[0]
    with pytest.raises(sl.Unvalidated): sl.mark_done(led, sid, meta, validation_errors=None)
    with pytest.raises(sl.Unvalidated): sl.mark_done(led, sid, meta, validation_errors=["NaN"])
    assert sl.read(led).set_index("shard_id").at[sid, "status"] == "running"
    calls = []
    assert sl.mark_done(led, sid, meta, validation_errors=[], install=lambda: calls.append(1), owner="a") is True
    assert sl.mark_done(led, sid, meta, validation_errors=[], install=lambda: calls.append(2)) is False
    assert calls == [1]                                            # first valid result wins; no second install
    assert sl.mark_failed(led, sid, "late bad copy") is False      # done is never downgraded
    r = sl.read(led).set_index("shard_id").loc[sid]
    assert r.status == "done" and r.owner == "" and r.gpu == "X x1"


def test_install_failure_leaves_ledger_unchanged(led):
    meta = dict(platform="c", gpu="X", n_gpus_used=1, config_hash="c", t_start=1.0, t_end=2.0, wall_sec=1.0, games_per_sec_wall=2.0)
    sid = sl.claim(led, "a", 1, 100)[0]
    def boom(): raise OSError("disk full")
    with pytest.raises(OSError): sl.mark_done(led, sid, meta, validation_errors=[], install=boom)
    assert sl.read(led).set_index("shard_id").at[sid, "status"] == "running"


def _hold_lock_forever(path, ev):
    with sl.locked(path):
        # also leave a half-written tmp behind, like a writer killed mid-write
        open(path + ".tmp.99999", "wb").write(b"garbage")
        ev.set(); time.sleep(600)


def test_kill_minus_9_holder_leaves_consistent_ledger(led):
    ctx = mp.get_context("spawn"); ev = ctx.Event()
    p = ctx.Process(target=_hold_lock_forever, args=(led, ev)); p.start()
    assert ev.wait(30)
    with pytest.raises(sl.LockTimeout):
        with sl.locked(led, timeout=0.3): pass
    os.kill(p.pid, signal.SIGKILL); p.join(10)
    got = sl.claim(led, "a", 2, 100)                               # lock freed by the kernel; stale tmp swept
    assert len(got) == 2 and not list(Path(led).parent.glob("ledger.parquet.tmp.*"))
    df = sl.read(led); assert not df.shard_id.duplicated().any()


def _claim_forever(path, ev):
    ev.set()
    while True: sl.claim(path, "k", 1, 3600)


def test_kill_minus_9_during_claims(tmp_path):
    p = str(tmp_path / "ledger.parquet"); make_ledger(p, n2=2000); sl.ensure_schema(p)
    ctx = mp.get_context("spawn"); ev = ctx.Event()
    pr = ctx.Process(target=_claim_forever, args=(p, ev)); pr.start(); ev.wait(30); time.sleep(1.0)
    os.kill(pr.pid, signal.SIGKILL); pr.join(10)
    df = sl.read(p)                                                # readable, consistent
    run = df[df.status == "running"]
    assert len(df) == 2000 + 4 and not df.shard_id.duplicated().any()
    assert (run.owner == "k").all() and (run.attempts == 1).all()
    assert len(sl.claim(p, "z", 3, 10)) == 3                       # and still usable
    ev = [json.loads(l) for l in open(sl.events_path(p))]
    claimed = [s for e in ev if e["op"] == "claim" and e["owner"] == "k" for s in e["shards"]]
    assert len(claimed) == len(set(claimed)) == len(run)           # event log agrees with the ledger


def test_reconcile_records_valid_unrecorded_results(led, tmp_path):
    res = tmp_path / "results"; res.mkdir()
    meta = dict(platform="colab", gpu="X", n_gpus_used=1, config_hash="c", t_start=1.0, t_end=2.0, wall_sec=1.0, games_per_sec_wall=2.0)
    s_ok, s_bad, s_none = sl.claim(led, "a", 3, 100)
    for s in (s_ok, s_bad): (res / f"{s}_meta.json").write_text(json.dumps(meta))
    r = sl.reconcile(led, str(res), lambda sid, row, d: [] if sid == s_ok else ["bad"])
    assert r["repaired"] == [s_ok] and list(r["invalid"]) == [s_bad]
    df = sl.read(led).set_index("shard_id")
    assert df.at[s_ok, "status"] == "done" and df.at[s_bad, "status"] == "running" and df.at[s_none, "status"] == "running"
    assert sl.reconcile(led, str(res), lambda *a: [])["repaired"] == [s_bad]
    assert sl.reconcile(led, str(res), lambda *a: [])["repaired"] == []


def test_peek_does_not_write(led):
    before = Path(led).read_bytes()
    assert len(sl.peek(led, 4)) == 4 and Path(led).read_bytes() == before


# ------------------------------------------------------------------ scan_merge.cmd_ingest on the locked ledger
def _merge_env(tmp_path, monkeypatch):
    import scan_merge as sm
    root = tmp_path / "scan"; root.mkdir()
    make_ledger(str(root / "ledger.parquet"))
    sid = "S0004"
    pd.DataFrame({"uid": ["u1", "u2"], "shard_id": sid}).to_parquet(root / "manifest.parquet")
    monkeypatch.setattr(sm, "SCAN", str(root)); monkeypatch.setattr(sm, "LEDGER", str(root / "ledger.parquet"))
    monkeypatch.setattr(sm, "RES", str(root / "results"))
    inbox = tmp_path / "inbox"; inbox.mkdir()
    meta = dict(platform="colab", gpu="X", n_gpus_used=1, config_hash="c", t_start=1.0, t_end=2.0, wall_sec=1.0, games_per_sec_wall=2.0)
    (inbox / f"{sid}_meta.json").write_text(json.dumps(meta))
    for nm in ("moves", "games"): (inbox / f"{sid}_{nm}.parquet").write_bytes(b"first")
    return sm, root, inbox, sid


def test_scan_merge_ingest_first_valid_wins_and_never_downgrades(tmp_path, monkeypatch, capsys):
    sm, root, inbox, sid = _merge_env(tmp_path, monkeypatch)
    monkeypatch.setattr(sm, "validate", lambda *a: [])
    sm.cmd_ingest(str(inbox), "cfg")
    assert "ACCEPT" in capsys.readouterr().out and (root / "results" / f"{sid}_games.parquet").read_bytes() == b"first"
    (inbox / f"{sid}_games.parquet").write_bytes(b"second")
    sm.cmd_ingest(str(inbox), "cfg")
    assert "already done" in capsys.readouterr().out and (root / "results" / f"{sid}_games.parquet").read_bytes() == b"first"
    monkeypatch.setattr(sm, "validate", lambda *a: ["NaN"])
    sm.cmd_ingest(str(inbox), "cfg")
    assert sl.read(str(root / "ledger.parquet")).set_index("shard_id").at[sid, "status"] == "done"


def test_scan_merge_ingest_rejects_and_reconcile(tmp_path, monkeypatch, capsys):
    sm, root, inbox, sid = _merge_env(tmp_path, monkeypatch)
    monkeypatch.setattr(sm, "validate", lambda *a: ["NaN"])
    sm.cmd_ingest(str(inbox), "cfg")
    row = sl.read(sm.LEDGER).set_index("shard_id").loc[sid]
    assert row.status == "failed" and row.note == "NaN"
    assert not list((root / "results").glob("S*"))
    # a complete result set sitting in results/ with the ledger not yet written (crash between copy and record)
    res = root / "results"
    for f in inbox.iterdir(): (res / f.name).write_bytes(f.read_bytes())
    monkeypatch.setattr(sm, "validate", lambda *a: [])
    r = sm.cmd_reconcile("cfg")
    assert r["repaired"] == [sid] and sl.read(sm.LEDGER).set_index("shard_id").at[sid, "status"] == "done"
    assert sm.cmd_reconcile("cfg")["repaired"] == []


# ------------------------------------------------------------------ scan_runner --queue-file
def test_shard_stream_old_behaviour_and_queue(tmp_path):
    from scan_runner import shard_stream
    assert list(shard_stream(["S1", "S2"])) == ["S1", "S2"]
    assert list(shard_stream(["S1"], None)) == ["S1"]
    f = tmp_path / "q.txt"; f.write_text("S2\n"); t = {"now": 0.0}
    def sleep(_):                                   # the "controller" appends while the runner waits
        t["now"] += 5
        if t["now"] == 10: f.write_text(f.read_text() + "S3\n")
        if t["now"] == 20: f.write_text(f.read_text() + "END\n")
    assert list(shard_stream(["S1"], str(f), 10, 5, sleep=sleep, clock=lambda: t["now"])) == ["S1", "S2", "S3"]


def test_shard_stream_idle_timeout(tmp_path):
    from scan_runner import shard_stream
    f = tmp_path / "q.txt"; f.write_text("")
    t = {"now": 0.0}
    def sleep(_): t["now"] += 60
    assert list(shard_stream([], str(f), 10, 60, sleep=sleep, clock=lambda: t["now"])) == []
    assert t["now"] > 600


def test_hold_takes_all_todo_with_long_lease_and_spares_others(led):
    a = sl.claim(led, "colab-a", 2, 3600)            # another owner's running rows must stay theirs
    mine = sl.claim(led, "school", 1, 60)            # school's own short lease gets extended
    until = time.time() + 20 * 86400
    held = sl.hold(led, "school", until, platform="school")
    df = sl.read(led); t2 = df[df.tier == 2]
    assert set(held) == set(t2[t2.owner == "school"].shard_id) and len(held) == 38
    assert set(mine) <= set(held) and not set(a) & set(held)
    assert (t2[t2.owner == "school"].lease_until == until).all()
    assert (t2[t2.owner == "colab-a"].lease_until < until).all()
    assert (t2.status == "todo").sum() == 0
    assert sl.claim(led, "other", 5, 3600) == []     # nothing left to take while the lease is valid
    assert df[df.tier == 1].status.eq("done").all()
    assert held == sorted(held, key=lambda s: float(t2.set_index("shard_id").at[s, "prio"]))
    assert sl.hold(led, "school", until, platform="school") == held     # idempotent
