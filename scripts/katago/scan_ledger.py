"""Locked, atomic ledger access for the multi-owner tier-2 scan (the only writer used by scan_pool / scan_merge).

Every mutation is: flock(<ledger>.lock) -> read parquet -> change -> write <ledger>.tmp.<pid> -> fsync -> os.replace,
and one JSON line is appended to <ledger dir>/ledger_events.jsonl (inside the same lock).  The lock path is derived from the
ledger path that is passed in, so callers that redirect their ledger (tests, predict_hsl) get their own lock.

Extra columns (added lazily by `ensure_schema`, only by claim/renew/expire): owner, lease_until, attempts, claim_id, prio.
`prio` is the rank of a tier-2 shard in a fixed pseudo-random order (PRIO_SEED), so any prefix of work is an unbiased
subset of players.  Rows with status `done` are never modified by anything here; tier-1 rows only gain default values
in the new columns.

  python scan_ledger.py migrate|status|expire|peek N   [--ledger PATH]
"""
import argparse, contextlib, fcntl, glob, json, os, time
import numpy as np, pandas as pd

PRIO_SEED = 20261019          # named fixed seed for the tier-2 claim order
MAX_ATTEMPTS = 3
LOCK_TIMEOUT = 60.0
NEW_COLS = ("owner", "lease_until", "attempts", "claim_id", "prio")


class LedgerError(RuntimeError): pass
class LockTimeout(LedgerError): pass
class Unvalidated(LedgerError): pass


def events_path(path):
    return os.path.join(os.path.dirname(os.path.abspath(path)), "ledger_events.jsonl")


@contextlib.contextmanager
def locked(path, timeout=LOCK_TIMEOUT):
    """Exclusive flock on <path>.lock (non-blocking poll so a stuck holder gives an error, not a hang)."""
    lp = path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(lp)), exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o644)
    t0 = time.time()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except OSError:
                if time.time() - t0 > timeout:
                    raise LockTimeout(f"ledger lock {lp} not acquired in {timeout:.0f}s")
                time.sleep(0.005)
        yield
    finally:
        os.close(fd)      # closing drops the flock (also on kill -9: the kernel releases it)


def _write(path, df):
    for stale in glob.glob(path + ".tmp.*"):          # leftovers of a killed writer; we hold the lock, so nobody is mid-write
        try: os.remove(stale)
        except OSError: pass
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_parquet(tmp, index=False)
    with open(tmp, "rb") as f: os.fsync(f.fileno())
    os.replace(tmp, path)


def _event(path, owner, op, shards, **kw):
    rec = {"t": time.time(), "owner": owner, "op": op, "shards": list(shards), **kw}
    with open(events_path(path), "a") as f: f.write(json.dumps(rec) + "\n")


def rmw(path, fn, *, owner="", op="rmw", log_shards=None):
    """Under the lock: df = read(path); r = fn(df) (mutates df in place); write atomically.  Returns r.
    If fn returns None-ish nothing is special; if fn raises, the file is untouched."""
    with locked(path):
        df = pd.read_parquet(path)
        before = df.shape
        r = fn(df)
        _write(path, df)
        _event(path, owner, op, log_shards(r) if log_shards else (r if isinstance(r, (list, tuple)) else []))
        return r


def read(path):
    """Lock-free read: the file is only ever replaced atomically, so it is always a complete parquet."""
    return pd.read_parquet(path)


# ------------------------------------------------------------------------------------------------ schema
def prio_order(shard_ids, seed=PRIO_SEED):
    """Deterministic rank per shard id (sorted ids shuffled with default_rng(seed)); same ids -> same ranks."""
    ids = sorted(shard_ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {sid: int(r) for sid, r in zip(ids, perm)}


def _fill_schema(df):
    """In-place; idempotent; never touches existing values of existing columns."""
    changed = False
    defaults = {"owner": "", "lease_until": np.nan, "attempts": 0, "claim_id": ""}
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v; changed = True
    if df["attempts"].dtype.kind != "i": df["attempts"] = df["attempts"].fillna(0).astype("int64")
    if "prio" not in df.columns:
        df["prio"] = np.nan; changed = True
    t2 = df.index[(df.tier == 2) & df["prio"].isna()]
    if len(t2):
        order = prio_order(df.loc[df.tier == 2, "shard_id"].tolist())
        df.loc[t2, "prio"] = [float(order[s]) for s in df.loc[t2, "shard_id"]]; changed = True
    return changed


def ensure_schema(path):
    """Add the lease columns if missing (backup <ledger>.premigrate once).  Idempotent."""
    cur = pd.read_parquet(path)       # lock-free fast path: the file is only ever replaced atomically
    if all(c in cur.columns for c in NEW_COLS) and not cur.loc[cur.tier == 2, "prio"].isna().any():
        return False
    with locked(path):
        df = pd.read_parquet(path)
        if all(c in df.columns for c in NEW_COLS) and not df.loc[df.tier == 2, "prio"].isna().any():
            return False
        bak = path + ".premigrate"
        if not os.path.exists(bak): _write_copy(path, bak)
        _fill_schema(df); _write(path, df); _event(path, "", "migrate", [])
        return True


def _write_copy(src, dst):
    import shutil
    shutil.copy2(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)


# ------------------------------------------------------------------------------------------------ claim / lease
def _candidates(df, tier, now):
    """todo first, then expired running; each by prio.  Legacy running rows (no lease) are never taken."""
    t = df[df.tier == tier]
    todo = t[t.status == "todo"].sort_values("prio", kind="stable")
    exp = t[(t.status == "running") & t.lease_until.notna() & (t.lease_until < now)].sort_values("prio", kind="stable")
    return list(todo.index) + list(exp.index)


def peek(path, n, tier=2, now=None):
    """Read-only: the shard ids a claim(n) would take right now (dry-run)."""
    now = time.time() if now is None else now
    df = pd.read_parquet(path)
    if not all(c in df.columns for c in NEW_COLS): df = df.copy(); _fill_schema(df)
    return [df.loc[i, "shard_id"] for i in _candidates(df, tier, now)[:n]]


def claim(path, owner, n, lease_seconds, tier=2, platform="", now=None, max_attempts=MAX_ATTEMPTS):
    """Take up to n shards for `owner` (todo by prio, then expired leases).  Shards that already used max_attempts
    claims are marked failed (note) instead and skipped.  Returns the claimed shard ids."""
    ensure_schema(path)
    def fn(df):
        t = time.time() if now is None else now
        got = []
        for i in _candidates(df, tier, t):
            if len(got) >= n: break
            if int(df.at[i, "attempts"]) >= max_attempts:
                df.loc[i, ["status", "owner", "note"]] = ["failed", "", f"attempts>={max_attempts} (last owner {df.at[i, 'owner']})"]
                df.loc[i, "lease_until"] = np.nan
                continue
            att = int(df.at[i, "attempts"]) + 1
            df.loc[i, ["status", "owner", "platform", "claim_id"]] = ["running", owner, platform, f"{owner}-{int(t)}-{att}"]
            df.loc[i, ["lease_until", "started"]] = [t + lease_seconds, t]
            df.loc[i, "attempts"] = att
            got.append(df.at[i, "shard_id"])
        return got
    return rmw(path, fn, owner=owner, op="claim")


def renew(path, owner, shards, lease_seconds, now=None):
    """Extend the lease of the shards still running under `owner`; return that list (the gap = taken by someone else)."""
    ensure_schema(path)
    shards = set(shards)
    def fn(df):
        t = time.time() if now is None else now
        m = df.shard_id.isin(shards) & (df.status == "running") & (df.owner == owner)
        df.loc[m, "lease_until"] = t + lease_seconds
        return df.loc[m, "shard_id"].tolist()
    return rmw(path, fn, owner=owner, op="renew")


def release(path, owner, shards=None):
    """owner's running shards (optionally only `shards`) -> todo, attempts-1 (a voluntary give-back is not a failure).
    Never touches done, nor another owner's rows.  Works on a ledger without lease columns (matches nothing)."""
    def fn(df):
        if "owner" not in df.columns: return []
        m = (df.status == "running") & (df.owner == owner)
        if shards is not None: m &= df.shard_id.isin(set(shards))
        df.loc[m, "status"] = "todo"; df.loc[m, "owner"] = ""; df.loc[m, "lease_until"] = np.nan
        df.loc[m, "attempts"] = (df.loc[m, "attempts"] - 1).clip(lower=0)
        return df.loc[m, "shard_id"].tolist()
    return rmw(path, fn, owner=owner, op="release")


def hold(path, owner, until, tier=2, platform="", now=None):
    """Long-lease reservation for an autonomous owner: every `todo` shard of `tier` becomes running under `owner` with
    lease_until=`until` (epoch seconds), and `owner`'s own running rows get the same lease.  Other owners' rows and done/failed
    rows are never touched.  Returns the shard ids `owner` holds afterwards, in prio order."""
    ensure_schema(path)
    def fn(df):
        t = time.time() if now is None else now
        m = (df.tier == tier) & (df.status == "todo")
        df.loc[m, ["status", "owner", "platform"]] = ["running", owner, platform]
        df.loc[m, "started"] = t
        df.loc[m, "attempts"] = df.loc[m, "attempts"].clip(lower=0) + 1
        for i in df.index[m]: df.at[i, "claim_id"] = f"{owner}-hold-{int(t)}"
        mine = (df.tier == tier) & (df.status == "running") & (df.owner == owner)
        df.loc[mine, "lease_until"] = float(until)
        return df.loc[mine].sort_values("prio", kind="stable").shard_id.tolist()
    return rmw(path, fn, owner=owner, op="hold")


def expire_leases(path, now=None, max_attempts=MAX_ATTEMPTS):
    """running + lease_until < now -> todo (or failed when attempts>=max).  Returns the expired shard ids."""
    ensure_schema(path)
    def fn(df):
        t = time.time() if now is None else now
        m = (df.status == "running") & df.lease_until.notna() & (df.lease_until < t)
        bad = m & (df.attempts >= max_attempts)
        df.loc[m, "status"] = "todo"
        df.loc[bad, "status"] = "failed"; df.loc[bad, "note"] = f"lease expired, attempts>={max_attempts}"
        df.loc[m, ["owner"]] = ""; df.loc[m, "lease_until"] = np.nan
        return df.loc[m, "shard_id"].tolist()
    return rmw(path, fn, op="expire")


# ------------------------------------------------------------------------------------------------ results
def _apply_done(df, i, meta):
    df.loc[i, ["status", "platform", "gpu", "config_hash", "note"]] = [
        "done", meta["platform"], meta["gpu"] + f" x{meta['n_gpus_used']}", meta["config_hash"], ""]
    df.loc[i, ["started", "finished", "wall_sec", "games_per_sec"]] = [
        meta["t_start"], meta["t_end"], meta["wall_sec"], meta["games_per_sec_wall"]]
    if "owner" in df.columns:
        df.loc[i, "owner"] = ""; df.loc[i, "lease_until"] = np.nan


def mark_done(path, sid, meta, *, validation_errors, install=None, owner=""):
    """Record `sid` as done.  `validation_errors` MUST be the (empty) list returned by scan_merge.validate: None means
    "not validated" -> Unvalidated; a non-empty list -> Unvalidated and nothing changes.  `install()` runs inside the
    lock (copy result files into results/) only if the shard is not already done.  First valid result wins: returns
    True if this call recorded it, False if the shard was already done (install is then skipped)."""
    if validation_errors is None:
        raise Unvalidated(f"{sid}: mark_done called without validation")
    if len(validation_errors):
        raise Unvalidated(f"{sid}: validation errors: {list(validation_errors)}")
    def fn(df):
        i = df.index[df.shard_id == sid]
        if len(i) != 1: raise LedgerError(f"{sid} not in ledger")
        i = i[0]
        if df.at[i, "status"] == "done": return False
        if install is not None: install()
        _apply_done(df, i, meta)
        return True
    r = rmw(path, fn, owner=owner, op="done", log_shards=lambda r: [sid] if r else [])
    return r


def mark_failed(path, sid, note, owner=""):
    """status=failed + note, unless the shard is done (done is never downgraded).  Returns True if changed."""
    def fn(df):
        i = df.index[df.shard_id == sid]
        if len(i) != 1 or df.at[i[0], "status"] == "done": return False
        df.loc[i[0], ["status", "note"]] = ["failed", note]
        if "owner" in df.columns: df.loc[i[0], "owner"] = ""; df.loc[i[0], "lease_until"] = np.nan
        return True
    return rmw(path, fn, owner=owner, op="failed", log_shards=lambda r: [sid] if r else [])


def reconcile(path, res_dir, validate_fn, load_meta=None):
    """Make the ledger agree with results/: a shard with a complete, valid result set that is not `done` is marked done
    (covers a crash between "files copied" and "ledger written").  validate_fn(sid, row, res_dir) -> list of errors
    (run outside the lock).  Also reports done shards whose files are missing.  Returns dict(repaired, invalid, missing)."""
    df = read(path)
    rep, bad, missing = [], {}, []
    metas = sorted(glob.glob(os.path.join(res_dir, "S*_meta.json")))
    have = {os.path.basename(p).split("_")[0] for p in metas}
    for jp in metas:
        sid = os.path.basename(jp).split("_")[0]
        rows = df[df.shard_id == sid]
        if len(rows) != 1 or rows.iloc[0].status == "done": continue
        errs = validate_fn(sid, rows.iloc[0], res_dir)
        if errs: bad[sid] = errs; continue
        meta = json.load(open(jp))
        if mark_done(path, sid, meta, validation_errors=[], owner="reconcile"):
            rep.append(sid)
    for _, r in df[df.status == "done"].iterrows():
        if r.shard_id != "P0000" and r.tier == 2 and r.shard_id not in have: missing.append(r.shard_id)
    return {"repaired": rep, "invalid": bad, "missing": missing}


def summary(path, now=None):
    df = read(path); now = time.time() if now is None else now
    out = {"by_status": df.groupby(["tier", "status"]).size().to_dict()}
    if "owner" in df.columns:
        r = df[df.status == "running"]
        out["running_by_owner"] = r.groupby("owner").size().to_dict()
        out["expired_leases"] = int((r.lease_until < now).sum())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["migrate", "status", "expire", "peek"])
    ap.add_argument("n", nargs="?", type=int, default=5)
    ap.add_argument("--ledger", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "scan", "ledger.parquet")))
    a = ap.parse_args()
    if a.cmd == "migrate": print("migrated" if ensure_schema(a.ledger) else "already migrated")
    elif a.cmd == "expire": print("expired", expire_leases(a.ledger))
    elif a.cmd == "peek": print(peek(a.ledger, a.n))
    else: print(summary(a.ledger))
