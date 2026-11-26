"""Local background controller: Colab L4 sessions -> KataGo HumanSL full scan (default tier 1), restartable from the ledger.

  scan_controller.py run [--tier 1] [--max-units 40] [--hours 10] [--per-session 26]
  scan_controller.py status        one-line progress + controller state
Stop gracefully:  touch .scratch/scan_tier1/STOP   (or gpujob kill / SIGTERM): collects finished shards, stops the session.
State: .scratch/scan_tier1/state.json   Log: .scratch/scan_tier1/controller.log   Inbox: .scratch/scan_tier1/inbox
Only sessions named gogame-scan-* opened by this controller are ever touched.
"""
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tarfile, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D = f"{ROOT}/.scratch/scan_tier1"; STATE = f"{D}/state.json"; LOG = f"{D}/controller.log"; INBOX = f"{D}/inbox"
SESS_CFG = f"{D}/sessions.json"; LEDGER = f"{ROOT}/data/scan/ledger.parquet"
COLAB = [os.environ.get("COLAB_CLI", "colab"), "--auth=oauth2", "--config", SESS_CFG]
PY = f"{ROOT}/.venv/bin/python"; PLAT = "colab-l4"; R = "/content/scan"
CODE = ["kgio.py", "scan_runner.py", "scan_merge.py", "scan_smoke.py", "scan_setup.sh", "scan_pins.json", "analysis.cfg"]
STOPPING = False


def log(*a):
    s = time.strftime("%F %T ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as f: f.write(s + "\n")


def sh(cmd, timeout=120, check=False):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or ""); rc = p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "TIMEOUT", 124
    if check and rc: raise RuntimeError(f"{cmd[:4]} rc={rc}: {out[-500:]}")
    return rc, out


def colab(*args, timeout=120):
    return sh(COLAB + list(args), timeout)


def load_state():
    try: return json.load(open(STATE))
    except Exception: return {}


def save_state(s):
    json.dump(s, open(STATE + ".tmp", "w"), indent=1); os.replace(STATE + ".tmp", STATE)


def balance():
    rc, out = colab("usage", timeout=60)
    m = re.search(r"Current balance:\s*([\d.]+)", out)
    return float(m.group(1)) if m else None


def ledger():
    return pd.read_parquet(LEDGER)


def led_set(sids, status, platform=None):
    l = ledger()
    l.loc[l.shard_id.isin(sids), "status"] = status
    if platform is not None: l.loc[l.shard_id.isin(sids), "platform"] = platform
    if status == "running": l.loc[l.shard_id.isin(sids), "started"] = time.time()
    l.to_parquet(LEDGER + ".tmp", index=False); os.replace(LEDGER + ".tmp", LEDGER)


def release_mine():
    l = ledger(); mine = l[(l.status == "running") & (l.platform == PLAT)].shard_id.tolist()
    if mine: led_set(mine, "todo", ""); log("released", mine)


def rexec(sess, code, timeout=90):
    f = f"{D}/_cell.py"; open(f, "w").write(code)
    return colab("exec", "-s", sess, "-f", f, "--timeout", str(timeout), timeout=timeout + 60)


def session_alive(sess):
    rc, out = colab("status", "-s", sess, timeout=60)
    return rc == 0 and ("IDLE" in out.upper() or "BUSY" in out.upper())


def build_pkg(sess, shards, smoke, hours, start_epoch):
    pkg = f"{D}/pkg_{sess}.tar.gz"; sc = f"{D}/job_{sess}.sh"
    open(sc, "w").write(f"""#!/bin/bash
set -uo pipefail
R={R}; cd $R; mkdir -p out work logs
export PYTHON=python3 CU12_DIR=$R/cu12
echo setup > phase
bash code/scan_setup.sh $R > logs/setup.log 2>&1 || {{ echo FAIL_SETUP > phase; exit 1; }}
export LD_LIBRARY_PATH="$(ls -d $R/cu12/nvidia/*/lib | tr '\\n' ':')${{LD_LIBRARY_PATH:-}}"
if [ "{int(smoke)}" = "1" ]; then
  echo smoke > phase
  python3 -u code/scan_smoke.py --katago-dir $R/katago --bench-dir $R/bench --out $R/smoke.json > logs/smoke.log 2>&1 || {{ echo FAIL_SMOKE > phase; exit 1; }}
fi
echo run > phase
EL=$(( ($(date +%s) - {int(start_epoch)}) / 60 ))
python3 -u code/scan_runner.py --shards {','.join(shards)} --sgf-dir $R/sgf --shards-table $R/shards.parquet --katago-dir $R/katago \\
  --cu12 $R/cu12 --out $R/out --work $R/work --platform colab --time-budget-hours {hours} --elapsed-before-min $EL --reserve-minutes 20 > logs/runner.log 2>&1
echo DONE_rc=$? > phase
""")
    with tarfile.open(pkg, "w:gz") as t:
        for f in CODE: t.add(f"{HERE}/{f}", arcname=f"code/{f}")
        t.add(sc, arcname="job.sh")
        t.add(f"{ROOT}/data/scan/shards.parquet", arcname="shards.parquet")
        for f in ("bench_games.parquet", "bench_pilot_moves.parquet"): t.add(f"{ROOT}/data/scan/bench/{f}", arcname=f"bench/{f}")
        for s in shards: t.add(f"{ROOT}/data/scan/sgf/{s}.parquet", arcname=f"sgf/{s}.parquet")
    return pkg


POLL = """
import os, glob, json, subprocess, hashlib
R='/content/scan'; o={}
rd=lambda p:(open(p).read().strip() if os.path.exists(p) else None)
o['phase']=rd(R+'/phase')
o['alive']=subprocess.run("pgrep -f 'scan_runner.py|scan_smoke.py|job.sh' | head -3",shell=True,capture_output=True,text=True).stdout.split()
o['metas']=sorted(os.path.basename(p).split('_')[0] for p in glob.glob(R+'/out/S*_meta.json'))
def tail(p,n=3):
    return open(p).read().strip().splitlines()[-n:] if os.path.exists(p) else []
o['runner']=tail(R+'/logs/runner.log'); o['setup']=tail(R+'/logs/setup.log',2); o['smoke']=tail(R+'/logs/smoke.log',2)
o['smoke_json']=json.load(open(R+'/smoke.json')) if os.path.exists(R+'/smoke.json') else None
print('POLL:'+json.dumps(o))
"""
SHA = """
import hashlib,sys,json
out={}
for sid in %s:
    for nm in ('moves','games','meta'):
        p='/content/scan/out/%%s_%%s.%%s'%%(sid,nm,'json' if nm=='meta' else 'parquet')
        h=hashlib.sha256(open(p,'rb').read()).hexdigest(); out[sid+'_'+nm]=h
print('SHA:'+json.dumps(out))
"""


def poll(sess):
    rc, out = rexec(sess, POLL, 90)
    m = re.search(r"POLL:(\{.*\})", out)
    return json.loads(m.group(1)) if m else None


def sha_local(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def collect(sess, st, metas):
    """Download + verify + ingest every finished shard on the VM that is not yet done in the ledger."""
    done = set(ledger().query("status == 'done'").shard_id)
    new = [s for s in metas if s not in done]
    got = []
    for sid in new:
        rc, out = rexec(sess, SHA % json.dumps([sid]), 180)
        m = re.search(r"SHA:(\{.*\})", out)
        if not m: log("sha query failed", sid, out[-200:]); continue
        want = json.loads(m.group(1)); ok = True
        os.makedirs(INBOX, exist_ok=True)
        for nm, ext in (("moves", "parquet"), ("games", "parquet"), ("meta", "json")):
            fn = f"{INBOX}/{sid}_{nm}.{ext}"
            for attempt in range(3):
                rc, o2 = colab("download", "-s", sess, f"{R}/out/{sid}_{nm}.{ext}", fn, timeout=900)
                if os.path.exists(fn) and sha_local(fn) == want[f"{sid}_{nm}"]: break
                log("download retry", sid, nm, attempt, o2[-150:])
            else: ok = False
        if ok: got.append(sid)
    if got:
        rc, out = sh([PY, f"{HERE}/scan_merge.py", "ingest", INBOX], 600)
        log("ingest:", " | ".join(l for l in out.splitlines() if l.startswith(("ACCEPT", "REJECT", "already"))))
        l = ledger().set_index("shard_id")
        for sid in got:
            st_ = l.loc[sid, "status"]
            if st_ == "done":
                for nm, ext in (("moves", "parquet"), ("games", "parquet"), ("meta", "json")):
                    try: os.remove(f"{INBOX}/{sid}_{nm}.{ext}")
                    except OSError: pass
                st["shards_done_this_run"] = st.get("shards_done_this_run", []) + [sid]
            else:
                st["rejected"] = st.get("rejected", []) + [sid]; st["halt"] = f"{sid} rejected: {l.loc[sid, 'note']}"
                log("HALT", st["halt"])
    save_state(st)
    return got


def stop_stuff():
    return STOPPING or os.path.exists(f"{D}/STOP")


def run_session(a, st):
    global STOPPING
    l = ledger()
    todo = l[(l.tier == a.tier) & (l.status.isin(["todo", "failed"])) & (l.shard_id != "P0000")].shard_id.sort_values().tolist()
    fails = st.get("fail_count", {})
    todo = [s for s in todo if fails.get(s, 0) < 2]
    if not todo: log("no todo shards for tier", a.tier); return "finished"
    bal = balance()
    if bal is None: log("balance unreadable"); return "error"
    spent = st["baseline_balance"] - bal
    if spent >= a.max_units - 4: log(f"units cap: spent {spent:.2f} of {a.max_units}"); return "cap"
    shards = todo[:a.per_session]
    sess = f"gogame-scan-t1-{time.strftime('%m%d%H%M%S')}"
    smoke = not st.get("smoke_passed", False)
    log(f"opening {sess}: {len(shards)} shards {shards[0]}..{shards[-1]} smoke={smoke} balance={bal}")
    led_set(shards, "running", PLAT)
    t_open = time.time()
    st.update(session=sess, session_open=t_open, shards=shards, phase="opening"); save_state(st)
    rc, out = colab("new", "-s", sess, "--gpu", "L4", timeout=300)
    if rc != 0:
        log("colab new failed", out[-300:]); release_mine(); st.pop("session", None); save_state(st); return "error"
    return monitor(a, st, sess, shards, smoke, t_open)


def monitor(a, st, sess, shards, smoke, t_open, resumed=False):
    global STOPPING
    result = "ok"
    try:
        if not resumed:
            pkg = build_pkg(sess, shards, smoke, a.hours, t_open)
            rc, out = colab("upload", "-s", sess, pkg, "/content/pkg.tar.gz", timeout=900)
            if rc: raise RuntimeError("upload failed " + out[-300:])
            rc, out = rexec(sess, f"""
import subprocess,os
os.makedirs('{R}',exist_ok=True)
subprocess.run(['tar','xzf','/content/pkg.tar.gz','-C','{R}'],check=True)
subprocess.Popen(['bash','{R}/job.sh'],cwd='{R}',stdin=subprocess.DEVNULL,stdout=open('{R}/job.log','ab'),stderr=subprocess.STDOUT,start_new_session=True)
print('LAUNCHED')""", 120)
            if "LAUNCHED" not in out: raise RuntimeError("launch failed " + out[-300:])
            log("launched on", sess)
        bad = 0; last_bal = 0; n_seen = set()
        while True:
            time.sleep(180)
            if stop_stuff(): log("STOP requested"); result = "stopped"; break
            p = poll(sess)
            if p is None:
                bad += 1; log("poll failed", bad)
                if bad >= 5 and not session_alive(sess): log("session lost"); result = "lost"; break
                continue
            bad = 0; st["phase"] = p["phase"]
            if smoke and p.get("smoke_json") and "smoke_done" not in st.get(f"x{sess}", {}):
                sj = p["smoke_json"]; json.dump(sj, open(f"{D}/smoke_{sess}.json", "w"), indent=1)
                log("SMOKE", "PASS" if sj["pass"] else "FAIL", "ab_mad %.4f pilotA_mad %.4f pilotB_mad %.4f agree ab %.2f pA %.2f pB %.2f" % (
                    sj["ab"]["move_mad"], sj["pilot_a"]["move_mad"], sj["pilot_b"]["move_mad"], sj["ab"]["game_argmax_agree"],
                    sj["pilot_a"]["game_argmax_agree"], sj["pilot_b"]["game_argmax_agree"]), "evals/s", [int(x) for x in sj["evals_per_sec"]])
                st["smoke_passed"] = bool(sj["pass"]); st[f"x{sess}"] = {"smoke_done": 1}
            if p["phase"] and p["phase"].startswith("FAIL"):
                log("remote failure", p["phase"], p["setup"], p["smoke"]); st["halt"] = p["phase"]; result = "fail"; break
            if p["runner"]: log("runner:", p["runner"][-1][:160])
            collect(sess, st, p["metas"])
            if st.get("halt"): result = "halt"; break
            if p["phase"] and p["phase"].startswith("DONE"): log("runner finished", p["phase"]); collect(sess, st, p["metas"]); break
            if time.time() - t_open > (a.hours + 0.4) * 3600: log("session hard time limit"); result = "timelimit"; break
            b = balance()
            if b is not None: st["last_balance"] = b; save_state(st)
            if b is not None and st["baseline_balance"] - b >= a.max_units: log(f"units cap hit: balance {b}"); result = "cap"; break
    except BaseException as e:
        log("monitor exception", repr(e)); result = "error"
    finally:
        try:
            p = poll(sess)
            if p: collect(sess, st, p["metas"])
        except BaseException as e: log("final collect failed", repr(e))
        rc, out = colab("stop", "-s", sess, timeout=120)
        log("stop", sess, rc, out.strip()[-100:])
        rc, out = colab("sessions", timeout=60); log("sessions after stop:", out.strip().replace("\n", " | ")[-200:])
        release_mine()
        b = balance(); st["last_balance"] = b
        st.setdefault("history", []).append({"session": sess, "result": result, "hours": round((time.time() - t_open) / 3600, 2), "balance_after": b})
        st.pop("session", None); st["phase"] = "idle"; save_state(st)
        log(f"session {sess} ended: {result}; hours {(time.time() - t_open) / 3600:.2f}; balance {b}; spent {st['baseline_balance'] - b if b else '?'}")
    return result


def cmd_run(a):
    os.makedirs(D, exist_ok=True); os.makedirs(INBOX, exist_ok=True)
    def term(*_):
        global STOPPING; STOPPING = True; log("signal: graceful stop")
    signal.signal(signal.SIGTERM, term); signal.signal(signal.SIGINT, term)
    st = load_state()
    if os.path.exists(f"{D}/STOP"): os.remove(f"{D}/STOP")
    if "baseline_balance" not in st:
        st["baseline_balance"] = balance(); st["started"] = time.strftime("%F %T")
        log("baseline balance", st["baseline_balance"])
    st.pop("halt", None)
    # restart: reattach to a live session of ours, else release stale claims
    if st.get("session"):
        sess = st["session"]
        if session_alive(sess):
            log("reattaching", sess); r = monitor(a, st, sess, st["shards"], not st.get("smoke_passed", False), st["session_open"], resumed=True)
            if r in ("cap", "halt", "fail", "stopped", "error"): return
        else:
            log("stale session in state, not alive:", sess); st.pop("session", None); save_state(st)
    release_mine()
    errors = 0
    while not stop_stuff():
        n0 = len(ledger().query("tier == %d and status == 'done'" % a.tier))
        r = run_session(a, st)
        if r == "ok" and len(ledger().query("tier == %d and status == 'done'" % a.tier)) == n0: r = "error"
        if r in ("finished", "cap", "halt", "fail", "stopped"): log("controller exit:", r); break
        if r in ("error", "lost", "timelimit", "ok"):
            errors += r in ("error", "lost")
            if errors >= 3: log("too many errors; exit"); break
        if st.get("halt"): break
    log("controller done")


def cmd_status():
    st = load_state(); l = ledger(); t = l[l.tier == 1]
    done = t[t.status == "done"]; run = t[t.status == "running"]
    bal = st.get("last_balance")
    gps = done.games_per_sec.dropna()
    print(f"tier1 {len(done)}/{len(t)} shards done ({int(done.n_games.sum())}/{int(t.n_games.sum())} games), running {len(run)}, failed {int((t.status=='failed').sum())}; "
          f"session={st.get('session')} phase={st.get('phase')} smoke_passed={st.get('smoke_passed')} halt={st.get('halt')} "
          f"spent={'%.1f' % (st['baseline_balance'] - bal) if bal and st.get('baseline_balance') else '?'}u mean_gps={gps.mean():.2f}" if len(gps) else
          f"tier1 {len(done)}/{len(t)} shards done, running {len(run)}; session={st.get('session')} phase={st.get('phase')} smoke_passed={st.get('smoke_passed')} halt={st.get('halt')}")
    if os.path.exists(LOG): print(open(LOG).read().strip().splitlines()[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--tier", type=int, default=1); ap.add_argument("--max-units", type=float, default=40)
    ap.add_argument("--hours", type=float, default=10); ap.add_argument("--per-session", type=int, default=26)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_status()
