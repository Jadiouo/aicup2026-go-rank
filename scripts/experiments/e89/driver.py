"""E89 S line: Colab-side driver for ONE stage (A or B).  Runs scripts that sit next to it (train_s.py) as an ordinary subprocess.

  python driver.py --stage A --root /content/e89 --blocks 10 --ch 128 --accum 1 --workers 8
Loop: start train_s.py (it resumes from ckpt_latest.pt by itself) -> watch it; if the process exits non-zero, or status.json has not moved for
20 minutes, kill + restart from the last checkpoint (at most 3 restarts).  On success write ROOT/DONE_<stage>.json (val h, test h, gate flag)
and, for stage B, ROOT/summary.json (A and B val/test, selected setting = higher val h).  Stage A never starts B (the operator starts it
with start_b.sh after reviewing A).  Gate information only: gate_test_h_lt_.15 is recorded, the driver does not decide anything with it.
Files written: ROOT/driver_<stage>.log, ROOT/driver_state_<stage>.json (pid, restarts, state).
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["A", "B"]); ap.add_argument("--root", default="/content/e89")
ap.add_argument("--blocks", type=int, required=True); ap.add_argument("--ch", type=int, required=True)
ap.add_argument("--accum", type=int, default=1); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--stall-min", type=float, default=20.0); ap.add_argument("--max-restarts", type=int, default=3)
a = ap.parse_args()
root = Path(a.root); run = root / f"run{a.stage}"; run.mkdir(parents=True, exist_ok=True)
logf = open(root / f"driver_{a.stage}.log", "a"); state_p = root / f"driver_state_{a.stage}.json"
def log(*x):
    s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(i) for i in x); logf.write(s + "\n"); logf.flush()
def state(**kw):
    st = dict(stage=a.stage, driver_pid=os.getpid(), updated=time.time(), restarts=restarts, **kw); state_p.write_text(json.dumps(st)); return st
restarts = 0
cmd = [sys.executable, str(root / "train_s.py"), "--data", str(root / "s_games.npz"), "--out", str(run), "--blocks", str(a.blocks), "--ch", str(a.ch), "--accum", str(a.accum),
       "--workers", str(a.workers), "--epochs", str(a.epochs), "--seed", str(a.seed), "--name", a.stage]
ok = False
while True:
    p = subprocess.Popen(cmd, stdout=open(run / "stdout.log", "a"), stderr=subprocess.STDOUT); log("started train_s.py pid", p.pid, "attempt", restarts, " ".join(cmd))
    state(state="running", train_pid=p.pid); last_beat = time.time(); last_seen = None; reason = None
    while p.poll() is None:
        time.sleep(30)
        try: t = json.loads((run / "status.json").read_text())["time"]
        except Exception: t = None
        if t is not None and t != last_seen: last_seen, last_beat = t, time.time()
        state(state="running", train_pid=p.pid)
        if time.time() - last_beat > a.stall_min * 60:
            reason = f"no progress for {a.stall_min} min"; log("STALL:", reason, "-> kill"); p.kill(); p.wait(); break
    rc = p.returncode; log("train_s.py ended rc", rc, reason or "")
    if rc == 0 and (run / "results.json").exists(): ok = True; break
    restarts += 1
    if restarts > a.max_restarts:
        log("giving up after", a.max_restarts, "restarts"); state(state="failed"); (root / f"FAILED_{a.stage}").write_text(f"rc {rc} {reason}"); sys.exit(1)
    log("restart", restarts, "from last checkpoint"); time.sleep(10)
res = json.loads((run / "results.json").read_text())
done = dict(stage=a.stage, val_h=res["val_h"], test_h=res["test_h"], test_ci95=res["test"].get("h_ci95_player_bootstrap"), test_top5=res["test"]["top5_all"], blocks=a.blocks, ch=a.ch,
            gate_test_h_lt_015=bool(res["test_h"] < 0.15), restarts=restarts, seconds_total_train_script=res["seconds"], finished=time.time())
(root / f"DONE_{a.stage}.json").write_text(json.dumps(done, indent=1)); log("DONE", json.dumps(done)); state(state="done")
if a.stage == "B":
    A = json.loads((root / "DONE_A.json").read_text()); sel = "A" if A["val_h"] >= done["val_h"] else "B"
    summ = dict(A=A, B=done, selected_by_higher_val_h=sel, selected_test_h=(A if sel == "A" else done)["test_h"], all_done=True)
    (root / "summary.json").write_text(json.dumps(summ, indent=1)); log("SUMMARY", json.dumps(summ))
