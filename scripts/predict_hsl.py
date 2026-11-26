"""Test-day KataGo HumanSL extraction for the Rank formal model (E06 / hsl_lgbm).  CPU side + Colab controller wrapper.

  prepare   --data-dir DIR [--root data/test_scan]   official rank_prediction_test_{public,private}.csv -> scan-format sgf
                                                     segments, shards table, manifest, ledger (never touches data/scan/)
  run --backend school [--hours 3] [--dry-run|--preflight-only]   MAIN PATH: the school RTX8000 via scan_pool.SchoolBackend + job_school.sh in its own
                                                     remote dir ~/gogame_scan_1104 (needs SCAN_SCHOOL_LIVE=1; refuses while tier-2 runs there)
  run       [--root ..] [--max-units 12] [--hours 3] [--dry-run]    BACKUP: Colab L4 session(s) through the existing
                                                     scripts/katago/scan_controller.py machinery, results ingested with the
                                                     SAME validation (scan_merge.validate: pinned versions/hashes/config_hash)
  status    [--root ..]                              ledger + controller state
  ingest    DIR [--root ..]                          manual fallback: validate + accept shard files from DIR
  features  [--root ..] [--allow-partial]            accepted shards -> <root>/hsl_test_games.parquet (+ features_manifest.json),
                                                     same columns/dtypes as data/cache/e06/hsl_games.parquet
  sim-results --source-scan data/scan --root DIR     CPU rehearsal only: fabricate shard results by copying already-scanned
                                                     training games with the same (colour, sgf)   (refuses the default root)
  crosscheck  --root DIR                             rehearsal check against data/scan (manifest fields, HSL rows, bag aggregates)

Fixed setting = scripts/katago/scan_pins.json + analysis.cfg (the same files the tier-1 training scan used); nothing here
can change it: the runner is the unchanged scan_runner.py and the ingest is the unchanged scan_merge.validate.
Test uid = '<pub|priv>_<question_id>#<slot>' (predict.py explode_question), the key hsl_lgbm joins on.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KATAGO = ROOT / "scripts" / "katago"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(KATAGO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_ROOT = ROOT / "data" / "test_scan"
PINS_PATH, CFG_PATH = KATAGO / "scan_pins.json", KATAGO / "analysis.cfg"
PILOT_HSL = ROOT / "data" / "cache" / "katago" / "pilot_hsl_games.parquet"
SHARD_TARGET = 3500
LEDGER_COLS = ["status", "platform", "gpu", "config_hash", "note"]
PROVENANCE_KEYS = ("scan_version", "katago_version", "katago_executable_sha256", "main_net_content_sha256",
                   "human_net_content_sha256", "config_hash", "pins_query", "engine_overrides")
RANK_FILES = ("rank_pub", "rank_priv")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _runner():
    import scan_runner
    return scan_runner


def pins() -> dict:
    return json.loads(PINS_PATH.read_text())


def current_config_hash() -> str:
    return _runner().config_hash(str(CFG_PATH), pins())


def expected_own_moves(sgf: str, color: int, n_own: int | None = None) -> int:
    """Number of own moves the runner will query (first n_own_moves own moves); 0 if the runner will skip the game.
    Uses the runner's own kgio.normalize_game (missing SZ = 19, HD>=2 swap + stones), so the two cannot disagree."""
    from kgio import normalize_game
    n_own = n_own or pins()["query"]["n_own_moves"]
    ng = normalize_game(sgf, color)
    if ng.skip:
        return 0
    col = "BW"[ng.eff_color]
    return min(n_own, sum(1 for c, _ in ng.moves if c == col))


def predicted_skip_reason(sgf: str, color: int) -> str:
    """'' = the runner will evaluate the game, else the runner's skip reason (same function as the runner)."""
    from kgio import normalize_game
    ng = normalize_game(sgf, color)
    if ng.skip:
        return ng.skip
    return "" if expected_own_moves(sgf, color) > 0 else "no_own_moves"


# ================================================================================ prepare
def collect_test_games(data_dir: Path) -> pd.DataFrame:
    """uid, color(int8 0/1), sgf, qid, split, slot for every game of the Rank test files (public then private, file order)."""
    import predict as P
    rows = []
    for key in RANK_FILES:
        path = Path(data_dir) / P.FILES[key]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: put the official rank_prediction_test_{{public,private}}.csv in {data_dir}")
        split = key.split("_")[1]
        for q in P.load_questions(path, split, "rank"):
            for i, g in enumerate(q.games, 1):
                rows.append((g.uid, g.color, g.sgf, q.qid, split, i))
    df = pd.DataFrame(rows, columns=["uid", "color", "sgf", "qid", "split", "slot"])
    df["color"] = df["color"].astype(np.int8)
    if df["uid"].duplicated().any():
        raise ValueError("duplicate test game uid")
    return df


def make_ledger(shards: pd.DataFrame) -> pd.DataFrame:
    """Same columns as scan_merge.cmd_init (minus the pilot row): all shards `todo`."""
    l = shards[["shard_id", "tier", "n_games", "content_hash"]].copy()
    for c in LEDGER_COLS:
        l[c] = ""
    for c in ["started", "finished", "wall_sec", "games_per_sec"]:
        l[c] = np.nan
    l["status"] = "todo"
    return l


def prepare(data_dir: Path, root: Path, shard_target: int = SHARD_TARGET, force: bool = False, log=print,
            strict_format: bool = False) -> dict:
    root = Path(root)
    if root.resolve() == (ROOT / "data" / "scan").resolve():
        raise ValueError("refusing to write into data/scan")
    led_p = root / "ledger.parquet"
    if led_p.exists() and not force:
        l = pd.read_parquet(led_p)
        if (l["status"] != "todo").any():
            raise RuntimeError(f"{led_p} already has non-todo shards; refusing to overwrite (use --force only if you really want to redo)")
    g = collect_test_games(data_dir)
    from gogame.sgf import format_gate, format_report, get_color_semantics
    cs = get_color_semantics()
    frep = format_report(g["sgf"], g["color"], cs)
    format_gate(frep, strict_format, log, title="SGF format report (Rank test files)")     # stops BEFORE anything is written when strict
    sr = _runner()
    g["n_own"] = [expected_own_moves(s, c) for s, c in zip(g["sgf"], g["color"])]
    g["predicted_skip"] = g["n_own"] == 0
    g["predicted_skip_reason"] = [predicted_skip_reason(s, c) for s, c in zip(g["sgf"], g["color"])]
    g["order"] = np.arange(len(g), dtype=np.int32)
    n_sh = max(1, int(np.ceil(len(g) / shard_target)))
    g["shard_id"] = ""
    for k, part in enumerate(np.array_split(np.arange(len(g)), n_sh), 1):
        g.loc[part, "shard_id"] = f"S{9000 + k:04d}"      # S-prefix: scan_controller / scan_merge only look at S*_meta.json
    (root / "sgf").mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, d in g.groupby("shard_id", sort=True):
        d = d.sort_values("order")
        rows.append(dict(shard_id=sid, tier=1, n_games=len(d), n_pilot=0,
                         content_hash=sr.content_hash(d["uid"], d["color"], d["sgf"]),
                         uids_hash=hashlib.sha256("\n".join(d["uid"]).encode()).hexdigest()))
        d[["uid", "color", "sgf"]].to_parquet(root / "sgf" / f"{sid}.parquet", index=False, compression="zstd", compression_level=9)
    shards = pd.DataFrame(rows)
    shards.to_parquet(root / "shards.parquet", index=False)
    g.drop(columns=["sgf"]).to_parquet(root / "manifest.parquet", index=False, compression="zstd")
    make_ledger(shards).to_parquet(led_p, index=False)
    for sub in ("results", "inbox"):
        (root / sub).mkdir(exist_ok=True)
    info = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_games": int(len(g)), "n_questions": int(g["qid"].nunique()),
            "n_shards": int(n_sh), "shard_sizes": shards["n_games"].tolist(), "predicted_skipped_games": int(g["predicted_skip"].sum()),
            "n_own_moves_total": int(g["n_own"].sum()), "config_hash": current_config_hash(),
            "scan_pins_sha256": sha256_file(PINS_PATH), "analysis_cfg_sha256": sha256_file(CFG_PATH),
            "inputs": {k: sha256_file(Path(data_dir) / __import__("predict").FILES[k]) for k in RANK_FILES},
            "color_dist": {c: int((g["color"] == i).sum()) for i, c in enumerate("BW")},
            "color_semantics": cs, "format_report": frep,
            "predicted_skip_reasons": {k: int(v) for k, v in g.loc[g["predicted_skip_reason"] != "", "predicted_skip_reason"].value_counts().items()}}
    (root / "prepare_meta.json").write_text(json.dumps(info, indent=1, default=str))
    log(f"prepared {info['n_games']} games / {info['n_questions']} questions in {n_sh} shards -> {root}")
    log(f"  predicted engine-skips {info['predicted_skipped_games']} by reason {info['predicted_skip_reasons']}")
    log(f"  config_hash {info['config_hash'][:16]}…  predicted engine-skips {info['predicted_skipped_games']}  "
        f"own-move evals ~{info['n_own_moves_total'] * len(pins()['query']['profiles'])}")
    return info


# ================================================================================ scan_merge / scan_controller reuse
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scan_merge_for(root: Path):
    """scripts/katago/scan_merge.py with its data/scan paths pointed at `root` (original file untouched)."""
    m = _load_module("scan_merge_testscan", KATAGO / "scan_merge.py")
    m.SCAN, m.LEDGER, m.RES = str(root), str(root / "ledger.parquet"), str(root / "results")
    return m


def cmd_ingest(root: Path, d: Path) -> None:
    scan_merge_for(root).cmd_ingest(str(d), str(CFG_PATH))


def controller_for(root: Path):
    """scan_controller.py with ledger / state / log / sessions / package source redirected to `root`.  The job script,
    setup, smoke test, runner arguments and monitoring logic are the original ones."""
    m = _load_module("scan_controller_testscan", KATAGO / "scan_controller.py")
    ctl = root / "_controller"
    ctl.mkdir(parents=True, exist_ok=True)
    (ctl / "inbox").mkdir(exist_ok=True)
    m.D, m.STATE, m.LOG, m.INBOX = str(ctl), str(ctl / "state.json"), str(ctl / "controller.log"), str(ctl / "inbox")
    m.SESS_CFG, m.LEDGER = str(ctl / "sessions.json"), str(root / "ledger.parquet")
    m.COLAB = [os.environ.get("COLAB_CLI", "colab"), "--auth=oauth2", "--config", m.SESS_CFG]
    # build_pkg reads {ROOT}/data/scan/{shards.parquet,sgf/,bench/}: mirror the test files (and the pinned bench games) there
    pk = ctl / "pkgroot"
    (pk / "data" / "scan").mkdir(parents=True, exist_ok=True)
    m.ROOT = str(pk)
    orig_sh = m.sh

    def sh(cmd, timeout=120, check=False):          # controller -> `scan_merge.py ingest INBOX` becomes our ingest
        if len(cmd) >= 4 and str(cmd[1]).endswith("scan_merge.py") and cmd[2] == "ingest":
            cmd = [cmd[0], str(Path(__file__).resolve()), "--root", str(root), "ingest", cmd[3]]
        return orig_sh(cmd, timeout, check)
    m.sh = sh
    return m


def stage_pkgroot(m, root: Path) -> None:
    pk = Path(m.ROOT) / "data" / "scan"
    shutil.copy(root / "shards.parquet", pk / "shards.parquet")
    shutil.rmtree(pk / "sgf", ignore_errors=True)
    shutil.copytree(root / "sgf", pk / "sgf")
    shutil.rmtree(pk / "bench", ignore_errors=True)
    shutil.copytree(ROOT / "data" / "scan" / "bench", pk / "bench")


def cmd_run(root: Path, max_units: float, hours: float, dry_run: bool) -> int:
    from gogame.sgf import DEFAULT_COLOR_SEMANTICS, get_color_semantics
    if get_color_semantics() != DEFAULT_COLOR_SEMANTICS:
        raise RuntimeError("the Colab controller does not forward color_semantics to its job script; use --backend school or the default")
    if not (root / "ledger.parquet").exists():
        raise FileNotFoundError(f"{root}/ledger.parquet missing: run `predict_hsl.py prepare` first")
    m = controller_for(root)
    stage_pkgroot(m, root)
    l = pd.read_parquet(m.LEDGER)
    todo = l[l["status"].isin(["todo", "failed"])].shard_id.sort_values().tolist()
    est_h = l[l["status"].isin(["todo", "failed"])].n_games.sum() / 2.35 / 3600
    print(f"{len(todo)} shard(s) to run ({int(l[l.shard_id.isin(todo)].n_games.sum())} games): ~{est_h:.2f} h engine time at 2.35 games/s "
          f"+ ~0.15 h setup/smoke  =>  ~{(est_h + 0.2) * 1.54:.1f} units")
    if dry_run:
        pkg = m.build_pkg("dryrun", todo, True, hours, time.time())
        import tarfile
        with tarfile.open(pkg) as t:
            names = t.getnames()
        print("dry-run package:", pkg, f"{os.path.getsize(pkg) / 1e6:.1f} MB;", len(names), "members")
        print("  ", [n for n in names if not n.startswith("sgf/")][:20], "+ sgf:", [n for n in names if n.startswith("sgf/")])
        print(open(os.path.join(m.D, "job_dryrun.sh")).read())
        return 0
    a = argparse.Namespace(tier=1, max_units=max_units, hours=hours, per_session=max(len(todo), 1))
    m.cmd_run(a)
    return 0


# ================================================================================ school backend (main path from 2026-10-20)
TIER2_REMOTE = "~/gogame_scan"            # tier-2 supervisor's directory: NEVER written here; only read (katago/, cu12/, .venv symlink targets)
SCHOOL_REMOTE = "~/gogame_scan_1104"      # own remote work dir: own scripts/, sgf/, shards.parquet, queue.txt, out/, work/, logs/, phase, run.pid
SCHOOL_GPS = 1.9                          # games/s on the RTX8000 (tier-2 measurement, 4 engines); estimate only
LINKED = ("katago", "cu12", ".venv")      # big read-only assets shared with the tier-2 install through symlinks
OWN_PROC_PAT = "[s]upervisor.py|[j]ob_school.sh|[s]can_runner.py"    # bracket trick: the pgrep's own command line never matches
PREFLIGHT_SH = ("R=$(readlink -f {remote} 2>/dev/null); echo \"OURS=$R\"; "
                "for p in $(pgrep -u $(id -u) -f '" + OWN_PROC_PAT + "'); do echo \"PROC $p $(readlink /proc/$p/cwd) $(ps -p $p -o args= | cut -c1-100)\"; done; "
                "echo GPU; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; echo \"NVRC=$?\"; "
                "echo MEM; nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits; echo \"MEMRC=$?\"")
MIN_FREE_MIB = 4096                       # --share-gpu: minimum free VRAM (MiB) on top of a foreign GPU program (user decision 2026-10-31)


SHARE_GPU_ENV = "GOGAME_ALLOW_FOREIGN_GPU=1 "    # makes job_school.sh skip its foreign-GPU refusal; only set with --share-gpu


def school_backend_cls():
    sp = _load_module("scan_pool_hsl1104", KATAGO / "scan_pool.py")

    class Hsl1104Backend(sp.SchoolBackend):
        """scan_pool.SchoolBackend (rsync+sha, setsid launch, run.pid check, sha-verified fetch, FAIL_FOREIGN_GPU via job_school.sh)
        pointed at its own remote directory and at <root>/{shards.parquet,sgf/}.  Nothing under ~/gogame_scan is written."""
        def __init__(self, root, host=os.environ.get("REMOTE_HOST", "remote-host"), remote=SCHOOL_REMOTE, dry_run=True, share_gpu=False):
            super().__init__(host=host, remote=remote, dry_run=dry_run, owner="hsl1104")
            self.scan = str(root)
            from gogame.sgf import get_color_semantics
            self.launch_env = ("GOGAME_SCAN_ROOT=" + remote.replace("~", "$HOME", 1) + " "
                               + "GOGAME_COLOR_SEMANTICS=" + get_color_semantics() + " ")      # job_school.sh: R=${GOGAME_SCAN_ROOT:-$HOME/gogame_scan}
            if share_gpu:
                self.launch_env += SHARE_GPU_ENV
            self.session = "gogame-hsl1104"

        def link_assets(self):
            """Create OUR dir and symlink katago/cu12/.venv from the tier-2 install (read-only use), after checking they are complete."""
            r, t = self.remote, TIER2_REMOTE
            cmd = (f"mkdir -p {r} && for n in {' '.join(LINKED)}; do [ -e {r}/$n ] || ln -s \"$HOME/{t[2:]}/$n\" {r}/$n; done; "
                   f"test -x {r}/katago/katago && test -x {r}/.venv/bin/python && test -d {r}/cu12 && echo LINKED_OK")
            rc, out = self.ssh(cmd)
            if not self.dry_run and (rc or "LINKED_OK" not in out):
                raise RuntimeError(f"cannot set up {r}: katago/cu12/.venv of {t} missing or unusable: " + out[-200:])

        def sync_code(self):
            self.link_assets()
            super().sync_code()

        def preflight(self) -> dict:
            """Read-only look at the school machine -> {reachable, ours_dir, procs_ours, procs_other, gpu_apps}."""
            rc, out = self.ssh(PREFLIGHT_SH.format(remote=self.remote), 90)
            if rc or "NVRC=" not in out:
                return {"reachable": False, "detail": out[-200:]}
            ours, procs_ours, procs_other, gpu, in_gpu, in_mem, free_mib = "", [], [], [], False, False, None
            for line in out.splitlines():
                if line.startswith("OURS="): ours = line[5:].strip()
                elif line.startswith("PROC "):
                    f = line.split(None, 3); pid, cwd, args = f[1], (f[2] if len(f) > 2 else ""), (f[3] if len(f) > 3 else "")
                    (procs_ours if ours and cwd.startswith(ours) else procs_other).append(f"{pid} {cwd} {args}".strip())
                elif line.startswith("GPU"): in_gpu = True
                elif line.startswith("NVRC="): in_gpu = False
                elif line.startswith("MEMRC="): in_mem = False
                elif line.startswith("MEM"): in_mem = True
                elif in_mem and free_mib is None:      # first GPU only (the job uses GPU 0); unparsable -> stays None
                    m_ = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", line)
                    if m_: free_mib = int(m_.group(1))
                elif in_gpu and line.strip(): gpu.append(line.strip())
            return {"reachable": True, "ours_dir": ours, "procs_ours": procs_ours, "procs_other": procs_other, "gpu_apps": gpu, "gpu_free_mib": free_mib,
                    "nvidia_smi_ok": "NVRC=0" in out}

        def discard_remote(self, sid):
            """Move a result that failed local validation out of out/ (so a re-run redoes it); only OUR dir, only this shard."""
            assert sid.startswith("S") and sid[1:].isdigit()
            self.ssh(f"cd {self.remote} && mkdir -p out/rejected && for f in out/{sid}_*; do [ -e \"$f\" ] && mv \"$f\" out/rejected/; done; rm -rf work/{sid}; true", 60)

    return Hsl1104Backend


def judge_preflight(pf: dict, share_gpu: bool = False, min_free_mib: int = MIN_FREE_MIB) -> tuple[str, str]:
    """-> (verdict, message).  verdict: 'ok' | 'resume' (our own job is already running) | 'refuse' | 'wait' (someone else's GPU program)."""
    if not pf.get("reachable"):
        return "refuse", "cannot reach the school machine (ssh failed); not touching anything: " + pf.get("detail", "")
    if pf["procs_other"]:
        return "refuse", ("our tier-2 supervisor/runner (or another job_school.sh/scan_runner.py of ours) is still active on school: "
                          + " | ".join(pf["procs_other"][:4]) + ".  Wait until tier-2 has finished (docs/experiments/tier2_runbook.md section 8; "
                          "`scan_pool.py sync --backend school` shows ETA) or use the Colab backend")
    if pf["procs_ours"]:
        return "resume", "our own 1104 job is already running; it will be adopted: " + " | ".join(pf["procs_ours"][:2])
    if not pf.get("nvidia_smi_ok"):
        return "wait", "nvidia-smi query failed; treating the GPU as busy"
    if pf["gpu_apps"]:
        apps = "; ".join(pf["gpu_apps"][:3])
        if not share_gpu:
            return "wait", "another user's GPU program is present (" + apps + "); nothing launched, will look again"
        free = pf.get("gpu_free_mib")
        if free is None:
            return "wait", "another user's GPU program is present (" + apps + "); --share-gpu is on but free VRAM could not be read; nothing launched, will look again"
        if free < min_free_mib:
            return "wait", (f"another user's GPU program is present ({apps}); --share-gpu is on but only {free} MiB VRAM is free "
                            f"(need >= {min_free_mib}); nothing launched, will look again")
        return "ok", f"sharing the GPU with another user's program ({apps}); {free} MiB VRAM free (>= {min_free_mib}); that program is never touched"
    return "ok", "no process of ours is active and the GPU is free"


def _todo(root: Path) -> list[str]:
    l = pd.read_parquet(root / "ledger.parquet")
    return l[l["status"].isin(["todo", "failed"])].shard_id.sort_values().tolist()


def run_school(root: Path, hours: float = 3.0, dry_run: bool = False, preflight_only: bool = False, remote: str = SCHOOL_REMOTE,
               poll_seconds: float = 120.0, be=None, sleep=time.sleep, now=time.time, log=None, share_gpu: bool = False) -> int:
    """Rank-test HSL extraction on the school RTX8000.  Exit codes: 0 all shards ingested; 2 refused / unreachable;
    3 a result failed validation (stopped; nothing weakened); 4 runner failed / stalled / unreachable (use Colab); 5 time limit."""
    root = Path(root)
    sd = root / "_school"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "inbox").mkdir(exist_ok=True)

    def say(*a):
        msg = time.strftime("%F %T ") + " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(sd / "school.log", "a") as f:
            f.write(msg + "\n")
    log = log or say
    if not (root / "ledger.parquet").exists():
        raise FileNotFoundError(f"{root}/ledger.parquet missing: run `predict_hsl.py prepare` first")
    pm = json.loads((root / "prepare_meta.json").read_text())
    if pm["config_hash"] != current_config_hash():
        log("REFUSE: prepare_meta config_hash differs from the pinned config; re-run prepare"); return 2
    todo = _todo(root)
    n_games = int(pd.read_parquet(root / "ledger.parquet").query("shard_id in @todo").n_games.sum())
    log(f"{len(todo)} shard(s) to run ({n_games} games): ~{n_games / SCHOOL_GPS / 60:.0f} min engine time at {SCHOOL_GPS} games/s "
        f"(inference from the tier-2 measurement) + a few minutes setup/transfer; no Colab units")
    if not todo:
        log("nothing to do: every shard is done"); return 0
    live = os.environ.get("SCAN_SCHOOL_LIVE") == "1"
    if be is None:
        be = school_backend_cls()(root, remote=remote, dry_run=not (live or preflight_only), share_gpu=share_gpu)
    elif share_gpu and SHARE_GPU_ENV not in be.launch_env:
        be.launch_env += SHARE_GPU_ENV
    if share_gpu:
        log(f"--share-gpu ON: will run next to another user's GPU program if >= {MIN_FREE_MIB} MiB VRAM is free (never stopping or touching it)")
    if dry_run:
        be.dry_run = True
        log("DRY RUN (no ssh is executed): commands that a live run would issue")
        be.ssh(PREFLIGHT_SH.format(remote=be.remote), 90)
        be.start(str(sd), todo); be.poll(); be.fetch(todo[0], str(sd / "inbox")); be.drain(); be.stop()
        for c in be.cmds:
            print("  ", c if isinstance(c, str) else " ".join(map(str, c)))
        return 0
    if not live and not preflight_only:
        log("REFUSE: live school run needs SCAN_SCHOOL_LIVE=1 in the environment"); return 2
    verdict, msg = judge_preflight(be.preflight(), share_gpu)
    log(f"preflight: {verdict}: {msg}")
    if preflight_only or verdict == "refuse":
        return 0 if (preflight_only and verdict != "refuse") else 2
    deadline, started, restarts, bad_polls = now() + hours * 3600, False, 0, 0
    last_prog, last_change, done_seen, grace = None, now(), 0, False
    inbox = sd / "inbox"
    while True:
        if (sd / "STOP").exists():
            be.stop(); log("STOP file: our job stopped; shards already ingested stay ingested"); return 4
        if now() > deadline:
            be.stop(); log(f"time limit {hours} h reached: our job stopped (rerun the same command to continue, or use Colab)"); return 5
        if not started:
            verdict, msg = judge_preflight(be.preflight(), share_gpu)
            if verdict in ("refuse", "wait"):
                log(("REFUSE: " if verdict == "refuse" else "waiting: ") + msg)
                if verdict == "refuse": return 2
                sleep(600); continue
            try:
                be.start(str(sd), _todo(root)); be.drain(); started = True; last_change = now()
                log("job running on school (own dir " + be.remote + ")")
            except Exception as e:
                ph = getattr(e, "phase", "")
                if ph == "FAIL_FOREIGN_GPU":
                    log("FAIL_FOREIGN_GPU: another user's GPU program appeared at launch; nothing touched, retry in 10 min"); sleep(600); continue
                log("launch problem:", repr(e)[:300]); restarts += 1
                if restarts > 2: log("giving up on school: use the Colab backend"); return 4
                sleep(120); continue
        r = be.poll()
        if r is None:
            bad_polls += 1
            if bad_polls >= 10: log("school unreachable for 10 polls; our job (if any) keeps running; rerun later or use Colab"); return 4
            sleep(poll_seconds); continue
        bad_polls = 0
        if r.get("progress") != last_prog: last_prog, last_change = r.get("progress"), now()
        for sid in sorted(set(r["metas"]) & set(_todo(root))):
            if not be.fetch(sid, str(inbox)):
                log(f"{sid}: fetch/sha check failed (will retry next poll)"); continue
            cmd_ingest(root, inbox)
            st = pd.read_parquet(root / "ledger.parquet").set_index("shard_id").loc[sid]
            if st["status"] == "done":
                done_seen += 1; log(f"{sid}: validated + ingested ({done_seen} done)")
                for f in inbox.glob(f"{sid}_*"): f.unlink()
            else:
                rej = sd / "rejected"; rej.mkdir(exist_ok=True)
                for f in inbox.glob(f"{sid}_*"): f.replace(rej / f.name)
                be.discard_remote(sid); be.stop()
                log(f"REJECT {sid}: {str(st['note'])[:300]}.  Not weakening the validation; our job stopped.  Rerun the same command once "
                    f"(environment mismatch) - a second rejection means: use the Colab backend"); return 3
        left = _todo(root)
        if not left:
            log("all shards ingested; our job exits by itself (END queued)"); return 0
        ph = r.get("phase") or ""
        if ph.startswith("FAIL") or ph.startswith("DONE"):
            if ph.startswith("DONE") and not grace:
                grace = True; sleep(poll_seconds); continue                   # one more round to collect the last result
            grace = False; restarts += 1; started = False
            log(f"runner phase {ph} with shards left {left}: restart {restarts}/2")
            if restarts > 2: log("runner keeps failing: use the Colab backend"); return 4
            if ph == "FAIL_FOREIGN_GPU": sleep(600)
            continue
        if now() - last_change > 3600:
            be.stop(); log("no progress line for 60 min: our job stopped; use the Colab backend or rerun once"); return 4
        sleep(poll_seconds)


def cmd_status(root: Path) -> None:
    l = pd.read_parquet(root / "ledger.parquet")
    print(l[["shard_id", "n_games", "status", "platform", "games_per_sec", "note"]].to_string(index=False))
    st = root / "_controller" / "state.json"
    if st.exists():
        s = json.loads(st.read_text())
        print({k: s.get(k) for k in ("session", "phase", "smoke_passed", "halt", "baseline_balance", "last_balance")})
    lg = root / "_controller" / "controller.log"
    if lg.exists():
        print(lg.read_text().strip().splitlines()[-1])
    sl = root / "_school" / "school.log"
    if sl.exists():
        print("school:", sl.read_text().strip().splitlines()[-1])
    print("complete:", bool((l["status"] == "done").all()))


# ================================================================================ features
def _ref_columns() -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_schema(PILOT_HSL).names


def build_features(root: Path, allow_partial: bool = False, train_scan: Path | None = ROOT / "data" / "scan", log=print) -> dict:
    root = Path(root)
    led = pd.read_parquet(root / "ledger.parquet")
    man = pd.read_parquet(root / "manifest.parquet")
    ref = _ref_columns()
    frames, segs, problems_all = [], [], []
    rehearsal = False
    train_meta = None
    if train_scan is not None and (Path(train_scan) / "results").exists():
        led_t = pd.read_parquet(Path(train_scan) / "ledger.parquet")
        done_t = led_t[(led_t["status"] == "done") & (led_t["shard_id"] != "P0000")].shard_id.tolist()
        if done_t:
            train_meta = json.loads((Path(train_scan) / "results" / f"{done_t[0]}_meta.json").read_text())
    cfg_hash_now = current_config_hash()
    for r in led.itertuples(index=False):
        rec = {"shard_id": r.shard_id, "status": r.status, "n_rows": 0, "present": False, "reason": ""}
        segs.append(rec)
        if r.status != "done":
            rec["reason"] = f"ledger status {r.status}"
            continue
        gp, mp = root / "results" / f"{r.shard_id}_games.parquet", root / "results" / f"{r.shard_id}_meta.json"
        if not gp.exists() or not mp.exists():
            rec["reason"] = "ledger done but files missing"
            continue
        g, meta = pd.read_parquet(gp), json.loads(mp.read_text())
        rehearsal |= bool(meta.get("rehearsal_copy"))
        pr = []
        if list(g.columns) != ref:
            pr.append("columns differ from pilot/E06 hsl_games")
        if len(g) != meta.get("n_games_out"):
            pr.append(f"rows {len(g)} != meta n_games_out {meta.get('n_games_out')}")
        if g["uid"].duplicated().any():
            pr.append("duplicate uid")
        if g.drop(columns="uid").isna().any().any():
            pr.append("NaN in features")
        if not set(g["uid"]) <= set(man.loc[man["shard_id"] == r.shard_id, "uid"]):
            pr.append("uid outside shard manifest")
        if meta.get("config_hash") != cfg_hash_now:
            pr.append("config_hash != current pinned config")
        if train_meta is not None:
            diff = [k for k in PROVENANCE_KEYS if meta.get(k) != train_meta.get(k)]
            if diff:
                pr.append(f"provenance differs from the training scan: {diff}")
        exp = man.set_index("uid")["n_own"]
        nm = g.set_index("uid")["n_moves"]
        bad = nm.index[nm.to_numpy() != exp.loc[nm.index].to_numpy()]
        if len(bad):
            pr.append(f"{len(bad)} game(s) with n_moves != min(60, own moves), first {bad[0]}")
        if pr:
            rec["reason"] = "; ".join(pr)
            problems_all.append((r.shard_id, pr))
            continue
        rec.update(n_rows=len(g), present=True)
        frames.append(g)
    seg = pd.DataFrame(segs)
    complete = bool(seg["present"].all())
    if rehearsal and Path(root).resolve() == DEFAULT_ROOT.resolve():
        raise RuntimeError("rehearsal (sim-results) data found in the real test root; refusing")
    if problems_all:
        raise RuntimeError("shard validation failed: " + " | ".join(f"{s}: {p}" for s, p in problems_all))
    if not complete and not allow_partial:
        raise RuntimeError(f"incomplete: missing/not done shards {seg.loc[~seg.present, 'shard_id'].tolist()} "
                           f"(reasons {seg.loc[~seg.present, 'reason'].tolist()}); finish `run` or pass --allow-partial")
    if not frames:
        raise RuntimeError("no accepted shard")
    hsl = pd.concat(frames, ignore_index=True).sort_values("uid", kind="stable").reset_index(drop=True)
    hsl[ref[1:]] = hsl[ref[1:]].astype(np.float32)
    out = root / "hsl_test_games.parquet"
    tmp = out.with_suffix(".tmp")
    hsl.to_parquet(tmp, index=False)
    tmp.replace(out)
    have = set(hsl["uid"])
    q = man.assign(has=man["uid"].isin(have)).groupby("qid").agg(n=("uid", "size"), n_hsl=("has", "sum"))
    pm_p = root / "prepare_meta.json"
    info = {"color_semantics": (json.loads(pm_p.read_text()).get("color_semantics") if pm_p.exists() else None),
            "complete": complete, "rehearsal": rehearsal, "sha256": sha256_file(out), "n_games": int(len(hsl)),
            "n_games_manifest": int(len(man)), "n_games_without_hsl": int(len(man) - len(hsl)),
            "uids_without_hsl": sorted(set(man["uid"]) - have)[:200],
            "skip_reasons_without_hsl": ({k: int(v) for k, v in man.loc[~man["uid"].isin(have), "predicted_skip_reason"].replace("", "unexplained_missing").value_counts().items()}
                                         if "predicted_skip_reason" in man.columns else {}),
            "n_questions": int(len(q)), "questions_without_any_hsl": q.index[q["n_hsl"] == 0].tolist(),
            "questions_partial": int(((q["n_hsl"] > 0) & (q["n_hsl"] < q["n"])).sum()),
            "config_hash": cfg_hash_now, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "columns": ref[1:]}
    (root / "features_manifest.json").write_text(json.dumps(info, indent=1))
    log(f"features: {info['n_games']}/{info['n_games_manifest']} games with HSL, complete={complete}, "
        f"questions without any HSL game: {len(info['questions_without_any_hsl'])}, partial questions: {info['questions_partial']} -> {out}")
    log(f"  games without HSL by reason (no silent drops): {info['skip_reasons_without_hsl'] or 'none'}")
    return info


# ================================================================================ rehearsal (CPU only)
def _scanned_sgf(train_scan: Path) -> pd.DataFrame:
    led = pd.read_parquet(train_scan / "ledger.parquet")
    parts = []
    for sid in led[(led["status"] == "done") & (led["shard_id"] != "P0000")].shard_id:
        d = pd.read_parquet(train_scan / "sgf" / f"{sid}.parquet")
        parts.append(d.assign(src_shard=sid))
    return pd.concat(parts, ignore_index=True)


def match_scanned(root: Path, train_scan: Path) -> pd.DataFrame:
    """Test games whose (colour, sgf) is also a scanned training game: columns test_uid, shard_id, train_uid, src_shard."""
    sg = _scanned_sgf(train_scan)
    sg["k"] = sg["color"].astype(int).astype(str) + "\t" + sg["sgf"]
    sg = sg.drop_duplicates("k")
    man = pd.read_parquet(root / "manifest.parquet")
    sgf_by_uid = {}
    for sid in man["shard_id"].unique():
        d = pd.read_parquet(root / "sgf" / f"{sid}.parquet")
        sgf_by_uid.update(zip(d["uid"], d["color"].astype(int).astype(str) + "\t" + d["sgf"]))
    man["k"] = man["uid"].map(sgf_by_uid)
    mm = man.merge(sg[["k", "uid", "src_shard"]].rename(columns={"uid": "train_uid"}), on="k", how="inner")
    return mm[["uid", "shard_id", "train_uid", "src_shard"]].rename(columns={"uid": "test_uid"})


def sim_results(root: Path, train_scan: Path, out_dir: Path | None = None, log=print) -> dict:
    """Write <root>/inbox_sim/S9*_{moves,games,meta} exactly as the runner would (matched games = scanned, others = skipped)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root)
    if root.resolve() == DEFAULT_ROOT.resolve():
        raise RuntimeError("sim-results must never run on the real test root")
    mm = match_scanned(root, train_scan)
    led = pd.read_parquet(root / "ledger.parquet")
    out_dir = Path(out_dir or root / "inbox_sim")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, tuple] = {}
    n_tot = 0
    for r in led.itertuples(index=False):
        sid = r.shard_id
        mine = mm[mm["shard_id"] == sid]
        gs, ms = [], []
        for src, d in mine.groupby("src_shard"):
            if src not in cache:
                cache[src] = (pd.read_parquet(train_scan / "results" / f"{src}_games.parquet"),
                              pd.read_parquet(train_scan / "results" / f"{src}_moves.parquet"),
                              json.loads((train_scan / "results" / f"{src}_meta.json").read_text()))
            g, mv, meta0 = cache[src]
            mp_ = dict(zip(d["train_uid"], d["test_uid"]))
            gg = g[g["uid"].isin(mp_)].copy()
            gg["uid"] = gg["uid"].map(mp_)
            mm_ = mv[mv["uid"].isin(mp_)].copy()
            mm_["uid"] = mm_["uid"].map(mp_)
            gs.append(gg)
            ms.append(mm_)
        if not gs:
            raise RuntimeError(f"{sid}: no scanned training game matches any test game; cannot rehearse")
        games = pd.concat(gs, ignore_index=True).sort_values("uid").reset_index(drop=True)
        moves = pd.concat(ms, ignore_index=True).sort_values(["uid", "own_move_idx"]).reset_index(drop=True)
        sg_uids = set(pd.read_parquet(root / "sgf" / f"{sid}.parquet")["uid"])
        skipped = sorted(sg_uids - set(games["uid"]))
        meta0 = next(iter(cache.values()))[2]
        meta = {**meta0, "shard_id": sid, "content_hash": r.content_hash, "n_games_in": int(r.n_games), "n_games_out": len(games),
                "n_move_rows": len(moves), "n_skipped": len(skipped), "skipped": [(u, "rehearsal_not_scanned") for u in skipped[:50]],
                "rehearsal_copy": True, "platform": "rehearsal"}
        md = json.dumps(meta, default=str)
        for nm, df in (("moves", moves), ("games", games)):
            tb = pa.Table.from_pandas(df, preserve_index=False)
            tb = tb.replace_schema_metadata({**(tb.schema.metadata or {}), b"scan_meta": md.encode()})
            pq.write_table(tb, out_dir / f"{sid}_{nm}.parquet", compression="zstd")
        (out_dir / f"{sid}_meta.json").write_text(md)
        n_tot += len(games)
        log(f"  {sid}: {len(games)}/{r.n_games} games from copies of scanned training games")
    return {"dir": str(out_dir), "n_games": n_tot}


def crosscheck(root: Path, train_scan: Path, log=print) -> dict:
    """Rehearsal verification (needs prepare + sim-results + ingest + features on a rehearsal root)."""
    import predict as P
    from gogame.questions import Bag
    from gogame.aggregate import aggregate_many
    root = Path(root)
    mm = match_scanned(root, train_scan)
    sg = _scanned_sgf(train_scan).set_index("uid")
    res: dict = {"n_matched_games": int(len(mm))}
    # 1) manifest fields row by row: colour and sgf text of the prepared segment == the training scan segment, same content_hash fn
    t_sgf = pd.concat([pd.read_parquet(root / "sgf" / f"{s}.parquet") for s in sorted(mm["shard_id"].unique())]).set_index("uid")
    a, b = t_sgf.loc[mm["test_uid"]], sg.loc[mm["train_uid"]]
    res["sgf_equal"] = bool((a["sgf"].to_numpy() == b["sgf"].to_numpy()).all())
    res["color_equal"] = bool((a["color"].to_numpy() == b["color"].to_numpy()).all())
    res["dtypes_equal"] = {c: str(t_sgf[c].dtype) == str(sg[c].dtype) for c in ("color", "sgf")}
    res["columns_equal"] = list(t_sgf.reset_index().columns) == list(sg.reset_index().columns[:3])
    sr = _runner()
    import scan_manifest
    res["content_hash_fn_equal"] = sr.content_hash(a.index, a["color"], a["sgf"]) == scan_manifest.content_hash(a.index, a["color"], a["sgf"])
    # 2) per-game HSL rows: features output == training scan rows of the same game
    ft = pd.read_parquet(root / "hsl_test_games.parquet").set_index("uid")
    tr = pd.concat([pd.read_parquet(train_scan / "results" / f"{s}_games.parquet") for s in sorted(mm["src_shard"].unique())]).set_index("uid")
    got, want = ft.loc[mm["test_uid"]], tr.loc[mm["train_uid"]]
    res["hsl_rows_identical"] = bool(np.array_equal(got.to_numpy(), want.to_numpy()))
    res["hsl_columns_identical"] = list(got.columns) == list(want.columns)
    # 3) E06 side: per-game from lib.read_hsl_segments (what e06 `features` writes) and bag aggregation through E06's own construction
    sys.path.insert(0, str(ROOT))
    from scripts.experiments.e06 import lib as e06lib
    e06_hsl, _, _ = e06lib.read_hsl_segments(train_scan, PILOT_HSL)
    e06 = e06_hsl.set_index("uid").loc[mm["train_uid"]]
    res["e06_rows_identical"] = bool(np.array_equal(e06.to_numpy(), got.to_numpy()))
    names = tuple(e06_hsl.columns[1:])
    from gogame.aggregate import L0Store, make_spec
    hs_e06 = L0Store(e06_hsl["uid"].tolist(), e06_hsl[list(names)].to_numpy(np.float32), None, names)           # = pipeline.Ctx.hs
    sp_e06 = make_spec(names, np.zeros(0, np.int32))                                                                # = Ctx.sp_h
    man = pd.read_parquet(root / "manifest.parquet")
    t2t = dict(zip(mm["test_uid"], mm["train_uid"]))
    qb_t, qb_e = [], []
    for qid, d in man[man["uid"].isin(t2t)].groupby("qid", sort=True):
        col = int(d["color"].iloc[0])
        qb_t.append(Bag(qid, "", -1, col, len(d), tuple(d["uid"])))
        qb_e.append(Bag(qid, "", -1, col, len(d), tuple(t2t[u] for u in d["uid"])))
    Xt = aggregate_many(qb_t, P.hsl_store(ft.reset_index()), P.hsl_spec(P.hsl_store(ft.reset_index()).names))     # = HslLgbmRank path
    Xe = aggregate_many(qb_e, hs_e06, sp_e06)
    res["n_bags_compared"] = len(qb_t)
    res["bag_aggregates_identical"] = bool(np.array_equal(Xt, Xe))
    res["bag_columns"] = int(Xt.shape[1])
    res["all_ok"] = all(v is True for k, v in res.items() if isinstance(v, bool)) and all(res["dtypes_equal"].values())
    log(json.dumps(res, indent=1))
    return res


# ================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--color-semantics", choices=["as_annotated", "actual"], default=None,
                    help="color column of HD>=2 games: as_annotated = swapped together with the reversed B/W marks (default); "
                         "actual = real colour, kept.  Also GOGAME_COLOR_SEMANTICS.  Unconfirmed by the organisers (FAQ Q13).")
    ap.add_argument("--strict-format", action="store_true",
                    help="prepare: stop (nothing written) on unknown root properties or a skip rate > 1%% instead of warning")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--shard-target", type=int, default=SHARD_TARGET)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--max-units", type=float, default=12.0)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--backend", choices=["school", "colab"], default="colab")
    p.add_argument("--remote", default=SCHOOL_REMOTE, help="school: own remote work dir (never ~/gogame_scan)")
    p.add_argument("--share-gpu", action="store_true",
                   help="school: allow running next to another user's GPU program when >= 4096 MiB VRAM is free (default off; "
                        "env GOGAME_SHARE_GPU=1 does the same); the other program is never stopped or touched")
    p.add_argument("--poll-seconds", type=float, default=120.0)
    p.add_argument("--preflight-only", action="store_true", help="school: read-only ssh check (own processes? GPU busy?) and exit")
    sub.add_parser("status")
    p = sub.add_parser("ingest")
    p.add_argument("dir")
    p = sub.add_parser("features")
    p.add_argument("--allow-partial", action="store_true")
    p = sub.add_parser("sim-results")
    p.add_argument("--source-scan", default=str(ROOT / "data" / "scan"))
    p = sub.add_parser("crosscheck")
    p.add_argument("--source-scan", default=str(ROOT / "data" / "scan"))
    a = ap.parse_args(argv)
    root = Path(a.root)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "4")
    if a.color_semantics:
        os.environ["GOGAME_COLOR_SEMANTICS"] = a.color_semantics
    from gogame.sgf import get_color_semantics
    get_color_semantics()          # validates the env value early
    if a.cmd in ("run", "features") and (root / "prepare_meta.json").exists():
        pm_ = json.loads((root / "prepare_meta.json").read_text())
        if a.color_semantics and pm_.get("color_semantics") and a.color_semantics != pm_["color_semantics"]:
            ap.error(f"--color-semantics {a.color_semantics} differs from the one used by prepare ({pm_['color_semantics']}); re-run prepare")
        if pm_.get("color_semantics"):
            os.environ["GOGAME_COLOR_SEMANTICS"] = pm_["color_semantics"]
        if pm_.get("format_report"):
            from gogame.sgf import format_report_text
            print(format_report_text(pm_["format_report"], "SGF format report (from prepare)"), flush=True)
    if a.cmd == "prepare":
        prepare(Path(a.data_dir), root, a.shard_target, a.force, strict_format=a.strict_format)
    elif a.cmd == "run":
        if a.backend == "school":
            if a.remote.rstrip("/").endswith("gogame_scan"):
                ap.error("--remote must not be the tier-2 directory ~/gogame_scan")
            return run_school(root, a.hours, a.dry_run, a.preflight_only, a.remote, a.poll_seconds,
                              share_gpu=a.share_gpu or os.environ.get("GOGAME_SHARE_GPU") == "1")
        return cmd_run(root, a.max_units, a.hours, a.dry_run)
    elif a.cmd == "status":
        cmd_status(root)
    elif a.cmd == "ingest":
        cmd_ingest(root, Path(a.dir))
    elif a.cmd == "features":
        build_features(root, a.allow_partial)
    elif a.cmd == "sim-results":
        sim_results(root, Path(a.source_scan))
    elif a.cmd == "crosscheck":
        return 0 if crosscheck(root, Path(a.source_scan))["all_ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
