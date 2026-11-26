"""E96 export of the validation network `finh_v4`: class table (head) and per-game vectors of the fold-4 validation games, both sides.
Thin wrapper around scripts/experiments/e91/e91_embed_b.py (imported, unmodified: it IS the E91 v10b path -> e90/embed.py main() with checkpoint path / sha / class count patched, class -> player -> rank from the network's
own `players` list + training-CSV ranks, cross-checked against aux.npz and split.parquet).  Adds only: the fold-4 guard (no class may be a fold-4 player: .scratch/e96/fold4_pids.json), roster / steps / class-count
expectations, row alignment (rid) with the games file.  System python (torch + numba stub), no pandas.

  /usr/bin/python3 e96_embed.py head  --net-dir .scratch/e96/run_v4 --head .scratch/e96/head_v4.npz [--expect-roster finh_v4 --expect-steps 67020 --expect-classes 6039]
  /usr/bin/python3 e96_embed.py embed GAMES.npz OUT.npz --net-dir ... --head ... [--limit N] [--cpu] [same expectations]
      GAMES = .scratch/e90/stage0r/r0_games.npz (own side, = what E91 called S_r0) or .scratch/e91/opp_games.npz (opponent side); OUT = S_own_v4.npz / S_opp_v4.npz  (gv_all [41,589,128], rid, ... + .json).
"""
import os, sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".scratch/e96/numba_cache")); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/experiments/e91"))
import numpy as np
import e91_embed_b as B

FOLD4 = ROOT / ".scratch/e96/fold4_pids.json"


def guard_head(head, expect_classes=None):
    f4 = json.loads(FOLD4.read_text()); h = np.load(head, allow_pickle=True); pids = set(h["cls_pid"].astype(str).tolist()); codes = set(h["cls_code"].astype(int).tolist())
    assert len(f4["pid"]) == len(f4["code"]) == 1469
    assert not (pids & set(f4["pid"])), "a fold-4 player is a class of the head - STOP"
    assert not (codes & set(f4["code"])), "a fold-4 player code is a class of the head - STOP"
    assert len(pids) == len(codes) == int(h["n_classes"]) == h["W"].shape[0], "classes <-> players not one-to-one"
    if expect_classes: assert int(h["n_classes"]) == expect_classes, (int(h["n_classes"]), expect_classes)
    print(f"fold-4 guard OK: head has {int(h['n_classes'])} classes ({int(h['n_h_classes'])} H), 0 fold-4 players; rank counts {np.bincount(h['cls_rank'].astype(int), minlength=10).tolist()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["head", "embed"]); ap.add_argument("games", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--net-dir", default=str(ROOT / ".scratch/e96/run_v4")); ap.add_argument("--head", default=str(ROOT / ".scratch/e96/head_v4.npz")); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--expect-roster", default=None); ap.add_argument("--expect-steps", type=int, default=None); ap.add_argument("--expect-classes", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "head":
        B.export_head(a.net_dir, a.head, a.expect_roster, a.expect_steps); guard_head(a.head, a.expect_classes)
    else:
        assert a.games and a.out, "embed GAMES.npz OUT.npz"
        if not Path(a.head).exists(): B.export_head(a.net_dir, a.head, a.expect_roster, a.expect_steps)
        guard_head(a.head, a.expect_classes)
        B.embed(a.games, a.out, a.net_dir, a.head, a.limit, a.cpu, a.threads, a.chunk, a.expect_roster, a.expect_steps)
        g = np.load(a.games); o = np.load(a.out); n = a.limit or len(g["rid"])
        assert (o["rid"] == g["rid"][:n]).all() and o["gv_all"].shape == (n, 128), "rows of the vector file are not aligned 1:1 with the games file"
        print("rows aligned with", a.games, o["gv_all"].shape, flush=True)
