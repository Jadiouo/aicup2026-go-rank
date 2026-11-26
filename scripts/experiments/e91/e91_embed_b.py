"""E91 v10b step B: per-game vectors + class table of an E90 `finh_*` network (final_model.pt) for a game file (default use: both sides of the 10,000 Rank test games).
New file; reads (never edits) e91_embed.py / e90/embed.py / train_s.py.  Same encode -> planes -> Net.pos_vec -> mean -> L2 path as E91 step B (it IS e90/embed.py main(), with the
checkpoint path / sha / class count patched, exactly as `e91_final.py embed-test` does), but the class table comes from the NETWORK'S OWN roster (`players` in final_model.pt) and
the player ranks come from the TRAINING CSVs (player_id -> rank), cross-checked against aux.npz and split.parquet.  Needs system python (torch + numba stub); no pandas.

  /usr/bin/python3 e91_embed_b.py labels                                  # .scratch/e91/b/pid_rank.json  (player_id -> rank id, from data/training_v1.1/training/train_*.csv)
  /usr/bin/python3 e91_embed_b.py head  [--net-dir D] [--head F] [--expect-roster R --expect-steps N]
        -> head_b.npz: W [C,128], cls_rank [C], cls_code [C], cls_pid [C], ckpt_sha16, n_classes, n_h_classes ...  Every check failure is an assertion (nothing is written first).
  /usr/bin/python3 e91_embed_b.py embed GAMES.npz OUT.npz [--net-dir D] [--head F] [--limit N] [--cpu] [--threads 2] [--chunk 500] [--expect-roster R --expect-steps N]
        OUT.npz = embed.py's file (gv_all, gv_first60, has_first60, ids, rid).  Resumable (OUT.npz.chunks/).  --limit N: only the first N games (CPU tests).
  /usr/bin/python3 e91_embed_b.py fidelity [--net-dir D] [--n 24]          # re-embed n games of the net's OWN training-time export (vec_*.npz) on CPU and compare (mechanics check)
The H players' ranks are used as labels of LIBRARY classes only; no evaluation number is ever computed from them here.
"""
import os, sys, json, glob, argparse, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
B = ROOT / ".scratch/e91/b"
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".scratch/e91/numba_cache")); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
NET_DEFAULT = ROOT / ".scratch/e90/colab/finh_full"
PID_RANK = B / "pid_rank.json"; HEAD_B = B / "head_b.npz"
AUX = ROOT / ".scratch/e29/bundle/aux.npz"; STATS_H = ROOT / ".scratch/e90/data/data_stats_h.json"; SPLIT_JSON = ROOT / ".scratch/e91/split_ranks.json"
CSV_GLOB = str(ROOT / "data/training_v1.1/training/train_*.csv")
RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")      # = gogame.constants.RANKS (copied: this script runs without the project venv)


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------------------------ labels
def build_labels(force=False):
    """player_id -> rank id (0..9) of EVERY player of the training CSVs (file name = rank; each player must sit in exactly one file).  Cached in pid_rank.json."""
    if PID_RANK.exists() and not force: return json.loads(PID_RANK.read_text())
    res = {}; n_rows = {}
    for f in sorted(glob.glob(CSV_GLOB)):
        r = Path(f).stem.split("train_")[1]; assert r in RANKS, f
        with open(f, encoding="utf-8") as fh:
            next(fh); n = 0
            for line in fh:
                p = line.split(",", 4); assert p[2] == r, (f, p[:3]); n += 1
                if res.setdefault(p[0], RANKS.index(r)) != RANKS.index(r): raise AssertionError(f"player {p[0]} appears under two ranks")
            n_rows[r] = n
    assert len(n_rows) == 10 and len(res) == 8007, (len(n_rows), len(res))
    B.mkdir(parents=True, exist_ok=True); PID_RANK.write_text(json.dumps(res)); print("labels written", PID_RANK, "players", len(res), "games per rank file", n_rows); return res


def class_table(codes):
    """class j -> (player code, player_id, rank).  code < 7,367 = aux player (aux.npz player_ids); code >= 7,367 = H player (data_stats_h.json player_code_map).  Rank = training-CSV rank;
    cross-checked against aux.npz player_rank (aux players) and split.parquet rank_id (all, via the cached split_ranks.json dump)."""
    import numpy as np
    pr_csv = build_labels(); aux = np.load(AUX, allow_pickle=True); aux_ids = aux["player_ids"].astype(str); aux_rank = aux["player_rank"].astype(int); n_aux = len(aux_ids)
    cmap = json.loads(STATS_H.read_text())["player_code_map"]; inv = {int(v): k for k, v in cmap.items()}
    assert len(inv) == len(cmap) == 640 and min(inv) == n_aux and max(inv) == n_aux + 639, "H player_code_map is not the contiguous block after the aux players"
    assert not (set(cmap) & set(aux_ids.tolist())), "an H player is also an aux player"
    sp = json.loads(SPLIT_JSON.read_text()) if SPLIT_JSON.exists() else None; sprk = dict(zip(sp["pid"], sp["rank"])) if sp else {}
    codes = np.asarray(codes, np.int64); assert codes.min() >= 0 and codes.max() < n_aux + 640 and len(np.unique(codes)) == len(codes), "player codes out of range or repeated"
    pids = []; ranks = []; bad = dict(aux_vs_csv=0, split_vs_csv=0, missing_csv=0)
    for c in codes.tolist():
        pid = aux_ids[c] if c < n_aux else inv[c]; pids.append(str(pid))
        if pid not in pr_csv: bad["missing_csv"] += 1; ranks.append(-1); continue
        r = pr_csv[pid]; ranks.append(r)
        if c < n_aux and aux_rank[c] != r: bad["aux_vs_csv"] += 1
        if pid in sprk and sprk[pid] != r: bad["split_vs_csv"] += 1
    assert all(v == 0 for v in bad.values()), f"label cross-check failed {bad}"
    return codes, np.array(pids), np.array(ranks, np.int64), dict(n_aux_classes=int((codes < n_aux).sum()), n_h_classes=int((codes >= n_aux).sum()), crosscheck_failures=bad, split_json_used=bool(sp))


def load_net(net_dir, expect_roster=None, expect_steps=None):
    """final_model.pt + results.json of a finh_* run, with the identity checks.  Returns (ckpt dict, results dict, path)."""
    import torch
    net_dir = Path(net_dir); ck_path = net_dir / "final_model.pt"; rs_path = net_dir / "results.json"
    assert ck_path.exists() and rs_path.exists(), f"{ck_path} / {rs_path} missing"
    rs = json.loads(rs_path.read_text()); ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert "players" in ck and "model" in ck, "final_model.pt has no `players`/`model` (not an E90 final_model.pt)"
    assert rs["roster"] == ck["roster"], (rs["roster"], ck["roster"])
    if expect_roster: assert rs["roster"] == expect_roster, f"roster {rs['roster']} != expected {expect_roster}"
    if expect_steps: assert rs["steps"] == ck["step"] == expect_steps, (rs["steps"], ck["step"], expect_steps)
    assert rs["steps"] == rs["expected_steps"] == ck["step"], "network did not finish all planned steps"
    assert (rs["blocks"], rs["ch"]) == (10, 128), "architecture is not 10x128 (this script hard-codes Net(C, 10, 128) like E91)"
    C = ck["model"]["W"].shape[0]; assert C == rs["n_classes"] == len(ck["players"]) and ck["model"]["W"].shape[1] == 128, (C, rs["n_classes"], len(ck["players"]))
    return ck, rs, ck_path


def export_head(net_dir, out, expect_roster=None, expect_steps=None):
    import numpy as np
    ck, rs, ck_path = load_net(net_dir, expect_roster, expect_steps); W = ck["model"]["W"].float().numpy(); assert np.isfinite(W).all()
    codes, pids, ranks, info = class_table(ck["players"]); C = len(codes)
    cnt = np.bincount(ranks, minlength=10); assert (cnt > 0).all(), f"a rank has no library player {cnt.tolist()}"
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=W, cls_rank=ranks.astype(np.int8), cls_code=codes, cls_pid=pids, ckpt_sha16=sha16(ck_path), roster=rs["roster"], n_classes=C, n_h_classes=info["n_h_classes"], steps=rs["steps"])
    print("head exported", out, "roster", rs["roster"], "classes", C, "H classes", info["n_h_classes"], "rank counts", cnt.tolist(), "crosscheck", info["crosscheck_failures"], flush=True)
    return dict(path=str(out), roster=rs["roster"], n_classes=C, rank_counts=cnt.tolist(), ckpt_sha16=sha16(ck_path), **info)


# ------------------------------------------------------------------------------------------------------------------------------ embed
def embed(games, out, net_dir, head, limit, cpu, threads, chunk, expect_roster=None, expect_steps=None):
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts/experiments/e90"))
    import embed as e90, train_s as ts                       # embed.py sets sys.path (fastboard / train_s), numba stub, and sha-checks CKPT against SHA16 (patched below)
    ck, rs, ck_path = load_net(net_dir, expect_roster, expect_steps); C = ck["model"]["W"].shape[0]
    head = Path(head)
    if not head.exists(): export_head(net_dir, head, expect_roster, expect_steps)
    h = np.load(head, allow_pickle=True); assert int(h["n_classes"]) == C and h["W"].shape == (C, 128) and str(h["ckpt_sha16"]) == sha16(ck_path), "head file does not belong to this checkpoint"
    assert np.array_equal(h["W"], ck["model"]["W"].float().numpy())
    src = Path(games)
    if limit:
        import e91_prep as P; d = P.load(src); small = Path(out).with_name(Path(out).stem + f"_games_lim{limit}.npz"); np.savez(small, **P.subset(d, np.arange(limit))); src = small
    Orig = ts.Net; ts.Net = lambda n, r, d: Orig(C, r, d); e90.CKPT = ck_path; e90.SHA16 = sha16(ck_path)
    sys.argv = ["embed.py", str(src), str(out), "--chunk", str(chunk), "--threads", str(threads)] + (["--cpu"] if cpu else []); e90.main()
    o = np.load(out); gv = o["gv_all"]; assert gv.shape == (len(o["rid"]), 128) and np.isfinite(gv).all() and np.abs(np.linalg.norm(gv, axis=1) - 1).max() < 1e-3
    print("embed OK", out, gv.shape, flush=True)


def fidelity(net_dir, n):
    """Mechanics check: first n games of the net's own training-time export vec_*.npz (GPU fp16 inside train_e90h.py) vs this script's CPU fp32 re-embedding of the same rows."""
    import numpy as np, subprocess
    net_dir = Path(net_dir); vf = sorted(net_dir.glob("vec_*.npz")); assert vf, "no vec_*.npz in the net dir"
    v = np.load(vf[0]); ids = v["ids"][:n]; assert str(v["games_file"]) == "u"
    sys.path.insert(0, str(ROOT / "scripts/experiments/e91")); import e91_prep as P
    d = P.load(ROOT / ".scratch/e90/data/uh_games.npz"); small = B / "dry" / f"fid_games_{net_dir.name}.npz"; sub = P.subset(d, ids); np.savez(small, **sub)
    assert (sub["rid"] == v["rid"][:n]).all(); del d
    o = B / "dry" / f"fid_S_{net_dir.name}.npz"
    for p in (o,): (p.exists() and p.unlink())
    embed(small, o, net_dir, B / "dry" / f"fid_head_{net_dir.name}.npz", 0, True, 2, 500, None, None)
    a = np.load(o)["gv_all"]; b = v["gv_all"][:n]; cos = (a * b).sum(1)
    res = dict(net=net_dir.name, n=int(n), cos_min=float(cos.min()), cos_median=float(np.median(cos)), cos_max=float(cos.max())); print("fidelity", json.dumps(res)); return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["labels", "head", "embed", "fidelity"]); ap.add_argument("games", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--net-dir", default=str(NET_DEFAULT)); ap.add_argument("--head", default=str(HEAD_B)); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--chunk", type=int, default=500); ap.add_argument("--expect-roster"); ap.add_argument("--expect-steps", type=int); ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    if a.mode == "labels": build_labels(force=True)
    elif a.mode == "head": export_head(a.net_dir, a.head, a.expect_roster, a.expect_steps)
    elif a.mode == "embed":
        assert a.games and a.out, "embed GAMES.npz OUT.npz"; embed(a.games, a.out, a.net_dir, a.head, a.limit, a.cpu, a.threads, a.chunk, a.expect_roster, a.expect_steps)
    else: fidelity(a.net_dir, a.n)
