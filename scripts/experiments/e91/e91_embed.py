"""E91 step B: per-game vectors of the S network (E89 setting A) for a game file, plus what the classification head says about them.
Thin wrapper around scripts/experiments/e90/embed.py (imported read-only; same checkpoint sha check, same encode -> planes -> Net.pos_vec path,
same gv = L2-normalised mean of all positions of the side).  It does NOT change any logic of embed.py; it only (1) sets the numba cache dir to .scratch/e91,
(2) passes the command line through, (3) afterwards computes from the saved gv and the head W (numpy only): top-20 library classes + probabilities, p_max
and the 10-rank histogram, with the head exactly as E89 evaluates it: logits = 30 * cos(gv, W_j), softmax (T = 1).

  /usr/bin/python3 e91_embed.py head                                   # export head (W, class -> player -> rank) to .scratch/e91/head.npz  (CPU, seconds)
  /usr/bin/python3 e91_embed.py embed GAMES.npz OUT.npz [--cpu] [--threads 2] [--chunk 500] [--block 512]
        OUT.npz = embed.py's file (gv_all, gv_first60, has_first60, ids, rid);  OUT_head.npz = top20_cls/top20_p/pmax/rank_hist (+ n_games).
        embed.py skips chunks already cached in OUT.npz.chunks/, so an interrupted run can be resumed.
  /usr/bin/python3 e91_embed.py post OUT.npz                           # recompute OUT_head.npz from OUT.npz (no model run)
  /usr/bin/python3 e91_embed.py cmp A.npz B.npz                        # cosine of gv_all rows (e.g. the own-side re-embedding of 200 games vs S_r0.npz rows)
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".scratch/e91/numba_cache"))
sys.path.insert(0, str(ROOT / "scripts/experiments/e90"))
import numpy as np
SCALE = 30.0; HEAD = ROOT / ".scratch/e91/head.npz"


def export_head():
    import embed as e90                      # noqa: F401  (sets sys.path for fastboard/train_s, numba stub)
    import torch
    ck = torch.load(e90.CKPT, map_location="cpu", weights_only=False)
    assert e90.sha16(e90.CKPT) == e90.SHA16
    W = ck["model"]["W"].float().numpy(); assert W.shape == (5700, 128)
    S = np.load(e90.E89_DATA); aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=True)
    cls, pl = S["cls"].astype(np.int64), S["player"].astype(np.int64)
    cls_player = np.full(5700, -1, np.int64); cls_player[cls] = pl
    assert (cls_player >= 0).all() and (cls_player[cls] == pl).all(), "class -> player map inconsistent"
    pr = aux["player_rank"].astype(np.int64); pf = aux["player_fold"]; cls_rank = pr[cls_player]
    assert (pf[cls_player] <= 3).all() and len(np.unique(cls_player)) == 5700
    HEAD.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HEAD, W=W, cls_player=cls_player, cls_rank=cls_rank.astype(np.int8), ckpt_sha16=e90.SHA16)
    print("head exported", HEAD, "rank counts", np.bincount(cls_rank, minlength=10).tolist())


def head_stats(gv, W, chunk=2000):
    """gv [n,128] unit vectors -> dict(top20_cls int16, top20_p float32, pmax float32, rank_hist float32 [n,10]).  Pure numpy, softmax(30 cos)."""
    h = np.load(HEAD); cls_rank = h["cls_rank"].astype(np.int64)
    Wn = (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32); n = len(gv)
    t_c = np.zeros((n, 20), np.int16); t_p = np.zeros((n, 20), np.float32); hist = np.zeros((n, 10), np.float32)
    onehot = np.zeros((5700, 10), np.float32); onehot[np.arange(5700), cls_rank] = 1
    for a in range(0, n, chunk):
        lg = SCALE * (gv[a:a + chunk].astype(np.float32) @ Wn.T); lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
        top = np.argpartition(-p, 19, axis=1)[:, :20]; tp = np.take_along_axis(p, top, 1); o = np.argsort(-tp, 1)
        t_c[a:a + chunk] = np.take_along_axis(top, o, 1); t_p[a:a + chunk] = np.take_along_axis(tp, o, 1); hist[a:a + chunk] = p @ onehot
    return dict(top20_cls=t_c, top20_p=t_p, pmax=t_p[:, 0].copy(), rank_hist=hist)


def post(out):
    d = np.load(out); st = head_stats(d["gv_all"], np.load(HEAD)["W"])
    assert np.allclose(st["rank_hist"].sum(1), 1, atol=1e-3)
    p = Path(str(out)[:-4] + "_head.npz"); np.savez(p, rid=d["rid"], n_games=len(d["rid"]), **st); print("wrote", p, "pmax quantiles", np.quantile(st["pmax"], [.1, .5, .9, .99]).round(3).tolist())


def main():
    m = sys.argv[1]
    if m == "head": export_head()
    elif m == "embed":
        import embed as e90
        if not HEAD.exists(): export_head()
        sys.argv = ["embed.py"] + sys.argv[2:]; e90.main(); post(Path(sys.argv[2]))
    elif m == "post": post(sys.argv[2])
    elif m == "cmp":
        a, b = np.load(sys.argv[2]), np.load(sys.argv[3]); ra = a["rid"]; rb = b["rid"]
        ia = np.searchsorted(ra, rb) if len(ra) > len(rb) else np.arange(len(ra)); ib = np.arange(len(rb)) if len(ra) > len(rb) else np.searchsorted(rb, ra)
        A = a["gv_all"][ia]; B = b["gv_all"][ib]; assert (ra[ia] == rb[ib]).all()
        cos = (A * B).sum(1); print(json.dumps(dict(n=int(len(cos)), cos_min=float(cos.min()), cos_median=float(np.median(cos)), cos_max=float(cos.max()))))
    else: raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
