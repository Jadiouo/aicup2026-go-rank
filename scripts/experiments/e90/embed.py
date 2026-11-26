"""E90 stage 0: per-game vector export from the E89 setting-A checkpoint (no training).
For every game: 128-d position vector of ALL own positions (E89 evaluate(): fastboard encode -> build_planes -> Net.pos_vec, eval mode, no_grad,
fp16 autocast on CUDA), averaged, L2-normalised  (= E89 `gv`, bucket 'all');  plus the same restricted to own-move ordinal < 60 (E89 bucket 'first60').
Imports (read-only) the E89 modules fastboard / train_s so the code path is the one used for E89's own evaluation.

  embed.py games.npz out.npz [--sample-val N --seed S] [--chunk 500] [--cpu]
    games.npz : structure of s_games.npz (moves, off, eff, hd, ...); all games are embedded, or with --sample-val N a random sample (seed S, sorted)
                of the role==1 games of s_games.npz (check B1; also compares top-1 of the classification head with runA/preds.npz val_top5[:,0]).
Partial results are cached per chunk in <out>.chunks/ so the run can be resumed.
"""
import os, sys, time, json, hashlib, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".scratch/e90/numba_cache")); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.modules["coverage"] = None                      # numba needs this stub with the system python (see .scratch/e89/run_local.py)
sys.path[:0] = [str(ROOT / ".scratch/e89/pylib"), str(ROOT / "scripts/experiments/e89")]
import numpy as np, torch, torch.nn.functional as Fn
import fastboard as fb, train_s as ts

CKPT = ROOT / ".scratch/e89/colab/runA/ckpt_latest.pt"; SHA16 = "90c99d7e36bf4533"; E89_DATA = ROOT / ".scratch/e89/data/s_games.npz"


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("games"); ap.add_argument("out"); ap.add_argument("--sample-val", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=500); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--block", type=int, default=512); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    s = sha16(CKPT); print("checkpoint", CKPT, "sha256[:16]", s, "expected", SHA16, flush=True); assert s == SHA16, "checkpoint sha mismatch"
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda"); amp = dev.type == "cuda"; print("device", dev, "amp", amp, flush=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False); print("ckpt epoch", ck["epoch"], "step", ck["step"], flush=True)
    model = ts.Net(5700, 10, 128); model.load_state_dict(ck["model"]); model = model.to(dev).to(memory_format=torch.channels_last).eval()
    G = ts.load_games(a.games)
    if a.sample_val:
        va = np.flatnonzero(G["role"] == 1); ids = np.sort(np.random.default_rng(a.seed).choice(va, a.sample_val, replace=False))
    else:
        ids = np.arange(len(G["rid"]))
    out = Path(a.out); cdir = Path(str(out) + ".chunks"); cdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); npos = 0
    with torch.no_grad():
        for c0 in range(0, len(ids), a.chunk):
            cf = cdir / f"c{c0:07d}.npz"
            if cf.exists(): continue
            cid = ids[c0:c0 + a.chunk]; n = len(cid); sums = torch.zeros(2, n, ts.EMB, device=dev); cnt = torch.zeros(2, n, device=dev)
            for g0 in range(0, n, 16):
                grp = cid[g0:g0 + 16]; b = ts._encode_ids(G, grp, None, flags=False); P = len(b["gid"]); gid = torch.from_numpy(b["gid"]).to(dev) + g0
                t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}; oi = t["oi"].to(dev).long()
                flags = torch.stack([torch.ones_like(oi, dtype=torch.bool), oi < 60])
                for x in range(0, P, a.block):
                    sl = slice(x, min(x + a.block, P)); m = sl.stop - sl.start
                    sub = {k: v[sl] for k, v in t.items()}
                    if m < a.block:
                        for k in ("grid", "lib", "prev_opp", "prev_own", "ko", "move", "ply", "oi", "eff"):
                            sub[k] = torch.cat([sub[k], sub[k][:1].expand(a.block - m, *sub[k].shape[1:])])
                    X = ts.build_planes(sub, dev, torch.float16 if amp else torch.float32)
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                        v = model.pos_vec(X, sub["move"].to(dev).long())[:m]
                    for bk in range(2):
                        mk = flags[bk, sl]; ix = gid[sl][mk]
                        sums[bk].index_add_(0, ix, v[mk]); cnt[bk].index_add_(0, ix, torch.ones(len(ix), device=dev))
                npos += P
            gv = Fn.normalize(sums / cnt.clamp(min=1)[..., None], dim=-1)
            np.savez(cf, gv_all=gv[0].cpu().numpy(), gv_first60=gv[1].cpu().numpy(), has_all=(cnt[0] > 0).cpu().numpy(), has_first60=(cnt[1] > 0).cpu().numpy(), ids=cid)
            print(f"chunk {c0}/{len(ids)} {npos / (time.time() - t0):.0f} pos/s", flush=True)
    parts = [np.load(cdir / f"c{c0:07d}.npz") for c0 in range(0, len(ids), a.chunk)]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    gv_all, gv_f60, has_a, has_f = cat("gv_all"), cat("gv_first60"), cat("has_all"), cat("has_first60"); assert (cat("ids") == ids).all()
    assert has_a.all() and np.isfinite(gv_all).all() and np.abs(np.linalg.norm(gv_all, axis=1) - 1).max() < 1e-3
    n_nof60 = int((~has_f).sum())
    res = dict(ckpt_sha16=s, device=str(dev), amp=amp, n_games=int(len(ids)), n_without_first60=n_nof60, n_positions_this_run=int(npos))
    if a.sample_val:
        W = Fn.normalize(model.W.float(), dim=-1).detach().cpu(); lg = ts.SCALE * torch.from_numpy(gv_all) @ W.T; top1 = lg.argmax(1).numpy()
        P = np.load(ROOT / ".scratch/e89/colab/runA/preds.npz"); pos = np.searchsorted(P["val_ids"], ids); assert (P["val_ids"][pos] == ids).all()
        ref = P["val_top5"][pos, 0]; agree = float((top1 == ref).mean()); acc = float((top1 == G["cls"][ids]).mean())
        top5 = torch.from_numpy(lg.topk(5, dim=1).indices.numpy()).numpy(); agree5 = float((top5 == P["val_top5"][pos]).all(1).mean())
        res.update(check="B1", n=int(len(ids)), seed=a.seed, top1_agreement=agree, top5_list_agreement=agree5, own_top1_acc=acc, preds_top1_acc=float((ref == G["cls"][ids]).mean()), n_disagree=int((top1 != ref).sum()))
        print(json.dumps(res, indent=1))
    np.savez(out, gv_all=gv_all, gv_first60=gv_f60, has_first60=has_f, ids=ids, rid=G["rid"][ids])
    Path(str(out) + ".json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1), "seconds", round(time.time() - t0))


if __name__ == "__main__":
    main()
