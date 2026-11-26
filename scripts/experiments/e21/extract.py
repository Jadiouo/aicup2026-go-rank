"""E21 extraction on the school GPU: SE `score` per game.
Design (commander, fixed): per game 32 own-colour (csv `color`) positions, evenly spread from first to last own move,
board BEFORE the move (pre); per game save mean / std (ddof 0) / count of the per-position `score`.
  run   --shards DIR --out DIR   resumable: a shard is done iff out/se_<name>.parquet exists (atomic rename)
  bench --sanity PQ --sanity-csv CSV --model PT   fp32 vs fp16, speed, local-CPU comparison (n=30)
"""
from __future__ import annotations
import argparse, os, sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p in (HERE, os.environ.get("E21_SRC", str(HERE / "src"))):
    sys.path.insert(0, p)
import numpy as np, pandas as pd, torch
import se_encode as se
from gogame.sgf import parse_sgf


class GameDS(torch.utils.data.Dataset):
    def __init__(self, df, n_max, opp=False):
        self.uid, self.col, self.sgf, self.n = df.uid.tolist(), ((1 - df.color.astype(int)) if opp else df.color.astype(int)).tolist(), df.sgf.tolist(), n_max
    def __len__(self): return len(self.uid)
    def __getitem__(self, i):
        try:
            x = se.encode_game(self.sgf[i], self.col[i], "pre", self.n)
        except Exception:
            x = np.zeros((0, 18, 19, 19), np.float32)
        return i, torch.from_numpy(x)


def collate(b):
    return [i for i, _ in b], [len(x) for _, x in b], torch.cat([x for _, x in b])


@torch.inference_mode()
def score_df(model, df, n_max, half, workers, gb=48, sub=512, rec=None, opp=False):
    """returns arrays mean,std,n aligned with df; rec: dict to accumulate timing."""
    dl = torch.utils.data.DataLoader(GameDS(df, n_max, opp), batch_size=gb, num_workers=workers, collate_fn=collate, prefetch_factor=4 if workers else None)
    mean = np.full(len(df), np.nan); std = np.full(len(df), np.nan); cnt = np.zeros(len(df), int)
    npos = 0; tg = 0.0
    for idx, ns, x in dl:
        outs = []
        t = time.time()
        for k in range(0, len(x), sub):
            xb = x[k:k + sub].cuda(non_blocking=True)
            if half: xb = xb.half()
            outs.append(model(xb)["score"].float().squeeze(1).cpu())
        torch.cuda.synchronize(); tg += time.time() - t
        s = torch.cat(outs).numpy().astype(np.float64) if outs else np.zeros(0)
        npos += len(s); o = 0
        for i, n in zip(idx, ns):
            if n > 0:
                v = s[o:o + n]; mean[i] = v.mean(); std[i] = v.std(); cnt[i] = n
            o += n
    if rec is not None: rec.update(npos=npos, gpu_s=tg)
    return mean, std, cnt


def load_model(path, half):
    m = torch.jit.load(path, map_location="cuda").eval()
    return m.half() if half else m


def cmd_run(a):
    shards = sorted(Path(a.shards).glob("shard_*.parquet")); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.model, a.half)
    t0 = time.time(); done_pos = 0
    for sp in shards:
        o = out / f"se_{sp.stem}.parquet"
        if o.exists(): continue
        if a.stop_at and time.time() > a.stop_at:
            print("STOP_AT reached, exiting before", sp.name, flush=True); return 3
        df = pd.read_parquet(sp); rec = {}
        t = time.time()
        m, s, n = score_df(model, df, a.n, a.half, a.workers, rec=rec, opp=a.opp)
        res = pd.DataFrame({"uid": df.uid.values, "se_mean": m, "se_std": s, "se_n": n.astype(np.int16)})
        tmp = out / f".tmp_{sp.stem}.parquet"; res.to_parquet(tmp, index=False); os.replace(tmp, o)
        done_pos += rec["npos"]
        print(f"{sp.name} games={len(df)} pos={rec['npos']} nan={int(np.isnan(m).sum())} {time.time()-t:.0f}s "
              f"pos/s_total={rec['npos']/(time.time()-t):.0f} gpu_pos/s={rec['npos']/rec['gpu_s']:.0f} elapsed={time.time()-t0:.0f}s", flush=True)
    print("ALL_DONE", flush=True); return 0


def cmd_bench(a):
    from scipy.stats import spearmanr
    df = pd.read_parquet(a.sanity); ref = pd.read_csv(a.sanity_csv)
    df["gid"] = df.uid.str.split("_", n=1).str[1]
    res = {}
    for half in (False, True):
        model = load_model(a.model, half)
        score_df(model, df.iloc[:8], a.n, half, 0)            # warmup
        rec = {}; t = time.time()
        m, s, n = score_df(model, df, a.n, half, a.workers, rec=rec)
        res[half] = (m, rec, time.time() - t); del model; torch.cuda.empty_cache()
        print(f"half={half} npos={rec['npos']} wall={time.time()-t:.1f}s pos/s_total={rec['npos']/(time.time()-t):.0f} gpu_pos/s={rec['npos']/rec['gpu_s']:.0f} peakVRAM={torch.cuda.max_memory_allocated()/2**20:.0f}MiB", flush=True)
    m32, m16 = res[False][0], res[True][0]
    print("fp16 vs fp32 max|diff| of per-game mean:", float(np.abs(m32 - m16).max()), "spearman:", spearmanr(m32, m16)[0])
    r = df.merge(ref[["game_id", "rank", "own_pre", "own_pre_n"]].assign(uid=lambda d: d["rank"] + "_" + d.game_id), on="uid", validate="1:1")
    ix = pd.Series(range(len(df)), index=df.uid)[r.uid].values
    d32 = np.abs(m32[ix] - r.own_pre.values); d16 = np.abs(m16[ix] - r.own_pre.values)
    print(f"vs local CPU sanity (n=30): fp32 max|diff|={d32.max():.2e} mean={d32.mean():.2e}; fp16 max|diff|={d16.max():.2e}; count={len(r)}")
    return 0


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("run", "bench"):
        p = sub.add_parser(nm); p.add_argument("--model", required=True); p.add_argument("--n", type=int, default=32)
        p.add_argument("--workers", type=int, default=3); p.add_argument("--opp", action="store_true", help="score the opponent (1-color) side"); p.add_argument("--half", action="store_true")
        if nm == "run": p.add_argument("--shards", required=True); p.add_argument("--out", required=True); p.add_argument("--stop-at", type=float, default=0)
        else: p.add_argument("--sanity", required=True); p.add_argument("--sanity-csv", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    sys.exit(cmd_run(a) if a.cmd == "run" else cmd_bench(a))

if __name__ == "__main__":
    main()
