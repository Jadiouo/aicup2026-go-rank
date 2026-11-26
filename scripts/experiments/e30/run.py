"""E30 driver.  Usage:
  run.py check                      pipeline checks (S0 reproduction vs player_sim3 eval_scores, N=1,2,3 finite)
  run.py tune [scn ...]  [--reps 16]   tune scenarios (tuneA)         -> .scratch/e30/tune_<scn>.parquet / _stats.parquet
  run.py eval [scn ...]  [--reps 20]   eval scenarios (dev, S0..S7)    -> .scratch/e30/eval_<scn>.parquet / _stats.parquet
Resumable per rep.  Protocol: experiments/E30_player_robustness/PLAN.md"""
import sys, time, json
from pathlib import Path
import common as C
import numpy as np, pandas as pd

kind = sys.argv[1]
args = [a for a in sys.argv[2:] if not a.startswith("--")]
reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else (16 if kind == "tune" else 20)
if "--reps" in sys.argv:
    args = [a for a in args if a != str(reps)]


def check():
    model = C.get_model()
    D = C.Data()
    print("dev", len(D.dev), "tuneA", len(D.tuneA), "tuneB", len(D.tuneB), flush=True)
    ds = C.R.get_ds("main")
    pools = C.s0_pools(D, ds, 2)
    old = pd.read_parquet(C.ROOT / ".scratch" / "player_sim3" / "eval_scores.parquet")
    for p in pools:
        q, st = C.run_pool(p, model, D.sel, gammas=(2.0,))
        o = old[old.pool == p.rep]
        print("S0 pool", p.rep, "mine B", q.B.mean(), "old base", o.base.mean(), "max|diff|", float(np.abs(q.B.to_numpy() - o.base.to_numpy()).max()),
              "n equal", bool((q.n.to_numpy() == o.n.to_numpy()).all()), "A", q.A.mean(), "stats", {k: round(v, 4) for k, v in st.items() if isinstance(v, float)}, flush=True)
    # N=1,2,3 finite check on real vectors
    rng = np.random.default_rng(0)
    lib = [D.z(D.black[p][:40]) for p in D.dev[:60]]
    for n in (1, 2, 3):
        qz = [D.z(D.black[p][40:40 + n]) for p in D.dev[:60]]
        f = C.prr.candidate_features(lib, qz, D.sel, C.prr.K, C.prr.N_DENSE)
        e = model.predict(f["X"], num_threads=4)
        print("N", n, "X finite", bool(np.isfinite(f["X"]).all()), "e finite", bool(np.isfinite(e).all()), "e std", float(e.std()), flush=True)


def runner():
    model = C.get_model()
    D = C.Data()
    if kind == "tune":
        scn = C.tune_scenarios(D)
    else:
        scn = C.eval_scenarios(D)
        scn["S0"] = None
    names = args or list(scn)
    ds0 = None
    for name in names:
        fq, fs = C.SCR / f"{kind}_{name}.parquet", C.SCR / f"{kind}_{name}_stats.parquet"
        qs = [pd.read_parquet(fq)] if fq.exists() else []
        ss = pd.read_parquet(fs).to_dict("records") if fs.exists() else []
        done = {s["rep"] for s in ss}
        if name == "S0":
            ds0 = ds0 or C.R.get_ds("main")
            gen = ((p.rep, p) for p in C.s0_pools(D, ds0, reps))
        else:
            gen = ((r, None) for r in range(1, reps + 1))
        t0 = time.perf_counter()
        for rep, pool in gen:
            if rep in done:
                continue
            if pool is None:
                pool = scn[name](rep)
            q, st = C.run_pool(pool, model, D.sel)
            qs.append(q); ss.append(st)
            pd.concat(qs, ignore_index=True).to_parquet(fq); pd.DataFrame(ss).to_parquet(fs)
            print(kind, name, "rep", rep, f"nq={len(pool.n)} nc={len(pool.cand)} A={st['A_score']:.4f} B={st['B_score']:.4f} c_all={st['c_all']:.3f} {time.perf_counter() - t0:.0f}s", flush=True)


{"check": check}.get(kind, runner)()
