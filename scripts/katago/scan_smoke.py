"""Smoke gate for a fresh scan platform: 120 pilot games re-run with the exact runner path, twice with FRESH engines
(A, B; random symmetry, no shared nn cache), compared with the stored pilot moves.
Pass if pilot-vs-A deviation is within symmetry noise: it must not exceed the A-vs-B deviation by more than 25%
(move-level mean abs logp diff), and game-level argmax agreement must be >= A-vs-B agreement - 0.05, no NaN, n_illegal == 0.
Usage: scan_smoke.py --katago-dir D --bench-dir DIR --out smoke.json [--ngames 120]. Exit 1 on failure."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from kgio import Engine
from scan_runner import process_chunk, cast_moves

ap = argparse.ArgumentParser()
ap.add_argument("--katago-dir", required=True); ap.add_argument("--bench-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cfg", default=f"{HERE}/analysis.cfg"); ap.add_argument("--cu12", default=os.environ.get("CU12_DIR"))
ap.add_argument("--ngames", type=int, default=120)
a = ap.parse_args()
pins = json.load(open(f"{HERE}/scan_pins.json")); kd = a.katago_dir; P = 11
games = pd.read_parquet(f"{a.bench_dir}/bench_games.parquet").iloc[:a.ngames]
pil = pd.read_parquet(f"{a.bench_dir}/bench_pilot_moves.parquet")
key = ["uid", "own_move_idx"]; LC = [f"logp_{i}" for i in range(P)]


def run_pass(tag):
    e = Engine(f"{kd}/katago", f"{kd}/{pins['main_net']['name'][:-3]}/model.bin", f"{kd}/{pins['human_net']['name'][:-3]}/model.bin", a.cfg,
               pins["engine_overrides"], log=f"/content/scan/logs/smoke_katago_{tag}.log", cu12=a.cu12)
    t0 = time.time(); mvs = []; ne = 0
    for s in range(0, len(games), 50):
        mv, n, dt, _ = process_chunk(e, games.iloc[s:s + 50], pins); mvs.append(mv); ne += n
    e.close()
    mv = cast_moves(pd.concat(mvs), P).sort_values(key).reset_index(drop=True)
    return mv, ne / (time.time() - t0)


def cmp(x, y):
    ix = x.set_index(key); iy = y.set_index(key); idx = ix.index.intersection(iy.index)
    X = ix.loc[idx, LC].to_numpy(np.float64); Y = iy.loc[idx, LC].to_numpy(np.float64)
    g = idx.get_level_values(0)
    gx = pd.DataFrame(X).groupby(g).mean().to_numpy(); gy = pd.DataFrame(Y).groupby(g).mean().to_numpy()
    cx = gx - gx.mean(1, keepdims=True); cy = gy - gy.mean(1, keepdims=True)
    return {"n": int(len(idx)), "move_mad": float(np.abs(X - Y).mean()), "move_rmse": float(np.sqrt(((X - Y) ** 2).mean())),
            "game_centered_rmse": float(np.sqrt(((cx - cy) ** 2).mean())), "game_argmax_agree": float((gx.argmax(1) == gy.argmax(1)).mean())}


A, eps_a = run_pass("a"); B, eps_b = run_pass("b")
r = {"ab": cmp(A, B), "pilot_a": cmp(A, pil), "pilot_b": cmp(B, pil), "evals_per_sec": [eps_a, eps_b], "n_games": len(games),
     "n_illegal": [int(A.n_illegal.sum()), int(B.n_illegal.sum())], "nan": bool(A[LC].isna().any().any() or B[LC].isna().any().any())}
ab = r["ab"]; ok = not r["nan"] and r["n_illegal"] == [0, 0]
for k in ("pilot_a", "pilot_b"):
    ok &= r[k]["n"] == ab["n"] and r[k]["move_mad"] <= 1.25 * ab["move_mad"] and r[k]["game_argmax_agree"] >= ab["game_argmax_agree"] - 0.05
r["pass"] = bool(ok)
json.dump(r, open(a.out, "w"), indent=1); print(json.dumps(r, indent=1))
sys.exit(0 if ok else 1)
