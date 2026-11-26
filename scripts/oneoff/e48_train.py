"""E48 set-model training (E29 setmodel unchanged, imported).  Configs: C0 CE/139 cols, C1 soft-bootstrap/139, C2 CE/140 (+lastown), C3 boot/140.
Width per fold = v6 A1full chosen_W (not re-selected); seeds 0-2; saves E29-format models, setprobs npz (val bags), norm, select json under .scratch/e48/out/<cfg>/.
  E29_HOME=.scratch/e48/home139 PYTHONPATH=src:. python3 scripts/oneoff/e48_train.py --cfg C0 C1 [--folds 0 1 2 3 4] [--seeds 0 1 2]   (home140 for C2 C3)"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import torch
from scripts.experiments.e29 import setmodel as sm
OUTR = ROOT / ".scratch/e48/out"; A1 = ROOT / ".scratch/e37/runs/v6/out/A1full"
CFG = {"C0": ("ce", 139), "C1": ("boot", 139), "C2": ("ce", 140), "C3": ("boot", 140)}

def loss_fn(mode, lp, yb):
    if mode == "ce": return -lp.gather(1, yb[:, None]).mean()
    t = 0.7 * torch.nn.functional.one_hot(yb, sm.K).float() + 0.3 * lp.detach().exp()      # soft bootstrapping (E43 'boot')
    return -(t * lp).sum(1).mean()

def train_var(D, W, mode, seed, hp=sm.HP):
    kind = "ds"; torch.manual_seed(seed * 1009 + sm.KIND_ID[kind] * 31 + W); rng = np.random.default_rng(seed * 7919 + sm.KIND_ID[kind] * 131 + W + 1_000_000 * D.fold)
    model = sm.build(kind, D.F, W, hp["drop"]); npar = sum(p.numel() for p in model.parameters()); opt = sm.make_opt(model, hp["lr"], hp["wd"])
    best = {"nll": float("inf"), "state": None, "step": 0}; bad = 0; hist = []; t0 = time.perf_counter(); chunk = hp["eval_every"] * hp["batch"]
    for step in range(1, hp["max_steps"] + 1):
        if (step - 1) % hp["eval_every"] == 0:
            rows_np, y_np = D.sampler.sample(rng, chunk); rows_c, y_c = torch.from_numpy(rows_np).to(sm.DEV), torch.from_numpy(y_np).to(sm.DEV)
        j = ((step - 1) % hp["eval_every"]) * hp["batch"]; model.train(); x, m = D.gather(rows_c[j:j + hp["batch"]]); yb = y_c[j:j + hp["batch"]]
        lp = model(x, m); loss = loss_fn(mode, lp, yb); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % hp["eval_every"] == 0:
            em = sm.es_metrics(model, D); hist.append({"step": step, "nll": em["nll"], "score": em["score"]})
            if em["nll"] < best["nll"] - 1e-5: best = {"nll": em["nll"], "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "step": step}; bad = 0
            else:
                bad += 1
                if bad >= hp["patience"]: break
    model.load_state_dict(best["state"]); em = sm.es_metrics(model, D)
    return model, {"kind": kind, "W": W, "mode": mode, "seed": seed, "fold": D.fold, "n_params": npar, "best_step": best["step"], "steps_run": step, "es": em, "hist": hist, "seconds": round(time.perf_counter() - t0),
                   "status": "TRAIN_FAIL" if (em["modal_share"] > 0.5 or em["n_pred_classes"] <= 2) else "OK"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", nargs="+", required=True); ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); a = ap.parse_args()
    for k in a.folds:
        D = sm.Data(k)
        for c in a.cfg:
            mode, F = CFG[c]; assert D.F == F, (c, D.F, F)
            out = OUTR / c; (out / "models").mkdir(parents=True, exist_ok=True)
            sel = json.loads((A1 / f"select_ds_fold{k}.json").read_text()); W = sel["chosen_W"]
            (out / f"select_ds_fold{k}.json").write_text(json.dumps({**sel, "note": "chosen_W copied from v6 A1full (not re-selected)"}, indent=1)); np.savez(out / f"norm_fold{k}.npz", **D.norm)
            for s in a.seeds:
                f = out / f"setprobs_ds_W{W}_fold{k}_s{s}.npz"
                if f.exists(): continue
                model, info = train_var(D, W, mode, s)
                torch.save({"state": {kk: v.detach().cpu() for kk, v in model.state_dict().items()}, "kind": "ds", "W": W, "seed": s, "fold": k, "F": D.F, "info": {x: info[x] for x in ("best_step", "steps_run", "status", "es")}}, out / "models" / f"ds_W{W}_fold{k}_s{s}.pt")
                lp = sm.predict_logp(model, D, D.val["rows"]); assert np.isfinite(lp).all(); p = np.exp(lp).astype(np.float64); p /= p.sum(1, keepdims=True)
                np.savez(str(f) + ".tmp.npz", qid=D.val["qid"], p=p.astype(np.float32), info=json.dumps(info)); Path(str(f) + ".tmp.npz").rename(f)
                print(c, "fold", k, "seed", s, "W", W, "best_step", info["best_step"], "es_nll", round(info["es"]["nll"], 4), info["status"], info["seconds"], "s", flush=True)
        del D; torch.cuda.empty_cache()
