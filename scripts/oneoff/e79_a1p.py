"""E79 A1P: A1 set model (DeepSets + CORAL, E29 setmodel unchanged for Data/Sampler/es_metrics/predict) whose per-game input is [X139 | attention-pooled embedding of the game's <=24 per-position KataGo vectors].
PLAN section 2.  Per fold k: 4-cell grid p_in {0.1,0.3} x tdrop {0,0.25} (seed 0, ES players only), best ES NLL -> seeds 0,1,2 (seed 0 = the grid model).  Outputs .scratch/e79/out/A1P (E55 format: select_ds_fold{k}.json, setprobs_ds_W{W}_fold{k}_s{s}.npz, models/, tokstats_fold{k}.npz).
  gpujob systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python3 scripts/oneoff/e79_a1p.py [--folds 0 1 2 3 4] [--smoke N]"""
import argparse, json, os, sys, gc, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; S = ROOT / ".scratch/e79"; TOK = Path(os.environ.get("E79_TOKDIR", S)); HOME = S / "home_a1p"
os.environ["E29_HOME"] = str(HOME); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, torch.nn as nn
(HOME / "data").mkdir(parents=True, exist_ok=True)
for nm, tgt in (("aux.npz", ROOT / ".scratch/e29/bundle/aux.npz"), ("X.npy", ROOT / ".scratch/e36/bundle/X120_full.npy")):
    if not (HOME / "data" / nm).exists(): (HOME / "data" / nm).symlink_to(tgt)
from scripts.experiments.e29 import setmodel as sm
A1 = ROOT / ".scratch/e37/runs/v6/out/A1full"; OUT = Path(os.environ.get("E79_OUT", S / "out/A1P")); DEV = sm.DEV; TD = 128
GRID = [(p, t) for p in (0.1, 0.3) for t in (0.0, 0.25)]
class TokenBank:
    """Per-game tokens on the device (fp16 values, int8 own-move bucket, bool mask) + the per-fold standardisation."""
    def __init__(self, prefix="tok_full"):
        self.X = torch.from_numpy(np.load(TOK / f"{prefix}.npy", mmap_mode="r")[:]).to(DEV); self.O = torch.from_numpy(np.load(TOK / f"{prefix}_oidx.npy")).to(DEV); self.M = torch.from_numpy(np.load(TOK / f"{prefix}_mask.npy")).to(DEV)
def fit_stats(bank_np, fit_games, seed):
    rng = np.random.default_rng(seed); g = np.sort(rng.choice(fit_games, 100_000, replace=False)); X = np.asarray(bank_np[0][g], np.float32); M = bank_np[1][g]; v = X[M]
    return v.mean(0).astype(np.float32), (v.std(0) + 1e-6).astype(np.float32)
class DataP(sm.Data):
    """sm.Data with gather -> x = [X139 standardised | row id (float32, exact below 2^24)]."""
    def gather(self, rows):
        r = rows.clamp(min=0); return torch.cat([self.Xg[r], r.float()[..., None]], -1), rows >= 0
class A1P(nn.Module):
    def __init__(self, F, W, drop, p_in, tdrop, bank, mu, sd):
        super().__init__(); self.F, self.tdrop, self.bank = F, tdrop, bank
        self.register_buffer("mu", torch.from_numpy(mu)); self.register_buffer("sd", torch.from_numpy(sd))
        self.proj = nn.Sequential(nn.Linear(64, TD), nn.LayerNorm(TD)); self.pos = nn.Embedding(24, TD); self.col = nn.Embedding(2, TD); self.pin = nn.Dropout(p_in)
        self.mlp = nn.Sequential(nn.Linear(TD, 256), nn.GELU(), nn.Linear(256, TD)); self.q = nn.Parameter(torch.randn(1, 1, TD) * 0.02)
        self.attn = nn.MultiheadAttention(TD, 4, batch_first=True); self.out = nn.Linear(TD, TD)
        self.ds = sm.DeepSets(F + TD, W, drop); self.head = self.ds.head
    def embed(self, rows, color):
        X = self.bank.X[rows].float(); O = self.bank.O[rows].long(); M = self.bank.M[rows]
        if self.training and self.tdrop > 0:
            r = torch.rand(M.shape, device=M.device).masked_fill(~M, 2.0); M = M & ((r > self.tdrop) | (r == r.min(1, keepdim=True).values))
        h = self.pin(self.proj((X - self.mu) / self.sd) + self.pos(O) + self.col(color)[:, None, :]); h = h + self.mlp(h)
        o, _ = self.attn(self.q.expand(len(h), -1, -1), h, h, key_padding_mask=~M, need_weights=False); return self.out(o[:, 0])
    def forward(self, x, mask):
        feats, rows = x[..., :self.F], x[..., self.F].long(); B, N = mask.shape; e = torch.zeros(B, N, TD, device=x.device)
        rf = rows[mask]
        if len(rf): e[mask] = self.embed(rf, self.bank.gcol[rf])
        return self.ds(torch.cat([feats, e], -1), mask)
def train_cfg(D, W, seed, p_in, tdrop, bank, mu, sd, hp=sm.HP):
    torch.manual_seed(seed * 1009 + 31 + W + int(p_in * 100) * 7 + int(tdrop * 100) * 13); rng = np.random.default_rng(seed * 7919 + 131 + W + 1_000_000 * D.fold + int(p_in * 100) * 17 + int(tdrop * 100))
    model = A1P(D.F, W, hp["drop"], p_in, tdrop, bank, mu, sd).to(DEV); npar = sum(p.numel() for p in model.parameters()); opt = sm.make_opt(model, hp["lr"], hp["wd"])
    best = {"nll": float("inf"), "state": None, "step": 0}; bad = 0; hist = []; t0 = time.perf_counter(); chunk = hp["eval_every"] * hp["batch"]
    for step in range(1, hp["max_steps"] + 1):
        if (step - 1) % hp["eval_every"] == 0:
            rows_np, y_np = D.sampler.sample(rng, chunk); rows_c, y_c = torch.from_numpy(rows_np).to(DEV), torch.from_numpy(y_np).to(DEV)
        j = ((step - 1) % hp["eval_every"]) * hp["batch"]; model.train(); x, m = D.gather(rows_c[j:j + hp["batch"]]); yb = y_c[j:j + hp["batch"]]
        lp = model(x, m); loss = -lp.gather(1, yb[:, None]).mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % hp["eval_every"] == 0:
            em = sm.es_metrics(model, D); hist.append({"step": step, "nll": em["nll"], "score": em["score"]})
            if em["nll"] < best["nll"] - 1e-5: best = {"nll": em["nll"], "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "step": step}; bad = 0
            else:
                bad += 1
                if bad >= hp["patience"]: break
    model.load_state_dict(best["state"]); em = sm.es_metrics(model, D)
    return model, {"kind": "ds", "W": W, "mode": "ce", "seed": seed, "fold": D.fold, "p_in": p_in, "tdrop": tdrop, "n_params": npar, "best_step": best["step"], "steps_run": step, "es": em, "hist": hist, "seconds": round(time.perf_counter() - t0),
                   "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6), "status": "TRAIN_FAIL" if (em["modal_share"] > 0.5 or em["n_pred_classes"] <= 2) else "OK"}
def save(model, info, D, k, W, s):
    (OUT / "models").mkdir(parents=True, exist_ok=True); f = OUT / f"setprobs_ds_W{W}_fold{k}_s{s}.npz"
    sd_ = {kk: v.detach().cpu() for kk, v in model.state_dict().items() if not kk.startswith("bank")}
    torch.save({"state": sd_, "kind": "ds", "W": W, "seed": s, "fold": k, "F": D.F, "p_in": info["p_in"], "tdrop": info["tdrop"], "info": {x: info[x] for x in ("best_step", "steps_run", "status", "es")}}, OUT / "models" / f"a1p_W{W}_fold{k}_s{s}.pt")
    lp = sm.predict_logp(model, D, D.val["rows"]); assert np.isfinite(lp).all(); p = np.exp(lp).astype(np.float64); p /= p.sum(1, keepdims=True)
    np.savez(str(f) + ".tmp.npz", qid=D.val["qid"], p=p.astype(np.float32), info=json.dumps(info)); Path(str(f) + ".tmp.npz").rename(f)
    print("A1P fold", k, "seed", s, "W", W, "p_in", info["p_in"], "tdrop", info["tdrop"], "best_step", info["best_step"], "es_nll", round(info["es"]["nll"], 4), "es_score", round(info["es"]["score"], 4), info["status"], info["seconds"], "s", "vram", info["peak_vram_mb"], "MB", flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--smoke", type=int, default=0); a = ap.parse_args()
    if a.smoke: sm.HP.update(max_steps=a.smoke, eval_every=10, patience=3)
    OUT.mkdir(parents=True, exist_ok=True); bank = TokenBank(); aux = np.load(HOME / "data/aux.npz", allow_pickle=False); bank.gcol = torch.from_numpy(aux["game_color"].astype(np.int64)).to(DEV)
    tokX = np.load(TOK / "tok_full.npy", mmap_mode="r"); tokM = np.load(TOK / "tok_full_mask.npy")
    for k in a.folds:
        sel = json.loads((A1 / f"select_ds_fold{k}.json").read_text()); W = sel["chosen_W"]
        if all((OUT / f"setprobs_ds_W{W}_fold{k}_s{s}.npz").exists() for s in a.seeds): continue
        t0 = time.perf_counter(); D = DataP(k); assert D.F == 139; mu, sd = fit_stats((tokX, tokM), D.fit_games, 790 + k); np.savez(OUT / f"tokstats_fold{k}.npz", mu=mu, sd=sd); np.savez(OUT / f"norm_fold{k}.npz", **D.norm)
        gf = OUT / f"grid_fold{k}.json"; models = {}
        if gf.exists() and (OUT / f"setprobs_ds_W{W}_fold{k}_s0.npz").exists(): ch = json.loads(gf.read_text())["chosen"]; p_in, tdrop = ch["p_in"], ch["tdrop"]
        else:
            res = {}
            for (p_in, tdrop) in GRID:
                m_, info = train_cfg(D, W, 0, p_in, tdrop, bank, mu, sd); res[(p_in, tdrop)] = (m_, info); print("grid fold", k, p_in, tdrop, "es_nll", round(info["es"]["nll"], 4), "best_step", info["best_step"], info["status"], info["seconds"], "s", flush=True)
            order = sorted(GRID, key=lambda c: (round(res[c][1]["es"]["nll"], 5), c[0], c[1])); p_in, tdrop = order[0]   # lowest ES NLL; ties -> smaller p_in, smaller tdrop
            gf.write_text(json.dumps({"fold": k, "W": W, "chosen": {"p_in": p_in, "tdrop": tdrop}, "es_nll": {f"{c[0]}_{c[1]}": res[c][1]["es"]["nll"] for c in GRID}, "rule": "lowest ES balanced NLL, seed 0; ties -> smaller p_in, tdrop"}, indent=1))
            (OUT / f"select_ds_fold{k}.json").write_text(json.dumps({**sel, "note": "chosen_W copied from v6 A1full (not re-selected); X139 + attention-pooled per-position tokens; grid_fold json = (p_in, tdrop)"}, indent=1))
            save(res[(p_in, tdrop)][0], res[(p_in, tdrop)][1], D, k, W, 0); del res
        (OUT / f"select_ds_fold{k}.json").write_text(json.dumps({**sel, "note": "chosen_W copied from v6 A1full (not re-selected); X139 + attention-pooled per-position tokens"}, indent=1))
        for s in a.seeds:
            if s == 0 or (OUT / f"setprobs_ds_W{W}_fold{k}_s{s}.npz").exists(): continue
            m_, info = train_cfg(D, W, s, p_in, tdrop, bank, mu, sd); save(m_, info, D, k, W, s)
        del D; gc.collect(); torch.cuda.empty_cache(); print("fold", k, "done", round(time.perf_counter() - t0), "s", flush=True)
    print("A1P_DONE")
