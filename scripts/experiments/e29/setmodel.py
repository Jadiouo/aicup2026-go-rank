"""E29 stage 2: set models (DeepSets / small attention, CORAL head) on per-game features.  Plan: experiments/E29_set_model/PLAN.md.
School:  cd ~/gogame_e21/e29 && nice -n 10 ~/gogame_ft/.venv/bin/python -u code/setmodel.py sanity --fold 0
         ... code/setmodel.py run --fold 0 --kinds ds attn --seeds 0 1 2
Never scores val bags (only writes their probabilities); selection / early stopping use the ES players of the fold only.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HOME = Path(os.environ.get("E29_HOME", Path.home() / "gogame_e21" / "e29"))
DATA, OUT = HOME / "data", Path(os.environ.get("E29_OUT", HOME / "out"))
E1 = math.exp(-1.0)
K = 10
GRID = {"ds": (128, 256), "attn": (64, 128)}
KIND_ID = {"ds": 1, "attn": 2}
HP = dict(lr=1e-3, wd=1e-2, drop=0.2, batch=256, max_steps=6000, eval_every=200, patience=8)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------- data
class Data:
    """Bundle + fold roles + per-fold standardised features on the device."""

    def __init__(self, fold: int):
        t0 = time.perf_counter()
        self.fold = fold
        self.aux = {k: v for k, v in np.load(DATA / "aux.npz", allow_pickle=False).items()}
        a = self.aux
        self.X = np.load(DATA / "X.npy", mmap_mode="r")
        self.n, self.F = self.X.shape
        role = a[f"role_f{fold}"]
        assert set(np.unique(role).tolist()) == {1, 2, 3}
        self.role = role
        grole = role[a["game_player"]]
        self.fit_games = np.flatnonzero(grole == 1)
        self.prank = a["player_rank"]
        self.sampler = Sampler(a["game_player"], a["game_color"], self.prank, grole == 1)
        # bags
        for nm in ("es", "val"):
            rows = a[f"{nm}_rows_f{fold}"]
            pl = a[f"{nm}_player_f{fold}"]
            want = 2 if nm == "es" else 3
            assert (role[pl] == want).all(), f"{nm} bags contain non-{nm} players"
            valid = rows[rows >= 0]
            assert (role[a["game_player"][valid]] == want).all(), f"{nm} bag games belong to wrong players"
        self.es = self._bags("es")
        self.val = self._bags("val")
        self.stats = self._standardise()
        log(f"data fold {fold}: n_fit_games {len(self.fit_games)} groups {len(self.sampler.cnt)} es {len(self.es['y'])} val {len(self.val['y'])} "
            f"X on {DEV} {self.Xg.numel() * 4 / 1e6:.0f} MB  ({time.perf_counter() - t0:.0f}s)")

    def _bags(self, nm):
        a, k = self.aux, self.fold
        return {"rows": torch.from_numpy(a[f"{nm}_rows_f{k}"].astype(np.int64)).to(DEV), "n": a[f"{nm}_n_f{k}"].astype(np.int64),
                "y": a[f"{nm}_rank_f{k}"].astype(np.int64), "qid": a[f"{nm}_qid_f{k}"], "player": a[f"{nm}_player_f{k}"]}

    def _standardise(self):
        X, fit = self.X, self.fit_games
        F = self.F
        lo, hi, mu, sd = (np.zeros(F) for _ in range(4))
        for j in range(F):
            v = np.asarray(X[:, j])[fit].astype(np.float64)
            v = v[~np.isnan(v)]
            lo[j], hi[j] = np.percentile(v, [0.5, 99.5])
            v = np.clip(v, lo[j], hi[j])
            mu[j], sd[j] = v.mean(), v.std()
            if not sd[j] > 0:
                sd[j] = 1.0
        self.Xg = torch.empty((self.n, F), dtype=torch.float32, device=DEV)
        for a in range(0, self.n, 100_000):
            x = np.asarray(X[a:a + 100_000], dtype=np.float64)
            x = (np.clip(x, lo, hi) - mu) / sd
            self.Xg[a:a + len(x)] = torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float32)).to(DEV)
        self.norm = {"lo": lo, "hi": hi, "mu": mu, "sd": sd}        # saved for test-time inference (predict_test.py); no effect on training
        return {"n_fit_games": int(len(fit)), "mu_head": mu[:3].tolist()}

    def gather(self, rows):
        mask = rows >= 0
        return self.Xg[rows.clamp(min=0)], mask


class Sampler:
    """Training bags: class uniform -> N uniform in [5,20] -> (fit player, colour) group with >= N games, uniform -> N distinct games."""

    def __init__(self, game_player, game_color, prank, fit_mask):
        idx = np.flatnonzero(fit_mask)
        key = game_player[idx].astype(np.int64) * 2 + game_color[idx].astype(np.int64)
        o = np.argsort(key, kind="stable")
        idx, key = idx[o], key[o]
        uk, st, cnt = np.unique(key, return_index=True, return_counts=True)
        self.rows, self.start, self.cnt = idx.astype(np.int64), st.astype(np.int64), cnt.astype(np.int64)
        self.grank = prank[uk // 2]
        self.elig = {}
        for r in range(K):
            for n in range(5, 21):
                e = np.flatnonzero((self.grank == r) & (self.cnt >= n))
                assert len(e), f"no eligible group for class {r} N {n}"
                self.elig[(r, n)] = e

    def sample(self, rng, m):
        r = rng.integers(0, K, m)
        N = rng.integers(5, 21, m)
        out = np.full((m, 20), -1, np.int64)
        for i in range(m):
            e = self.elig[(int(r[i]), int(N[i]))]
            g = e[rng.integers(len(e))]
            pick = rng.choice(int(self.cnt[g]), int(N[i]), replace=False)
            out[i, :N[i]] = self.rows[self.start[g] + pick]
        return out, r.astype(np.int64)


# --------------------------------------------------------------------------------------------- model
class Coral(nn.Module):
    """Single score s + monotone thresholds b_0 > ... > b_8; P(y>k) = sigmoid(s + b_k).  Init b so that s=0 gives uniform probabilities."""

    def __init__(self):
        super().__init__()
        c = 1.0 - (np.arange(1, K)) / K                     # P(y>k) = 1-(k+1)/10
        b = np.log(c / (1 - c))
        gaps = b[:-1] - b[1:]
        self.b0 = nn.Parameter(torch.tensor(float(b[0])))
        self.gap_raw = nn.Parameter(torch.tensor(np.log(np.expm1(gaps)), dtype=torch.float32))

    def logp(self, s):
        b = torch.cat([self.b0[None], self.b0 - torch.cumsum(Fn.softplus(self.gap_raw), 0)])
        c = torch.sigmoid(s[:, None] + b[None, :])         # B,9 non-increasing
        p = torch.cat([1 - c[:, :1], c[:, :-1] - c[:, 1:], c[:, -1:]], 1)
        return torch.log(p.clamp_min(1e-7))


class DeepSets(nn.Module):
    def __init__(self, F, W, drop):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(F, W), nn.GELU(), nn.Dropout(drop), nn.Linear(W, W), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * W + 1, W), nn.GELU(), nn.Dropout(drop), nn.Linear(W, 1))
        self.head = Coral()
        nn.init.normal_(self.rho[-1].weight, std=0.01); nn.init.zeros_(self.rho[-1].bias)

    def forward(self, x, mask):
        h = self.phi(x)
        m = mask.unsqueeze(-1)
        cnt = m.sum(1).clamp_min(1)
        mean = (h * m).sum(1) / cnt
        mx = h.masked_fill(~m, -1e4).max(1).values
        z = torch.cat([mean, mx, (cnt.float().log() / math.log(20.0))], -1)
        return self.head.logp(self.rho(z).squeeze(-1))


class SmallAttn(nn.Module):
    def __init__(self, F, W, drop):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(F, W), nn.GELU())
        layer = nn.TransformerEncoderLayer(W, 4, 2 * W, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(W), enable_nested_tensor=False)
        self.rho = nn.Sequential(nn.Linear(W + 1, W), nn.GELU(), nn.Dropout(drop), nn.Linear(W, 1))
        self.head = Coral()
        nn.init.normal_(self.rho[-1].weight, std=0.01); nn.init.zeros_(self.rho[-1].bias)

    def forward(self, x, mask):
        h = self.enc(self.emb(x), src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1)
        cnt = m.sum(1).clamp_min(1)
        mean = (h * m).sum(1) / cnt
        z = torch.cat([mean, (cnt.float().log() / math.log(20.0))], -1)
        return self.head.logp(self.rho(z).squeeze(-1))


HEAD_LR_MULT = 10.0     # CORAL thresholds must widen from gap 0.44 to several units; at the base lr they move <=1e-3 per step (sanity run 1: NLL floor 1.01 with top-1 1.0)


def make_opt(model, lr, wd):
    head = list(model.head.parameters())
    ids = {id(p) for p in head}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": rest, "lr": lr, "weight_decay": wd}, {"params": head, "lr": lr * HEAD_LR_MULT, "weight_decay": 0.0}])


def build(kind, F, W, drop):
    return (DeepSets if kind == "ds" else SmallAttn)(F, W, drop).to(DEV)


# --------------------------------------------------------------------------------------------- eval helpers
def predict_logp(model, D, rows, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(rows), bs):
            x, m = D.gather(rows[a:a + bs])
            out.append(model(x, m).float().cpu())
    return torch.cat(out).numpy()


def ev_decode(p):
    S = np.where(np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]) == 0, 1.0,
                 np.where(np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]) == 1, E1, 0.0))
    return np.argmax(np.round(p @ S, 12), axis=1)


def balanced_w(y):
    cnt = np.bincount(y, minlength=K).astype(float)
    w = 1.0 / cnt[y]
    return w / w.mean()


def es_metrics(model, D):
    lp = predict_logp(model, D, D.es["rows"])
    y = D.es["y"]
    w = balanced_w(y)
    nll = float(-(lp[np.arange(len(y)), y] * w).mean())
    p = np.exp(lp)
    pred = ev_decode(p)
    sc = np.where(pred == y, 1.0, np.where(np.abs(pred - y) == 1, E1, 0.0))
    ns = D.es["n"]
    bal = float(np.mean([(sc[ns == n] * balanced_w(y[ns == n])).mean() for n in (5, 10, 15, 20)]))
    share = np.bincount(pred, minlength=K) / len(pred)
    return {"nll": nll, "score": bal, "modal_share": float(share.max()), "n_pred_classes": int((share > 0).sum()), "pred_share": share.round(3).tolist()}


# --------------------------------------------------------------------------------------------- training
def train_one(D, kind, W, seed, hp=HP, rng_tag=0):
    torch.manual_seed(seed * 1009 + KIND_ID[kind] * 31 + W)
    rng = np.random.default_rng(seed * 7919 + KIND_ID[kind] * 131 + W + 1_000_000 * D.fold)
    model = build(kind, D.F, W, hp["drop"])
    npar = sum(p.numel() for p in model.parameters())
    opt = make_opt(model, hp["lr"], hp["wd"])
    best = {"nll": float("inf"), "state": None, "step": 0}
    bad, hist, t0 = 0, [], time.perf_counter()
    chunk = hp["eval_every"] * hp["batch"]
    for step in range(1, hp["max_steps"] + 1):
        if (step - 1) % hp["eval_every"] == 0:
            rows_np, y_np = D.sampler.sample(rng, chunk)
            rows_c, y_c = torch.from_numpy(rows_np).to(DEV), torch.from_numpy(y_np).to(DEV)
        j = ((step - 1) % hp["eval_every"]) * hp["batch"]
        model.train()
        x, m = D.gather(rows_c[j:j + hp["batch"]])
        yb = y_c[j:j + hp["batch"]]
        lp = model(x, m)
        loss = -lp.gather(1, yb[:, None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % hp["eval_every"] == 0:
            em = es_metrics(model, D)
            hist.append({"step": step, "train_loss_last": float(loss), **{k: em[k] for k in ("nll", "score", "modal_share")}})
            log(f"{kind} W{W} s{seed} f{D.fold} step {step} train {float(loss):.4f} es_nll {em['nll']:.4f} es_score {em['score']:.4f} modal {em['modal_share']:.2f}")
            if em["nll"] < best["nll"] - 1e-5:
                best = {"nll": em["nll"], "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "step": step}
                bad = 0
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
    model.load_state_dict(best["state"])
    em = es_metrics(model, D)
    info = {"kind": kind, "W": W, "seed": seed, "fold": D.fold, "n_params": npar, "best_step": best["step"], "steps_run": step, "es": em, "hist": hist,
            "seconds": round(time.perf_counter() - t0), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0,
            "status": "TRAIN_FAIL" if (em["modal_share"] > 0.5 or em["n_pred_classes"] <= 2) else "OK"}
    return model, info


def run_cached(D, kind, W, seed):
    f = OUT / f"setprobs_{kind}_W{W}_fold{D.fold}_s{seed}.npz"
    if f.exists():
        return json.loads(str(np.load(f, allow_pickle=False)["info"]))
    model, info = train_one(D, kind, W, seed)
    (OUT / "models").mkdir(parents=True, exist_ok=True)           # trained weights kept for test inference (no effect on training)
    torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "kind": kind, "W": W, "seed": seed, "fold": D.fold, "F": D.F,
                "info": {k: info[k] for k in ("best_step", "steps_run", "status", "es")}}, OUT / "models" / f"{kind}_W{W}_fold{D.fold}_s{seed}.pt")
    lp = predict_logp(model, D, D.val["rows"])
    assert np.isfinite(lp).all()
    p = np.exp(lp).astype(np.float64)
    p /= p.sum(1, keepdims=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp.npz")
    np.savez(tmp, qid=D.val["qid"], p=p.astype(np.float32), info=json.dumps(info))
    tmp.replace(f)
    log(f"saved {f.name}: best_step {info['best_step']} es_nll {info['es']['nll']:.4f} es_score {info['es']['score']:.4f} status {info['status']} {info['seconds']}s vram {info['peak_vram_mb']}MB")
    return info


def cmd_run(a):
    D = Data(a.fold)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"norm_fold{a.fold}.npz", **D.norm)
    for kind in a.kinds:
        sel_f = OUT / f"select_{kind}_fold{a.fold}.json"
        if sel_f.exists():
            sel = json.loads(sel_f.read_text())
        else:
            cand = {W: run_cached(D, kind, W, 0) for W in GRID[kind]}
            W = min(cand, key=lambda w: cand[w]["es"]["nll"])
            sel = {"kind": kind, "fold": a.fold, "chosen_W": W, "es_nll": {str(w): cand[w]["es"]["nll"] for w in cand}, "rule": "lowest ES balanced NLL, seed 0"}
            sel_f.write_text(json.dumps(sel, indent=1))
            log("selected", sel)
        for s in a.seeds:
            run_cached(D, kind, sel["chosen_W"], s)


# --------------------------------------------------------------------------------------------- sanity (PLAN: before any val score)
def cmd_sanity(a):
    D = Data(a.fold)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"fold": a.fold, "ln10": math.log(10), "checks": {}}
    rng = np.random.default_rng(123)
    rows_np, y_np = D.sampler.sample(rng, 64)
    rows, y = torch.from_numpy(rows_np).to(DEV), torch.from_numpy(y_np).to(DEV)
    for kind in ("ds", "attn"):
        W = GRID[kind][0]
        torch.manual_seed(0)
        model = build(kind, D.F, W, 0.0)
        init_es = es_metrics(model, D)
        # un-weighted ES nll too (should be ln10 for any class mix when p is uniform)
        lp = predict_logp(model, D, D.es["rows"])
        init_nll = float(-lp[np.arange(len(lp)), D.es["y"]].mean())
        opt = make_opt(model, 3e-3, 0.0)
        losses = []
        for step in range(1, a.steps + 1):
            model.train()
            x, m = D.gather(rows)
            l = -model(x, m).gather(1, y[:, None]).mean()
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(l))
        model.eval()
        with torch.no_grad():
            x, m = D.gather(rows)
            lp = model(x, m)
        top1 = float((lp.argmax(1) == y).float().mean())
        res["checks"][kind] = {"W": W, "init_es_nll_unweighted": init_nll, "init_ok": abs(init_nll - math.log(10)) <= 0.05,
                               "overfit64_train_nll": float(-lp.gather(1, y[:, None]).mean()), "overfit64_top1": top1, "loss_first": losses[0],
                               "loss_last": losses[-1], "steps": a.steps,
                               "overfit_ok": bool(float(-lp.gather(1, y[:, None]).mean()) < 0.05 and top1 >= 0.98)}
        log(kind, res["checks"][kind])
    (OUT / f"sanity_fold{a.fold}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--fold", type=int, default=0); r.add_argument("--kinds", nargs="+", default=["ds", "attn"]); r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s = sp.add_parser("sanity"); s.add_argument("--fold", type=int, default=0); s.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    if os.environ.get("E29_SMOKE"):
        HP.update(max_steps=int(os.environ["E29_SMOKE"]), eval_every=10, patience=3)
    cmd_run(a) if a.cmd == "run" else cmd_sanity(a)
