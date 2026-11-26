"""E70 GPU half (system python3 + torch; submit with gpujob).  KataGo b18c384nbt-humanv0 trunk (forward cut after act_trunkfinal), position vector u = [at | mean] (768).
Subcommands:
  feats   : frozen u for every train/val position -> .scratch/e70/tr_u0.npy (fp16), tr_stats.npz (game-level mean/std of train games); also pilot frozen game vectors check vs E65 pilot_gv.npz
  lp      : linear probe (frozen trunk, head 768->256), episodic prototype loss -> lp_head.pt ; prints val score
  lora    : LoRA (r=8, all Conv2d in blocks 14..17) + head init from LP, --lr, --steps; ckpt every 1000 steps, val each ckpt; at the end picks best-val ckpt and extracts pilot game vectors
  extract : pilot game vectors with a given ckpt (--ckpt) -> pilot_ft_<tag>.npz
Data/index: e70_prep.py / e70_index.py outputs.  Game vector = mean over the game's 12 positions of u.  Head is linear -> embedding of mean == mean of embeddings."""
import sys, os, time, math, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; K = ROOT / ".scratch" / "e70"; KS = str(ROOT / ".scratch" / "e52" / "katago_src")
dev = os.environ.get("E70_DEV", "cuda")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class Stop(Exception):
    def __init__(self, x): self.x = x

def load(ckpt):
    sys.path.insert(0, str(Path(KS) / "python"))
    from katago.train.load_model import load_model
    model, _, _ = load_model(ckpt, use_swa=False, device=dev, pos_len=19); model.eval()
    def hook(m, i, o): raise Stop(o)
    model.act_trunkfinal.register_forward_hook(hook)
    for p in model.parameters(): p.requires_grad_(False)
    return model

class LoRAConv(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__(); self.base = base; k = base.kernel_size
        self.A = nn.Conv2d(base.in_channels, r, k, stride=base.stride, padding=base.padding, bias=False); self.B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5)); nn.init.zeros_(self.B.weight); self.s = alpha / r
    def forward(self, x): return self.base(x) + self.s * self.B(self.A(x))

def add_lora(model, first=14, r=8, alpha=16):
    n = 0
    for bi in range(first, len(model.blocks)):
        blk = model.blocks[bi]
        for name, mod in list(blk.named_modules()):
            for cn, ch in list(mod.named_children()):
                if isinstance(ch, nn.Conv2d): setattr(mod, cn, LoRAConv(ch, r, alpha).to(dev)); n += 1
    ps = [p for n_, p in model.named_parameters() if (".A." in n_ or ".B." in n_) and "blocks" in n_]
    for p in ps: p.requires_grad_(True)
    return n, ps

def lora_state(model): return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

def meta_rows():
    mr = np.load(ROOT / ".scratch" / "e52" / "kata" / "meta_rows.npy"); return {0: torch.tensor(mr[0]), 1: torch.tensor(mr[1])}
MROWS = meta_rows()

class Data:
    def __init__(self, pre):
        self.bits = np.load(K / f"{pre}_bits.npy"); self.glob = np.load(K / f"{pre}_glob.npy"); self.loc = np.load(K / f"{pre}_loc.npy").astype(np.int64); self.color = np.load(K / f"{pre}_color.npy")
        z = np.load(K / f"{pre}_idx.npz"); self.gid, self.player, self.split, self.pos, self.cnt = z["gid"], z["player"], z["split"], z["pos"], z["cnt"]; self.N = len(self.bits)
    def batch(self, idx):
        b = np.unpackbits(self.bits[idx], axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
        return (torch.from_numpy(b).to(dev).float(), torch.from_numpy(self.glob[idx]).to(dev), torch.stack([MROWS[int(c)] for c in self.color[idx]]).to(dev), torch.from_numpy(self.loc[idx]).to(dev))

def trunk_u(model, bx, gx, mx, loc, autocast=True):
    """-> (B,768) float32 = [at | mean] of act_trunkfinal.  at = vector at the played point (board mean on a pass, E52 convention)."""
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast): model(bx, gx, input_meta=mx)
        raise RuntimeError("hook did not fire")
    except Stop as e: f = e.x
    f = f.float(); ff = f.flatten(2); bm = ff.mean(2); pas = loc < 0; l = loc.clamp(min=0)
    at = ff.gather(2, l[:, None, None].expand(-1, f.shape[1], 1))[:, :, 0]; at = torch.where(pas[:, None], bm, at)
    return torch.cat([at, bm], 1)

@torch.no_grad()
def extract_positions(model, D, idx_all, bs=512, tag=""):
    out = np.zeros((len(idx_all), 768), np.float32); t0 = time.time()
    for s in range(0, len(idx_all), bs):
        idx = idx_all[s:s + bs]; out[s:s + len(idx)] = trunk_u(model, *D.batch(idx)).cpu().numpy()
        if (s // bs) % 100 == 0: log(tag, s, len(idx_all), f"{time.time() - t0:.0f}s")
    return out

def game_means(U, D, games):
    """U: position vectors aligned with D.pos rows of the given games -> (len(games), 768) mean over the game's positions (handles pad)"""
    return U

def val_score(E, D, games, seed=0, reps=5):
    """E: (len(games),d) game embeddings of the val games (order = games).  Players: D.player[games].  Retrieval score on 200 players, lib 10 games, N queries from the rest."""
    pl = D.player[games]; ups = np.unique(pl); byp = [np.flatnonzero(pl == p) for p in ups]; rng = np.random.default_rng(seed); sc = {1: [], 2: [], 3: [], 5: []}
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    for _ in range(reps):
        libs, rests = [], []
        for ix in byp: r = rng.permutation(ix); libs.append(En[r[:10]].mean(0)); rests.append(r[10:])
        L = np.stack(libs); L /= np.linalg.norm(L, axis=1, keepdims=True)
        for n in (1, 2, 3, 5):
            q = np.stack([En[r[:n]].mean(0) for r in rests]); q /= np.linalg.norm(q, axis=1, keepdims=True); S = q @ L.T
            t = S[np.arange(len(S)), np.arange(len(S))]; rk = 1 + (S > t[:, None]).sum(1); sc[n].append(np.where(rk <= 5, np.exp(-(rk - 1.0)), 0.0).mean())
    return {n: float(np.mean(v)) for n, v in sc.items()} | {"all": float(np.mean([np.mean(v) for v in sc.values()]))}

def standardize(Gv, st): return (Gv - st["mu"]) / st["sd"]

def episode_loss(e, nq, scale=20.0):
    """e: (P,4,d) game embeddings; nq: (P,) in {1,2}: query = mean of first nq games, prototype = mean of the rest; cross-entropy over players of scale*cosine"""
    P = e.shape[0]; w = (torch.arange(4, device=e.device)[None, :] < nq[:, None]).float(); wq = w / w.sum(1, keepdim=True); wc = (1 - w) / (1 - w).sum(1, keepdim=True)
    q = F.normalize((e * wq[:, :, None]).sum(1), dim=1); c = F.normalize((e * wc[:, :, None]).sum(1), dim=1); lg = scale * q @ c.T
    return F.cross_entropy(lg, torch.arange(P, device=e.device)), (lg.argmax(1) == torch.arange(P, device=e.device)).float().mean()

def sample_episode(rng, eligible, games_of, P, k):
    """-> list of (P,4) game ids, (P,4,k) position slot choices"""
    pl = rng.choice(len(eligible), P, replace=False); gids = np.stack([rng.choice(games_of[eligible[p]], 4, replace=False) for p in pl])
    slots = np.stack([[rng.choice(12, k, replace=False) for _ in range(4)] for _ in range(P)]); return gids, slots

def build_train_index(D):
    ok = (D.cnt == 12) & (D.split == 0); games_of = {}
    for g in np.flatnonzero(ok): games_of.setdefault(D.player[g], []).append(g)
    elig = [p for p, v in games_of.items() if len(v) >= 4]; return elig, {p: np.array(v) for p, v in games_of.items()}

def lp_train(args):
    D = Data("tr"); U0 = torch.from_numpy(np.load(K / "tr_u0.npy")).to(dev); st = {k: torch.from_numpy(v).to(dev) for k, v in np.load(K / "tr_stats.npz").items()}
    elig, games_of = build_train_index(D); log("LP players", len(elig))
    head = nn.Linear(768, 256, bias=False).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    posT = torch.from_numpy(D.pos).to(dev); t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups: g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        gids, slots = sample_episode(rng, elig, games_of, args.P, 6); gi = torch.from_numpy(gids).to(dev); sl = torch.from_numpy(slots).to(dev)
        rows = posT[gi[:, :, None], sl]; u = U0[rows].float().mean(2); e = head((u - st["mu"]) / st["sd"]); nq = torch.from_numpy(rng.integers(1, 3, args.P)).to(dev)
        loss, acc = episode_loss(e, nq); opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps - 1: log(f"LP step {step} loss {loss.item():.3f} acc {acc.item():.3f} {time.time() - t0:.0f}s")
    torch.save(head.state_dict(), K / "lp_head.pt"); np.save(K / "lp_head.npy", head.weight.detach().cpu().numpy()); report_val_from_u0(D, U0, st, head, "LP")

def game_vec_from_u(U, D, games):
    """U (N,768) position vectors (torch) -> (len(games),768) game means"""
    return U[torch.from_numpy(D.pos[games, :12]).to(U.device)].float().mean(1)

def report_val_from_u0(D, U0, st, head, name):
    vg = np.flatnonzero((D.split == 1) & (D.cnt == 12)); Gv = game_vec_from_u(U0, D, vg); Z = (Gv - st["mu"]) / st["sd"]
    raw = val_score(Z.cpu().numpy(), D, vg); log(f"VAL frozen standardized cosine (no head): {raw}")
    if head is not None:
        with torch.no_grad(): E = head(Z).cpu().numpy()
        r = val_score(E, D, vg); log(f"VAL {name}: {r}"); return r

def cmd_feats(args):
    model = load(args.ckpt); D = Data("tr"); U0 = extract_positions(model, D, np.arange(D.N), tag="tr")
    np.save(K / "tr_u0.npy", U0.astype(np.float16)); U0t = torch.from_numpy(U0.astype(np.float16)).to(dev); tr = np.flatnonzero((D.split == 0) & (D.cnt == 12)); Gv = game_vec_from_u(U0t, D, tr).cpu().numpy()
    np.savez(K / "tr_stats.npz", mu=Gv.mean(0).astype(np.float32), sd=(Gv.std(0) + 1e-6).astype(np.float32)); log("feats saved", U0.shape)
    # pilot sanity vs E65
    P = Data("pilot"); z = np.load(ROOT / ".scratch" / "e65" / "pilot_gv.npz"); gid65 = z["gid"]; V65 = z["V"][:, :768]; sub = np.arange(0, len(P.gid), 52)[:300]
    idx = P.pos[sub].reshape(-1); Up = extract_positions(model, P, idx, tag="pilot-check"); Gp = Up.reshape(len(sub), 12, 768).mean(1)
    assert (gid65[sub] == P.gid[sub]).all(); d = np.abs(Gp - V65[sub]).max(); cc = np.corrcoef(Gp.ravel(), V65[sub].ravel())[0, 1]; log(f"pilot check vs E65: maxdiff {d:.4g} corr {cc:.8f}")

def load_head(): h = nn.Linear(768, 256, bias=False).to(dev); h.load_state_dict(torch.load(K / "lp_head.pt")); return h

def cmd_lp(args): lp_train(args)

def cmd_lora(args):
    tag = args.tag; model = load(args.ckpt); n, lps = add_lora(model); log("LoRA convs", n, "params", sum(p.numel() for p in lps))
    D = Data("tr"); st = {k: torch.from_numpy(v).to(dev) for k, v in np.load(K / "tr_stats.npz").items()}; head = load_head()
    for p in head.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": lps}, {"params": list(head.parameters())}], lr=args.lr, weight_decay=1e-2); scaler = torch.amp.GradScaler()
    elig, games_of = build_train_index(D); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed); step0 = 0; ck = K / f"lora_{tag}_last.pt"
    if ck.exists():
        s = torch.load(ck, weights_only=False); model.load_state_dict({**model.state_dict(), **{k: v.to(dev) for k, v in s["lora"].items()}}, strict=True); head.load_state_dict(s["head"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"]); step0 = s["step"]; rng = np.random.default_rng(args.seed + step0); log("resumed at", step0)
    vg = np.flatnonzero((D.split == 1) & (D.cnt == 12)); vidx = D.pos[vg, :12].reshape(-1); history = json.load(open(K / f"lora_{tag}_hist.json")) if (K / f"lora_{tag}_hist.json").exists() else []
    t0 = time.time(); P, k = args.P, args.k
    for step in range(step0, args.steps):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups: g["lr"] = lr
        gids, slots = sample_episode(rng, elig, games_of, P, k); idx = D.pos[gids[:, :, None], slots].reshape(-1)
        u = trunk_u(model, *D.batch(idx)).reshape(P, 4, k, 768).mean(2); e = head((u - st["mu"]) / st["sd"]); nq = torch.from_numpy(rng.integers(1, 3, P)).to(dev)
        loss, acc = episode_loss(e, nq); opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(lps + list(head.parameters()), 1.0); scaler.step(opt); scaler.update()
        if step % 100 == 0: log(f"{tag} step {step} loss {loss.item():.3f} acc {acc.item():.3f} lr {lr:.2e} {time.time() - t0:.0f}s maxmem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            torch.save(dict(lora=lora_state(model), head=head.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step + 1), ck)
            torch.save(dict(lora=lora_state(model), head=head.state_dict(), step=step + 1), K / f"lora_{tag}_s{step + 1}.pt")
            Ug = extract_positions(model, D, vidx, tag="val").reshape(len(vg), 12, 768).mean(1); Z = (torch.from_numpy(Ug).to(dev) - st["mu"]) / st["sd"]
            with torch.no_grad(): E = head(Z).cpu().numpy()
            r = val_score(E, D, vg); r["step"] = step + 1; history.append(r); json.dump(history, open(K / f"lora_{tag}_hist.json", "w")); log(f"VAL {tag} step {step + 1}: {r}")
    best = max(history, key=lambda r: r["all"]); log("best val", best); sd = torch.load(K / f"lora_{tag}_s{best['step']}.pt", weights_only=False)
    model.load_state_dict({**model.state_dict(), **{k_: v.to(dev) for k_, v in sd["lora"].items()}}, strict=True); head.load_state_dict(sd["head"]); pilot_extract(model, tag + "_best")
    np.save(K / f"head_{tag}_best.npy", head.weight.detach().cpu().numpy()); json.dump(dict(best=best), open(K / f"lora_{tag}_best.json", "w"))

def pilot_extract(model, tag):
    P = Data("pilot"); idx = P.pos.reshape(-1); Up = extract_positions(model, P, idx, tag="pilot"); G = Up.reshape(len(P.gid), 12, 768).mean(1)
    np.savez(K / f"pilot_ft_{tag}.npz", gid=P.gid, V=G.astype(np.float32)); log("pilot extracted", tag, G.shape)

def cmd_extract(args):
    model = load(args.ckpt); n, _ = add_lora(model); sd = torch.load(args.lora, weights_only=False); model.load_state_dict({**model.state_dict(), **{k: v.to(dev) for k, v in sd["lora"].items()}}, strict=True); pilot_extract(model, args.tag)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--ckpt", default=str(K / "b18c384nbt-humanv0.ckpt")); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--P", type=int, default=128); ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--lora", default="")
    a = ap.parse_args(); log("args", vars(a)); {"feats": cmd_feats, "lp": cmd_lp, "lora": cmd_lora, "extract": cmd_extract}[a.cmd](a); log("done")
