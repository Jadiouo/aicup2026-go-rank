"""E89 S line: from-scratch pre-act ResNet, closed-set single-game player identification (PLAN experiments/E89_full_data/PLAN.md s1, s3.1).

Split/labels come from s_games.npz (built by data_prep.py = E87 step-0 split).  val/test games (role 1/2) are never touched by the
training loop: the train sampler only draws ids with role == 0 (see TrainSet / main, search "ROLE_GUARD").
Per epoch: every train game once, 256 games per optimiser step, 16 random own positions per game.  Evaluation: ALL own positions of a game.
  python train_s.py --data s_games.npz --out DIR --blocks 10 --ch 128 [--resume] [--bench 100] [--smoke ...]
"""
import os, sys, json, time, math, argparse, subprocess, threading
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
import fastboard as fb

SCALE, LS, LR, WD, GAMES_PER_STEP, POS_PER_GAME, EMB = 30.0, 0.1, 2e-3, 1e-2, 256, 16, 128
NPLANES = 18


# ------------------------------------------------------------------ data
def load_games(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def e79_idx(n_own):
    """E79 token positions (scripts/oneoff/e79_prep_local.py work24): window0 = 12 evenly spaced own moves in [2,60), window1 = 12 in [60,120)
    (else the last 58 own moves).  Own-move ordinals count passes."""
    def ev(n, k, lo, hi):
        top = min(hi, n)
        if top <= lo: return np.zeros(0, np.int64)
        if top - lo <= k: return np.arange(lo, top)
        return np.unique(np.round(np.linspace(lo, top - 1, k)).astype(np.int64))
    late = ev(n_own, 12, 60, 120)
    if len(late) == 0: late = ev(n_own, 12, max(2, n_own - 58), n_own)
    return np.concatenate([ev(n_own, 12, 2, 60), late])


def _encode_ids(G, ids, ppg, rng_for=None, flags=False):
    """Encode games `ids` -> dict of concatenated numpy arrays. ppg None = all own positions; else random ppg (seeded by rng_for(gid))."""
    parts = [[] for _ in range(8)]; gid, e_col, e79 = [], [], []
    for j, g in enumerate(ids):
        mv = G["moves"][G["off"][g]:G["off"][g + 1]]; e = int(G["eff"][g]); h = int(G["hd"][g])
        vo = fb.valid_ordinals(mv, e)
        if ppg is not None and len(vo) > ppg:
            sel = np.sort(rng_for(int(g)).choice(vo, ppg, replace=False)).astype(np.int32)
        else:
            sel = vo
        out = fb.encode(mv, e, h, sel)
        for a, o in zip(parts, out): a.append(o)
        gid.append(np.full(len(sel), j, np.int64)); e_col.append(np.full(len(sel), e, np.int8))
        if flags: e79.append(np.isin(sel, e79_idx(fb.count_own(mv, e))))
    names = ["grid", "lib", "prev_opp", "prev_own", "ko", "move", "ply", "oi"]
    b = {n: np.concatenate(p) for n, p in zip(names, parts)}
    b["gid"] = np.concatenate(gid); b["eff"] = np.concatenate(e_col)
    if flags: b["e79"] = np.concatenate(e79)
    return b


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, G, seed, ppg):
        self.G, self.seed, self.ppg = G, seed, ppg
    def __getitem__(self, item):
        epoch, ids, frac, last = item
        b = _encode_ids(self.G, ids, self.ppg, rng_for=lambda g: np.random.default_rng([self.seed, epoch, g]))
        P = len(ids) * self.ppg; n = len(b["gid"])
        if n < P:                                   # fixed shape: pad with copies of row 0, gid = dummy game
            pad = P - n
            for k in list(b):
                b[k] = np.concatenate([b[k], np.repeat(b[k][:1], pad, 0)])
            b["gid"][n:] = len(ids)
        t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}
        t["y"] = torch.from_numpy(self.G["cls"][ids].astype(np.int64)); t["frac"] = frac; t["last"] = last; t["ngames"] = len(ids)
        return t


class EvalSet(torch.utils.data.Dataset):
    def __init__(self, G, groups):
        self.G, self.groups = G, groups
    def __len__(self): return len(self.groups)
    def __getitem__(self, i):
        ids = self.groups[i]; b = _encode_ids(self.G, ids, None, flags=True)
        t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}; t["ngames"] = len(ids); return t


# ------------------------------------------------------------------ planes (GPU twin of fastboard.build_planes_np)
def build_planes(b, dev, dtype):
    g = b["grid"].to(dev, non_blocking=True); l = b["lib"].to(dev, non_blocking=True); P = g.shape[0]
    X = torch.zeros(P, NPLANES, 362, dtype=dtype, device=dev)
    own, opp = g == 1, g == 2
    X[:, 0, :361] = own; X[:, 1, :361] = opp; X[:, 2, :361] = g == 0
    for i in range(3):
        X[:, 3 + i, :361] = own & (l == i + 1); X[:, 7 + i, :361] = opp & (l == i + 1)
    X[:, 6, :361] = own & (l >= 4); X[:, 10, :361] = opp & (l >= 4)
    ar = torch.arange(P, device=dev)
    for ch, key in ((11, "prev_opp"), (12, "prev_own"), (13, "ko"), (14, "move")):
        idx = b[key].to(dev, non_blocking=True).long(); idx = torch.where(idx < 0, torch.full_like(idx, 361), idx)
        X[ar, ch, idx] = 1
    X = X[:, :, :361]
    X[:, 15] = (b["eff"].to(dev) == 0).to(dtype)[:, None]
    X[:, 16] = (b["ply"].to(dev).to(dtype) / 200.0)[:, None]
    X[:, 17] = (b["oi"].to(dev).to(dtype) / 120.0).clamp(max=1.0)[:, None]
    return X.reshape(P, NPLANES, 19, 19).contiguous(memory_format=torch.channels_last)


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn1, self.c1, self.bn2, self.c2 = nn.BatchNorm2d(c), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.Conv2d(c, c, 3, padding=1, bias=False)
    def forward(self, x):
        return x + self.c2(Fn.relu(self.bn2(self.c1(Fn.relu(self.bn1(x))))))


class Net(nn.Module):
    def __init__(self, n_cls, blocks, ch):
        super().__init__()
        self.stem = nn.Conv2d(NPLANES, ch, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        self.bnf = nn.BatchNorm2d(ch); self.head = nn.Linear(2 * ch, EMB)
        self.W = nn.Parameter(torch.randn(n_cls, EMB) * 0.05)       # same init scale as E87 step 0
    def pos_vec(self, X, move):
        f = Fn.relu(self.bnf(self.blocks(self.stem(X))))              # [P,C,19,19] channels_last
        P, C = f.shape[0], f.shape[1]
        flat = f.permute(0, 2, 3, 1).reshape(P, 361, C)
        at = flat[torch.arange(P, device=f.device), move.clamp(min=0)]
        return self.head(torch.cat([at, flat.mean(1)], 1)).float()
    def game_logits(self, gv):
        return SCALE * Fn.normalize(gv, dim=-1) @ Fn.normalize(self.W, dim=-1).T


def lr_at(step, nb, total):
    if step < nb: return LR * (step + 1) / nb
    return LR * 0.5 * (1 + math.cos(math.pi * (step - nb) / max(1, total - nb)))


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, G, ids, dev, amp, log, tag, workers, group=16, block=1024, hb=None):
    """Per-game mean of position vectors over ALL own positions (+ 3 info buckets) -> top5.  Returns dict of numpy arrays."""
    model.eval(); n = len(ids); groups = [ids[i:i + group] for i in range(0, n, group)]
    loader = torch.utils.data.DataLoader(EvalSet(G, groups), batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=4 if workers else None)
    sums = torch.zeros(4, n, EMB, device=dev); cnt = torch.zeros(4, n, device=dev); done = 0; t0 = time.time(); npos = 0
    for gi, b in enumerate(loader):
        base = gi * group; P = len(b["gid"]); gid_all = b["gid"].to(dev) + base
        oi = b["oi"].to(dev).long(); e79 = b["e79"].to(dev)
        flags = torch.stack([torch.ones_like(oi, dtype=torch.bool), oi < 60, oi >= 60, e79])   # all / first 60 own moves / own move 61+ / E79's 24 slots
        for a in range(0, P, block):
            sl = slice(a, min(a + block, P)); m = sl.stop - sl.start
            sub = {k: (v[sl] if torch.is_tensor(v) and v.ndim >= 1 and len(v) == P else v) for k, v in b.items()}
            if m < block:                                         # pad to a fixed shape with copies of row 0
                for k in ("grid", "lib", "prev_opp", "prev_own", "ko", "move", "ply", "oi", "eff"):
                    sub[k] = torch.cat([sub[k], sub[k][:1].expand(block - m, *sub[k].shape[1:])])
            X = build_planes(sub, dev, torch.float16 if amp else torch.float32)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                v = model.pos_vec(X, sub["move"].to(dev).long())[:m]
            for bk in range(4):
                mk = flags[bk, sl]; idx = gid_all[sl][mk]
                sums[bk].index_add_(0, idx, v[mk]); cnt[bk].index_add_(0, idx, torch.ones(len(idx), device=dev))
        npos += P
        if hb is not None and gi % 20 == 0: hb()
        if gi % 200 == 0: log(f"[{tag}] eval group {gi}/{len(groups)} {npos / (time.time() - t0):.0f} pos/s")
    Wn = Fn.normalize(model.W.float(), dim=-1); out = {}
    for bk, nm in enumerate(["all", "first60", "late61", "e79_24"]):
        has = cnt[bk] > 0; gv = Fn.normalize(sums[bk] / cnt[bk].clamp(min=1)[:, None], dim=-1)
        top = torch.cat([(SCALE * gv[a:a + 4096] @ Wn.T).topk(5, dim=1).indices for a in range(0, n, 4096)]).cpu().numpy()
        out[nm + "_top5"] = top; out[nm + "_has"] = has.cpu().numpy()
    # loss on the 'all' bucket
    ls = []
    gv = Fn.normalize(sums[0] / cnt[0].clamp(min=1)[:, None], dim=-1); y = torch.from_numpy(G["cls"][ids].astype(np.int64)).to(dev)
    for a in range(0, n, 4096): ls.append(Fn.cross_entropy(SCALE * gv[a:a + 4096] @ Wn.T, y[a:a + 4096], reduction="none"))
    out["loss_all"] = torch.cat(ls).cpu().numpy(); model.train(); return out


def summarize(res, G, ids, which=("all", "first60", "late61", "e79_24"), full=False):
    y = G["cls"][ids].astype(np.int64); r = {}
    for nm in which:
        has = res[nm + "_has"]; top = res[nm + "_top5"]
        r[f"h_{nm}"] = float((top[has, 0] == y[has]).mean()) if has.any() else None; r[f"n_{nm}"] = int(has.sum())
    r["top5_all"] = float((res["all_top5"] == y[:, None]).any(1).mean()); r["loss_all"] = float(res["loss_all"].mean())
    if full:
        hit = res["all_top5"][:, 0] == y; pl = G["player"][ids]; ups, inv = np.unique(pl, return_inverse=True)
        s1 = np.bincount(inv, weights=hit); cn = np.bincount(inv).astype(float); br = np.random.default_rng(1); bs = []
        for _ in range(1000):
            k = br.integers(0, len(ups), len(ups)); bs.append(s1[k].sum() / cn[k].sum())
        r["h_ci95_player_bootstrap"] = np.percentile(bs, [2.5, 97.5]).tolist(); ngm = G["pcount"][ids].astype(float); q = np.quantile(ngm, [1 / 3, 2 / 3])
        r["h_by_player_games_tertile"] = {f"<= {q[0]:.0f}": float(hit[ngm <= q[0]].mean()), f"{q[0]:.0f}-{q[1]:.0f}": float(hit[(ngm > q[0]) & (ngm <= q[1])].mean()), f"> {q[1]:.0f}": float(hit[ngm > q[1]].mean())}
        col = G["eff"][ids]; hdv = G["hd"][ids]
        r["h_by_own_colour_black"] = float(hit[col == 0].mean()); r["h_by_own_colour_white"] = float(hit[col == 1].mean())   # test/val games have hd<=1 -> eff colour = annotated colour
        r["n_black"], r["n_white"] = int((col == 0).sum()), int((col == 1).sum())
        r["random_chance_h"] = 1.0 / int(G["n_classes"])
    return r


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--blocks", type=int, default=10); ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="split the 256-game step into this many micro-batches (memory only)")
    ap.add_argument("--val-every", type=int, default=4); ap.add_argument("--val-sub", type=int, default=4, help="periodic val uses every N-th val game (final val is full)")
    ap.add_argument("--bench", type=int, default=0, help="run N steps, print speed + GPU utilisation, exit"); ap.add_argument("--stop-after-epoch", type=int, default=0)
    ap.add_argument("--games-per-step", type=int, default=GAMES_PER_STEP); ap.add_argument("--pos-per-game", type=int, default=16)
    ap.add_argument("--smoke-players", type=int, default=0, help="debug only: 200-player subset carved from TRAIN games (never val/test)")
    ap.add_argument("--smoke-train-cap", type=int, default=30); ap.add_argument("--shuffle-labels", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-final", action="store_true"); ap.add_argument("--selftest-planes", action="store_true"); ap.add_argument("--name", default="run")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); logf = open(out / "train.log", "a")
    def log(*x):
        s = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] " + " ".join(str(i) for i in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda"); amp = dev.type == "cuda"
    if dev.type == "cuda": torch.backends.cudnn.benchmark = True
    G = load_games(a.data); n_classes = int(G["n_classes"])
    role = G["role"].copy()
    # ---- smoke subset (debug): 200 players from TRAIN games only; their games are re-split inside train: 6 'val', 4 'test', <=cap train
    if a.smoke_players:
        rs = np.random.default_rng(7); trn = np.flatnonzero(G["role"] == 0); pl = G["player"][trn]; ups, cn = np.unique(pl, return_counts=True)
        chosen = rs.choice(ups[cn >= 10 + a.smoke_train_cap], a.smoke_players, replace=False); role[:] = -1; remap = {int(p): i for i, p in enumerate(chosen)}
        for p in chosen:
            g = trn[pl == p]; g = g[rs.permutation(len(g))]; role[g[:6]] = 1; role[g[6:10]] = 2; role[g[10:10 + a.smoke_train_cap]] = 0
        keep = role >= 0; cls = np.full(len(role), -1, np.int16); cls[keep] = [remap[int(p)] for p in G["player"][keep]]; G["cls"] = cls; G["n_classes"] = np.int64(a.smoke_players); n_classes = a.smoke_players
    if a.shuffle_labels:
        tr = np.flatnonzero(role == 0); cls = G["cls"].copy(); cls[tr] = cls[tr][np.random.default_rng(5).permutation(len(tr))]; G["cls"] = cls; log("LABELS SHUFFLED among train games")
    G["role"] = role
    # ROLE_GUARD: train ids are exactly role==0; val/test ids are role 1/2 and are only passed to evaluate()
    tr_ids, va_ids, te_ids = (np.flatnonzero(role == x) for x in (0, 1, 2))
    assert not (set(tr_ids) & set(va_ids)) and not (set(tr_ids) & set(te_ids)) and not (set(va_ids) & set(te_ids))
    if not a.smoke_players: assert (len(tr_ids), len(va_ids), len(te_ids), n_classes) == (459_916, 56_772, 56_772, 5_700), "split sizes differ from E87 step-0 log"
    log(f"games train {len(tr_ids)} val {len(va_ids)} test {len(te_ids)} classes {n_classes} device {dev} blocks {a.blocks} ch {a.ch} accum {a.accum} workers {a.workers}")
    if a.selftest_planes:
        ids = tr_ids[:20]; b = _encode_ids(G, ids, 16, rng_for=lambda g: np.random.default_rng([0, 0, g]))
        for e in (0, 1):
            m = b["eff"] == e
            if not m.any(): continue
            ref = fb.build_planes_np(b["grid"][m], b["lib"][m], b["prev_opp"][m], b["prev_own"][m], b["ko"][m], b["move"][m], b["ply"][m], b["oi"][m], e)
            t = {k: torch.from_numpy(v[m]) for k, v in b.items() if k in ("grid", "lib", "prev_opp", "prev_own", "ko", "move", "ply", "oi", "eff")}
            X = build_planes(t, torch.device("cpu"), torch.float32).numpy()
            log("selftest planes colour", e, "rows", int(m.sum()), "max abs diff", float(np.abs(X - ref).max()))
            assert np.abs(X - ref).max() < 1e-6
        return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = Net(n_classes, a.blocks, a.ch).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, fused=(dev.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    nb = math.ceil(len(tr_ids) / a.games_per_step); total = a.epochs * nb; start_ep = 0; step = 0; hist = []; vals = []
    ck = out / "ckpt_latest.pt"
    if ck.exists() and not a.bench:
        s = torch.load(ck, map_location=dev, weights_only=False); model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        start_ep, step, hist, vals = s["epoch"], s["step"], s["hist"], s["vals"]; torch.set_rng_state(s["rng_cpu"].cpu())
        if dev.type == "cuda": torch.cuda.set_rng_state(s["rng_cuda"].cpu())
        log(f"RESUMED from {ck} epoch {start_ep} step {step}")
    nparam = sum(p.numel() for p in model.parameters()); log(f"params {nparam / 1e6:.2f}M  steps/epoch {nb}  total steps {total}")
    sub_va = va_ids[::a.val_sub]
    ds = TrainSet(G, a.seed, a.pos_per_game)
    def status(phase, **kw):
        st = dict(name=a.name, phase=phase, epoch=ep_now[0], step=step, total_steps=total, time=time.time(), **kw)
        (out / "status.json.tmp").write_text(json.dumps(st)); os.replace(out / "status.json.tmp", out / "status.json")
    ep_now = [start_ep]; status("start")
    gpu_samples = []
    def poll_gpu(stop):
        while not stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.split()[0]
                gpu_samples.append(float(r))
            except Exception: pass
            stop.wait(0.5)
    model.train(); bench_done = False
    for ep in range(start_ep, a.epochs):
        ep_now[0] = ep
        perm = tr_ids[np.random.default_rng([a.seed, 1000, ep]).permutation(len(tr_ids))]
        assert (role[perm] == 0).all() and len(perm) == len(tr_ids)       # ROLE_GUARD: only train-role games are ever sampled for training
        items = []
        for s0 in range(nb):
            stp = perm[s0 * a.games_per_step:(s0 + 1) * a.games_per_step]; chunks = np.array_split(stp, min(a.accum, len(stp)))
            for ci, c in enumerate(chunks): items.append((ep, c, len(c) / len(stp), ci == len(chunks) - 1))
        loader = torch.utils.data.DataLoader(ds, batch_size=None, sampler=items, num_workers=a.workers, prefetch_factor=6 if a.workers else None, pin_memory=(dev.type == "cuda"), persistent_workers=False)
        acc = torch.zeros((), device=dev); n_acc = 0; t_wait = 0.0; t_log = time.time(); pos_log = 0; t_ep = time.time(); w0 = time.time()
        stop_ev = threading.Event(); th = None
        if a.bench and dev.type == "cuda": th = threading.Thread(target=poll_gpu, args=(stop_ev,), daemon=True); th.start()
        it = iter(loader)
        for _ in range(len(items)):
            t1 = time.time(); b = next(it); t_wait += time.time() - t1
            P = len(b["gid"]); G_n = b["ngames"]
            gid = b["gid"].to(dev, non_blocking=True)
            X = build_planes(b, dev, torch.float16 if amp else torch.float32)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                v = model.pos_vec(X, b["move"].to(dev, non_blocking=True).long())
            sums = torch.zeros(G_n + 1, EMB, device=dev).index_add_(0, gid, v); cntv = torch.zeros(G_n + 1, device=dev).index_add_(0, gid, torch.ones(P, device=dev))
            gv = sums[:G_n] / cntv[:G_n].clamp(min=1)[:, None]
            loss = Fn.cross_entropy(model.game_logits(gv), b["y"].to(dev, non_blocking=True), label_smoothing=LS) * b["frac"]
            scaler.scale(loss).backward(); acc += loss.detach(); pos_log += P
            if b["last"]:
                for g_ in opt.param_groups: g_["lr"] = lr_at(step, nb, total)
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1; n_acc += 1
                if step % 50 == 0 or (a.bench and step == a.bench):
                    l = float(acc) / n_acc; acc.zero_(); n_acc = 0; dt = time.time() - t_log
                    log(f"ep {ep + 1}/{a.epochs} step {step}/{total} loss {l:.4f} lr {lr_at(step - 1, nb, total):.2e} {pos_log / dt:.0f} pos/s data-wait {t_wait / max(dt, 1e-9):.2f} (since last log)")
                    hist.append((step, l)); status("train", loss=l, pos_per_s=pos_log / dt); t_log = time.time(); pos_log = 0; t_wait = 0.0
                if a.bench and step >= a.bench:
                    if dev.type == "cuda": torch.cuda.synchronize()
                    el = time.time() - w0; stop_ev.set()
                    log(f"BENCH {step} steps in {el:.1f}s = {step * a.games_per_step * a.pos_per_game / el:.0f} pos/s (incl. warmup), {el / step:.3f} s/step; "
                        f"GPU util mean {np.mean(gpu_samples) if gpu_samples else float('nan'):.0f}% (n={len(gpu_samples)}); peak mem {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.1f} GiB")
                    bench_done = True; break
        stop_ev.set()
        if bench_done: return
        del it, loader
        log(f"epoch {ep + 1} done in {time.time() - t_ep:.0f}s")
        if (ep + 1) % a.val_every == 0 and (ep + 1) < a.epochs:
            r = summarize(evaluate(model, G, sub_va, dev, amp, log, f"val-sub ep{ep + 1}", a.workers, hb=lambda: status("val_eval")), G, sub_va, which=("all",)); r["epoch"] = ep + 1; r["n_games"] = int(len(sub_va)); vals.append(r)
            log(f"VAL(sub 1/{a.val_sub}) epoch {ep + 1}: h {r['h_all']:.4f} top5 {r['top5_all']:.4f} loss {r['loss_all']:.3f}"); status("val", val_h=r["h_all"])
        s = dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=ep + 1, step=step, hist=hist, vals=vals, rng_cpu=torch.get_rng_state(),
                 rng_cuda=torch.cuda.get_rng_state() if dev.type == "cuda" else torch.zeros(1, dtype=torch.uint8), args=vars(a))
        torch.save(s, out / "ckpt_tmp.pt"); os.replace(out / "ckpt_tmp.pt", ck); log(f"checkpoint saved epoch {ep + 1}"); status("ckpt")
        if a.stop_after_epoch and ep + 1 >= a.stop_after_epoch: log("stop-after-epoch reached"); return
    if a.no_final: return
    status("final_eval")
    rv = evaluate(model, G, va_ids, dev, amp, log, "val-full", a.workers, hb=lambda: status("final_eval_val")); sv = summarize(rv, G, va_ids, full=True); log("VAL full", json.dumps(sv))
    rt = evaluate(model, G, te_ids, dev, amp, log, "test", a.workers, hb=lambda: status("final_eval_test")); st = summarize(rt, G, te_ids, full=True); log("TEST", json.dumps(st))
    np.savez(out / "preds.npz", val_ids=va_ids, test_ids=te_ids, val_top5=rv["all_top5"], test_top5=rt["all_top5"], val_loss=rv["loss_all"], test_loss=rt["loss_all"])
    res = dict(name=a.name, blocks=a.blocks, ch=a.ch, epochs=a.epochs, seed=a.seed, params_M=nparam / 1e6, n_train=int(len(tr_ids)), n_val=int(len(va_ids)), n_test=int(len(te_ids)), n_classes=n_classes,
               val=sv, test=st, test_h=st["h_all"], val_h=sv["h_all"], periodic_val_subset=vals, train_loss_curve=hist, args=vars(a), seconds=round(time.time() - t0), steps=step)
    (out / "results.json").write_text(json.dumps(res, indent=1)); log("RESULTS written", out / "results.json"); status("done", test_h=st["h_all"], val_h=sv["h_all"])


if __name__ == "__main__":
    main()
