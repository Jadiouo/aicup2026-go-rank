"""E90 stages 1-3: S-network training on a ROSTER (copy of scripts/experiments/e89/train_s.py; the model / planes / encoder / evaluate code is
byte-identical to E89 (the block between the 'data' and 'main' banners is pasted verbatim, V1 checks this), only main() differs).

Recipe = E89 setting A: 10 blocks x 128 ch, 18 input planes, 256 games x 16 positions per step, AdamW lr 2e-3 / wd 1e-2, 1 epoch warm-up + cosine, fp16,
label smoothing .1, cosine head x30, grad clip 1.0, accum 2, seed 0, fixed 20 epochs, no checkpoint selection.
Differences from E89 (all forced by the task, nothing else changed):
  * `--roster NAME` (sim_full sim_h0 sim_h1 fin_full fin_h0 fin_h1, masks in rosters.npz): EVERY game of the roster trains (no per-player val/test games);
    classes = the roster's players (dense rank of aux game_player).  Games outside the roster are never read by the training loop (ROLE_GUARD).
  * step count asserted against the PLAN table; sim rosters assert that none of the 1,481 sim-eval players is a class or has a training game.
  * no val: monitoring = training loss every 50 steps (+ steps_loss.tsv, one line per step) and an in-sample probe (fixed 512 roster games, every 4 epochs,
    observation only; torch RNG state is saved/restored around it so the training trajectory is the same as without it).
  * after training: per-game vectors (mean of all own-position 128-d vectors, L2-normalised; = E89 gv bucket 'all', plus 'first60') for the games listed in
    export_spec.json -> OUT/vec_<name>.npz (chunk-cached, resumable).  results.json is written LAST (driver's success marker).
  python train_e90.py --data u_games.npz --roster sim_full --out DIR [--rosters rosters.npz --tdata t_games.npz --export auto]   (resume from ckpt_latest.pt is automatic)
"""
import os, sys, json, time, math, argparse, subprocess, threading
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if not (HERE / "fastboard.py").exists(): sys.path.append(str(HERE.parent / "e89"))       # local checkout: unmodified E89 fastboard.py (VM: upload it next to this file)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
import fastboard as fb

SCALE, LS, LR, WD, GAMES_PER_STEP, POS_PER_GAME, EMB = 30.0, 0.1, 2e-3, 1e-2, 256, 16, 128
NPLANES = 18
EXPECT_GAMES = dict(sim_full=355_947, sim_h0=177_882, sim_h1=178_065, fin_full=707_820, fin_h0=344_529, fin_h1=363_291,
                    finh_full=994_875, finh_h0=631_584, finh_h1=650_346)       # PLAN s2 table; finh_* = fin_* + all 287,055 H games (make_data_h.py)
EXPECT_STEPS = dict(sim_full=27_820, sim_h0=13_900, sim_h1=13_920, fin_full=55_300, fin_h0=26_920, fin_h1=28_400,
                    finh_full=77_740, finh_h0=49_360, finh_h1=50_820)   # PLAN s2 table says 28,380, but ceil(363,291/256)=1,420 steps/epoch x 20 = 28,400 (E89 recipe: ceil); PLAN figure is an arithmetic slip, reported
EXPECT_CLASSES = dict(sim_full=5_244, sim_h0=2_629, sim_h1=2_615, fin_full=6_725, fin_h0=3_352, fin_h1=3_373,
                      finh_full=7_365, finh_h0=3_992, finh_h1=4_013)                  # .scratch/e80/meta_log.json
SIM_ROSTERS = ("sim_full", "sim_h0", "sim_h1")


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


# ------------------------------------------------------------------ per-game vector export (same maths as E89 evaluate() / scripts/experiments/e90/embed.py)
@torch.no_grad()
def embed_games(model, G, ids, dev, amp, log, tag, workers, cache_dir, chunk=4000, group=16, block=1024, hb=None):
    """ids = rows of G.  Returns dict: gv_all (n,128) float32 L2-normalised mean of ALL own-position vectors, gv_first60 (own-move ordinal < 60),
    has_first60.  Chunk results are cached in cache_dir so that an interrupted export resumes."""
    model.eval(); cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True); n = len(ids); t0 = time.time(); npos = 0
    for c0 in range(0, n, chunk):
        cf = cache_dir / f"c{c0:08d}.npz"
        if cf.exists(): continue
        cid = ids[c0:c0 + chunk]; m_ = len(cid); groups = [cid[i:i + group] for i in range(0, m_, group)]
        loader = torch.utils.data.DataLoader(EvalSet(G, groups), batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=4 if workers else None)
        sums = torch.zeros(2, m_, EMB, device=dev); cnt = torch.zeros(2, m_, device=dev)
        for gi, b in enumerate(loader):
            base = gi * group; P = len(b["gid"]); gid_all = b["gid"].to(dev) + base; oi = b["oi"].to(dev).long()
            flags = torch.stack([torch.ones_like(oi, dtype=torch.bool), oi < 60])
            for a in range(0, P, block):
                sl = slice(a, min(a + block, P)); m = sl.stop - sl.start
                sub = {k: (v[sl] if torch.is_tensor(v) and v.ndim >= 1 and len(v) == P else v) for k, v in b.items()}
                if m < block:
                    for k in ("grid", "lib", "prev_opp", "prev_own", "ko", "move", "ply", "oi", "eff"):
                        sub[k] = torch.cat([sub[k], sub[k][:1].expand(block - m, *sub[k].shape[1:])])
                X = build_planes(sub, dev, torch.float16 if amp else torch.float32)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    v = model.pos_vec(X, sub["move"].to(dev).long())[:m]
                for bk in range(2):
                    mk = flags[bk, sl]; idx = gid_all[sl][mk]
                    sums[bk].index_add_(0, idx, v[mk]); cnt[bk].index_add_(0, idx, torch.ones(len(idx), device=dev))
            npos += P
            if hb is not None and gi % 20 == 0: hb()
        gv = Fn.normalize(sums / cnt.clamp(min=1)[..., None], dim=-1)
        np.savez(cf, gv_all=gv[0].cpu().numpy(), gv_first60=gv[1].cpu().numpy(), has_all=(cnt[0] > 0).cpu().numpy(), has_first60=(cnt[1] > 0).cpu().numpy(), ids=cid)
        log(f"[{tag}] export chunk {c0}/{n} {npos / max(1e-9, time.time() - t0):.0f} pos/s")
    parts = [np.load(cache_dir / f"c{c0:08d}.npz") for c0 in range(0, n, chunk)]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    out = dict(gv_all=cat("gv_all"), gv_first60=cat("gv_first60"), has_all=cat("has_all"), has_first60=cat("has_first60"), ids=cat("ids"))
    assert (out["ids"] == ids).all() and out["has_all"].all() and np.isfinite(out["gv_all"]).all() and out["gv_all"].shape == (n, EMB)
    assert np.abs(np.linalg.norm(out["gv_all"], axis=1) - 1).max() < 1e-3, "per-game vectors are not unit length"
    model.train(); return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="u_games.npz (s_games.npz-compatible)"); ap.add_argument("--out", required=True)
    ap.add_argument("--roster", required=True, help="sim_full sim_h0 sim_h1 fin_full fin_h0 fin_h1 (or any key of --rosters with --custom-roster)")
    ap.add_argument("--rosters", default=None, help="rosters.npz (default: next to --data)"); ap.add_argument("--tdata", default=None, help="t_games.npz (default: next to --data)")
    ap.add_argument("--export", default="", help="'auto' = every required entry of export_spec.json for this roster; or comma list of names; '' = none")
    ap.add_argument("--export-spec", default=None, help="export_spec.json (default: next to --data); arrays in export_spec.npz next to it")
    ap.add_argument("--export-only", action="store_true", help="never train: need a finished checkpoint (epoch == epochs); only (re)write the exports")
    ap.add_argument("--blocks", type=int, default=10); ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2, help="split the 256-game step into this many micro-batches (recipe: 2; the 12GB school GPU needs 4 = BN batch differs)")
    ap.add_argument("--probe-every", type=int, default=4); ap.add_argument("--probe-games", type=int, default=512)
    ap.add_argument("--bench", type=int, default=0, help="run N steps, print speed + GPU utilisation, exit"); ap.add_argument("--stop-after-epoch", type=int, default=0)
    ap.add_argument("--games-per-step", type=int, default=GAMES_PER_STEP); ap.add_argument("--pos-per-game", type=int, default=16)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--selftest-planes", action="store_true"); ap.add_argument("--name", default=None)
    ap.add_argument("--custom-roster", action="store_true", help="TEST ONLY: a roster that is not one of the six PLAN rosters (no step/size asserts, recipe flags free)")
    a = ap.parse_args(); a.name = a.name or a.roster
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); logf = open(out / "train.log", "a")
    def log(*x):
        s = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] " + " ".join(str(i) for i in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda"); amp = dev.type == "cuda"
    if dev.type == "cuda": torch.backends.cudnn.benchmark = True
    data_dir = Path(a.data).resolve().parent
    G = load_games(a.data); n_games_file = len(G["rid"])
    R = np.load(a.rosters or (data_dir / "rosters.npz")); assert a.roster in R.files, f"roster {a.roster} not in rosters file"
    official = a.roster in EXPECT_GAMES and not a.custom_roster
    if a.custom_roster: assert a.roster not in EXPECT_GAMES, "--custom-roster is for non-PLAN rosters only"
    else: assert a.roster in EXPECT_GAMES, f"{a.roster} is not a PLAN roster (use --custom-roster for tests)"
    mask = R[a.roster].astype(bool); assert len(mask) == n_games_file, "roster mask length != number of games in the data file"
    # ---- roster -> training ids, classes (dense rank of aux game_player among the roster's players); ROLE_GUARD: only mask games ever train
    tr_ids = np.flatnonzero(mask); pl = G["player"][tr_ids].astype(np.int64); ups = np.unique(pl); n_classes = len(ups)
    cls = np.full(n_games_file, -1, np.int16); cls[tr_ids] = np.searchsorted(ups, pl).astype(np.int16); G["cls"] = cls; G["n_classes"] = np.int64(n_classes)
    role = np.full(n_games_file, -1, np.int8); role[tr_ids] = 0; G["role"] = role
    nb = math.ceil(len(tr_ids) / a.games_per_step); total = a.epochs * nb
    if official:
        assert (a.blocks, a.ch, a.epochs, a.games_per_step, a.pos_per_game, a.seed) == (10, 128, 20, 256, 16, 0), "recipe flags differ from E89 setting A"
        assert a.accum in (2, 4), "accum must be 2 (recipe) or 4 (school 12GB GPU, documented deviation)"
        assert len(tr_ids) == EXPECT_GAMES[a.roster], (len(tr_ids), EXPECT_GAMES[a.roster]); assert n_classes == EXPECT_CLASSES[a.roster], (n_classes, EXPECT_CLASSES[a.roster])
        assert total == EXPECT_STEPS[a.roster], (total, EXPECT_STEPS[a.roster])
        if a.roster in SIM_ROSTERS:
            ev = R["eval_players"].astype(np.int64); assert len(ev) == 1481 and not np.isin(ups, ev).any() and not np.isin(G["player"][tr_ids], ev).any(), "sim eval player in a sim roster"
        assert (G["player"][tr_ids] >= 0).all() and (R["u_rid"] == G["rid"]).all(), "player/rid alignment"
    assert len(np.unique(tr_ids)) == len(tr_ids) and (G["cls"][tr_ids] >= 0).all()
    log(f"roster {a.roster}{' (custom)' if not official else ''}: games {len(tr_ids)} classes {n_classes} steps/epoch {nb} total steps {total} (PLAN {EXPECT_STEPS.get(a.roster)}) device {dev} "
        f"blocks {a.blocks} ch {a.ch} accum {a.accum} workers {a.workers}")
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
    start_ep = 0; step = 0; hist = []; vals = []
    ck = out / "ckpt_latest.pt"
    if ck.exists() and not a.bench:
        s = torch.load(ck, map_location=dev, weights_only=False); assert s["args"]["roster"] == a.roster and s["args"]["blocks"] == a.blocks and s["args"]["ch"] == a.ch, "checkpoint belongs to another run"
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        start_ep, step, hist, vals = s["epoch"], s["step"], s["hist"], s["vals"]; torch.set_rng_state(s["rng_cpu"].cpu())
        if dev.type == "cuda": torch.cuda.set_rng_state(s["rng_cuda"].cpu())
        log(f"RESUMED from {ck} epoch {start_ep} step {step}")
    if a.export_only: assert start_ep == a.epochs, "--export-only needs a finished checkpoint"
    nparam = sum(p.numel() for p in model.parameters()); log(f"params {nparam / 1e6:.2f}M  steps/epoch {nb}  total steps {total}")
    # steps_loss.tsv: one line per optimiser step; on resume drop the lines of the epoch that was lost
    sl_path = out / "steps_loss.tsv"
    if sl_path.exists() and not a.bench:
        keep = [l for l in sl_path.read_text().splitlines() if l and int(l.split("\t")[0]) <= step]; sl_path.write_text("".join(l + "\n" for l in keep))
    sl_f = open(sl_path, "a")
    rs_probe = np.random.default_rng(12345); probe_ids = np.sort(rs_probe.choice(tr_ids, min(a.probe_games, len(tr_ids)), replace=False))
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
    def probe(tag):
        st = torch.get_rng_state(); r = summarize(evaluate(model, G, probe_ids, dev, amp, log, tag, a.workers, hb=lambda: status("probe")), G, probe_ids, which=("all",)); torch.set_rng_state(st)
        r["n_games"] = int(len(probe_ids)); return r
    model.train(); bench_done = False
    if not a.export_only:
      for ep in range(start_ep, a.epochs):
        ep_now[0] = ep
        perm = tr_ids[np.random.default_rng([a.seed, 1000, ep]).permutation(len(tr_ids))]
        assert mask[perm].all() and len(perm) == len(tr_ids)               # ROLE_GUARD: only roster games are ever sampled for training
        items = []
        for s0 in range(nb):
            stp = perm[s0 * a.games_per_step:(s0 + 1) * a.games_per_step]; chunks = np.array_split(stp, min(a.accum, len(stp)))
            for ci, c in enumerate(chunks): items.append((ep, c, len(c) / len(stp), ci == len(chunks) - 1))
        loader = torch.utils.data.DataLoader(ds, batch_size=None, sampler=items, num_workers=a.workers, prefetch_factor=6 if a.workers else None, pin_memory=(dev.type == "cuda"), persistent_workers=False)
        acc = torch.zeros((), device=dev); acc_step = torch.zeros((), device=dev); n_acc = 0; t_wait = 0.0; t_log = time.time(); pos_log = 0; t_ep = time.time(); w0 = time.time()
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
            scaler.scale(loss).backward(); acc += loss.detach(); acc_step += loss.detach(); pos_log += P
            if b["last"]:
                for g_ in opt.param_groups: g_["lr"] = lr_at(step, nb, total)
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1; n_acc += 1
                sl_f.write(f"{step}\t{float(acc_step):.7f}\n"); sl_f.flush(); acc_step.zero_()
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
        if a.probe_every and (ep + 1) % a.probe_every == 0 and (ep + 1) < a.epochs:
            r = probe(f"probe ep{ep + 1}"); r["epoch"] = ep + 1; vals.append(r)
            log(f"PROBE(in-sample, {r['n_games']} train games, observation only) epoch {ep + 1}: h {r['h_all']:.4f} top5 {r['top5_all']:.4f} loss {r['loss_all']:.3f}"); status("probe_done", probe_h=r["h_all"])
        s = dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=ep + 1, step=step, hist=hist, vals=vals, rng_cpu=torch.get_rng_state(),
                 rng_cuda=torch.cuda.get_rng_state() if dev.type == "cuda" else torch.zeros(1, dtype=torch.uint8), args=vars(a))
        torch.save(s, out / "ckpt_tmp.pt"); os.replace(out / "ckpt_tmp.pt", ck); log(f"checkpoint saved epoch {ep + 1}"); status("ckpt")
        if a.stop_after_epoch and ep + 1 >= a.stop_after_epoch: log("stop-after-epoch reached"); return
    if not a.export_only:
        assert step == total, (step, total)
    # ---- finished training: final in-sample probe, final model + class list, exports, results.json last
    status("final")
    pf = probe("probe final"); log("PROBE final (in-sample, observation only)", json.dumps(pf))
    torch.save(dict(model=model.state_dict(), players=ups, roster=a.roster, epoch=a.epochs, step=step), out / "final_model.pt")
    exports = []
    if a.export:
        sp_path = Path(a.export_spec or (data_dir / "export_spec.json")); spec = json.loads(sp_path.read_text()); arr = np.load(sp_path.with_suffix(".npz"))
        entries = spec["exports"].get(a.roster, [])
        want = [e for e in entries if (a.export == "auto" and e["required"]) or e["name"] in a.export.split(",")]
        assert want, f"no export entries selected for roster {a.roster} (--export {a.export})"
        TG = None
        for e in want:
            if e["games"] == "t":
                TG = TG or load_games(a.tdata or (data_dir / "t_games.npz")); GG = TG
            else: GG = G
            rows = arr[e["rows"]].astype(np.int64); ev = out / f"vec_{e['name']}.npz"
            if ev.exists(): log("export exists, skip", e["name"]); exports.append(e["name"]); continue
            res = embed_games(model, GG, rows, dev, amp, log, e["name"], a.workers, out / f"vec_{e['name']}.chunks", hb=lambda: status("export"))
            np.savez(ev, gv_all=res["gv_all"], gv_first60=res["gv_first60"], has_first60=res["has_first60"], ids=rows, rid=GG["rid"][rows], net=a.roster, games_file=e["games"], epoch=a.epochs)
            log("export written", ev, res["gv_all"].shape); exports.append(e["name"])
    res = dict(name=a.name, roster=a.roster, blocks=a.blocks, ch=a.ch, epochs=a.epochs, seed=a.seed, accum=a.accum, params_M=nparam / 1e6, n_train=int(len(tr_ids)), n_classes=n_classes,
               steps=step, expected_steps=EXPECT_STEPS.get(a.roster), probe_final=pf, probes=vals, train_loss_curve=hist, exports=exports, args=vars(a), seconds=round(time.time() - t0))
    (out / "results.json").write_text(json.dumps(res, indent=1)); log("RESULTS written", out / "results.json"); status("done", probe_h=pf["h_all"])


if __name__ == "__main__":
    main()
