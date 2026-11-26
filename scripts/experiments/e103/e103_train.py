"""E103 交件版訓練（COPY of e98_train.py；原檔與 scripts/experiments/e98/ 未動）。與原檔的差別（全部，其餘逐字相同）：
  1. TrainData：原本「fold 4 不可進訓練」的斷言改成「訓練玩家集合 ＝ 原 5,898 位（E103_OLD_PLAYERS 檔）∪ fold 4 的 1,469 位（E103_FOLD4 檔）」；fold 只能是 0-4，且 fold 4 的盤＝fold 4 玩家。
  2. --run-name：run 資料夾名（預設仍是 <arm>_s<seed>），交件版用 final_s0。
  3. 監看的 log 行標明 IN-SAMPLE（fold 4 的 600 題現在是訓練內資料，只當分數有在動的健康檢查；不影響訓練：每步的亂數只由 (seed, step) 決定）。
以下為原檔說明：
E98 Rank 全量版：KataGo humanv0 主幹「解凍最後 4 區塊」(ft4) vs「全凍結」(frozen)；E94-C（scripts/experiments/e94/e94_c_train.py）的複本，只改規模與資料讀法。
PLAN: experiments/E98_rank_ft4_full/PLAN.md。模型、頭、CORAL、抽樣、優化、種子規則與 E94-C 完全相同；差別只有：
  * 資料：e98_data.py 產生的 bundle 目錄（.npy，mmap 讀取，不整個載入記憶體）；訓練 bundle ＝ 全部非 fold 4 玩家的盤，每盤 12 個局面。
  * 監看：fold 4 驗證題的一個固定子集（--monitor-q 題），每 --eval-every 步評一次，只看不選（匯出一律用最後一步）。正式評估是 e98_export.py 匯出後在本機 e98_score.py 算。
  * 檢查點：每 --ckpt-every 步（預設 500）原子地覆寫 last.pt（只留最新一個），可續跑（抽樣與 dropout 亂數只由 (seed, step) 決定）。
  * 總步數：speed 子命令量速度後定成「約 --target-h 小時跑完」，上限 12,000、下限 4,000，寫 speed.json。
系統 python3（有 torch；只需 numpy＋torch，不需 pandas／pyarrow）。
子命令 train / speed / stats / dry
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[3]
W = Path(os.environ.get("E98_WORK", ROOT / ".scratch" / "e98"))                    # 工作資料夾（學校：~/gogame_e98）
KS = Path(os.environ.get("E98_KS", ROOT / ".scratch" / "e52" / "katago_src" / "python"))
META_ROWS = Path(os.environ.get("E98_META_ROWS", ROOT / ".scratch" / "e52" / "kata" / "meta_rows.npy"))
NS = (5, 10, 15, 20)
K = 10
FIRST_FT = 14                     # ft4 = blocks 14,15,16,17 (+ norm_trunkfinal); b18c384nbt has 18 blocks
E1 = math.exp(-1.0)
BITS_W = 993
MIN_FREE_GB = float(os.environ.get("E98_MIN_FREE_GB", "6"))
STEPS_MIN, STEPS_MAX = 4000, 12000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_gb(p=W) -> float:
    return shutil.disk_usage(p).free / 1e9


# ----------------------------------------------------------------------------------------------- data
class Bundle:
    """e98_data.py 的 bundle 目錄（bits/glob/loc/ply/gid .npy＋meta.npz）。bits／glob 以 mmap 讀。局面依 gid 排序，每盤一段連續列。"""

    def __init__(self, bdir, limit_games: int = 0):
        bdir = Path(bdir)
        m = np.load(bdir / "meta.npz", allow_pickle=False)
        self.meta = {k: m[k] for k in m.files}
        gid = np.load(bdir / "gid.npy")
        assert (np.diff(gid) >= 0).all()
        G = len(self.meta["g_uid"])
        if limit_games:
            G = min(G, limit_games)
        self.G = G
        self.start = np.searchsorted(gid, np.arange(G))
        self.end = np.searchsorted(gid, np.arange(G), side="right")
        assert ((self.end - self.start) > 0).all(), "game without positions"
        nrows = int(self.end[-1])
        self.bits = np.load(bdir / "bits.npy", mmap_mode="r")
        self.glob = np.load(bdir / "glob.npy", mmap_mode="r")
        assert self.bits.shape[1] == BITS_W and self.bits.shape[0] >= nrows
        self.loc = np.load(bdir / "loc.npy")[:nrows].astype(np.int64)
        self.nrows = nrows
        self.player = self.meta["g_player"][:G]
        self.rank = self.meta["g_rank"][:G].astype(np.int64)
        self.fold = self.meta["g_fold"][:G]
        self.color = self.meta["g_color"][:G].astype(np.int64)
        self.row_game = np.repeat(np.arange(G), self.end - self.start)


class TrainData(Bundle):
    def __init__(self, bdir, limit_games: int = 0):
        super().__init__(bdir, limit_games)
        assert set(self.fold.tolist()) <= {0, 1, 2, 3, 4}
        if not limit_games:                                                # E103: 訓練玩家集合＝原 5,898 位 ∪ fold 4 的 1,469 位
            old_p = set(Path(os.environ["E103_OLD_PLAYERS"]).read_text().split())
            f4_p = set(Path(os.environ["E103_FOLD4"]).read_text().split())
            ps = set(self.player.tolist())
            assert len(old_p) == 5898 and len(f4_p) == 1469 and not (old_p & f4_p), (len(old_p), len(f4_p))
            assert ps == old_p | f4_p, f"train players {len(ps)} != old 5898 + fold4 1469"
            assert set(self.player[self.fold == 4].tolist()) == f4_p and set(self.player[self.fold != 4].tolist()) == old_p, "fold-4 games/players mismatch"
            log(f"E103 player-set assertion OK: {len(ps)} train players = {len(old_p)} old + {len(f4_p)} fold-4; games {self.G} (fold-4 games {int((self.fold == 4).sum())})")
        self.by_rank = []
        for r in range(K):
            ps = {}
            for g in np.flatnonzero(self.rank == r):
                ps.setdefault(self.player[g], []).append(g)
            self.by_rank.append([np.array(v) for _, v in sorted(ps.items())])
        assert all(len(b) > 0 for b in self.by_rank), "a rank has no training player"

    def sample_step(self, seed: int, step: int, n_bags: int, pos_per_game: int, max_n: int | None = None):
        """一步的訓練袋。只由 (seed, step) 決定。-> rows(局面列號), game_of_row, bag_of_game, labels   （與 E94-C 相同）"""
        rng = np.random.default_rng([seed, 5000 + step])
        rows, game_of_row, bag_of_game, labels = [], [], [], []
        gi = 0
        for b in range(n_bags):
            r = int(rng.integers(K))
            pl = self.by_rank[r][int(rng.integers(len(self.by_rank[r])))]
            n = NS[int(rng.integers(len(NS)))]
            if max_n:
                n = min(n, max_n)
            gs = pl[rng.permutation(len(pl))[:n]]
            for g in gs:
                pos = np.arange(self.start[g], self.end[g])
                if len(pos) > pos_per_game:
                    pos = np.sort(rng.permutation(pos)[:pos_per_game])
                rows.append(pos)
                game_of_row.append(np.full(len(pos), gi))
                bag_of_game.append(b)
                gi += 1
            labels.append(r)
        return np.concatenate(rows), np.concatenate(game_of_row), np.array(bag_of_game), np.array(labels)


class ValData(Bundle):
    """驗證 bundle（fold 4 驗證題的 41,589 盤）＋題目結構。questions: val_questions.npz（qid, rows=rid 的 Nq×20（-1 補）, ns, y, player）。
    rid ＝ E29 bundle 的列號（＝ E92／E95 匯出檔的 rid）。synth=True（只給乾跑）：由現有的盤湊出 5 盤的假題。"""

    def __init__(self, bdir, qfile=None, limit_games: int = 0, synth: bool = False):
        super().__init__(bdir, limit_games)
        assert set(self.fold.tolist()) == {4}, "validation bundle must be fold 4 only"
        self.rid = self.meta["g_row"][:self.G].astype(np.int64)
        o = np.argsort(self.rid)
        self.rid_sorted, self.order = self.rid[o], o
        if synth:
            qs, y, pl_ = [], [], []
            for r in range(K):                                   # 假題：同段位的 5 盤一袋（可能不同玩家；只為了在小資料上走通程式）
                gs = np.flatnonzero(self.rank == r)
                for i in range(0, len(gs) - 4, 5):
                    qs.append(self.rid[gs[i:i + 5]])
                    y.append(r)
                    pl_.append(self.player[gs[i]])
            rows = np.full((len(qs), 20), -1, np.int64)
            for i, r in enumerate(qs):
                rows[i, :5] = r
            self.q = dict(rows=rows, ns=np.full(len(qs), 5), y=np.array(y), player=np.array(pl_), qid=np.array([f"syn_{i}" for i in range(len(qs))]))
        else:
            z = np.load(qfile, allow_pickle=False)
            self.q = {k: z[k] for k in ("rows", "ns", "y", "player", "qid")}
            need = np.unique(self.q["rows"][self.q["rows"] >= 0])
            assert len(need) == self.G and (need == self.rid_sorted).all(), f"validation bundle games != the games of the questions ({self.G} vs {len(need)})"
        rows = self.q["rows"]
        j = np.searchsorted(self.rid_sorted, np.where(rows >= 0, rows, 0))
        assert (self.rid_sorted[j][rows >= 0] == rows[rows >= 0]).all(), "question game missing from the validation bundle"
        self.q_gidx = np.where(rows >= 0, self.order[j], -1)               # 題 -> bundle 內的盤序號（-1 補）
        self.nq = len(rows)


MROWS = None


def meta_rows():
    global MROWS
    if MROWS is None:
        MROWS = {c: torch.tensor(r) for c, r in enumerate(np.load(META_ROWS))}
    return MROWS


def make_inputs(D: Bundle, rows, dev):
    rows = np.asarray(rows)
    b = np.unpackbits(np.asarray(D.bits[rows]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
    mr = meta_rows()
    return (torch.from_numpy(b).to(dev).float(), torch.from_numpy(np.asarray(D.glob[rows])).to(dev).float(),
            torch.stack([mr[int(c)] for c in D.color[D.row_game[rows]]]).to(dev), torch.from_numpy(D.loc[rows]).to(dev))


# ----------------------------------------------------------------------------------------------- trunk（與 E94-C 相同）
class Stop(Exception):
    def __init__(self, x):
        self.x = x


class Trunk:
    def __init__(self, arm: str, dev: str, ckpt: Path):
        sys.path.insert(0, str(KS))
        from katago.train.load_model import load_model
        self.model, _, _ = load_model(str(ckpt), use_swa=False, device=dev, pos_len=19)
        self.model.eval()
        self.dev, self.arm = dev, arm
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = []
        if arm == "ft4":
            self.params = [p for blk in self.model.blocks[FIRST_FT:] for p in blk.parameters()] + list(self.model.norm_trunkfinal.parameters())
            for p in self.params:
                p.requires_grad_(True)
        self.amp = dev.startswith("cuda")

    @torch.no_grad()
    def front(self, bx, gx, mx):
        h = self.model.blocks[FIRST_FT].register_forward_pre_hook(lambda m, a, kw: (_ for _ in ()).throw(Stop(a[0] if a else kw["x"])), with_kwargs=True)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                self.model(bx, gx, input_meta=mx)
            raise RuntimeError("hook did not fire")
        except Stop as e:
            return e.x
        finally:
            h.remove()

    def tail(self, x, grad: bool):
        B = x.shape[0]
        mask = torch.ones(B, 1, 19, 19, device=x.device)
        kw = dict(mask=mask, mask_sum_hw=mask.sum(dim=(2, 3), keepdim=True), mask_sum=mask.sum(), extra_outputs=None)
        out = x
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            for i in range(FIRST_FT, len(self.model.blocks)):
                blk = self.model.blocks[i]
                if grad:
                    res = checkpoint(lambda o, _b=blk: _b(o, block_shared_data={}, **kw), out, use_reentrant=False)
                else:
                    res = blk(out, block_shared_data={}, **kw)
                out = out + res
            out = self.model.norm_trunkfinal(out, mask=mask, mask_sum_hw=kw["mask_sum_hw"], mask_sum=kw["mask_sum"])
            out = self.model.act_trunkfinal(out)
        return out

    def u(self, bx, gx, mx, loc, grad: bool):
        x = self.front(bx, gx, mx)
        if grad:
            f = self.tail(x, True)
        else:
            with torch.no_grad():
                f = self.tail(x, False)
        f = f.float()
        ff = f.flatten(2)
        bm = ff.mean(2)
        l = loc.clamp(min=0)
        at = ff.gather(2, l[:, None, None].expand(-1, ff.shape[1], 1))[:, :, 0]
        at = torch.where((loc < 0)[:, None], bm, at)
        return torch.cat([at, bm], 1)

    def param_hash(self):
        groups = {"front": [], "ft4_params": [], "other": []}
        ft_ids = {id(p) for blk in self.model.blocks[FIRST_FT:] for p in blk.parameters()} | {id(p) for p in self.model.norm_trunkfinal.parameters()}
        for n, p in sorted(self.model.named_parameters()):
            k = "ft4_params" if id(p) in ft_ids else ("front" if not n.startswith(("policy_head", "value_head", "intermediate", "norm_intermediate")) else "other")
            groups[k].append(p)
        out = {}
        for k, ps in groups.items():
            h = hashlib.sha256()
            for p in ps:
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            out[k] = h.hexdigest()[:16]
        return out


# ----------------------------------------------------------------------------------------------- head（與 E94-C 相同）
class Head(nn.Module):
    def __init__(self, mu, sd, D=256, hid=256, drop=0.1):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        self.proj = nn.Linear(768, D)
        self.phi = nn.Sequential(nn.Linear(D, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, hid), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, 1))
        c = 1.0 - np.arange(1, K) / K
        b = np.log(c / (1 - c))
        self.b0 = nn.Parameter(torch.tensor(float(b[0])))
        self.gap_raw = nn.Parameter(torch.tensor(np.log(np.expm1(b[:-1] - b[1:])), dtype=torch.float32))

    def pos_vec(self, u):
        return self.proj((u - self.mu) / self.sd)

    def game_vec(self, pv, game_of_row, G):
        S = torch.zeros(G, pv.shape[1], device=pv.device, dtype=pv.dtype).index_add_(0, game_of_row, pv)
        C_ = torch.zeros(G, device=pv.device, dtype=pv.dtype).index_add_(0, game_of_row, torch.ones(len(game_of_row), device=pv.device, dtype=pv.dtype))
        return S / C_[:, None]

    def score(self, gv, bag_idx, bag_mask):
        h = self.phi(gv)[bag_idx] * bag_mask[..., None]
        bar = h.sum(1) / bag_mask.sum(1, keepdim=True)
        return self.rho(bar)[:, 0]

    def thresholds(self):
        return torch.cat([self.b0[None], self.b0 - torch.cumsum(F.softplus(self.gap_raw), 0)])

    def cum(self, s):
        return torch.sigmoid(s[:, None] + self.thresholds()[None, :])

    def probs(self, s):
        c = self.cum(s)
        p = torch.cat([1 - c[:, :1], c[:, :-1] - c[:, 1:], c[:, -1:]], 1)
        return p.clamp_min(0) / p.clamp_min(0).sum(1, keepdim=True)

    def coral_loss(self, s, y):
        lv = (y[:, None] > torch.arange(K - 1, device=y.device)[None, :]).float()
        return F.binary_cross_entropy(self.cum(s).clamp(1e-6, 1 - 1e-6), lv)


def pad_bags(bag_of_game, B, dev):
    n = np.bincount(bag_of_game, minlength=B)
    nmax = int(n.max())
    idx = np.zeros((B, nmax), np.int64)
    mask = np.zeros((B, nmax), np.float32)
    for b in range(B):
        g = np.flatnonzero(bag_of_game == b)
        idx[b, :len(g)] = g
        mask[b, :len(g)] = 1
    return torch.from_numpy(idx).to(dev), torch.from_numpy(mask).to(dev)


# ----------------------------------------------------------------------------------------------- scoring（官方計分；類別平衡；與 E94-C 相同）
def _smat():
    d = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def decide_ev(p):
    return np.argmax(np.round(np.asarray(p, np.float64) @ _smat(), 12), axis=1)


def rank_scores(pred, truth):
    d = np.abs(pred - truth)
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def balanced(sc, truth):
    _, inv, cnt = np.unique(truth, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return float((sc * w).sum() / w.sum())


# ----------------------------------------------------------------------------------------------- embedding / bag outputs
@torch.no_grad()
def embed_games(V: ValData, trunk: Trunk, head: Head, dev, games, chunk_pos=256, gbs=48, progress=None):
    """games（bundle 內盤序號的陣列）-> (len(games), D) 盤向量（該盤全部局面的平均）。"""
    head.eval()
    out = torch.zeros(len(games), head.proj.out_features, device=dev)
    for s in range(0, len(games), gbs):
        gs = games[s:s + gbs]
        rows = np.concatenate([np.arange(V.start[g], V.end[g]) for g in gs])
        gor = torch.from_numpy(np.concatenate([np.full(V.end[g] - V.start[g], j) for j, g in enumerate(gs)])).to(dev)
        u = torch.cat([trunk.u(*make_inputs(V, rows[m:m + chunk_pos], dev), grad=False) for m in range(0, len(rows), chunk_pos)])
        out[s:s + len(gs)] = head.game_vec(head.pos_vec(u), gor, len(gs))
        if progress:
            progress(s + len(gs), len(games))
    return out


@torch.no_grad()
def question_outputs(head: Head, gv_full, q_gidx, dev):
    """gv_full: (G_total, D)（bundle 盤序號）；q_gidx: (Nq,20) -> s (Nq,), probs (Nq,10)。"""
    head.eval()
    idx = torch.from_numpy(np.where(q_gidx >= 0, q_gidx, 0)).to(dev)
    mask = torch.from_numpy((q_gidx >= 0).astype(np.float32)).to(dev)
    s = head.score(gv_full, idx, mask)
    return s.float().cpu().numpy(), head.probs(s).float().cpu().numpy()


def bal_by_n(p, y, ns):
    sc = rank_scores(decide_ev(p), y)
    per = {n: balanced(sc[ns == n], y[ns == n]) for n in NS if (ns == n).any()}
    return {"mean": float(np.mean(list(per.values()))), "per_N": per, "argmax_mean": float(np.mean([balanced(rank_scores(p.argmax(1), y)[ns == n], y[ns == n]) for n in per]))}


def monitor_eval(V: ValData, qsel, trunk, head, dev):
    """監看：固定題子集 qsel（只看不選）。"""
    games = np.unique(V.q_gidx[qsel][V.q_gidx[qsel] >= 0])
    gv = embed_games(V, trunk, head, dev, games)
    full = torch.zeros(V.G, gv.shape[1], device=dev)
    full[torch.from_numpy(games).to(dev)] = gv
    s, p = question_outputs(head, full, V.q_gidx[qsel], dev)
    head.train()
    r = bal_by_n(p, V.q["y"][qsel], V.q["ns"][qsel])
    r["n_questions"], r["n_games"] = int(len(qsel)), int(len(games))
    return r


def monitor_questions(V: ValData, n: int):
    n = min(n, V.nq)
    return np.sort(np.random.default_rng([0, 98]).choice(V.nq, n, replace=False))


# ----------------------------------------------------------------------------------------------- stats
def cmd_stats(a):
    dev = a.device
    D = TrainData(a.train, a.limit_games)
    trunk = Trunk("frozen", dev, Path(a.ckpt))
    rng = np.random.default_rng([0, 999])
    rows = np.sort(rng.choice(D.nrows, min(a.n, D.nrows), replace=False))
    U = []
    for s in range(0, len(rows), 128):
        U.append(trunk.u(*make_inputs(D, rows[s:s + 128], dev), grad=False).cpu().numpy())
    U = np.concatenate(U)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mu=U.mean(0).astype(np.float32), sd=(U.std(0) + 1e-6).astype(np.float32), n=len(rows))
    log("stats saved", out, "from", len(rows), "frozen-trunk positions")


# ----------------------------------------------------------------------------------------------- train
def build(a, dev):
    z = np.load(a.stats)
    torch.manual_seed(a.seed)
    head = Head(z["mu"], z["sd"]).to(dev)
    trunk = Trunk(a.arm, dev, Path(a.ckpt))
    groups = [{"params": list(head.parameters()), "lr": a.lr_head, "base": a.lr_head}]
    if trunk.params:
        groups.append({"params": trunk.params, "lr": a.lr_trunk, "base": a.lr_trunk})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.startswith("cuda"))
    return trunk, head, opt, scaler


def lr_scale(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def save_ckpt(path, trunk, head, opt, scaler, step, hist, a):
    if free_gb(path.parent) < MIN_FREE_GB:
        raise SystemExit(f"disk free {free_gb(path.parent):.2f}GB < {MIN_FREE_GB}GB: stop (checkpoint not written)")
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "head": head.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "trunk_ft": [p.detach().cpu() for p in trunk.params], "hist": hist, "args": {k: v for k, v in vars(a).items() if not k.startswith("_")}}, tmp)
    os.replace(tmp, path)


def train_loop(a, D: TrainData, V: ValData | None, dev):
    run = Path(a.out) / (a.run_name or f"{a.arm}_s{a.seed}")
    run.mkdir(parents=True, exist_ok=True)
    trunk, head, opt, scaler = build(a, dev)
    total = a.steps
    warm = max(1, int(0.05 * total))
    qsel = monitor_questions(V, a.monitor_q) if V is not None else None
    step0, hist = 0, []
    ck = run / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        assert s["args"].get("steps") == total and s["args"].get("seed") == a.seed and s["args"].get("arm") == a.arm, "checkpoint was made with different --steps/--seed/--arm"
        head.load_state_dict(s["head"])
        opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"])
        for p, v in zip(trunk.params, s["trunk_ft"]):
            p.data.copy_(v.to(dev))
        step0, hist = s["step"], s["hist"]
        log(f"resumed {run.name} at step {step0}/{total}")
    all_params = list(head.parameters()) + trunk.params
    head.train()
    t0, tl = time.time(), []
    h0 = trunk.param_hash() if a.hash_check else None
    for step in range(step0, total):
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_scale(step, total, warm)
        torch.manual_seed(a.seed * 1_000_003 + step)
        rows, gor, bog, lab = D.sample_step(a.seed, step, a.bags, a.pos_per_game, a.max_n)
        G, B = len(bog), len(lab)
        gor_t = torch.from_numpy(gor).to(dev)
        idx, msk = pad_bags(bog, B, dev)
        grad = a.arm == "ft4"
        u = torch.cat([trunk.u(*make_inputs(D, rows[m:m + a.micro], dev), grad=grad) for m in range(0, len(rows), a.micro)])
        gv = head.game_vec(head.pos_vec(u), gor_t, G)
        loss = head.coral_loss(head.score(gv, idx, msk), torch.from_numpy(lab).to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        if a.hash_check and step == step0:
            a._gradnorm = {"head": float(sum(p.grad.norm() ** 2 for p in head.parameters() if p.grad is not None) ** .5),
                           "blocks": {i: float(sum(p.grad.norm() ** 2 for p in trunk.model.blocks[i].parameters() if p.grad is not None) ** .5) for i in range(len(trunk.model.blocks))},
                           "norm_trunkfinal": float(sum(p.grad.norm() ** 2 for p in trunk.model.norm_trunkfinal.parameters() if p.grad is not None) ** .5),
                           "n_trunk_params_with_grad": int(sum(p.grad is not None for p in trunk.model.parameters()))}
        scaler.step(opt)
        scaler.update()
        tl.append(loss.item())
        if not math.isfinite(tl[-1]):
            raise SystemExit(f"non-finite loss at step {step + 1}: stop (checkpoint from step {step0} or earlier is intact)")
        if (step + 1) % a.log_every == 0:
            log(f"{run.name} step {step + 1}/{total} loss {np.mean(tl[-a.log_every:]):.4f} pos/step {len(rows)} gn {float(gn):.2f} {time.time() - t0:.0f}s"
                + (f" vram {torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if dev.startswith("cuda") else ""))
        if V is not None and ((step + 1) % a.eval_every == 0 or step + 1 == total):
            te = time.time()
            r = monitor_eval(V, qsel, trunk, head, dev)
            hist.append({"step": step + 1, "train_loss": float(np.mean(tl[-a.eval_every:])), "eval_secs": round(time.time() - te), "monitor": r})
            log(f"{run.name} MONITOR[IN-SAMPLE: fold-4 questions are training data; health check only] step {step + 1}: mean ev {r['mean']:.4f} argmax {r['argmax_mean']:.4f} " + " ".join(f"N{n}={v:.3f}" for n, v in r["per_N"].items()) + f" ({r['n_questions']} q)")
            (run / "hist.json").write_text(json.dumps(hist, indent=1))
        if ((step + 1) % a.ckpt_every == 0 and step + 1 < total) or (a.stop_after and step + 1 == a.stop_after):
            save_ckpt(ck, trunk, head, opt, scaler, step + 1, hist, a)
        if a.stop_after and step + 1 == a.stop_after:
            return None, trunk, head, opt
    fin = {"arm": a.arm, "seed": a.seed, "steps": total, "secs_this_call": round(time.time() - t0), "last_monitor": hist[-1]["monitor"] if hist else None,
           "note": "export uses the LAST step (no selection on any step). monitor = fixed fold-4 question subset, watching only."}
    if h0 is not None:
        fin["param_hash_before"], fin["param_hash_after"] = h0, trunk.param_hash()
    save_ckpt(ck, trunk, head, opt, scaler, total, hist, a)           # 最後一步（匯出用）
    (run / "final.json").write_text(json.dumps(fin, indent=1))
    return fin, trunk, head, opt


def add_common(p):
    p.add_argument("--arm", choices=["frozen", "ft4"], default="ft4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=0, help="total steps (train needs it; take it from speed.json)")
    p.add_argument("--train", default=str(W / "bundle_train"))
    p.add_argument("--val", default=str(W / "bundle_val12"))
    p.add_argument("--val-q", default=str(W / "sel" / "val_questions.npz"))
    p.add_argument("--ckpt", default=str(W / "ckpt.ckpt"))
    p.add_argument("--stats", default=str(W / "stats.npz"))
    p.add_argument("--out", default=str(W / "runs"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--bags", type=int, default=12)
    p.add_argument("--pos-per-game", type=int, default=8)
    p.add_argument("--max-n", type=int, default=0)
    p.add_argument("--micro", type=int, default=240)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-trunk", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--monitor-q", type=int, default=600)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--hash-check", action="store_true")
    p.add_argument("--limit-games", type=int, default=0, help="TEST ONLY: use only the first N games of the train bundle")
    p.add_argument("--synth-val", action="store_true", help="TEST ONLY: synthetic questions from the games present in a partial validation bundle")
    p.add_argument("--stop-after", type=int, default=0)
    p.add_argument("--run-name", default="", help="run folder name under --out (default <arm>_s<seed>)")


def load_val(a):
    if not a.val or not Path(a.val).exists():
        return None
    return ValData(a.val, None if a.synth_val else a.val_q, synth=a.synth_val)


def cmd_train(a):
    assert a.steps > 0, "--steps required"
    assert STEPS_MIN <= a.steps <= STEPS_MAX or a.limit_games, f"steps {a.steps} outside [{STEPS_MIN},{STEPS_MAX}] (PLAN)"
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"disk free {free_gb():.2f}GB < {MIN_FREE_GB}GB")
    D = TrainData(a.train, a.limit_games)
    V = load_val(a)
    log(f"data: train games {D.G} positions {D.nrows} players {len(set(D.player.tolist()))} | val games {V.G if V else None} questions {V.nq if V else None}")
    fin, *_ = train_loop(a, D, V, a.device)
    log("final", json.dumps({k: fin[k] for k in ("arm", "seed", "steps", "secs_this_call")}))


def cmd_speed(a):
    """ft4 臂 200 步量速度＋監看與匯出的推論速度 -> 依「約 --target-h 小時」決定總步數（1,000 的倍數，4,000–12,000），寫 speed.json。"""
    D = TrainData(a.train, a.limit_games)
    V = load_val(a)
    trunk, head, opt, scaler = build(a, a.device)
    all_params = list(head.parameters()) + trunk.params
    head.train()
    t0, pps = None, []
    cuda = a.device.startswith("cuda")
    for step in range(a.speed_steps):
        if step == 10:
            torch.cuda.synchronize() if cuda else None
            t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_scale(step, 3000, 150)
        torch.manual_seed(step)
        rows, gor, bog, lab = D.sample_step(a.seed, step, a.bags, a.pos_per_game, a.max_n)
        pps.append(len(rows))
        idx, msk = pad_bags(bog, len(lab), a.device)
        u = torch.cat([trunk.u(*make_inputs(D, rows[m:m + a.micro], a.device), grad=(a.arm == "ft4")) for m in range(0, len(rows), a.micro)])
        gv = head.game_vec(head.pos_vec(u), torch.from_numpy(gor).to(a.device), len(bog))
        loss = head.coral_loss(head.score(gv, idx, msk), torch.from_numpy(lab).to(a.device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(opt)
        scaler.update()
        if (step + 1) % 25 == 0:
            log(f"speed step {step + 1} loss {loss.item():.4f}")
    if cuda:
        torch.cuda.synchronize()
    t_step = (time.time() - t0) / (a.speed_steps - 10)
    # 推論速度：用驗證 bundle 的前 --speed-eval-pos 個局面
    t_pos, n_mon, n_exp = None, 0, 0
    if V is not None:
        rows = np.arange(min(a.speed_eval_pos, V.nrows))
        t1 = time.time()
        with torch.no_grad():
            for m in range(0, len(rows), 256):
                trunk.u(*make_inputs(V, rows[m:m + 256], a.device), grad=False)
        if cuda:
            torch.cuda.synchronize()
        t_pos = (time.time() - t1) / len(rows)
        qsel = monitor_questions(V, a.monitor_q)
        g_mon = np.unique(V.q_gidx[qsel][V.q_gidx[qsel] >= 0])
        n_mon = int(sum(V.end[g] - V.start[g] for g in g_mon))
        n_exp = int(V.nrows) + int(sum(int(x) for x in a.extra_export_positions.split(",") if x))
    t_mon = (t_pos or 0) * n_mon
    t_export = (t_pos or 0) * n_exp
    budget = a.target_h * 3600 - t_export
    per1000 = 1000 * t_step + (1000 // a.eval_every) * t_mon
    steps = int(budget / per1000) * 1000
    clamp = max(STEPS_MIN, min(STEPS_MAX, steps))
    info = {"t_step_ft4": t_step, "sec_per_position_infer": t_pos, "monitor_positions": n_mon, "t_monitor": t_mon, "export_positions": n_exp, "t_export_est": t_export,
            "target_h": a.target_h, "steps_uncapped": steps, "steps": clamp, "limits": [STEPS_MIN, STEPS_MAX],
            "est_h_total": (clamp * t_step + (clamp // a.eval_every) * t_mon + t_export) / 3600, "pos_per_step_mean": float(np.mean(pps)),
            "vram_peak_GB": (torch.cuda.max_memory_allocated() / 1e9) if cuda else None, "device": a.device, "when": time.strftime("%F %T")}
    Path(a.speed_out).write_text(json.dumps(info, indent=1))
    log("speed", json.dumps(info))


# ----------------------------------------------------------------------------------------------- dry run (CPU)
def cmd_dry(a):
    """兩臂各：3 步＋1 次監看＋存讀檢查點；ft4 後 4 區塊有梯度；frozen 主幹參數雜湊不變；中斷續跑與不中斷逐位相同。"""
    global MIN_FREE_GB
    MIN_FREE_GB = float(os.environ.get("E98_MIN_FREE_GB", "0.3"))
    dev = "cpu"
    a.device, a.hash_check = dev, True
    base = Path(a.out)
    shutil.rmtree(base / "dry", ignore_errors=True)
    D = TrainData(a.train, a.limit_games)
    V = load_val(a)
    log(f"DRY data: train games {D.G} positions {D.nrows} | val games {V.G} synthetic questions {V.nq}")
    if not Path(a.stats).exists():
        a.stats = str(base / "dry" / "stats_dry.npz")
        cmd_stats(argparse.Namespace(device=dev, train=a.train, limit_games=a.limit_games, ckpt=a.ckpt, n=120, out=a.stats))
    report = {}
    for arm in ("frozen", "ft4"):
        a.arm = arm
        a.seed, a.steps, a.bags, a.pos_per_game, a.max_n = 0, 3, 3, 2, 10
        a.eval_every, a.ckpt_every, a.log_every, a.micro, a.monitor_q = 3, 2, 1, 64, 4
        a.stop_after = 0
        a.out = str(base / "dry" / "full")
        fin, trunk, head, opt = train_loop(a, D, V, dev)
        head_full = {k: v.clone() for k, v in head.state_dict().items()}
        grads = dict(a._gradnorm)
        before, after = fin["param_hash_before"], fin["param_hash_after"]
        ck = Path(a.out) / f"{arm}_s0" / "last.pt"
        a.out = str(base / "dry" / "resume")
        a.stop_after = 2
        train_loop(a, D, V, dev)
        a.stop_after = 0
        fin2, trunk2, head2, _ = train_loop(a, D, V, dev)
        same = all(torch.equal(head_full[k], head2.state_dict()[k]) for k in head_full)
        same_ft = all(torch.equal(p1.cpu(), p2.cpu()) for p1, p2 in zip(trunk.params, trunk2.params)) if trunk.params else True
        report[arm] = {"gradnorm_step0": grads, "hash_before": before, "hash_after": after,
                       "front_unchanged": before["front"] == after["front"], "ft4_params_changed": before["ft4_params"] != after["ft4_params"],
                       "resume_head_equal": same, "resume_ft4_params_equal": same_ft, "ckpt_MB": round(ck.stat().st_size / 1e6, 1), "ckpt_files_in_run_dir": sorted(p.name for p in ck.parent.iterdir()),
                       "monitor_last": fin["last_monitor"], "n_trainable_trunk": len(trunk.params), "n_trunk_params_with_requires_grad": sum(p.requires_grad for p in trunk.model.parameters())}
        log("DRY", arm, json.dumps(report[arm], default=str))
        del trunk, trunk2
    (base / "dry" / "dry_report.json").write_text(json.dumps(report, indent=1, default=str))
    ok = (report["frozen"]["front_unchanged"] and report["frozen"]["hash_before"]["ft4_params"] == report["frozen"]["hash_after"]["ft4_params"]
          and report["frozen"]["n_trunk_params_with_requires_grad"] == 0
          and report["ft4"]["front_unchanged"] and report["ft4"]["ft4_params_changed"]
          and all(report["ft4"]["gradnorm_step0"]["blocks"][i] > 0 for i in range(FIRST_FT, 18))
          and all(report["ft4"]["gradnorm_step0"]["blocks"][i] == 0 for i in range(FIRST_FT))
          and all(report[x]["resume_head_equal"] and report[x]["resume_ft4_params_equal"] for x in ("frozen", "ft4")))
    log("DRY RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("train", "speed", "dry"):
        p = sub.add_parser(nm)
        add_common(p)
        if nm == "speed":
            p.add_argument("--speed-steps", type=int, default=200)
            p.add_argument("--speed-eval-pos", type=int, default=3000)
            p.add_argument("--target-h", type=float, default=10.0)
            p.add_argument("--extra-export-positions", default="", help="comma-separated position counts of additional export bundles (e.g. the 48-position validation bundle)")
            p.add_argument("--speed-out", default=str(W / "speed.json"))
    p = sub.add_parser("stats")
    p.add_argument("--train", default=str(W / "bundle_train")); p.add_argument("--ckpt", default=str(W / "ckpt.ckpt"))
    p.add_argument("--out", default=str(W / "stats.npz")); p.add_argument("--n", type=int, default=4000); p.add_argument("--device", default="cuda"); p.add_argument("--limit-games", type=int, default=0)
    a = ap.parse_args()
    if getattr(a, "max_n", None) == 0:
        a.max_n = None
    {"train": cmd_train, "speed": cmd_speed, "dry": cmd_dry, "stats": cmd_stats}[a.cmd](a)
