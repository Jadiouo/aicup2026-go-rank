"""Training loop (checkpoint / resume) + backends.

`run_training` is backend-agnostic: all state that decides what happens next (epoch, batch cursor, step, history) lives in
`state.json`; the epoch order and per-step symmetry come from named seeds, so an interrupted run that resumes from
the last checkpoint reproduces the uninterrupted run (tests/test_e13_train.py).  Backends:
  ReferenceBackend  numpy softmax regression on pooled planes (tests, CPU dry run; not a model of record)
  TorchBackend      KataGo b18c384nbt trunk (forward hook cuts at act_trunkfinal) + mean-pool + Linear(384,10)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gogame.constants import SEED
from gogame.seeds import derive_seed

from . import lib
from .shards import ShardSet, unpack_bits


@dataclass
class TrainConfig:
    scope: str = "head"                 # head | last4 | full
    epochs: int = 3
    batch: int = 256
    lr_head: float = 1e-3
    lr_trunk: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    seed: int = 0
    ckpt_every: int = 1000              # steps
    keep_last: int = 2
    label_sigma: float = 0.0
    sym_aug: bool = True
    class_weighted: bool = True
    patience: int = 2                   # epochs without es-NLL improvement
    max_steps: int = 0                  # 0 = full schedule; >0 stops early WITHOUT changing the lr schedule (tests / smoke)
    amp: bool = True
    log_every: int = 100                # steps between loss / throughput log lines (and loss_log entries in state.json)


def lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(derive_seed(SEED, "e13", "order", str(seed), str(epoch))).permutation(n)


def step_symmetry(seed: int, step: int) -> int:
    return int(np.random.default_rng(derive_seed(SEED, "e13", "sym", str(seed), str(step))).integers(8))


def apply_symmetry(x: np.ndarray, k: int) -> np.ndarray:
    """D4 element k (0..7) on the last two axes of (..., 19, 19): k&4 transpose, then k&3 rotations."""
    if k & 4:
        x = np.swapaxes(x, -1, -2)
    return np.rot90(x, k & 3, axes=(-2, -1))


def _atomic_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def run_training(backend, train: ShardSet, cfg: TrainConfig, outdir: Path, es_eval=None, log=print, stop_file=None, es_every: int = 0,
                 ckpt_every_sec: float = 900.0) -> dict:
    """es_eval(backend) -> float (bag-level NLL on the early-stopping players; lower is better) or None.
    Resumes from outdir/state.json if present.
    stop_file: if this path exists (or SIGTERM/SIGINT arrives) the loop saves a checkpoint after the current step and returns with
      done=False, st["stopped_by"] set (graceful pause for GPU hand-over; rerun the same command to continue).
    es_every: >0 logs es NLL every that many steps into st["es_log"] (monitoring only; never touches best / early stopping).
    ckpt_every_sec: also checkpoint on wall-clock (the step-based cfg.ckpt_every may be slow when the GPU is shared)."""
    import signal
    sig = {"hit": None}
    old = {}
    for sg in (signal.SIGTERM, signal.SIGINT):
        try:
            old[sg] = signal.signal(sg, lambda n, f: sig.__setitem__("hit", signal.Signals(n).name))
        except ValueError:          # not in the main thread
            pass
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    spe = train.n // cfg.batch
    total = cfg.epochs * spe
    sp = outdir / "state.json"
    st = {"step": 0, "epoch": 0, "batch": 0, "ckpt": None, "history": [], "best": None, "bad_epochs": 0, "cfg": asdict(cfg), "done": False, "loss_log": [],
          "es_log": [], "stopped_by": None}
    if sp.exists():
        st = json.loads(sp.read_text())
        st.setdefault("loss_log", [])
        st.setdefault("es_log", [])
        st["stopped_by"] = None
        if {**st["cfg"], "max_steps": 0, "log_every": 0} != {**asdict(cfg), "max_steps": 0, "log_every": 0}:
            raise RuntimeError("config differs from the checkpointed run; use a new --out or delete state.json")
        if st["ckpt"]:
            backend.load(outdir / st["ckpt"])
            log(f"resumed at step {st['step']} (epoch {st['epoch']}, batch {st['batch']})")
        if st["done"]:
            return st
    t0, loss_acc, n_acc = time.time(), 0.0, 0
    last_save = time.time()

    def save():
        nonlocal last_save
        last_save = time.time()
        name = f"ckpt_{st['step']:08d}.bin"
        backend.save(outdir / name)
        st["ckpt"] = name
        _atomic_json(sp, st)
        olds = sorted(outdir.glob("ckpt_*.bin"))
        for f in olds[:-cfg.keep_last]:
            if f.name != (st["best"] or {}).get("ckpt"):
                f.unlink()

    stopped = False
    while st["epoch"] < cfg.epochs and not st["done"] and not stopped:
        order = epoch_order(train.n, cfg.seed, st["epoch"])
        while st["batch"] < spe:
            if cfg.max_steps and st["step"] >= cfg.max_steps:
                stopped = True            # simulated kill / smoke limit: not "done", a later run resumes
                break
            idx = order[st["batch"] * cfg.batch:(st["batch"] + 1) * cfg.batch]
            bits, glob, color, y = train.get(np.sort(idx))
            sym = step_symmetry(cfg.seed, st["step"]) if cfg.sym_aug else 0
            loss = backend.train_step(bits, glob, color, y.astype(np.int64), lr_scale(st["step"], total, cfg.warmup_steps), sym)
            st["step"] += 1
            st["batch"] += 1
            loss_acc, n_acc = loss_acc + loss, n_acc + 1
            if cfg.log_every and st["step"] % cfg.log_every == 0:
                rate = n_acc * cfg.batch / (time.time() - t0)
                log(f"step {st['step']}/{total} loss {loss_acc / n_acc:.4f} {rate:.0f} pos/s")
                st["loss_log"].append({"step": st["step"], "loss": loss_acc / n_acc, "pos_per_s": rate})
                loss_acc, n_acc, t0 = 0.0, 0, time.time()
            if st["batch"] < spe and (st["step"] % cfg.ckpt_every == 0 or time.time() - last_save >= ckpt_every_sec):
                save()
            if es_every and es_eval is not None and st["step"] % es_every == 0 and st["batch"] < spe:
                v = float(es_eval(backend))
                st["es_log"].append({"step": st["step"], "es_nll": v})
                log(f"step {st['step']} es_nll {v:.4f} (monitor only)")
                save()
                loss_acc, n_acc, t0 = 0.0, 0, time.time()
            why = sig["hit"] or ("STOP file" if stop_file and Path(stop_file).exists() else None)
            if why:
                st["stopped_by"] = why
                stopped = True
                log(f"graceful stop ({why}) at step {st['step']}: checkpointing")
                break
        if stopped:
            break
        st["epoch"] += 1
        st["batch"] = 0
        rec = {"epoch": st["epoch"], "step": st["step"]}
        if es_eval is not None:
            rec["es_nll"] = float(es_eval(backend))
            if st["best"] is None or rec["es_nll"] < st["best"]["es_nll"]:
                st["best"] = {"es_nll": rec["es_nll"], "epoch": st["epoch"], "ckpt": None}
                st["bad_epochs"] = 0
                improved = True
            else:
                st["bad_epochs"] += 1
                improved = False
        else:
            improved = False
        st["history"].append(rec)
        log(f"epoch {st['epoch']} {rec}")
        save()
        loss_acc, n_acc, t0 = 0.0, 0, time.time()      # the es evaluation and checkpoint write must not count as training time
        if improved:
            shutil.copyfile(outdir / st["ckpt"], outdir / "best.bin")
            st["best"]["ckpt"] = "best.bin"
            _atomic_json(sp, st)
        if es_eval is not None and st["bad_epochs"] >= cfg.patience:
            st["done"] = True
    save()
    st["done"] = not stopped
    _atomic_json(sp, st)
    for sg, h in old.items():
        signal.signal(sg, h)
    return st


# ------------------------------------------------------------------------------------------------ reference backend (numpy)
class ReferenceBackend:
    """Softmax regression on [mean of each plane, global features, 1].  Plain SGD with the same lr schedule hooks."""

    def __init__(self, n_channels: int, n_global: int, class_w: np.ndarray, label_sigma: float = 0.0, lr: float = 0.5, seed: int = 0):
        self.C, self.G, self.cw, self.sigma, self.lr = n_channels, n_global, class_w, label_sigma, lr
        self.W = np.random.default_rng(seed).normal(0, 0.01, (n_channels + n_global + 1, lib.N_CLASSES))

    def _x(self, bits, glob, sym):
        planes = apply_symmetry(unpack_bits(bits, self.C).astype(np.float32), sym)
        return np.concatenate([planes.mean(axis=(2, 3)), glob, np.ones((len(bits), 1), np.float32)], axis=1)

    def train_step(self, bits, glob, color, y, scale, sym):
        x = self._x(bits, glob, sym)
        p = lib.softmax(x @ self.W)
        t = lib.soft_targets(y, self.sigma)
        w = self.cw[y] / self.cw[y].mean()
        loss = float((-(t * np.log(np.clip(p, 1e-12, 1))).sum(1) * w).mean())
        self.W -= self.lr * scale * (x.T @ ((p - t) * w[:, None])) / len(y)
        return loss

    def predict_logits(self, bits, glob, color):
        return (self._x(bits, glob, 0) @ self.W).astype(np.float32)

    def save(self, path):
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, W=self.W, scope=np.array("reference"), n_trunk=np.array(0))
        os.replace(tmp, path)

    def load(self, path):
        self.W = np.load(path)["W"]


# ------------------------------------------------------------------------------------------------ torch backend
SCOPES = ("head", "last4", "full")


class TorchBackend:
    """model: any torch module; trunk(bx, gx, mx) -> (B, D, 19, 19) features.  For KataGo see `load_katago_trunk`."""

    def __init__(self, model, trunk, trunk_dim: int, meta_rows: dict, scope: str, cfg: TrainConfig, class_w: np.ndarray,
                 device: str = "cuda", trunk_params=None, n_channels: int = 22):
        import torch
        self.torch, self.model, self.trunk_fn, self.cfg, self.device = torch, model, trunk, cfg, device
        self.meta_rows = {int(k): np.asarray(v, np.float32) for k, v in meta_rows.items()}
        self.head = torch.nn.Linear(trunk_dim, lib.N_CLASSES).to(device)
        self.cw = torch.tensor(class_w, dtype=torch.float32, device=device)
        for p in model.parameters():
            p.requires_grad = False
        self.trunk_params = list(trunk_params) if scope != "head" else []
        for p in self.trunk_params:
            p.requires_grad = True
        groups = [{"params": list(self.head.parameters()), "lr": cfg.lr_head, "base": cfg.lr_head}]
        if self.trunk_params:
            groups.append({"params": self.trunk_params, "lr": cfg.lr_trunk, "base": cfg.lr_trunk})
        self.opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler(device.split(":")[0], enabled=cfg.amp and device.startswith("cuda"))
        self.scope, self._C = scope, n_channels
        self.chunks = 1          # >1 after a GPU OOM: the (frozen) trunk forward is split into this many pieces (head scope only)
        self.model.eval()        # KataGo b18c384nbt trunk is normless (fixscale) [inference from config name]: no train/eval difference

    def _tensors(self, bits, glob, color, sym):
        t = self.torch
        planes = apply_symmetry(unpack_bits(bits, self._C), sym) if sym else unpack_bits(bits, self._C)
        bx = t.from_numpy(np.ascontiguousarray(planes)).to(self.device).float()
        gx = t.from_numpy(np.asarray(glob)).to(self.device)
        mx = t.from_numpy(np.stack([self.meta_rows[int(c)] for c in color])).to(self.device)
        return bx, gx, mx

    def _logits(self, bx, gx, mx, grad_trunk):
        t = self.torch
        with t.autocast(self.device.split(":")[0], dtype=t.float16, enabled=self.cfg.amp and self.device.startswith("cuda")):
            if grad_trunk:
                f = self.trunk_fn(bx, gx, mx).float().mean(dim=(2, 3))
            else:
                with t.no_grad():
                    n = len(bx)
                    step = -(-n // self.chunks)
                    f = t.cat([self.trunk_fn(bx[i:i + step], gx[i:i + step], mx[i:i + step]).float().mean(dim=(2, 3)) for i in range(0, n, step)])
            z = self.head(f)
        return z.float()

    def train_step(self, bits, glob, color, y, scale, sym):
        while True:
            try:
                return self._train_step(bits, glob, color, y, scale, sym)
            except self.torch.cuda.OutOfMemoryError:
                # someone else grew on the shared GPU.  Head scope: split the frozen-trunk forward and retry the same step (numerically the same).
                # Trunk scopes cannot shrink without changing the batch: re-raise so the caller stops with the last checkpoint intact.
                self.opt.zero_grad(set_to_none=True)
                self.torch.cuda.empty_cache()
                if self.trunk_params or self.chunks >= 16:
                    raise
                self.chunks *= 2
                print(f"GPU OOM -> trunk forward chunks={self.chunks}", flush=True)

    def _train_step(self, bits, glob, color, y, scale, sym):
        t = self.torch
        bx, gx, mx = self._tensors(bits, glob, color, sym)
        for g in self.opt.param_groups:
            g["lr"] = g["base"] * scale
        z = self._logits(bx, gx, mx, bool(self.trunk_params))
        tgt = t.from_numpy(lib.soft_targets(y, self.cfg.label_sigma).astype(np.float32)).to(self.device)
        w = self.cw[t.from_numpy(y).to(self.device)] if self.cfg.class_weighted else t.ones(len(y), device=self.device)
        loss = (-(tgt * t.log_softmax(z, dim=1)).sum(1) * w).sum() / w.sum()
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(loss.detach())

    def predict_logits(self, bits, glob, color):
        t = self.torch
        with t.no_grad():
            bx, gx, mx = self._tensors(bits, glob, color, 0)
            return self._logits(bx, gx, mx, False).cpu().numpy()

    def save(self, path):
        t = self.torch
        state = {"scope": self.scope, "n_trunk_tensors": len(self.trunk_params), "trunk_shapes": [tuple(p.shape) for p in self.trunk_params],
                 "head": self.head.state_dict(), "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                 "trunk": [p.detach().cpu() for p in self.trunk_params]}
        tmp = str(path) + ".tmp"
        t.save(state, tmp)
        os.replace(tmp, path)

    def load(self, path):
        t = self.torch
        s = t.load(path, map_location=self.device)
        if s.get("scope") != self.scope:
            raise ValueError(f"{path}: checkpoint scope {s.get('scope')!r} != backend scope {self.scope!r}")
        if len(s["trunk"]) != len(self.trunk_params) or [tuple(v.shape) for v in s["trunk"]] != [tuple(p.shape) for p in self.trunk_params]:
            raise ValueError(f"{path}: checkpoint has {len(s['trunk'])} trunk tensors, model scope {self.scope!r} selects {len(self.trunk_params)} "
                             f"(or the shapes differ)")
        self.head.load_state_dict(s["head"])
        for p, v in zip(self.trunk_params, s["trunk"]):
            p.data.copy_(v.to(self.device))
        self.opt.load_state_dict(s["opt"])
        self.scaler.load_state_dict(s["scaler"])


def read_ckpt_scope(path, backend: str = "torch") -> str:
    """The scope a checkpoint was trained with (inference trusts this, not the command line)."""
    if backend == "reference":
        return str(np.load(path)["scope"])
    import torch
    s = torch.load(path, map_location="cpu", weights_only=False)
    if "scope" not in s:
        raise ValueError(f"{path}: checkpoint has no scope (written by an older version); retrain or re-save")
    return s["scope"]


def select_trunk_params(model, scope: str):
    """Same selection as the smoke bench: last4 = blocks 14..17 + norm_trunkfinal; full = everything but policy/value heads."""
    if scope == "head":
        return []
    if scope == "last4":
        n = len(model.blocks)
        return [p for blk in list(model.blocks)[n - 4:] for p in blk.parameters()] + list(model.norm_trunkfinal.parameters())
    if scope == "full":
        return [p for n, p in model.named_parameters() if not n.startswith(("policy_head", "value_head", "intermediate_", "norm_intermediate"))]
    raise ValueError(scope)


def load_katago_trunk(ckpt: str, katago_src: str, device: str):
    """Returns (model, trunk_fn, model_config).  Hook on act_trunkfinal raises to cut the forward (policy/value heads unused)."""
    import sys
    sys.path.insert(0, str(Path(katago_src) / "python"))
    from katago.train.load_model import load_model
    model, _, _ = load_model(ckpt, use_swa=False, device=device, pos_len=19)

    class StopFwd(Exception):
        def __init__(self, x):
            self.x = x

    def hook(m, i, o):
        raise StopFwd(o)

    model.act_trunkfinal.register_forward_hook(hook)

    def trunk(bx, gx, mx):
        try:
            model(bx, gx, input_meta=mx)
        except StopFwd as e:
            return e.x
        raise RuntimeError("trunk hook did not fire")

    return model, trunk, model.config
