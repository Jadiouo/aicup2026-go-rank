"""E43 shared loader: v6 OOF probabilities (arms), question metadata, question-level aggregated features, scoring helpers.
Only OOF (non-H) data. Never touches H / h_eval_ledger.  Run with PYTHONPATH=src:. .venv/bin/python
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from gogame.metrics import balanced_weights, rank_scores, E1, rank_score_matrix
from gogame.models import decide_ev

E29B = ROOT / ".scratch/e29/bundle"; X120 = ROOT / ".scratch/e36/bundle/X120_full.npy"
A1 = ROOT / ".scratch/e37/runs/v6/out/A1full"; W120 = ROOT / ".scratch/e37/runs/v6_dse/out/W120"
S43 = ROOT / ".scratch/e43"; EXP = ROOT / "experiments/E43_rank_ten_directions"; RES = EXP / "results"
NS = (5, 10, 15, 20); PC = [f"p_{i}" for i in range(10)]
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


class Q:
    """OOF question frame, fold-concatenated in aux order (= E42 P_v6 order)."""
    def __init__(self):
        a = np.load(E29B / "aux.npz"); self.aux = a
        parts = []
        for k in range(5):
            parts.append(pd.DataFrame({"qid": a[f"val_qid_f{k}"].astype(str), "player": a[f"val_player_f{k}"], "truth": a[f"val_rank_f{k}"].astype(int),
                                       "n": a[f"val_n_f{k}"].astype(int), "fold": k, "color": a[f"val_color_f{k}"].astype(int)}))
        self.q = pd.concat(parts, ignore_index=True)
        self.truth = self.q.truth.to_numpy(); self.ns = self.q.n.to_numpy(); self.pl = self.q.player.to_numpy(); self.fold = self.q.fold.to_numpy()
        self.rows = np.concatenate([a[f"val_rows_f{k}"] for k in range(5)])      # [Nq,20] game rows (-1 pad)
        self.nq = len(self.q)

    def align(self, df):
        d = df.set_index("qid").loc[self.q.qid]; assert (d["rank_id"].to_numpy() == self.truth).all(); return d[PC].to_numpy(np.float64)

    def set_probs(self, d: Path, seed):
        out = []
        for k in range(5):
            W = json.loads((d / f"select_ds_fold{k}.json").read_text())["chosen_W"]
            z = np.load(d / f"setprobs_ds_W{W}_fold{k}_s{seed}.npz", allow_pickle=False)
            assert (z["qid"].astype(str) == self.q.qid[self.q.fold == k].to_numpy()).all()
            out.append(z["p"].astype(np.float64))
        return np.concatenate(out)

    def arms(self):
        """dict of OOF probabilities: D120, M03, M04, A1 (3-seed mean), A1_s0..2, F_main (=.5 D120+.5 A1)."""
        rd = lambda nm: self.align(pd.concat([pd.read_parquet(W120 / f"oof_{nm}_fold{k}.parquet") for k in range(5)], ignore_index=True))
        d120, m03, m04 = rd("Dse"), rd("M03SE"), rd("M04")
        a1s = [self.set_probs(A1, s) for s in (0, 1, 2)]; a1 = np.mean(a1s, 0)
        out = {"D120": d120, "M03": m03, "M04": m04, "A1": a1, "A1_s0": a1s[0], "A1_s1": a1s[1], "A1_s2": a1s[2]}
        out["F_main"] = 0.5 * d120 + 0.5 * a1; out["F_main"] /= out["F_main"].sum(1, keepdims=True)
        return out


def bal(sc, truth, ns, mask=None):
    """class-balanced per N, N equal weight."""
    m0 = np.ones(len(sc), bool) if mask is None else mask; out = []
    for n in NS:
        m = m0 & (ns == n)
        if m.any():
            w = balanced_weights(truth[m]); out.append((sc[m] * w).sum() / w.sum())
    return float(np.mean(out))


def score_of(P, Qf, mask=None):
    return bal(rank_scores(decide_ev(P), Qf.truth), Qf.truth, Qf.ns, mask)


def boot_delta(sc_a, sc_b, Qf, B=1000, seed=0, mask=None):
    """paired player-cluster bootstrap of bal(sc_a)-bal(sc_b); returns (delta, lo, hi)."""
    pl = Qf.pl; u, inv = np.unique(pl, return_inverse=True)
    order = np.argsort(inv, kind="stable"); starts = np.r_[0, np.cumsum(np.bincount(inv))]
    d0 = bal(sc_a, Qf.truth, Qf.ns, mask) - bal(sc_b, Qf.truth, Qf.ns, mask)
    r = np.random.default_rng(seed); vals = []
    for _ in range(B):
        pick = r.integers(0, len(u), len(u))
        idx = np.concatenate([order[starts[i]:starts[i + 1]] for i in pick])
        m = None if mask is None else mask[idx]
        vals.append(bal(sc_a[idx], Qf.truth[idx], Qf.ns[idx], m) - bal(sc_b[idx], Qf.truth[idx], Qf.ns[idx], m))
    return float(d0), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def qfeatures(Qf, force=False):
    """question-level mean/std of the 139 per-game features (X120_full) over the bag, cached."""
    f = S43 / "qfeat.npy"
    if f.exists() and not force: return np.load(f)
    X = np.load(X120, mmap_mode="r"); out = np.zeros((Qf.nq, 139 * 2), np.float32)
    for i in range(Qf.nq):
        r = Qf.rows[i]; r = r[r >= 0]; x = np.asarray(X[np.sort(r)], np.float64)
        out[i, :139] = np.nanmean(x, 0); out[i, 139:] = np.nanstd(x, 0)
    np.save(f, out); return out
