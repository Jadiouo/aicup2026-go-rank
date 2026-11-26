"""E90 stage 0R: rebuild of F_E79 (a copy of the rebuild in scripts/oneoff/e88_score.py lines 14-50, with e88's E-leg and CLI removed; the original is untouched).
Asserts the same reproductions as e88_score.py: F_T .6837, F_E66 .6913 (+-.0005) and F_E79 .6997 (+-.0005, the overall figure of experiments/E79_rank_pos_attn/results/final.json).
Run with PYTHONPATH=src:scripts:. """
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
from scripts.oneoff.e43_common import Q, bal, rank_scores, decide_ev, NS, ROOT
from scripts.experiments.e35.score import boot, diff, ci

S55 = ROOT / ".scratch/e55"; S57 = ROOT / ".scratch/e57"; S63 = ROOT / ".scratch/e63"; S79 = ROOT / ".scratch/e79"
nrm = lambda p: p / p.sum(1, keepdims=True); mix = lambda x, b, w: nrm((1 - w) * x + w * b)


def rebuild():
    Qf = Q(); A = Qf.arms(); y = Qf.truth; ns = Qf.ns; fold = Qf.fold
    sc = lambda p: rank_scores(decide_ev(p), y)
    def oof(d, nm): return Qf.align(pd.concat([pd.read_parquet(Path(d) / f"oof_{nm}_fold{k}.parquet") for k in range(5)], ignore_index=True))
    def set_arm(d):
        out = np.zeros((Qf.nq, 10))
        for k in range(5):
            m = fold == k; W = json.loads((d / f"select_ds_fold{k}.json").read_text())["chosen_W"]; ps = []
            for s in (0, 1, 2):
                z = np.load(d / f"setprobs_ds_W{W}_fold{k}_s{s}.npz", allow_pickle=False); assert (z["qid"].astype(str) == Qf.q.qid[m].to_numpy()).all(); ps.append(z["p"].astype(np.float64))
            out[m] = np.mean(ps, 0)
        return out
    M03T = oof(S55 / "dse_out/W120T", "M03SE"); M04 = A["M04"]; A1T = set_arm(S55 / "out/A1T"); M03T2 = oof(S63 / "dse_out/M03TL", "M03SE"); A1T2 = set_arm(S63 / "out/A1T2"); M04T = oof(S57 / "dse_out/M04T", "M04T"); A1P = set_arm(S79 / "out/A1P")
    D = lambda m3, m4, v=.5: nrm(v * m3 + (1 - v) * m4); FT = mix(D(M03T, M04), A1T, .5); r_ft = bal(sc(FT), y, ns); assert abs(r_ft - .6837) <= .0005, r_ft
    W_ = (.3, .4, .5, .6, .7, .8, .9); U_ = (0, .25, .5, .75, 1)
    GRID66 = [(w, v) for w in W_ for v in W_]; ORDER66 = sorted(GRID66, key=lambda t: abs(t[0] - .5) + abs(t[1] - .5))
    GRID79 = [(w, v, u) for w in W_ for v in W_ for u in U_]; ORDER79 = sorted(GRID79, key=lambda t: abs(t[0] - .5) + abs(t[1] - .5) + abs(t[2] - .5))
    au = lambda a1t2, a1p, u: nrm((1 - u) * a1t2 + u * a1p)
    def FA(m3, m4, a2, a1p, w, v, u): return mix(D(m3, m4, v), au(a2, a1p, u), w)
    def pick66(mask):
        best = None
        for w, v in ORDER66:
            s = bal(sc(mix(D(M03T2, M04T, v), A1T2, w)), y, ns, mask)
            if best is None or s > best[0] + 1e-12: best = (s, w, v)
        return best
    def pick79(mask):
        best = None
        for w, v, u in ORDER79:
            s = bal(sc(FA(M03T2, M04T, A1T2, A1P, w, v, u)), y, ns, mask)
            if best is None or s > best[0] + 1e-12: best = (s, w, v, u)
        return best
    out66 = np.zeros_like(A1T2)
    for k in range(5):
        _, w, v = pick66(fold != k); m = fold == k; out66[m] = mix(D(M03T2[m], M04T[m], v), A1T2[m], w)
    r66 = bal(sc(out66), y, ns); assert abs(r66 - .6913) <= .0005, r66
    F79 = np.zeros_like(A1T2); ch79 = {}
    for k in range(5):
        _, w, v, u = pick79(fold != k); ch79[k] = (w, v, u); m = fold == k; F79[m] = FA(M03T2[m], M04T[m], A1T2[m], A1P[m], w, v, u)
    b79 = bal(sc(F79), y, ns); assert abs(b79 - .6997) <= .0005, b79
    F = SimpleNamespace(pl=Qf.pl, ns=ns, truth=y, fold=fold)
    return dict(Qf=Qf, F=F, F79=F79, A1P=A1P, A1T2=A1T2, y=y, ns=ns, fold=fold, sc=sc, repro=dict(F_T=r_ft, F_E66=r66, F_E79=b79), ch79=ch79)
