"""E22: per-N calibration of D probabilities then EV decode.  Plan: experiments/E22_calibration/PLAN.md.
Run: .venv/bin/python scripts/experiments/e22/run.py   (CPU only, reads E10 oof)"""
import os, json, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
from sklearn.isotonic import IsotonicRegression
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from gogame.metrics import balanced_weights, rank_scores
from gogame.models import decide_ev
from gogame.constants import SEED
from gogame.seeds import derive_seed

OOF = ROOT / "experiments/E10_tier2_retrain/oof"; OUT = ROOT / "experiments/E22_calibration/results"; OUT.mkdir(parents=True, exist_ok=True)
PC = [f"p_{i}" for i in range(10)]; NS = (5, 10, 15, 20); EPS = 1e-9
H = set(pd.read_parquet(ROOT / "data/splits/split.parquet").query("is_holdout")["player_id"])
D = {k: pd.read_parquet(OOF / f"oof_D_fold{k}.parquet").query("qset == 'q2'").reset_index(drop=True) for k in range(5)}
allp = [set(D[k].player_id) for k in range(5)]
for k in range(5):
    assert not (allp[k] & H), "H in OOF"
    for j in range(k):
        assert not (allp[k] & allp[j]), "folds share players"

def tp(p, T):
    q = np.power(np.clip(p, EPS, 1), 1.0 / T); return q / q.sum(1, keepdims=True)
def nll(p, y, w): return float(-(w * np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))).sum() / w.sum())

def fit_temp(p, y, w):
    r = minimize_scalar(lambda T: nll(tp(p, T), y, w), bounds=(0.3, 3.0), method="bounded"); return float(r.x)
def fit_bias(p, y, w, lam=1.0):
    def f(th):
        T, b = th[0], th[1:]; q = np.log(np.clip(tp(p, np.exp(T)), EPS, 1)) + b
        q = np.exp(q - q.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        return nll(q, y, w) + lam * (b ** 2).sum() / len(y) * 0 + lam * (b ** 2).sum() * 1e-3
    r = minimize(f, np.zeros(11), method="L-BFGS-B", bounds=[(-1.1, 1.1)] + [(-3, 3)] * 10); return r.x
def apply_bias(p, th):
    q = np.log(np.clip(tp(p, np.exp(th[0])), EPS, 1)) + th[1:]; q = np.exp(q - q.max(1, keepdims=True)); return q / q.sum(1, keepdims=True)
def fit_iso(p, y, w):
    return [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[:, c], (y == c).astype(float), sample_weight=w) for c in range(10)]
def apply_iso(p, ms):
    q = np.column_stack([m.predict(p[:, c]) for c, m in enumerate(ms)]) + 1e-6; return q / q.sum(1, keepdims=True)

def calibrate(target, fit_folds):
    tg = D[target]; fit = pd.concat([D[j] for j in fit_folds], ignore_index=True)
    assert not (set(tg.player_id) & set(fit.player_id)) and not (set(tg.player_id) & H) and not (set(fit.player_id) & H)
    out = {m: tg[PC].to_numpy().copy() for m in ("temp", "bias", "iso")}; info = []
    for n in NS:
        fm, tm = (fit.n == n).to_numpy(), (tg.n == n).to_numpy()
        pf, yf = fit.loc[fm, PC].to_numpy(), fit.loc[fm, "rank_id"].to_numpy(); wf = balanced_weights(yf)
        T = fit_temp(pf, yf, wf); th = fit_bias(pf, yf, wf); ms = fit_iso(pf, yf, wf)
        pt = tg.loc[tm, PC].to_numpy()
        out["temp"][tm] = tp(pt, T); out["bias"][tm] = apply_bias(pt, th); out["iso"][tm] = apply_iso(pt, ms)
        info.append({"target": target, "N": n, "T": T, "bias_logT": float(th[0]), "nll_before_fit": nll(pf, yf, wf), "nll_after_fit": nll(tp(pf, T), yf, wf)})
    return out, info

def evaluate(targets, tag):
    rows, drows, infos = [], [], []
    frames, preds = [], {}
    cal = {m: [] for m in ("temp", "bias", "iso")}
    for k in targets:
        out, info = calibrate(k, [j for j in range(5) if j != k]); infos += info
        for m in cal: cal[m].append(out[m])
    T = pd.concat([D[k] for k in targets], ignore_index=True)
    truth, ns, pl = T.rank_id.to_numpy(), T.n.to_numpy(), T.player_id.to_numpy()
    P = {"D": T[PC].to_numpy(), **{f"D_{m}": np.vstack(cal[m]) for m in cal}}
    sc = {nm: rank_scores(decide_ev(p), truth) for nm, p in P.items()}
    hit = {nm: (decide_ev(p) == truth).astype(float) for nm, p in P.items()}
    chg = {nm: float((decide_ev(p) != decide_ev(P["D"])).mean()) for nm, p in P.items()}
    u, gi = np.unique(pl, return_inverse=True); Pn = len(u)
    C = np.random.default_rng(derive_seed(SEED, "e22", "boot", tag)).multinomial(Pn, np.full(Pn, 1 / Pn), size=4000).astype(float)
    def nd(v, kind):
        o = []
        for n in NS:
            mk = ns == n; w = balanced_weights(truth[mk]) if kind == "balanced" else np.ones(mk.sum())
            o.append((np.bincount(gi[mk], weights=v[mk] * w, minlength=Pn), np.bincount(gi[mk], weights=w, minlength=Pn)))
        return o
    for kind in ("balanced", "raw", "hit"):
        vs = {"balanced": sc, "raw": sc, "hit": hit}[kind]
        base = nd(vs["D"], kind)
        for nm in P:
            a = nd(vs[nm], kind)
            if nm == "D":
                for i, n in enumerate(NS): rows.append({"tag": tag, "model": nm, "N": n, "metric": kind, "value": a[i][0].sum() / a[i][1].sum()})
                rows.append({"tag": tag, "model": nm, "N": "eq", "metric": kind, "value": float(np.mean([x.sum() / y.sum() for x, y in a]))})
                continue
            dn = [(x[0] - y[0], x[1]) for x, y in zip(a, base)]
            for i, n in enumerate(NS):
                bs = (C @ dn[i][0]) / (C @ dn[i][1]); rows.append({"tag": tag, "model": nm, "N": n, "metric": kind, "value": a[i][0].sum() / a[i][1].sum()})
                drows.append({"tag": tag, "model": nm, "N": n, "metric": kind, "delta": dn[i][0].sum() / dn[i][1].sum(), "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975), "frac_pred_changed": chg[nm]})
            bb = np.mean([(C @ x) / (C @ y) for x, y in dn], axis=0)
            rows.append({"tag": tag, "model": nm, "N": "eq", "metric": kind, "value": float(np.mean([x.sum() / y.sum() for x, y in a]))})
            drows.append({"tag": tag, "model": nm, "N": "eq", "metric": kind, "delta": float(np.mean([x.sum() / y.sum() for x, y in dn])), "lo": np.quantile(bb, .025), "hi": np.quantile(bb, .975), "frac_pred_changed": chg[nm]})
    # per-fold point deltas (balanced eq)
    for k in targets:
        mk0 = (T.fold == k).to_numpy() if "fold" in T else None
    return pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(infos), (T, P, truth, ns)

def perfold(T, P, truth, ns, targets):
    out = []
    for k in targets:
        mk = (T.fold == k).to_numpy()
        for nm in P:
            if nm == "D": continue
            v = []
            for n in NS:
                m = mk & (ns == n); w = balanced_weights(truth[m])
                v.append((w * (rank_scores(decide_ev(P[nm][m]), truth[m]) - rank_scores(decide_ev(P["D"][m]), truth[m]))).mean())
            out.append({"fold": k, "model": nm, "delta_bal_eq": float(np.mean(v))})
    return out

if __name__ == "__main__":
    os.nice(10)
    res = {}
    for tag, tg in (("main_f01", [0, 1]), ("extra_f234", [2, 3, 4])):
        s, d, i, ctx = evaluate(tg, tag); res[tag] = (s, d, i)
        pf = pd.DataFrame(perfold(*ctx, tg)); pf.to_csv(OUT / f"perfold_{tag}.csv", index=False)
        s.to_csv(OUT / f"scores_{tag}.csv", index=False); d.to_csv(OUT / f"deltas_{tag}.csv", index=False); i.to_csv(OUT / f"fitinfo_{tag}.csv", index=False)
        pd.set_option("display.width", 220)
        print(tag); print(d[d.metric == "balanced"].to_string()); print(d[(d.N == "eq")].to_string()); print(i.to_string()); print(pf.to_string())
        print(s[(s.N == "eq")].to_string())
