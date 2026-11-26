"""11/04 offline Player-contest simulation: compare decoders (independent / Hungarian / multi-round / transductive / fusion).
Pools: E23b protocol (dev 450 players = not H, not in E04v2 fit set; frozen E05 model; 400 queries/pool, one per pool player,
N uniform 5..20, library 100 black games, query games disjoint). Usage: player_sim.py [n_pools]"""
import sys, json, hashlib, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "experiments" / "e23"), str(ROOT / "src")]
import numpy as np, pandas as pd
import run as R
from run import prr, S, paired_bootstrap, unit, zrow, norm_score, E05
from scipy.optimize import linear_sum_assignment
import run_b as RB

OUT = ROOT / ".scratch" / "player_sim"
BIG = 1e6
NPOOL = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20


def sm(x):
    z = np.exp(x - x.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)


def hung(C):
    r, c = linear_sum_assignment(C)
    a = np.empty(len(C), int); a[r] = c
    return a


def cost_from(top, val, nc):
    C = np.full((len(top), nc), BIG)
    for i in range(len(top)):
        C[i, top[i]] = val[i]
    return C


def order_rest(top, e, a1):
    """a1 first, then original reranked order without a1"""
    out = []
    for i in range(len(top)):
        rest = [j for j in top[i][np.argsort(-e[i], kind="stable")] if j != a1[i]]
        out.append([a1[i]] + rest[:4])
    return np.array(out)


def multiround(top, e, base_cost, nc, rounds=5):
    C = base_cost.copy()
    A = []
    for k in range(rounds):
        a = hung(C)
        A.append(a)
        C[np.arange(len(C)), a] = BIG * 10
    out = []
    for i in range(len(top)):
        lst = []
        for k in range(rounds):
            j = A[k][i]
            if base_cost[i, j] < BIG and j not in lst:
                lst.append(j)
        for j in top[i][np.argsort(-e[i], kind="stable")]:
            if j not in lst: lst.append(j)
        out.append(lst[:5])
    return np.array(out)


def transduce(lib_z, q_z, ds, truth_unused, tau, iters=3):
    """Iteratively fix confident mutual-best pairs, remove them, re-shortlist/re-score the rest."""
    model = MODEL
    nq = len(q_z)
    cand = np.arange(len(lib_z)); qry = np.arange(nq)
    fixed = {}                      # query -> candidate
    first = None
    stat = []
    for it in range(iters + 1):
        f = prr.candidate_features([lib_z[c] for c in cand], [q_z[i] for i in qry], ds.sel, prr.K, prr.N_DENSE)
        e = model.predict(f["X"], num_threads=4).reshape(len(qry), -1)
        top = cand[f["top"]]                      # global candidate ids
        if first is None:
            first = (qry.copy(), top.copy(), e.copy())
        ordr = np.argsort(-e, axis=1, kind="stable")
        best = np.take_along_axis(top, ordr[:, :1], 1)[:, 0]
        p1 = sm(e).max(1)
        if it == iters:
            break
        # candidate-side best query by e (among queries shortlisting it)
        cb = {}
        for ii in range(len(qry)):
            for kk in range(top.shape[1]):
                c = top[ii, kk]
                if c not in cb or e[ii, kk] > cb[c][0]:
                    cb[c] = (e[ii, kk], ii)
        sel_ = [ii for ii in range(len(qry)) if p1[ii] >= tau and cb[best[ii]][1] == ii]
        if not sel_: break
        stat.append(len(sel_))
        for ii in sel_: fixed[qry[ii]] = best[ii]
        keepq = np.array([ii for ii in range(len(qry)) if ii not in set(sel_)], int)
        gone = set(best[sel_])
        cand = np.array([c for c in cand if c not in gone]); qry = qry[keepq]
        if len(qry) == 0: break
    return fixed, (qry, top, e), first, stat


def run_pool(ds, sim, st, pool_i):
    pool = list(sim.players); qs = sim.queries
    nc = len(pool)
    assert len(qs) == 400 and all(p in st for p in pool)
    for q in qs:
        assert not (set(q.uids) & set(sim.library[q.player_id])) and len(sim.library[q.player_id]) == 100
    lib_z = [np.asarray(ds.Z[ds.rows(sim.library[p])]) for p in pool]
    q_z = [np.asarray(ds.Z[ds.rows(q.uids)]) for q in qs]
    pos = {p: i for i, p in enumerate(pool)}
    truth = np.array([pos[q.player_id] for q in qs])
    f = prr.candidate_features(lib_z, q_z, ds.sel, prr.K, prr.N_DENSE)
    Cc, top, ns = f["cos"], f["top"], f["ns"]
    e = MODEL.predict(f["X"], num_threads=4).reshape(len(qs), prr.K)
    res = {}
    def put(name, t5):
        h = np.asarray(t5)[:, :5] == truth[:, None]
        res[name] = np.where(h.any(1), np.exp(-h.argmax(1).astype(float)), 0.0)
        res[name + "|t1"] = h[:, 0].astype(float)
    orig = np.take_along_axis(top, np.argsort(-e, axis=1, kind="stable"), 1)[:, :5]
    put("a_indep", orig)
    # (b) current v2: cost -log(clip(raw score))
    Cb = cost_from(top, -np.log(np.clip(e, 1e-9, 1)), nc)
    a1 = hung(Cb)
    put("b_v2", order_rest(top, e, a1))
    res["b_v2|outside"] = (Cb[np.arange(400), a1] >= BIG).astype(float)
    res["b_v2|scoreneg"] = np.full(400, float((e <= 0).mean()))
    # b2: proper log-softmax cost
    lsm = np.log(sm(e))
    a2 = hung(cost_from(top, -lsm, nc)); put("b2_logsoftmax", order_rest(top, e, a2))
    # (c) multiround (on v2 cost, and on logsoftmax cost)
    put("c_multi_v2cost", multiround(top, e, Cb, nc))
    put("c_multi_lsm", multiround(top, e, cost_from(top, -lsm, nc), nc))
    # (e) fusion with M01 cosine, Hungarian
    ze = zrow(e); zc = zrow(np.take_along_axis(Cc, top, 1))
    for beta in (0.5, 1.0):
        fz = ze + beta * zc
        a = hung(cost_from(top, -np.log(sm(fz)), nc)); put(f"e_fuse{beta}", order_rest(top, fz, a))
    # CSLS-qq fused (E23 (a) setting) then Hungarian
    G = unit(np.stack([z.mean(0) for z in lib_z])); G = G @ G.T
    Sm = norm_score("csls_qq", 5, 1.0, Cc, G, np.zeros(len(ns), int))
    fz = ze + 1.0 * zrow(np.take_along_axis(Sm, top, 1))
    put("e_csls_indep", np.take_along_axis(top, np.argsort(-fz, axis=1, kind="stable"), 1)[:, :5])
    a = hung(cost_from(top, -np.log(sm(fz)), nc)); put("e_csls_hung", order_rest(top, fz, a))
    # sinkhorn soft assignment
    for T in (1.0, 0.5):
        P = np.zeros((400, nc))
        for i in range(400): P[i, top[i]] = np.exp((e[i] - e[i].max()) / T)
        for _ in range(100):
            P /= np.maximum(P.sum(1, keepdims=True), 1e-300)
            P /= np.maximum(P.sum(0, keepdims=True), 1e-300)
        sp = np.take_along_axis(P, top, 1)
        put(f"e_sinkhorn_T{T}", np.take_along_axis(top, np.argsort(-sp, axis=1, kind="stable"), 1)[:, :5])
    # (d) transductive removal
    for name, tau in (("d_trans_hi", 0.9), ("d_trans_lo", 0.5), ("d_trans_t80", 0.8), ("d_trans_t95", 0.95), ("d_trans_t99", 0.99)):
        fixed, (qry, top2, e2), first, stat = transduce(lib_z, q_z, ds, None, tau)
        t5 = orig.copy()
        for qi, c in fixed.items():
            rest = [j for j in orig[qi] if j != c]
            t5[qi] = [c] + rest[:4]
        # remaining queries: Hungarian on last stage over remaining candidates
        if len(qry):
            remc = np.array(sorted(set(range(nc)) - set(fixed.values())))
            cm = {c: k for k, c in enumerate(remc)}
            Cr = np.full((len(qry), len(remc)), BIG)
            lg = -np.log(sm(e2))
            for ii in range(len(qry)):
                for kk in range(top2.shape[1]):
                    Cr[ii, cm[top2[ii, kk]]] = lg[ii, kk]
            ar = hung(Cr) if len(qry) <= len(remc) else None
            for ii, qi in enumerate(qry):
                o = top2[ii][np.argsort(-e2[ii], kind="stable")]
                c = remc[ar[ii]] if ar is not None and Cr[ii, ar[ii]] < BIG else o[0]
                lst = [c] + [j for j in o if j != c]
                lst += [j for j in orig[qi] if j not in lst]
                t5[qi] = lst[:5]
        put(name, t5)
        conf = np.zeros(400); prec = np.full(400, np.nan)
        for qi, c in fixed.items(): conf[qi] = 1; prec[qi] = float(c == truth[qi])
        res[name + "|conf"] = conf; res[name + "|confprec"] = prec
    df = pd.DataFrame({**res, "qplayer": [q.player_id for q in qs], "n": ns, "recall": (top == truth[:, None]).any(1).astype(float)})
    df["pool"] = pool_i
    return df


def main():
    import lightgbm as lgb
    global MODEL
    OUT.mkdir(parents=True, exist_ok=True)
    fro = json.loads((E05 / "frozen.json").read_text())
    assert hashlib.sha256((E05 / "model.txt").read_bytes()).hexdigest() == fro["model_sha256"]
    MODEL = lgb.Booster(model_file=str(E05 / "model.txt"))
    st = R.sets()
    assert not (set(st["dev"]) & st["H"]) and not (set(st["dev"]) & st["fit"])
    ds = R.get_ds("main")
    sims = R.build_player_pool_sim(ds.index, st["dev"], 400, NPOOL, 100, tuple(range(5, 21)), seed=R.derive_seed(2026, "player_sim", "1"))
    rng = np.random.default_rng(R.derive_seed(2026, "player_sim", "N"))
    frames, t0 = [], time.perf_counter()
    for s in sims:
        frames.append(run_pool(ds, RB.thin(s, rng), set(st["dev"]), s.draw))
        print("pool", s.draw, f"{time.perf_counter()-t0:.0f}s", flush=True)
        pd.concat(frames, ignore_index=True).to_parquet(OUT / "scores.parquet")
    df = pd.concat(frames, ignore_index=True)
    methods = [c for c in df.columns if "|" not in c and c not in ("qplayer", "n", "recall", "pool")]
    rows = []
    for m in methods:
        r = paired_bootstrap(df[m].to_numpy(), df["b_v2"].to_numpy(), df["qplayer"].to_numpy(), n_boot=4000, seed=S("ps", m))
        row = {"method": m, "score": df[m].mean(), "top1": df[m + "|t1"].mean(), "d_vs_b": r.delta, "lo": r.ci_low, "hi": r.ci_high}
        for lo, hi in ((5, 5), (10, 10), (15, 15), (20, 20)):
            pass
        for b, (lo, hi) in {"N5-8": (5, 8), "N9-12": (9, 12), "N13-16": (13, 16), "N17-20": (17, 20)}.items():
            row[b] = df.loc[(df.n >= lo) & (df.n <= hi), m].mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    print("recall20", df.recall.mean(), "outside frac b_v2", df["b_v2|outside"].mean(), "neg score frac", df["b_v2|scoreneg"].mean())
    for n in ("d_trans_hi", "d_trans_lo", "d_trans_t80", "d_trans_t95", "d_trans_t99"):
        print(n, "conf frac", df[n + "|conf"].mean(), "conf precision", np.nanmean(df[n + "|confprec"]))


if __name__ == "__main__":
    main()
