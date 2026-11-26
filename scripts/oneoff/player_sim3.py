"""11/06 Player offline sim, round 3: do SE strength / HSL style features help the current v3 (E04v2 + transductive tau .9)?

Stages (argv[1]):  prep   build LDA/whitening metrics for SE and HSL from FIT players (not dev, not tune, not H) -> .scratch/player_sim3/
                   tune   pick lambda on independent TUNE players (lib 40, N in 5/10/15; pools of 400 tune players)
                   eval   dev 450 players, 40 pools, lib 100, N balanced {5,10,15,20} x100 per pool (same pools/libs as player_sim.py)
                   report paired comparisons -> REPORT tables (csv) in .scratch/player_sim3/
Fusion (no retrain): e' = e_E04v2 + B, B[q,c] = -0.5 * lam * D[q,c]; D = Mahalanobis^2 between bag mean and library mean in a whitened
(within-player variance = 1) LDA space fitted on FIT players, divided by (1/N_q + 1/n_lib) (Gaussian log-likelihood). Then the same
transductive decoder (tau .9) as player_sim.py::transduce.  Never uses H players.
"""
import os, sys, json, hashlib, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(ROOT / "scripts" / "experiments" / "e23"), str(ROOT / "src")]
STAGE = sys.argv[1] if len(sys.argv) > 1 else ""
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]          # player_sim reads argv[1] at import
import numpy as np, pandas as pd, scipy.linalg
import player_sim as PS
from player_sim import R, RB, prr, hung, sm, BIG, E05, S, paired_bootstrap
sys.argv = _argv

OUT = ROOT / ".scratch" / "player_sim3"
CACHE = ROOT / "data" / "cache"
SE_PATH = ROOT / ".scratch" / "e21" / "se_uid.parquet"
HSL_PATH = CACHE / "e10" / "hsl_games.parquet"
HMAX = 40


# ----------------------------------------------------------------------------- metrics
def fit_metric(X, pid, ridge=1e-3, mmax=None):
    """Generalised LDA: returns (gmean, V) with V^T Sw V = I and columns sorted by between/within ratio (desc)."""
    codes, inv = np.unique(pid, return_inverse=True)
    C, d = len(codes), X.shape[1]
    cnt = np.bincount(inv).astype(np.float64)
    mu = np.zeros((C, d))
    for j in range(d):
        mu[:, j] = np.bincount(inv, weights=X[:, j], minlength=C) / cnt
    R_ = X - mu[inv]
    Sw = R_.T @ R_ / (len(X) - C)
    Sw += ridge * np.trace(Sw) / d * np.eye(d)
    gm = mu.mean(0)
    Sb = np.cov(mu.T) if d > 1 else np.array([[mu.var(ddof=1)]])
    ev, V = scipy.linalg.eigh(Sb, Sw)
    o = np.argsort(-ev)
    V = V[:, o][:, :mmax or d]
    return gm, V, ev[o]


def meta_black():
    m = pd.read_parquet(CACHE / "games_meta.parquet")
    return m[m.color == 0].reset_index(drop=True)


def stage_prep():
    OUT.mkdir(parents=True, exist_ok=True)
    st = R.sets()
    assert not (st["fit"] & st["H"]) and not (set(st["dev"]) & st["fit"])
    mb = meta_black()
    need = mb[mb.player_id.isin(st["fit"] | set(st["dev"]) | set(st["tune"]))]
    assert not need.player_id.isin(st["H"]).any()
    # ---- SE
    se = pd.read_parquet(SE_PATH)
    fit_se = need[need.player_id.isin(st["fit"])].merge(se, on="uid")
    # cap games per player so a few heavy players do not dominate
    fit_se = fit_se.sample(frac=1.0, random_state=0)
    fit_se = fit_se[fit_se.groupby("player_id").cumcount() < 120]
    X1 = fit_se[["se_mean"]].to_numpy(float); X2 = fit_se[["se_mean", "se_std"]].to_numpy(float)
    g1, V1, e1 = fit_metric(X1, fit_se.player_id.to_numpy()); g2, V2, e2 = fit_metric(X2, fit_se.player_id.to_numpy())
    print("SE fit players", fit_se.player_id.nunique(), "games", len(fit_se), "LDA ratios se1", e1, "se2", e2)
    np.savez(OUT / "metric_se.npz", g1=g1, V1=V1, g2=g2, V2=V2)
    # ---- HSL (all black games of fit/dev/tune players; projected with LDA fitted on fit players only)
    h = pd.read_parquet(HSL_PATH)
    cols = [c for c in h.columns if c not in ("uid", "n_moves")]
    h = h[h.uid.isin(set(need.uid))].reset_index(drop=True)
    X = h[cols].to_numpy(np.float64)
    isfit = h.uid.map(dict(zip(need.uid, need.player_id.isin(st["fit"]).to_numpy()))).to_numpy(bool)
    pid = h.uid.map(dict(zip(need.uid, need.player_id))).to_numpy()
    mu, sd = X[isfit].mean(0), X[isfit].std(0) + 1e-9
    Xs = (X - mu) / sd
    # cap 120 games/player for the fit
    rng = np.random.default_rng(0)
    fi = np.flatnonzero(isfit)
    fi = rng.permutation(fi)
    keep = fi[(pd.Series(pid[fi]).groupby(pid[fi]).cumcount() < 120).to_numpy()]
    gm, V, ev = fit_metric(Xs[keep], pid[keep], ridge=1e-2, mmax=HMAX)
    print("HSL fit players", len(set(pid[keep])), "games", len(keep), "dims", len(cols), "top LDA ratios", np.round(ev[:10], 3), "ratio@40", ev[HMAX - 1])
    np.savez(OUT / "metric_hsl.npz", cols=np.array(cols), mu=mu, sd=sd, gm=gm, V=V, ev=ev)   # needed to project TEST games the same way
    Y = ((Xs - gm) @ V).astype(np.float32)
    np.savez(OUT / "hsl_proj.npz", uid=h.uid.to_numpy().astype(str), Y=Y, ev=ev)
    print("saved", Y.shape)


class Feat:
    """per-game SE vectors (standardised through the fitted metrics) and HSL LDA coordinates, looked up by uid."""
    def __init__(self):
        se = pd.read_parquet(SE_PATH)
        self.se_ix = pd.Index(se.uid)
        self.se_arr = se[["se_mean", "se_std"]].to_numpy(np.float64)
        m = np.load(OUT / "metric_se.npz")
        self.m = {k: m[k] for k in m.files}
        z = np.load(OUT / "hsl_proj.npz", allow_pickle=False)
        self.h_ix = pd.Index(z["uid"]); self.h_Y = z["Y"].astype(np.float64)

    def se(self, uids, kind):
        ix = self.se_ix.get_indexer(uids)
        a = np.where((ix >= 0)[:, None], self.se_arr[np.maximum(ix, 0)], np.nan)
        if kind == "se1":
            return (a[:, :1] - self.m["g1"]) @ self.m["V1"]
        return (a - self.m["g2"]) @ self.m["V2"]

    def hsl(self, uids):
        ix = self.h_ix.get_indexer(uids)
        assert (ix >= 0).all()
        return self.h_Y[ix]


def mean_n(arrs, m):
    mu = np.empty((len(arrs), m)); n = np.empty(len(arrs))
    for i, a in enumerate(arrs):
        a = a[:, :m]; ok = ~np.isnan(a).any(1)
        n[i] = ok.sum()
        mu[i] = a[ok].mean(0) if ok.any() else np.nan
    return mu, n


def dist_matrix(lib_arrs, q_arrs, m):
    """Gaussian log-lik Mahalanobis^2 (nq,nc); NaN-safe (rows/cols without data -> row median)."""
    mc, nc = mean_n(lib_arrs, m); mq, nq = mean_n(q_arrs, m)
    ok_c, ok_q = nc > 0, nq > 0
    mc0, mq0 = np.nan_to_num(mc), np.nan_to_num(mq)
    d2 = (mq0 ** 2).sum(1)[:, None] + (mc0 ** 2).sum(1)[None, :] - 2 * mq0 @ mc0.T
    D = np.maximum(d2, 0) / (1.0 / np.maximum(nq, 1)[:, None] + 1.0 / np.maximum(nc, 1)[None, :])
    bad = ~(ok_q[:, None] & ok_c[None, :])
    med = np.median(D[~bad]) if (~bad).any() else 0.0
    D[bad] = med
    return D


SRC = {"se1": ("se1", 1, 0), "se2": ("se2", 2, 0)}
for _m in (5, 15, 40):
    SRC[f"hsl{_m}"] = ("hsl", _m, 0)
for _sub in (10, 20, 40):          # library games that have HSL: a random subset of the candidate's 100 games
    SRC[f"hsl5s{_sub}"] = ("hsl", 5, _sub)
    SRC[f"hsl15s{_sub}"] = ("hsl", 15, _sub)


def pool_D(F, lib_u, q_u, names):
    """lib_u: list (per candidate) of uid arrays; q_u: list of uid arrays."""
    out = {}
    cache = {}
    for nm in names:
        kind, m, sub = SRC[nm]
        key = (kind, sub)
        if key not in cache:
            fn = (lambda u, k=kind: F.hsl(u)) if kind == "hsl" else (lambda u, k=kind: F.se(u, k))
            lu = lib_u
            if sub:
                rg = np.random.default_rng(sub)
                lu = [u[np.sort(rg.permutation(len(u))[:sub])] for u in lib_u]
            cache[key] = ([fn(u) for u in lu], [fn(u) for u in q_u])
        out[nm] = dist_matrix(cache[key][0], cache[key][1], m)
    return out


# ----------------------------------------------------------------------------- transductive decoder with additive bonus
def trans_t5(model, ds, lib_z, q_z, B, tau=0.9, iters=3, rescore=None):
    """Same logic as player_sim.transduce + run_pool decode; B (nq,nc) additive score bonus (None = current v3)."""
    nq, nc = len(q_z), len(lib_z)
    cand = np.arange(nc); qry = np.arange(nq); fixed = {}; first = None
    for it in range(iters + 1):
        f = prr.candidate_features([lib_z[c] for c in cand], [q_z[i] for i in qry], ds.sel, prr.K, prr.N_DENSE)
        e = model.predict(f["X"], num_threads=4).reshape(len(qry), -1)
        top = cand[f["top"]]
        if B is not None:
            e = e + B[qry[:, None], top]
        if rescore is not None:
            e = rescore(e, top, qry)
        if first is None:
            first = (top.copy(), e.copy())
        ordr = np.argsort(-e, axis=1, kind="stable")
        best = np.take_along_axis(top, ordr[:, :1], 1)[:, 0]
        p1 = sm(e).max(1)
        if it == iters:
            break
        cb = {}
        for ii in range(len(qry)):
            for kk in range(top.shape[1]):
                c = top[ii, kk]
                if c not in cb or e[ii, kk] > cb[c][0]:
                    cb[c] = (e[ii, kk], ii)
        sel_ = [ii for ii in range(len(qry)) if p1[ii] >= tau and cb[best[ii]][1] == ii]
        if not sel_:
            break
        for ii in sel_:
            fixed[qry[ii]] = best[ii]
        ss = set(sel_)
        keepq = np.array([ii for ii in range(len(qry)) if ii not in ss], int)
        gone = set(best[sel_])
        cand = np.array([c for c in cand if c not in gone]); qry = qry[keepq]
        if len(qry) == 0:
            break
    top0, e0 = first
    orig = np.take_along_axis(top0, np.argsort(-e0, axis=1, kind="stable"), 1)[:, :5]
    t5 = orig.copy()
    for qi, c in fixed.items():
        t5[qi] = [c] + [j for j in orig[qi] if j != c][:4]
    if len(qry):
        e2, top2 = e, top
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
    return t5, top0, e0


# ----------------------------------------------------------------------------- configs
def cfg_name(parts):
    return "+".join(f"{s}@{l:g}" for s, l in parts)


def run_pools(sims_q, ds, model, F, configs, label, assign=None):
    """sims_q: list of PoolSim with exactly one query per pool player (already thinned). configs: list of tuples of (src, lam)."""
    srcs = sorted({s for c in configs for s, _ in c} | set(SRC))
    frames, t0 = [], time.perf_counter()
    for sim in sims_q:
        pool = list(sim.players); qs = sim.queries; nc = len(pool)
        lib_z = [np.asarray(ds.Z[ds.rows(sim.library[p])]) for p in pool]
        q_z = [np.asarray(ds.Z[ds.rows(q.uids)]) for q in qs]
        pos = {p: i for i, p in enumerate(pool)}
        truth = np.array([pos[q.player_id] for q in qs])
        D = pool_D(F, [np.array(sim.library[p]) for p in pool], [np.array(q.uids) for q in qs], list(SRC))
        res = {}
        # diagnostics: each source alone over all candidates
        for nm, Dm in D.items():
            r1 = (Dm.argmin(1) == truth).astype(float)
            res[f"diag_{nm}|t1"] = r1
            res[f"diag_{nm}"] = r1
        def put(name, t5):
            h = t5[:, :5] == truth[:, None]
            res[name] = np.where(h.any(1), np.exp(-h.argmax(1).astype(float)), 0.0)
            res[name + "|t1"] = h[:, 0].astype(float)
        t5, top0, e0 = trans_t5(model, ds, lib_z, q_z, None)
        put("base", t5)
        res["recall20"] = (top0 == truth[:, None]).any(1).astype(float)
        for conf in configs:
            B = sum(-0.5 * lam * D[s] for s, lam in conf)
            t5, _, _ = trans_t5(model, ds, lib_z, q_z, B)
            put(cfg_name(conf), t5)
        df = pd.DataFrame({**res, "qplayer": [q.player_id for q in qs], "n": [len(z) for z in q_z], "pool": sim.draw})
        frames.append(df)
        print(label, "pool", sim.draw, f"{time.perf_counter() - t0:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def balanced_thin(sim, rng, ns=(5, 10, 15, 20)):
    """one query per pool player; exactly len(pool)/len(ns) players per N (random assignment)."""
    import dataclasses
    by = {}
    for q in sim.queries:
        by.setdefault((q.player_id, q.n), q)
    pl = list(sim.players)
    perm = rng.permutation(len(pl))
    k = len(pl) // len(ns)
    qs = []
    nn = {pl[perm[i]]: ns[min(i // k, len(ns) - 1)] for i in range(len(pl))}
    for p in pl:
        qs.append(by[(p, nn[p])])
    return dataclasses.replace(sim, queries=tuple(qs))


def get_model():
    import lightgbm as lgb
    fro = json.loads((E05 / "frozen.json").read_text())
    assert hashlib.sha256((E05 / "model.txt").read_bytes()).hexdigest() == fro["model_sha256"]
    return lgb.Booster(model_file=str(E05 / "model.txt"))


def grid():
    g = []
    for s in ("se1", "se2"):
        for lam in (0.1, 0.25, 0.5, 1.0, 2.0):
            g.append(((s, lam),))
    for s in ("hsl5", "hsl15", "hsl40"):
        for lam in (0.05, 0.1, 0.25, 0.5, 1.0):
            g.append(((s, lam),))
    return g


def stage_tune():
    model = get_model(); st = R.sets(); F = Feat()
    ds = R.get_ds("tune")
    npool = int(_argv[2]) if len(_argv) > 2 else 12
    mb = pd.read_parquet(CACHE / "games_meta.parquet")
    sims = RB_pools(ds.index, st["tune"], npool, 40, (5, 10, 15))
    rng = np.random.default_rng(R.derive_seed(2026, "player_sim3", "tuneN"))
    sims_q = [balanced_thin(s, rng, (5, 10, 15)) for s in sims]
    cf, tag = (_argv[3], _argv[4]) if len(_argv) > 4 else (None, "tune")
    cfgs = grid() if cf is None else [tuple((a, float(b)) for a, b in c) for c in json.loads(Path(cf).read_text())]
    df = run_pools(sims_q, ds, model, F, cfgs, "tune")
    df.to_parquet(OUT / f"{tag}_scores.parquet")


def RB_pools(index, players, npool, lib, ns):
    return R.build_player_pool_sim(index, players, 400, npool, lib, ns, seed=R.derive_seed(2026, "player_sim3", "tunepool"))


def stage_eval():
    model = get_model(); st = R.sets(); F = Feat()
    ds = R.get_ds("main")
    assert not (set(st["dev"]) & st["H"]) and not (set(st["dev"]) & st["fit"])
    npool = int(_argv[2]) if len(_argv) > 2 else 40
    cfile, tag = (_argv[3], _argv[4]) if len(_argv) > 4 else (str(OUT / "eval_configs.json"), "eval")
    cfgs = json.loads(Path(cfile).read_text())
    cfgs = [tuple((s, float(l)) for s, l in c) for c in cfgs]
    sims = R.build_player_pool_sim(ds.index, st["dev"], 400, npool, 100, (5, 10, 15, 20), seed=R.derive_seed(2026, "player_sim", "1"))
    rng = np.random.default_rng(R.derive_seed(2026, "player_sim3", "N"))
    sims_q = [balanced_thin(s, rng) for s in sims]
    df = run_pools(sims_q, ds, model, F, cfgs, "eval")
    df.to_parquet(OUT / f"{tag}_scores.parquet")


def stage_check():
    """sanity: reproduce player_sim d_trans_hi on the first 2 pools with the ORIGINAL N-uniform design (scores.parquet)."""
    model = get_model(); st = R.sets(); ds = R.get_ds("main")
    PS.MODEL = model
    sims = R.build_player_pool_sim(ds.index, st["dev"], 400, 40, 100, tuple(range(5, 21)), seed=R.derive_seed(2026, "player_sim", "1"))
    rng = np.random.default_rng(R.derive_seed(2026, "player_sim", "N"))
    old = pd.read_parquet(ROOT / ".scratch" / "player_sim" / "scores.parquet")
    for s in sims[:int(_argv[2]) if len(_argv) > 2 else 2]:
        sim = RB.thin(s, rng); pool = list(sim.players); qs = sim.queries
        lib_z = [np.asarray(ds.Z[ds.rows(sim.library[p])]) for p in pool]
        q_z = [np.asarray(ds.Z[ds.rows(q.uids)]) for q in qs]
        pos = {p: i for i, p in enumerate(pool)}; truth = np.array([pos[q.player_id] for q in qs])
        t5, _, _ = trans_t5(model, ds, lib_z, q_z, None)
        h = t5[:, :5] == truth[:, None]
        sc = np.where(h.any(1), np.exp(-h.argmax(1).astype(float)), 0.0)
        o = old[old.pool == s.draw]["d_trans_hi"].to_numpy()
        print("pool", s.draw, "mine", sc.mean(), "old", o.mean(), "max abs diff", np.abs(sc - o).max(), "n diff", int((np.abs(sc - o) > 1e-9).sum()))


# ----------------------------------------------------------------------------- stage 2: retrained reranker on top of e + distances
S2_SRC = ("se1", "se2", "hsl5", "hsl15", "hsl40")


def s2_features(e, top, qry, D, ns):
    """per (query, shortlisted candidate) features: frozen E04v2 score + SE/HSL distances (raw and relative to row min) + N."""
    k = e.shape[1]
    cols = [e, e - e.max(1, keepdims=True), np.argsort(np.argsort(-e, axis=1), axis=1).astype(float)]
    for nm in S2_SRC:
        d = D[nm][qry[:, None], top]
        cols += [d, d - d.min(1, keepdims=True)]
    cols.append(np.broadcast_to(ns[qry][:, None], e.shape).astype(float))
    return np.stack(cols, axis=2).reshape(len(qry) * k, -1)


def stage_s2():
    import lightgbm as lgb
    model = get_model(); st = R.sets(); F = Feat()
    dst = R.get_ds("tune")
    sims = RB_pools(dst.index, st["tune"], 36, 40, (5, 10, 15))
    rng = np.random.default_rng(R.derive_seed(2026, "player_sim3", "tuneN"))
    sims_q = [balanced_thin(s, rng, (5, 10, 15)) for s in sims]     # draws 1-12 reproduce the lambda-selection pools exactly
    Xs, ys, gs = [], [], []
    t0 = time.perf_counter()
    for sim in sims_q[12:]:                                         # train on draws 13-36 only
        pool = list(sim.players); qs = sim.queries
        lib_z = [np.asarray(dst.Z[dst.rows(sim.library[p])]) for p in pool]
        q_z = [np.asarray(dst.Z[dst.rows(q.uids)]) for q in qs]
        pos = {p: i for i, p in enumerate(pool)}; truth = np.array([pos[q.player_id] for q in qs])
        D = pool_D(F, [np.array(sim.library[p]) for p in pool], [np.array(q.uids) for q in qs], list(S2_SRC))
        f = prr.candidate_features(lib_z, q_z, dst.sel, prr.K, prr.N_DENSE)
        e = model.predict(f["X"], num_threads=4).reshape(len(qs), -1)
        ns = np.array([len(z) for z in q_z])
        X = s2_features(e, f["top"], np.arange(len(qs)), D, ns)
        Xs.append(X); ys.append((f["top"] == truth[:, None]).astype(int).reshape(-1))
        print("s2 train pool", sim.draw, f"{time.perf_counter() - t0:.0f}s", flush=True)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=7, min_data_in_leaf=100, feature_fraction=0.9, verbose=-1, seed=1, num_threads=4, deterministic=True)
    m2 = lgb.train(prm, lgb.Dataset(X, y), num_boost_round=200)
    m2.save_model(str(OUT / "s2_model.txt"))
    print("s2 trained rows", X.shape, "pos", y.sum(), flush=True)
    # ---- validate on tune draws 1-12 (independent of s2 training) and evaluate on dev 40 pools
    def run(sims_list, ds, label, extra_cfg):
        frames = []
        for sim in sims_list:
            pool = list(sim.players); qs = sim.queries; nc = len(pool)
            lib_z = [np.asarray(ds.Z[ds.rows(sim.library[p])]) for p in pool]
            q_z = [np.asarray(ds.Z[ds.rows(q.uids)]) for q in qs]
            pos = {p: i for i, p in enumerate(pool)}; truth = np.array([pos[q.player_id] for q in qs])
            D = pool_D(F, [np.array(sim.library[p]) for p in pool], [np.array(q.uids) for q in qs], list(SRC))
            ns = np.array([len(z) for z in q_z]); res = {}
            def put(name, t5):
                h = t5[:, :5] == truth[:, None]
                res[name] = np.where(h.any(1), np.exp(-h.argmax(1).astype(float)), 0.0); res[name + "|t1"] = h[:, 0].astype(float)
            put("base", trans_t5(model, ds, lib_z, q_z, None)[0])
            for conf in extra_cfg:
                B = sum(-0.5 * lam * D[s_] for s_, lam in conf); put(cfg_name(conf), trans_t5(model, ds, lib_z, q_z, B)[0])
            rs = lambda e, top, qry: m2.predict(s2_features(e, top, qry, D, ns), num_threads=4).reshape(e.shape)   # noqa: E731
            # s2 returns probabilities; use logit as the score so that softmax is a sensible confidence
            rs_logit = lambda e, top, qry: np.log(np.clip(rs(e, top, qry), 1e-9, 1 - 1e-9) / (1 - np.clip(rs(e, top, qry), 1e-9, 1 - 1e-9)))   # noqa: E731
            for tau in (0.9, 0.95):
                put(f"s2_tau{tau}", trans_t5(model, ds, lib_z, q_z, None, tau=tau, rescore=rs_logit)[0])
            frames.append(pd.DataFrame({**res, "qplayer": [q.player_id for q in qs], "n": ns, "pool": sim.draw, "recall20": 0.0}))
            print(label, "pool", sim.draw, f"{time.perf_counter() - t0:.0f}s", flush=True)
        return pd.concat(frames, ignore_index=True)
    extra = [(("se1", 0.5), ("hsl5", 0.25))]
    run(sims_q[:12], dst, "s2 tune-val", extra).to_parquet(OUT / "s2tune_scores.parquet")
    dsm = R.get_ds("main")
    sims = R.build_player_pool_sim(dsm.index, st["dev"], 400, 40, 100, (5, 10, 15, 20), seed=R.derive_seed(2026, "player_sim", "1"))
    rng = np.random.default_rng(R.derive_seed(2026, "player_sim3", "N"))
    run([balanced_thin(s_, rng) for s_ in sims], dsm, "s2 dev", extra + [(("se1", 0.5), ("hsl15", 0.25))]).to_parquet(OUT / "s2dev_scores.parquet")


# ----------------------------------------------------------------------------- reporting
def table(df, base="base", cols=None, n_boot=4000, tag="ps3"):
    cols = cols or [c for c in df.columns if "|" not in c and not c.startswith("diag_") and c not in ("qplayer", "n", "pool", "recall20")]
    ns = sorted(df.n.unique())
    rows = []
    for m in cols:
        r = paired_bootstrap(df[m].to_numpy(), df[base].to_numpy(), df["qplayer"].to_numpy(), n_boot=n_boot, seed=S(tag, m))
        row = dict(method=m, score=df[m].mean(), top1=df[m + "|t1"].mean(), d=r.delta, lo=r.ci_low, hi=r.ci_high)
        for n in ns:
            row[f"N{n}"] = df.loc[df.n == n, m].mean()
            row[f"d{n}"] = df.loc[df.n == n, m].mean() - df.loc[df.n == n, base].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def stage_report():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    which = _argv[2]
    df = pd.read_parquet(OUT / f"{which}_scores.parquet")
    t = table(df)
    t.to_csv(OUT / f"{which}_table.csv", index=False)
    print(t.round(4).to_string(index=False))
    d = [c for c in df.columns if c.startswith("diag_") and "|" not in c]
    print("alone top1 (all 400 candidates) by N:")
    print(df.groupby("n")[d].mean().round(3).to_string())
    print("base score by pool: mean", df.groupby("pool").base.mean().mean(), "sd", df.groupby("pool").base.mean().std(),
          "min", df.groupby("pool").base.mean().min(), "max", df.groupby("pool").base.mean().max())
    print("recall20", df.recall20.mean())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"prep": stage_prep, "tune": stage_tune, "eval": stage_eval, "check": stage_check, "report": stage_report, "s2": stage_s2}[STAGE]()
