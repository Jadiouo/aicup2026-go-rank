"""E30 shared code: data access, scenario builders, methods A/B/D, detection statistics.  CPU only, <=4 threads.
Protocol: experiments/E30_player_robustness/PLAN.md.  B = player_sim3.trans_t5 (v3), reused verbatim.
No H players are ever loaded as pool members or outsiders (asserted in Data.__init__)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(ROOT / "scripts" / "experiments" / "e23"), str(ROOT / "src")]
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]                      # player_sim / player_sim3 read argv at import
import numpy as np
import pandas as pd
import player_sim3 as P3
sys.argv = _argv
from gogame.seeds import derive_seed

R = P3.R
prr = P3.prr
sm = P3.sm
SCR = ROOT / ".scratch" / "e30"
SCR.mkdir(parents=True, exist_ok=True)
NS_STD = (5, 10, 15, 20)
NS_SMALL = (1, 2, 3, 5)
GAMMAS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
TAU = 0.9


def seed(*k):
    return derive_seed(2026, "E30", *[str(x) for x in k])


# ----------------------------------------------------------------------------- data
class Data:
    """uid -> Z row for dev black (E05 Z), tune black (Z_tune) and dev white (prep_white.py); player game lists by colour."""

    def __init__(self):
        st = R.sets()
        self.st = st
        meta = pd.read_parquet(R.CACHE / "games_meta.parquet")
        self.sel = np.load(R.E05 / "sel.npy")
        srcs = [(np.load(R.E05 / "Z.npy", mmap_mode="r"), json.loads((R.E05 / "uids.json").read_text())),
                (np.load(R.SCR / "Z_tune.npy", mmap_mode="r"), json.loads((R.SCR / "uids_tune.json").read_text())),
                (np.load(SCR / "Z_dev_white.npy", mmap_mode="r"), json.loads((SCR / "uids_dev_white.json").read_text()))]
        self.Zs = [s[0] for s in srcs]
        self.loc = {}
        for k, (_, uids) in enumerate(srcs):
            for i, u in enumerate(uids):
                self.loc.setdefault(u, (k, i))
        use = set(st["dev"]) | set(st["tune"])
        assert not (use & st["H"]) and not (use & st["fit"])
        m = meta[meta.player_id.isin(use) & meta.uid.isin(self.loc.keys())]
        self.black, self.white, self.rank = {}, {}, {}
        for pid, g in m.groupby("player_id"):
            self.rank[pid] = int(g.rank_id.iloc[0])
            self.black[pid] = sorted(g.loc[g.color == 0, "uid"])
            self.white[pid] = sorted(g.loc[g.color == 1, "uid"])
        self.dev = [p for p in st["dev"] if p in self.black]
        # tuneA: first 450 of (tune players with >=60 black games) by sha1; tuneB: the rest of tune
        tun = sorted(p for p in st["tune"] if p in self.black)
        el = sorted((p for p in tun if len(self.black[p]) >= 60), key=lambda p: hashlib.sha1(p.encode()).hexdigest())
        self.tuneA = sorted(el[:450])
        self.tuneB = sorted(set(tun) - set(self.tuneA))
        assert not (set(self.tuneA) & set(self.tuneB)) and not (set(self.tuneA) & set(self.dev))

    def z(self, uids) -> np.ndarray:
        locs = [self.loc[u] for u in uids]
        out = np.empty((len(locs), self.Zs[0].shape[1]), np.float32)
        ks = np.array([a for a, _ in locs]); rs = np.array([b for _, b in locs])
        for k in set(ks.tolist()):
            m = ks == k
            out[m] = self.Zs[k][rs[m]]
        return out

    def elig(self, players, need_b=0, need_w=0):
        return [p for p in players if len(self.black.get(p, ())) >= need_b and len(self.white.get(p, ())) >= need_w]


@dataclasses.dataclass
class Pool:
    name: str
    rep: int
    cand: list
    lib_z: list
    q_z: list
    truth: np.ndarray            # candidate index, -1 = outsider
    n: np.ndarray
    sib: np.ndarray              # number of queries sharing the same true candidate (0 for outsiders)
    overlap_q: float = 0.0       # fraction of queries whose games overlap a sibling's (insufficient games)


def _assign_n(rng, nq, ns):
    perm = rng.permutation(nq)
    out = np.empty(nq, int)
    for r, i in enumerate(perm):
        out[i] = ns[r % len(ns)]
    return out


def build_pool(D: Data, name, rep, rng, pool_pids, lib_spec, owners, ns, qcolor="B"):
    """pool_pids: candidate player ids.  lib_spec: {'B': k, 'W': k}.  owners: list of player ids (one entry per query;
    repeats = sibling queries; ids not in pool_pids = outsiders, black games only).  qcolor: 'B' | 'W' | 'mix'."""
    pos = {p: i for i, p in enumerate(pool_pids)}
    state, ptr = {}, {}
    lib_uids = {}
    for p in pool_pids:
        st_ = {"B": [D.black[p][j] for j in rng.permutation(len(D.black[p]))],
               "W": [D.white[p][j] for j in rng.permutation(len(D.white[p]))] if lib_spec.get("W") or qcolor != "B" else []}
        lu = []
        for c, k in lib_spec.items():
            assert len(st_[c]) >= k, (p, c, len(st_[c]), k)
            lu += st_[c][:k]
        lib_uids[p] = sorted(lu)
        state[p] = st_
        ptr[p] = {c: lib_spec.get(c, 0) for c in ("B", "W")}
    nq = len(owners)
    Ns = _assign_n(rng, nq, ns)
    q_uids, truth, sib_n, ovl = [], [], [], 0
    out_state = {}
    cnt_owner = {}
    for o in owners:
        cnt_owner[o] = cnt_owner.get(o, 0) + 1
    for i, o in enumerate(owners):
        n = int(Ns[i])
        if o in pos:
            st_, pt = state[o], ptr[o]
            if qcolor == "mix":
                need = {"B": n // 2, "W": n - n // 2}
            else:
                need = {qcolor: n}
            uu, ov = [], False
            for c, k in need.items():
                if pt[c] + k <= len(st_[c]):
                    uu += st_[c][pt[c]:pt[c] + k]; pt[c] += k
                else:                                           # not enough disjoint games left: sample from the remainder
                    pool_c = st_[c][lib_spec.get(c, 0):]
                    uu += [pool_c[j] for j in rng.permutation(len(pool_c))[:k]]; ov = True
            ovl += ov
            q_uids.append(uu); truth.append(pos[o]); sib_n.append(cnt_owner[o])
        else:
            if o not in out_state:
                out_state[o] = [D.black[o][j] for j in rng.permutation(len(D.black[o]))]
            assert len(out_state[o]) >= n
            q_uids.append(out_state[o][:n]); truth.append(-1); sib_n.append(0)
            out_state[o] = out_state[o][n:]
    lib_z = [D.z(lib_uids[p]) for p in pool_pids]
    q_z = [D.z(u) for u in q_uids]
    return Pool(name, rep, list(pool_pids), lib_z, q_z, np.array(truth), np.array([len(u) for u in q_uids]), np.array(sib_n), ovl / max(nq, 1))


# ----------------------------------------------------------------------------- scenario builders
def pick(rng, players, k):
    return sorted(players[i] for i in rng.choice(len(players), size=k, replace=False))


def sc_base(D, players, npool, lib, ns, name):
    def f(rep):
        rng = np.random.default_rng(seed(name, rep))
        pl = pick(rng, D.elig(players, lib + max(ns)), npool)
        return build_pool(D, name, rep, rng, pl, {"B": lib}, list(rng.permutation(pl)), ns)
    return f


def sc_outsiders(D, players, outs, npool, lib, frac, append, ns, name):
    def f(rep):
        rng = np.random.default_rng(seed(name, rep))
        pl = pick(rng, D.elig(players, lib + max(ns)), npool)
        if append:
            nout = int(round(frac * npool)); inn = list(pl)
        else:
            nout = int(round(frac * npool)); inn = list(rng.permutation(pl)[:npool - nout])
        outp = [p for p in outs if p not in set(pl)]
        outp = D.elig(outp, max(ns))
        oo = list(pick(rng, outp, nout))
        owners = list(rng.permutation(np.array(inn + oo, dtype=object)))
        return build_pool(D, name, rep, rng, pl, {"B": lib}, owners, ns)
    return f


def sc_siblings(D, players, npool, lib, ns, mode, name):
    def f(rep):
        rng = np.random.default_rng(seed(name, rep))
        pl = pick(rng, D.elig(players, lib + 3 * max(ns)), npool)
        if mode == "two":
            owners = list(pl) * 2
        else:                                                    # random assignment of npool queries, cap 3 per candidate
            cnt = {p: 0 for p in pl}; owners = []
            while len(owners) < npool:
                p = pl[int(rng.integers(len(pl)))]
                if cnt[p] < 3:
                    cnt[p] += 1; owners.append(p)
        owners = list(rng.permutation(np.array(owners, dtype=object)))
        return build_pool(D, name, rep, rng, pl, {"B": lib}, owners, ns)
    return f


def sc_band(D, lib, ns, name, control):
    def f(rep):
        rng = np.random.default_rng(seed(name, rep))
        w = rep % 8
        band = D.elig([p for p in D.dev if w <= D.rank[p] <= w + 2], lib + max(ns))
        pl = sorted(band)
        if control:
            allp = D.elig(D.dev, lib + max(ns))
            pl = pick(rng, allp, len(band))
        return build_pool(D, name, rep, rng, pl, {"B": lib}, list(rng.permutation(pl)), ns)
    return f


def sc_color(D, npool, kind, name):
    def f(rep):
        rng = np.random.default_rng(seed(name, rep))
        if kind == "a":      # queries white, library 100 black
            pl = pick(rng, D.elig(D.dev, 100, 20), npool); ls, qc = {"B": 100}, "W"
        elif kind == "b":    # mixed library 50B+50W, mixed queries
            pl = pick(rng, D.elig(D.dev, 50 + 10, 50 + 10), npool); ls, qc = {"B": 50, "W": 50}, "mix"
        else:                # library 50 white, queries white
            pl = pick(rng, D.elig(D.dev, 0, 50 + 20), npool); ls, qc = {"W": 50}, "W"
        return build_pool(D, name, rep, rng, pl, ls, list(rng.permutation(pl)), NS_STD, qcolor=qc)
    return f


def eval_scenarios(D: Data):
    sup = sorted(D.dev + D.tuneB)
    S = {
        "S1": sc_base(D, D.dev, 400, 100, NS_SMALL, "S1"),
        "S2a": sc_base(D, D.dev, 400, 20, NS_STD, "S2a"),
        "S2b": sc_base(D, D.dev, 400, 50, NS_STD, "S2b"),
        "S3": sc_base(D, sup, 1000, 30, NS_STD, "S3"),
        "S3c": sc_base(D, sup, 400, 30, NS_STD, "S3c"),
        "S3u": sc_base(D, D.dev, len(D.elig(D.dev, 120)), 100, NS_STD, "S3u"),
        "S4a": sc_outsiders(D, D.dev, D.tuneB, 400, 100, 0.2, False, NS_STD, "S4a"),
        "S4b": sc_outsiders(D, D.dev, D.tuneB, 400, 100, 0.5, False, NS_STD, "S4b"),
        "S4c": sc_outsiders(D, D.dev, D.tuneB, 400, 100, 0.5, True, NS_STD, "S4c"),
        "S5a": sc_siblings(D, D.dev, 400, 50, NS_STD, "rand", "S5a"),
        "S5c": sc_siblings(D, D.dev, 400, 50, NS_STD, "two", "S5c"),
        "S6": sc_band(D, 100, NS_STD, "S6", False),
        "S6c": sc_band(D, 100, NS_STD, "S6c", True),
        "S7a": sc_color(D, 400, "a", "S7a"),
        "S7b": sc_color(D, 400, "b", "S7b"),
        "S7c": sc_color(D, 400, "c", "S7c"),
    }
    return S


def tune_scenarios(D: Data):
    A = D.tuneA
    return {
        "T0": sc_base(D, A, 400, 40, NS_STD, "T0"),
        "T1": sc_base(D, A, 400, 40, NS_SMALL, "T1"),
        "T2": sc_base(D, A, 400, 20, NS_STD, "T2"),
        "T4a": sc_outsiders(D, A, A, 300, 40, 0.2, False, NS_STD, "T4a"),
        "T4b": sc_outsiders(D, A, A, 300, 40, 0.5, False, NS_STD, "T4b"),
        "T4c": sc_outsiders(D, A, A, 300, 40, 0.5, True, NS_STD, "T4c"),
        "T5a": sc_siblings(D, A, 400, 30, (5, 10), "rand", "T5a"),
        "T5c": sc_siblings(D, A, 400, 30, (5, 10), "two", "T5c"),
    }


def s0_pools(D: Data, ds, npool=20):
    """S0 = player_sim3 eval protocol (same seeds, first npool of the 40 pools)."""
    st = R.sets()
    sims = R.build_player_pool_sim(ds.index, st["dev"], 400, npool, 100, NS_STD, seed=R.derive_seed(2026, "player_sim", "1"))
    rng = np.random.default_rng(R.derive_seed(2026, "player_sim3", "N"))
    out = []
    for s in sims:
        s = P3.balanced_thin(s, rng)
        pool = list(s.players); pos = {p: i for i, p in enumerate(pool)}
        lib_z = [np.asarray(ds.Z[ds.rows(s.library[p])]) for p in pool]
        q_z = [np.asarray(ds.Z[ds.rows(q.uids)]) for q in s.queries]
        truth = np.array([pos[q.player_id] for q in s.queries])
        out.append(Pool("S0", s.draw, pool, lib_z, q_z, truth, np.array([len(z) for z in q_z]), np.ones(len(truth), int)))
    return out


# ----------------------------------------------------------------------------- methods
def top5_orig(top0, e0):
    return np.take_along_axis(top0, np.argsort(-e0, axis=1, kind="stable"), 1)[:, :5]


def cand_best_query(top, e, qmask=None, exclude=None):
    """for each candidate: (best score, query index) over queries (with qmask) whose shortlist contains it."""
    nq, K = top.shape
    qi = np.repeat(np.arange(nq), K); cf = top.ravel(); ef = e.ravel()
    ok = np.ones(len(cf), bool)
    if qmask is not None:
        ok &= qmask[qi]
    if exclude is not None:
        ok &= ~exclude[cf]
    qi, cf, ef = qi[ok], cf[ok], ef[ok]
    if len(cf) == 0:
        return cf, ef, qi
    o = np.lexsort((-ef, cf))
    cf, ef, qi = cf[o], ef[o], qi[o]
    first = np.r_[True, cf[1:] != cf[:-1]]
    return cf[first], ef[first], qi[first]


def decode_D(top0, e0, nc, gamma, tau=TAU, rounds=3):
    nq, K = e0.shape
    e = e0.copy()
    orig = top5_orig(top0, e0)
    active = np.ones(nq, bool); claimed_c = np.zeros(nc, bool); claim = {}
    for _ in range(rounds):
        order = np.argsort(-e, axis=1, kind="stable")
        best = np.take_along_axis(top0, order[:, :1], 1)[:, 0]
        p1 = sm(e).max(1)
        cc, _, cq = cand_best_query(top0, e, qmask=active, exclude=claimed_c)
        qbest = np.full(nc, -1); qbest[cc] = cq
        sel = [i for i in np.flatnonzero(active) if p1[i] >= tau and not claimed_c[best[i]] and qbest[best[i]] == i]
        if not sel:
            break
        for i in sel:
            claim[i] = best[i]; claimed_c[best[i]] = True; active[i] = False
        pen = np.zeros(nc, bool)
        for i in sel:
            pen[best[i]] = True
        e = e - gamma * (pen[top0] & active[:, None])
    order = np.argsort(-e, axis=1, kind="stable")
    t5 = np.take_along_axis(top0, order, 1)[:, :5].copy()
    for i, c in claim.items():
        t5[i] = ([c] + [j for j in orig[i] if j != c])[:5]
    return t5


def det_stats(top0, e0, nc, tau=TAU):
    nq, K = e0.shape
    order = np.argsort(-e0, axis=1, kind="stable")
    best = np.take_along_axis(top0, order[:, :1], 1)[:, 0]
    p1 = sm(e0).max(1)
    cnt = np.bincount(best, minlength=nc)
    coll = cnt[best] > 1
    c_all = float(coll.mean())
    conf = p1 >= tau
    c_conf = float(coll[conf].mean()) if conf.sum() >= 20 else c_all
    cc, _, cq = cand_best_query(top0, e0)
    qbest = np.full(nc, -1); qbest[cc] = cq
    mut = float((qbest[best] == np.arange(nq)).mean())
    return dict(c_all=c_all, c_conf=c_conf, s_mut=1.0 - mut, exc=c_all - (1.0 - float(p1.mean())),
                p1_mean=float(p1.mean()), conf_frac=float(conf.mean()), distinct_over_nc=float(len(np.unique(best)) / nc),
                nq=nq, nc=nc)


def score_t5(t5, truth):
    h = np.asarray(t5)[:, :5] == truth[:, None]
    return np.where(h.any(1), np.exp(-h.argmax(1).astype(float)), 0.0), h[:, 0].astype(float)


def run_pool(pool: Pool, model, sel, gammas=GAMMAS):
    ds = SimpleNamespace(sel=sel)
    nc = len(pool.cand)
    t5B, top0, e0 = P3.trans_t5(model, ds, pool.lib_z, pool.q_z, None)
    A = top5_orig(top0, e0)
    res = {}
    res["A"], res["A|t1"] = score_t5(A, pool.truth)
    res["B"], res["B|t1"] = score_t5(t5B, pool.truth)
    for g in gammas:
        res[f"D{g:g}"], res[f"D{g:g}|t1"] = score_t5(decode_D(top0, e0, nc, g), pool.truth)
    res["recall20"] = (top0 == pool.truth[:, None]).any(1).astype(float)
    q = pd.DataFrame({**res, "n": pool.n, "outsider": pool.truth < 0, "sib": pool.sib, "scn": pool.name, "rep": pool.rep})
    stt = det_stats(top0, e0, nc)
    stt.update(scn=pool.name, rep=pool.rep, overlap_q=pool.overlap_q, B_score=float(res["B"].mean()), A_score=float(res["A"].mean()),
               out_frac=float((pool.truth < 0).mean()))
    return q, stt


def get_model():
    return P3.get_model()


def decide_decoder(top0, e0, nc, tuned_json=None):
    """Day-of rule (labels never used).  Input: first-pass shortlist `top0` (nq,K) and reranker scores `e0` (nq,K) of ONE candidate file.
    Returns (decoder, stats): decoder 'B' (v3 transductive) if exc <= theta else 'D' (soft exclusion, gamma*), 'D' only if tuned stat/F say so.
    Also returns size_mismatch (nq != nc: not a bijection by construction); it is reported only, the decision uses exc alone."""
    tj = json.loads((ROOT / "experiments" / "E30_player_robustness" / "tuned.json").read_text()) if tuned_json is None else tuned_json
    ch = tj["chosen"]
    st = det_stats(top0, e0, nc)
    non11 = st[ch["stat"]] > ch["theta"]
    return (ch["F"] if non11 else "B"), dict(st, theta=ch["theta"], gamma=tj["gamma_star"], size_mismatch=bool(st["nq"] != nc))
