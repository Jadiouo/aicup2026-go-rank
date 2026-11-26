"""E102 Rank: vector bank (857,634 training games, own side + opponent side, finh_v4 vectors) and four readouts T1-T4.
Plan (frozen, never edited): experiments/E102_rank_vector_bank/PLAN.md.  Reuses scripts/experiments/e96/e96_score.py and e101_score.py READ-ONLY.
Reference arm REF = fixed-setting F_opp_x (gamma 0, tau 0, alpha .1, lam 2.0; votes of hd in (0,1) games, E96 shift_lift) = the current candidate v10d.  Reproduced first (< 1e-9 vs posthoc_x_fixed.json).
Rank index 0..9 = D, C, B, A, 1D..6D.  eff 0 = file owner / questioned player has BLACK.

Modes
  unit        : no data, no bank; tests the table t (table rank of an opponent), the kNN search, the prototype code, the slot-level vote path          -> stdout only
  repro       : reference arm must reproduce .scratch/e96/posthoc_x_fixed.json (< 1e-9); also checks that the slot-level vote path (w = 0) equals the reference
                                                                                                                                                  -> .scratch/e102/repro_check.json
  dry-noise   : FAKE bank (random unit opponent vectors; own side = head rows + noise) -> all arms; pass = |Delta| <= .005 for T2 / T3 / T4          -> .scratch/e102/dry_noise.json
  dry-signal  : FAKE bank with planted signal (labels used ONLY here); pass = T3 delta >= +.02 & lo > 0 & T3_shuf lo <= 0, T4 same             -> .scratch/e102/dry_signal.json
  rehearse    : the formal `run` path on a 300-row REAL-FORMAT bank (selftest vectors) in .scratch/e102/rehearsal; numbers meaningless -> .scratch/e102/rehearsal/result.json (may be re-run)
  bench       : timing of the kNN search / head pass on a random full-size bank (857,634 rows)                                                         -> stdout
  run         : REAL bank.  Checks bank files first, then repro assertion, then ONE formal scoring                                                    -> .scratch/e102/result.json (refuses to overwrite)

  CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 env PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/experiments/e102/e102_score.py <mode>
Never touches H, h_eval_ledger.jsonl, data/submissions, result_v4.json.  Writes only .scratch/e102/.
"""
import os, sys, json, hashlib, time, resource
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts/experiments/e101"), str(ROOT / "scripts/experiments/e96"), str(ROOT / "scripts/experiments/e90"), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]
import numpy as np
import e101_score as E1                                   # read-only: K_WHITE, K_BLACK, load_everything, reference_check, fixed setting

OUT = ROOT / ".scratch/e102"; NNC = OUT / "nn_cache"; PLAN = ROOT / "experiments/E102_rank_vector_bank/PLAN.md"
N_BANK = 857_634
GM, LO, HI, AL, LAM = E1.GM, E1.LO, E1.HI, E1.AL, E1.LAM
AL4 = .1                                                   # T4 alpha, fixed
K_NN3 = 50                                                 # T3 neighbours
T3_GRID = [(b, w) for b in (20., 40., 80.) for w in (.5, 1., 2.)]                                     # (beta, w) 9 combos
T4_GRID = [(k, b, l) for k in (10, 30, 100) for b in (20., 40., 80.) for l in (.5, 1., 2.)]            # (K, beta, lam4) 27 combos
NB_NEEDED = 4000
NOISE_TOL = .005; SIGNAL_MIN = .02; PLANT_FRAC = .30; PLANT_NOISE = .3; T4_PLANT_ROWS = 5; HI_PMAX_T2B = .8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ============================================================================================================================ rank table t (the table rank of the opponent of a bank game)
def make_T():
    """T[hd, eff, owner_rank] -> distribution over the OPPONENT's rank (opponent of the file owner = the person the neighbour vector stands for).
    hd 0: unit vector at the owner's rank.  hd 1: K_<owner colour>[owner_rank] normalised (eff 0 = owner black -> K_BLACK, eff 1 = owner white -> K_WHITE); rows with sum 0 stay 0 (no vote)."""
    T = np.zeros((2, 2, 10, 10))
    for k in range(10):
        T[0, :, k, k] = 1.
        for eff, K in ((0, E1.K_BLACK), (1, E1.K_WHITE)):
            s = K[k].sum()
            if s > 0: T[1, eff, k] = K[k] / s
    return T


T_TAB = make_T()


def unit_T():
    """Direction checks by hand (owner = file owner, t = opponent of the owner)."""
    # hd 1, owner WHITE (the owner is the stronger one, opponent one station-level lower): owner 3D (6) -> opponent 2D (5); owner 1D (4) -> A (3)
    assert abs(T_TAB[1, 1, 6][5] - 1) < 1e-12 and abs(T_TAB[1, 1, 4][3] - 1) < 1e-12
    # hd 1, owner BLACK (opponent one station-level higher): owner 3D (6) -> opponent 4D (7); owner 6D (9) -> zero row (outside the library)
    assert abs(T_TAB[1, 0, 6][7] - 1) < 1e-12 and T_TAB[1, 0, 9].sum() == 0
    # same band 2/3 + one band lower / higher 1/3 for A,B,C ; D black -> D 2/3 + C 1/3 ; D white -> only D (the rest outside) -> unit D
    assert np.allclose(T_TAB[1, 1, 3][[3, 2]], [2 / 3, 1 / 3]) and np.allclose(T_TAB[1, 0, 2][[2, 3]], [2 / 3, 1 / 3]) and abs(T_TAB[1, 1, 0][0] - 1) < 1e-12
    assert np.allclose(T_TAB[1, 0, 0][[0, 1]], [2 / 3, 1 / 3])
    assert np.allclose(T_TAB[0, 0].sum(1), 1) and np.allclose(T_TAB[0, 1], np.eye(10))
    # chain through shift_lift (opponent rank -> questioned player's rank, E96): questioned BLACK (eff 0), opponent = bank neighbour whose t is 'opponent at 3D (6)'
    # -> E96 shift for own black reads lift[min(k+1, 9)] => likelihood mass at k = 5 (2D) and 6 (3D): a black player facing 3D plays below 3D.  Opposite for white.
    import e96_score as S
    lift = np.zeros((2, 10)); lift[:, 6] = 1.
    x = S.shift_lift(lift, np.array([0, 1]), np.array([1, 1]))
    assert np.allclose(x[0][[5, 6]], [.5, .5]) and np.allclose(x[1][[6, 7]], [.5, .5])
    # end to end direction (table t of a bank neighbour -> E96 shift onto the questioned player's axis).  Questioned BLACK, hd 1, true rank r: the opponent (the neighbour) sits at r+1.
    # A bank game in which that person plays a WHITE owner of rank r+2 (hd 1) gives t = unit at r+1 -> shifted likelihood must put mass on r (and r+1), none on r+2.
    for r in range(3, 8):                                       # clean (1D and above) part of the station rule: K_WHITE[k, k-1] = 1 for k >= 5
        x = S.shift_lift(T_TAB[1, 1, r + 2][None], np.array([0]), np.array([1]))[0]; assert abs(x[r] - .5) < 1e-12 and abs(x[r + 1] - .5) < 1e-12 and x[r + 2] == 0 and x.sum() > .999, (r, x)
    # questioned WHITE, hd 1, true rank r: the opponent sits at r-1; a BLACK owner of rank r-2 (K_BLACK[k, k+1] = 1 for 4 <= k <= 8) -> t = unit at r-1 -> mass on r-1 and r, none on r-2
    for r in range(6, 10):
        x = S.shift_lift(T_TAB[1, 0, r - 2][None], np.array([1]), np.array([1]))[0]; assert abs(x[r] - .5) < 1e-12 and abs(x[r - 1] - .5) < 1e-12 and x[r - 2] == 0, (r, x)
    return True


# ============================================================================================================================ kNN, head pass
def knn(Q, B, k, qbatch=200, bblock=131072, row_map=None):
    """Exact top-k by inner product (unit vectors = cosine), numpy, blocked over the bank.  Q [nq,128] B [m,128] float32.  Returns (row index into B or row_map[...], cos) sorted descending."""
    nq = len(Q); oi = np.zeros((nq, k), np.int32); oc = np.zeros((nq, k), np.float32)
    for a in range(0, nq, qbatch):
        q = Q[a:a + qbatch]; bc = np.full((len(q), k), -9., np.float32); bi = np.zeros((len(q), k), np.int32)
        for s in range(0, len(B), bblock):
            sims = q @ B[s:s + bblock].T; kk = min(k, sims.shape[1])
            pi = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]; pc = np.take_along_axis(sims, pi, 1)
            ac = np.concatenate([bc, pc], 1); ai = np.concatenate([bi, pi.astype(np.int32) + s], 1)
            sel = np.argpartition(-ac, k - 1, axis=1)[:, :k]; bc = np.take_along_axis(ac, sel, 1); bi = np.take_along_axis(ai, sel, 1)
        o = np.argsort(-bc, 1); oc[a:a + len(q)] = np.take_along_axis(bc, o, 1); oi[a:a + len(q)] = np.take_along_axis(bi, o, 1)
    if row_map is not None: oi = row_map[oi].astype(np.int32)
    return oi, oc


def head_top1(gv, Wn, scale=30.0, chunk=4000):
    """top-1 class and p_max of softmax(30 cos) (same numbers as e96_score.head_stats pmax / top1, without the top-20 bookkeeping)."""
    n = len(gv); top = np.zeros(n, np.int32); pm = np.zeros(n, np.float32)
    for a in range(0, n, chunk):
        lg = scale * (gv[a:a + chunk].astype(np.float32) @ Wn.T); t = lg.argmax(1); mx = lg[np.arange(len(lg)), t]
        lg -= mx[:, None]; np.exp(lg, out=lg); top[a:a + chunk] = t; pm[a:a + chunk] = 1. / lg.sum(1)
    return top, pm


def unit_rows(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def cache_load(fn, key):
    if fn.exists():
        z = np.load(fn, allow_pickle=False)
        if str(z["key"]) == key: return {k: z[k] for k in z.files if k != "key"}
        log("cache key mismatch, recomputing:", fn.name)
    return None


def cache_save(fn, key, **kw):
    fn.parent.mkdir(parents=True, exist_ok=True); tmp = fn.with_name(fn.name + ".tmp.npz"); np.savez(tmp, key=np.array(key), **kw); os.replace(tmp, fn)


def sha_arr(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ============================================================================================================================ question structure
def slots(ctx, game):
    """Per question slot: game index (into game arrays), valid, voting (valid & hd in (0,1)), eff, hd.  Same rule as e96 question_votes with hd_ok (0,1)."""
    rows, rid = ctx["rows"], ctx["rid"]; valid = rows >= 0
    idx = np.searchsorted(rid, np.where(valid, rows, rid[0])); assert (rid[idx[valid]] == rows[valid]).all()
    hd = np.where(valid, game["hd"][idx], -1).astype(np.int64); eff = np.where(valid, game["eff"][idx], 0).astype(np.int64)
    return dict(idx=idx, valid=valid, vote=valid & np.isin(hd, (0, 1)), eff=eff, hd=hd)


def votes_slot(S, sl, lib_lift, nn_lift=None, w=0.):
    """Question vote at slot level.  per game x = normalise(lib + w * nn) over the OPPONENT's rank axis, then E96 shift_lift (hd 1 -> questioned player's axis), sum over voting slots, normalise (uniform .1 if none).
    nn_lift [Q,S,10] slot-level (zero rows = no neighbour vote).  w = 0 or nn_lift None -> exactly the reference path."""
    Q_, S_ = sl["idx"].shape; x = lib_lift[sl["idx"]]                                      # [Q,S,10]
    if nn_lift is not None and w > 0:
        x = x + w * nn_lift; x = x / x.sum(2, keepdims=True)
    x = S.shift_lift(x.reshape(-1, 10), sl["eff"].reshape(-1), sl["hd"].reshape(-1)).reshape(Q_, S_, 10)
    V = np.einsum("qs,qsk->qk", sl["vote"].astype(np.float64), x); tot = V.sum(1); has = tot > 0
    Vn = np.full(V.shape, .1); Vn[has] = V[has] / tot[has, None]
    return Vn, has


def fuse2(f79, V, lam, alpha, V4=None, lam4=0., alpha4=AL4):
    z = np.log(f79 + 1e-12) + lam * np.log(V + alpha)
    if V4 is not None: z = z + lam4 * np.log(V4 + alpha4)
    z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def nn_lift_games(nn_row, nn_cos, T_bank, beta, G):
    """Per queried game: normalise(sum_j exp(beta (cos_j - cos_max)) * t_j); zero where all neighbours are zero rows.  Returns [G,10] float64 (G rows of the query list)."""
    out = np.zeros((len(nn_row), 10))
    for a in range(0, len(nn_row), 5000):
        c = nn_cos[a:a + 5000].astype(np.float64); w = np.exp(beta * (c - c.max(1, keepdims=True)))
        v = np.einsum("gk,gkj->gj", w, T_bank[nn_row[a:a + 5000]].astype(np.float64)); s = v.sum(1); ok = s > 0
        v[ok] /= s[ok, None]; v[~ok] = 0.; out[a:a + 5000] = v
    return out


def nested_pick(S, ctx, Plist):
    """5-part nested selection over players (same parts / score / tie rule as e96 nested: best balanced score on the other 4 parts, ties -> earlier in the list)."""
    yq, nq, part = ctx["yq"], ctx["nq"], ctx["part"]; scq = [S.sc_q(P, yq) for P in Plist]; Pout = np.zeros_like(Plist[0]); chosen = []
    for j in range(5):
        tr = part != j; te = part == j; best = None
        for i, s in enumerate(scq):
            sc = S.bal(s, yq, nq, tr)
            if best is None or sc > best[0] + 1e-12: best = (sc, i)
        Pout[te] = Plist[best[1]][te]; chosen.append(dict(part=j, combo_index=int(best[1]), train_score=float(best[0])))
    return Pout, chosen


def verdict(d, shuf_d=None):
    """PLAN 'judgement': 1 shuffle control lower bound > 0 -> leak, arm void; 2 delta >= +.01 and lo > 0 -> established; 3 lo > 0 -> small gain; 4 otherwise not adopted."""
    if shuf_d is not None and shuf_d["lo"] > 0: return "VOID_leak_suspected(shuffle lo>0)"
    if d["delta"] >= .01 and d["lo"] > 0: return "established(make test-version bank)"
    if d["lo"] > 0: return "small_gain(only if test-version bank is made for another arm)"
    return "not_adopted"


# ============================================================================================================================ fake bank for the dry runs
def question_queries(ctx, sl, own_val):
    """T4 query: mean of the own-side vectors of the question's voting games, normalised; zero rows where a question has no voting game."""
    m = sl["vote"][..., None]; s = (own_val[sl["idx"]] * m).sum(1); nrm = np.linalg.norm(s, axis=1); has = nrm > 0
    q = np.zeros_like(s); q[has] = s[has] / nrm[has, None]; return q.astype(np.float32), has


def make_fake_bank(kind, ctx, game, sl, real, rng):
    """Returns dict(own, opp, meta{owner_cls, owner_rank, hd, eff}, planted info).  Own side = head row + noise (a null world for T2: prototypes ~ head rows);
    opp side = random unit vectors.  'signal': (a) 30 % of the voting validation games: their OPPONENT-side vector + small noise goes into the bank opp side, owner rank = the TRUE rank of the questioned player,
    hd 0; (b) 30 % of the questions: 5 rows = the question's own-side mean vector + noise, owner rank = true rank, hd 0.  Labels are used here and only here (dry run)."""
    M = np.load(OUT / "bank_meta.npz"); n0 = 50_000; pick = np.sort(rng.choice(len(M["rid"]), n0, replace=False))
    oc = M["owner_cls"][pick].astype(np.int64); orank = M["owner_rank"][pick].astype(np.int64); hd = M["hd"][pick].astype(np.int64); eff = M["eff"][pick].astype(np.int64)
    W, cls_rank = real["W"], real["cls_rank"]; Wn = unit_rows(W); info = dict(n_base=n0)
    if kind == "signal":
        by_rank = [np.flatnonzero(cls_rank == k) for k in range(10)]; true_g = ctx["true_rank_g"]
        qg = np.unique(sl["idx"][sl["vote"]]); pg = qg[rng.random(len(qg)) < PLANT_FRAC]
        vecs3 = unit_rows(real["opp"][pg] + PLANT_NOISE * rng.normal(size=(len(pg), 128)) / np.sqrt(128)); rk3 = true_g[pg]
        qv, has = question_queries(ctx, sl, real["own"]); qi = np.flatnonzero(has & (rng.random(len(has)) < PLANT_FRAC)); rep = np.repeat(qi, T4_PLANT_ROWS)
        vecs4 = unit_rows(qv[rep] + PLANT_NOISE * rng.normal(size=(len(rep), 128)) / np.sqrt(128)); rk4 = ctx["yq"][rep]
        pv = np.concatenate([vecs3, vecs4]); prk = np.concatenate([rk3, rk4]).astype(np.int64)
        oc = np.concatenate([oc, np.array([rng.choice(by_rank[k]) for k in prk])]); orank = np.concatenate([orank, prk]); hd = np.concatenate([hd, np.zeros(len(prk), np.int64)]); eff = np.concatenate([eff, rng.integers(0, 2, len(prk))])
        info.update(n_planted_T3=int(len(pg)), n_planted_T4_questions=int(len(qi)), n_planted_T4_rows=int(len(rep)))
    n = len(oc); opp = unit_rows(rng.normal(size=(n, 128)))
    if kind == "signal": opp[n0:] = pv
    own = unit_rows(Wn[oc] + .5 * rng.normal(size=(n, 128)) / np.sqrt(128))
    return dict(own=own, opp=opp, meta=dict(owner_cls=oc, owner_rank=orank, hd=hd, eff=eff), info=info, tag="dry-" + kind)


# ============================================================================================================================ prototypes (T2)
def prototypes(own_keep, own_cls, Wn, extra_vecs=None, extra_cls=None):
    """Class mean (normalised) of the own-side bank vectors of that class; classes with no bank row keep the head row (count reported).  extra_*: opponent-side games merged into their top-1 class (T2b)."""
    C = len(Wn); P = np.zeros((C, 128)); cnt = np.bincount(own_cls, minlength=C).astype(np.float64)
    for d in range(128): P[:, d] = np.bincount(own_cls, weights=own_keep[:, d].astype(np.float64), minlength=C)
    if extra_vecs is not None:
        cnt += np.bincount(extra_cls, minlength=C)
        for d in range(128): P[:, d] += np.bincount(extra_cls, weights=extra_vecs[:, d].astype(np.float64), minlength=C)
    empty = cnt == 0; nrm = np.linalg.norm(P, axis=1); bad = empty | (nrm < 1e-9)
    Pn = np.where(bad[:, None], Wn.astype(np.float64), P / np.where(bad, 1., nrm)[:, None]); return Pn.astype(np.float32), dict(n_classes=int(C), n_fallback_to_head_row=int(bad.sum()), n_rows=int(cnt.sum()))


# ============================================================================================================================ the evaluation shared by the dry runs and the real run
def evaluate(mode, S, R, game, ctx, ro_opp, real, bank, cache_dir, rng_seed=1020):
    from r0_common import bal, NS, boot, diff, ci
    t00 = time.time(); ny = len(R["y"]); m4 = R["fold"] == 4; idx4, yq, nq, rid = ctx["idx4"], ctx["yq"], ctx["nq"], ctx["rid"]
    sl = slots(ctx, game); W, cls_rank = real["W"], real["cls_rank"]; Wn = unit_rows(W); C = len(cls_rank)
    meta = bank["meta"]; keep = np.flatnonzero(meta["hd"] <= 1); n_hd2 = int((meta["hd"] >= 2).sum()); M = len(keep)
    opp_k = np.ascontiguousarray(bank["opp"][keep]); own_k = np.ascontiguousarray(bank["own"][keep]); bank["opp"] = bank["own"] = None; oc_k = meta["owner_cls"][keep].astype(np.int64)
    T_bank = T_TAB[meta["hd"][keep].astype(int), meta["eff"][keep].astype(int), meta["owner_rank"][keep].astype(int)].astype(np.float32)      # [M,10]; hd >= 2 rows are never in the bank used here
    res = dict(bank_tag=bank["tag"], n_bank_rows_total=int(len(meta["hd"])), n_bank_rows_used_hd_le_1=int(M), n_bank_rows_dropped_hd_ge_2=n_hd2, bank_info=bank.get("info"), n_zero_t_rows_hd1=int((T_bank.sum(1) == 0).sum()))
    log("bank used rows", M, "dropped hd>=2", n_hd2)
    own_val, opp_val = real["own"], real["opp"]
    # ---- reference arm and score helper
    def full(p):
        o = np.full((ny, 10), .1); o[idx4] = p; return o
    ro_x = dict(pmax=ro_opp["pmax"], lift=S.shift_lift(ro_opp["lift"], game["eff"], game["hd"]), top1_rank=ro_opp["top1_rank"])
    V0, _ = S.question_votes(ctx["rows"], rid, ctx["hd"], ro_x, GM, LO, HI, (0, 1)); P_ref = S.fuse(ctx["f79q"], V0, LAM, AL)
    Sc = {"REF": R["sc"](full(P_ref))}; Pq = {"REF": P_ref}; extra = {}
    # ---- T1 (information only)
    thr95 = float(np.quantile(S.game_readout(own_val, W, cls_rank)["pmax"][game["hd"] == 0], .95)); res["thresholds_pmax_own_q95"] = thr95
    kT1 = f"{bank['tag']}|head|{sha_arr(opp_k[:2000])}|{M}"; ch = cache_load(cache_dir / "bank_opp_head.npz", kT1)
    if ch is None:
        t = time.time(); top, pm = head_top1(opp_k, Wn); cache_save(cache_dir / "bank_opp_head.npz", kT1, top1=top, pmax=pm); log("T1 head pass (opp) %.0fs" % (time.time() - t))
    else: top, pm = ch["top1"], ch["pmax"]
    _, pm_own = head_top1(own_k, Wn); val_own_rate = float((S.game_readout(own_val, W, cls_rank)["pmax"][np.isin(game["hd"], (0, 1))] >= thr95).mean())
    hi_opp = pm >= thr95; self_hit = top == oc_k; rec = hi_opp & ~self_hit
    own_cnt = np.bincount(oc_k, minlength=C); rec_cnt = np.bincount(top[rec], minlength=C)
    has_own = own_cnt > 0; edges = np.quantile(own_cnt[has_own], np.linspace(0, 1, 6)); bins = np.clip(np.searchsorted(edges, own_cnt, side="right") - 1, 0, 4); bin_rows = []
    for b in range(5):
        mk = has_own & (bins == b)
        bin_rows.append(dict(bin=b, own_file_games_range=[int(own_cnt[mk].min()), int(own_cnt[mk].max())] if mk.any() else None, n_players=int(mk.sum()), recognized_in_others_median=float(np.median(rec_cnt[mk])) if mk.any() else None,
                             recognized_in_others_sum=int(rec_cnt[mk].sum()), own_games_sum=int(own_cnt[mk].sum()), ratio_sum=float(rec_cnt[mk].sum() / max(1, own_cnt[mk].sum()))))
    share_opp = float(hi_opp.mean()); recall_in = float((pm_own >= thr95).mean()); pi_hat = share_opp - val_own_rate
    res["T1"] = dict(note="information only, not a claim.  threshold = e96 Q95 of own-side p_max (validation hd==0 games); counts use bank games hd<=1 only; recognised = opponent-side p_max >= threshold and top-1 class != file owner",
                     threshold_q95=thr95, share_bank_opp_pmax_ge_thr=share_opp, share_validation_own_side_ge_thr_unseen_baseline=val_own_rate, pi_hat_excess=pi_hat,
                     share_bank_opp_pmax_ge_0p8=float((pm >= .8).mean()), n_bank_opp_pmax_ge_0p8=int((pm >= .8).sum()), share_bank_opp_top1_is_file_owner=float(self_hit.mean()),
                     recall_in_sample_bank_own_side_ge_thr=recall_in, per_player_bins_by_own_file_games=bin_rows,
                     rough_estimate_share_of_opponents_in_library=dict(
                         value=float(pi_hat / recall_in) if recall_in > 0 else None,
                         assumptions="(1) an opponent outside the library behaves like an unseen player's own side (share >= thr = baseline); (2) the recognition rate of an in-library opponent equals the IN-SAMPLE own-side rate of bank games (optimistic: the net saw those games) -> the value is a LOWER bound; "
                                     "(3) p_max >= Q95 is a low bar (about .25), so some excess is style similarity, not identity -> the value may be too high.  The two biases pull in opposite directions; treat as order of magnitude."))
    log("T1 done: share_opp", share_opp, "baseline", val_own_rate, "pi_hat", pi_hat)
    # ---- T2 / T2b
    P2, st2 = prototypes(own_k, oc_k, Wn); ro2 = S.game_readout(opp_val, P2, cls_rank); ro2x = dict(pmax=ro2["pmax"], lift=S.shift_lift(ro2["lift"], game["eff"], game["hd"]), top1_rank=ro2["top1_rank"])
    hi8 = pm >= HI_PMAX_T2B; P2b, st2b = prototypes(own_k, oc_k, Wn, extra_vecs=opp_k[hi8], extra_cls=top[hi8].astype(np.int64))
    ro2b = S.game_readout(opp_val, P2b, cls_rank); ro2bx = dict(pmax=ro2b["pmax"], lift=S.shift_lift(ro2b["lift"], game["eff"], game["hd"]), top1_rank=ro2b["top1_rank"])
    for nm, rx in (("T2", ro2x), ("T2b", ro2bx)):
        V, _ = S.question_votes(ctx["rows"], rid, ctx["hd"], rx, GM, LO, HI, (0, 1)); Pq[nm] = S.fuse(ctx["f79q"], V, LAM, AL); Sc[nm] = R["sc"](full(Pq[nm]))
    res["T2_info"] = dict(T2=st2, T2b=dict(st2b, n_opp_side_games_merged=int(hi8.sum())), top1_rank_changes_vs_head_share=float((ro2["top1_rank"] != ro_opp["top1_rank"]).mean()))
    log("T2 / T2b done")
    # ---- T3: neighbours of each queried validation game's opponent-side vector in the bank opponent side
    qg = np.unique(sl["idx"][sl["vote"]]); Gn = len(game["hd"])
    k3 = f"{bank['tag']}|t3|{sha_arr(opp_val[qg])}|{sha_arr(opp_k[:2000])}|{M}|k{K_NN3}"; c3 = cache_load(cache_dir / "t3_game_top50.npz", k3)
    if c3 is None:
        t = time.time(); nn_i, nn_c = knn(opp_val[qg], opp_k, K_NN3); dt = time.time() - t; cache_save(cache_dir / "t3_game_top50.npz", k3, nn_row=nn_i, nn_cos=nn_c); log("T3 kNN %d queries x %d rows: %.0fs (%.1f ms/query)" % (len(qg), M, dt, 1e3 * dt / len(qg)))
        res["T3_knn_seconds"] = dt
    else: nn_i, nn_c = c3["nn_row"], c3["nn_cos"]; log("T3 kNN from cache")
    assert nn_i.shape == (len(qg), K_NN3) and nn_i.max() < M and (nn_c[:, :-1] >= nn_c[:, 1:] - 1e-6).all()
    # neighbour rows are indices into the filtered bank (row_map None): hd <= 1 by construction
    P3 = []; P3s = []
    rng = np.random.default_rng(rng_seed); perm = np.arange(len(yq))                                  # same-N question permutation (T3_shuf / T4_shuf)
    for n in NS:
        ii = np.flatnonzero(nq == n); perm[ii] = rng.permutation(ii)
    qpos = np.full(Gn, -1); qpos[qg] = np.arange(len(qg)); lib = ro_opp["lift"]
    nn_by_beta = {}
    for beta in (20., 40., 80.):
        g_nn = np.zeros((Gn, 10)); g_nn[qg] = nn_lift_games(nn_i, nn_c, T_bank, beta, len(qg)); nn_by_beta[beta] = g_nn[sl["idx"]] * sl["vote"][..., None]      # slot level [Q,S,10]
    for beta, w in T3_GRID:
        nn_slot = nn_by_beta[beta]
        V, _ = votes_slot(S, sl, lib, nn_slot, w); P3.append(S.fuse(ctx["f79q"], V, LAM, AL))
        V, _ = votes_slot(S, sl, lib, nn_slot[perm], w); P3s.append(S.fuse(ctx["f79q"], V, LAM, AL))
    Pq["T3"], ch3 = nested_pick(S, ctx, P3); Pq["T3_shuf"], ch3s = nested_pick(S, ctx, P3s)
    for nm in ("T3", "T3_shuf"): Sc[nm] = R["sc"](full(Pq[nm]))
    res["T3_chosen"] = dict(T3=[dict(c, beta=T3_GRID[c["combo_index"]][0], w=T3_GRID[c["combo_index"]][1]) for c in ch3], T3_shuf=[dict(c, beta=T3_GRID[c["combo_index"]][0], w=T3_GRID[c["combo_index"]][1]) for c in ch3s])
    res["T3_each_combo_balanced_score_all_fold4"] = {f"beta{b:g}_w{w:g}": float(S.bal(S.sc_q(P, yq), yq, nq)) for (b, w), P in zip(T3_GRID, P3)}
    # diagnostic (labels used, description only, nothing tuned): hd0 validation games vs hd0 neighbours
    true_g = ctx["true_rank_g"][qg]; hd_nb = np.where(game["hd"][qg] == 0, True, False); nb_hd0 = meta["hd"][keep][nn_i] == 0; nb_rank = meta["owner_rank"][keep][nn_i].astype(int)
    edges_c = np.quantile(nn_c, [0., .5, .9, .99, 1.]); sel = hd_nb[:, None] & nb_hd0; diag = []
    pt = np.bincount(true_g[hd_nb], minlength=10) / max(1, hd_nb.sum()); pq = np.bincount(nb_rank[sel], minlength=10) / max(1, sel.sum())
    for a, b in zip(edges_c[:-1], edges_c[1:]):
        mk = sel & (nn_c >= a) & (nn_c <= b if b == edges_c[-1] else nn_c < b); d = np.abs(nb_rank - true_g[:, None])
        diag.append(dict(cos_lo=float(a), cos_hi=float(b), n_pairs=int(mk.sum()), same_rank=float((d[mk] == 0).mean()) if mk.any() else None, one_rank_off=float((d[mk] == 1).mean()) if mk.any() else None, within_one=float((d[mk] <= 1).mean()) if mk.any() else None))
    res["T3_neighbour_diag"] = dict(note="LABELED DIAGNOSTIC, description only.  validation games with hd==0 and neighbours with hd==0: neighbour's owner rank vs the questioned player's true rank (hd 0: the opponent sits at the same rank).  bins = pooled neighbour cosine quantiles (0,.5,.9,.99,1)",
                                    chance_same_rank=float((pt * pq).sum()), cos_quantiles_top1={str(q): float(np.quantile(nn_c[:, 0], q)) for q in (.01, .05, .25, .5, .75, .95, .99)},
                                    cos_quantiles_all50={str(q): float(np.quantile(nn_c, q)) for q in (.01, .05, .25, .5, .75, .95, .99)}, neighbour_rank_distribution=(np.bincount(nb_rank.ravel(), minlength=10) / nb_rank.size).tolist(), bins=diag,
                                    n_queried_games=int(len(qg)), share_neighbours_hd1=float((~nb_hd0).mean()))
    log("T3 done")
    # ---- T4: reverse vote (question mean own-side vector against bank opponent side)
    qv, qhas = question_queries(ctx, sl, own_val); qi = np.flatnonzero(qhas)
    k4 = f"{bank['tag']}|t4|{sha_arr(qv[qi])}|{sha_arr(opp_k[:2000])}|{M}|k100"; c4 = cache_load(cache_dir / "t4_question_top100.npz", k4)
    if c4 is None:
        t = time.time(); n4_i, n4_c = knn(qv[qi], opp_k, 100); dt = time.time() - t; cache_save(cache_dir / "t4_question_top100.npz", k4, nn_row=n4_i, nn_cos=n4_c); log("T4 kNN %d queries: %.0fs" % (len(qi), dt)); res["T4_knn_seconds"] = dt
    else: n4_i, n4_c = c4["nn_row"], c4["nn_cos"]; log("T4 kNN from cache")
    V4 = {}
    for K_, beta, lam4 in T4_GRID:
        if (K_, beta) not in V4:
            v = np.full((len(yq), 10), .1); vv = nn_lift_games(n4_i[:, :K_], n4_c[:, :K_], T_bank, beta, len(qi)); ok = vv.sum(1) > 0; v[qi[ok]] = vv[ok]; V4[(K_, beta)] = v
    P4 = [fuse2(ctx["f79q"], V0, LAM, AL, V4[(K_, b)], l4) for K_, b, l4 in T4_GRID]; P4s = [fuse2(ctx["f79q"], V0, LAM, AL, V4[(K_, b)][perm], l4) for K_, b, l4 in T4_GRID]
    Pq["T4"], ch4 = nested_pick(S, ctx, P4); Pq["T4_shuf"], ch4s = nested_pick(S, ctx, P4s)
    for nm in ("T4", "T4_shuf"): Sc[nm] = R["sc"](full(Pq[nm]))
    lab = lambda c: dict(c, K=T4_GRID[c["combo_index"]][0], beta=T4_GRID[c["combo_index"]][1], lam4=T4_GRID[c["combo_index"]][2])
    res["T4_chosen"] = dict(T4=[lab(c) for c in ch4], T4_shuf=[lab(c) for c in ch4s]); res["T4_n_questions_with_query"] = int(len(qi))
    res["T4_cos_quantiles_top1"] = {str(q): float(np.quantile(n4_c[:, 0], q)) for q in (.05, .5, .95)}
    log("T4 done")
    # ---- bootstrap (player-level, paired, all arms share one resample)
    r = boot(Sc, R["F"], m4, "balanced", "e102_rank_vector_bank"); res["n_bootstrap"] = NB_NEEDED
    res["scores"] = {k: dict(score=r[k][0], per_N={str(n): v for n, v in r[k][1].items()}, **ci(r[k][2])) for k in r}
    part_full = np.full(ny, -1); part_full[idx4] = ctx["part"]
    pairs = {nm: nm for nm in ("T2", "T2b", "T3", "T4", "T3_shuf", "T4_shuf")}; res["delta_vs_REF"] = {}
    for nm in pairs:
        d = diff(r, nm, "REF"); d["per_N"] = {str(n): float(r[nm][1][n] - r["REF"][1][n]) for n in r[nm][1]}
        d["per_part"] = [float(bal(Sc[nm], R["y"], R["ns"], part_full == j) - bal(Sc["REF"], R["y"], R["ns"], part_full == j)) for j in range(5)]; res["delta_vs_REF"][nm] = d
    res["delta_pairs_info"] = {"T3_minus_T3_shuf": diff(r, "T3", "T3_shuf"), "T4_minus_T4_shuf": diff(r, "T4", "T4_shuf")}
    dv = res["delta_vs_REF"]
    res["verdict"] = dict(T2=verdict(dv["T2"]), T2b=verdict(dv["T2b"]), T3=verdict(dv["T3"], dv["T3_shuf"]), T4=verdict(dv["T4"], dv["T4_shuf"]),
                          note="T2 / T2b have no shuffle control (no added vote exists to shuffle); T3_shuf / T4_shuf as defined in the task.  S2 start condition (PLAN): T3 delta >= +.005 -> %s" % (dv["T3"]["delta"] >= .005))
    res["wall_seconds"] = time.time() - t00; res["peak_rss_gb_so_far"] = rss_gb()
    return res, Sc, Pq


# ============================================================================================================================ real bank
def load_real_bank(S, ctx, real):
    h_sha = str(np.load(S.HEAD, allow_pickle=True)["ckpt_sha16"]); M = np.load(OUT / "bank_meta.npz"); out = {}
    assert len(M["rid"]) == N_BANK
    for side in ("own", "opp"):
        f = OUT / f"S_bank_{side}_v4.npz"; fj = Path(str(f) + ".json")
        if not f.exists() or not fj.exists(): raise SystemExit(f"STOP: {f.name} or its .json is missing - the vector bank export is not finished")
        sj = json.loads(fj.read_text()); assert sj["ckpt_sha16"] == h_sha, f"{f.name}: checkpoint sha {sj['ckpt_sha16']} != head {h_sha}"
        z = np.load(f); rid = z["rid"]
        if len(rid) != N_BANK: raise SystemExit(f"STOP: {f.name} has {len(rid)} rows, expected {N_BANK}")
        if not (rid.astype(np.int64) == M["rid"].astype(np.int64)).all(): raise SystemExit(f"STOP: rid of {f.name} != bank_meta rid")
        gv = z["gv_all"]; assert gv.shape == (N_BANK, 128) and gv.dtype == np.float32
        sub = np.random.default_rng(0).choice(N_BANK, min(20000, N_BANK), replace=False); assert np.isfinite(gv[sub]).all() and np.abs(np.linalg.norm(gv[sub], axis=1) - 1).max() < 1e-3
        out[side] = gv; out[side + "_sha16"] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    f4 = json.loads(S.FOLD4.read_text()); h = np.load(S.HEAD, allow_pickle=True); assert not (set(h["cls_pid"].astype(str).tolist()) & set(f4["pid"]))
    assert int(M["owner_cls"].max()) < len(h["cls_rank"]) and (h["cls_rank"][M["owner_cls"]] == M["owner_rank"]).all(), "bank_meta owner_cls/owner_rank do not match the head table"
    meta = {k: M[k] for k in ("owner_cls", "owner_rank", "hd", "eff")}
    return dict(own=out["own"], opp=out["opp"], meta=meta, tag="real-" + out["opp_sha16"] + "-" + out["own_sha16"], info=dict(S_bank_own_sha16=out["own_sha16"], S_bank_opp_sha16=out["opp_sha16"], head_ckpt_sha16=h_sha))


# ============================================================================================================================ modes
def unit_tests():
    unit_T(); log("table t direction checks OK")
    rng = np.random.default_rng(0); B = unit_rows(rng.normal(size=(3000, 128))); Q = unit_rows(rng.normal(size=(37, 128)))
    i, c = knn(Q, B, 50, qbatch=16, bblock=700); ref = np.argsort(-(Q @ B.T), 1)[:, :50]; assert (np.sort(i, 1) == np.sort(ref, 1)).all() and np.allclose(c, np.take_along_axis(Q @ B.T, i, 1), atol=1e-6)
    j, _ = knn(Q, B, 50, row_map=np.arange(len(B))[::-1].copy()); assert (j == (len(B) - 1 - i)).all(); log("kNN == brute force OK")
    Wn = unit_rows(rng.normal(size=(40, 128))); own = Wn[np.arange(200) % 40]; P, st = prototypes(own, np.arange(200) % 40, Wn); assert np.abs(P - Wn).max() < 1e-6 and st["n_fallback_to_head_row"] == 0
    P, st = prototypes(own[:100], np.arange(100) % 40, Wn); assert st["n_fallback_to_head_row"] == 0
    P, st = prototypes(own[:30], np.arange(30) % 40, Wn); assert st["n_fallback_to_head_row"] == 10; log("prototype code OK")
    top, pm = head_top1(Wn, Wn); assert (top == np.arange(40)).all(); import e96_score as S; st = S.head_stats(Wn, Wn, (np.arange(40) % 10)); assert np.allclose(st["pmax"], pm, atol=1e-4) and (st["top20_cls"][:, 0] == top).all(); log("head pass == e96 head_stats OK")
    g = np.zeros((2, 10)); g[:, 6] = 1.; nn = np.zeros((2, 10)); nn[:, 5] = 1.                       # shape check of the vote combination (w = 0 -> lib only)
    x = g + 1. * nn; x /= x.sum(1, keepdims=True); assert np.allclose(x[0][[5, 6]], .5)
    # the label never enters the vote path: votes_slot / nn_lift_games / knn take no label argument (checked by signature)
    import inspect
    for fn in (votes_slot, nn_lift_games, knn, question_queries, prototypes, head_top1): assert not any(p in ("y", "yq", "truth", "labels", "true_rank_g") for p in inspect.signature(fn).parameters)
    log("UNIT OK")


def setup():
    OUT.mkdir(parents=True, exist_ok=True); global S
    S, R, game, ctx, ro_opp, real = E1.load_everything(); return R, game, ctx, ro_opp, real


def do_repro(R, game, ctx, ro_opp, real, write=True):
    sha0 = hashlib.sha256((ROOT / ".scratch/e96/result_v4.json").read_bytes()).hexdigest()[:16]
    chk, Sc, msub, q_hd1, ro_x = E1.reference_check(S, R, game, ctx, ro_opp); chk["result_v4_sha16"] = sha0
    log("REPRO F_opp_x vs posthoc: all < 1e-9 =", chk["all_within_1e9"], "max absdiff", chk["max_absdiff"])
    # slot-level vote path (used by T3) with w = 0 must equal the reference votes
    sl = slots(ctx, game); Vs, _ = votes_slot(S, sl, ro_opp["lift"]); V0, _ = S.question_votes(ctx["rows"], ctx["rid"], ctx["hd"], ro_x, GM, LO, HI, (0, 1))
    P_a = S.fuse(ctx["f79q"], Vs, LAM, AL); P_b = S.fuse(ctx["f79q"], V0, LAM, AL); chk["slot_path_w0_max_abs_diff_P"] = float(np.abs(P_a - P_b).max()); chk["slot_path_ok"] = bool(chk["slot_path_w0_max_abs_diff_P"] < 1e-9)
    log("slot-level path (w=0) max |P diff|", chk["slot_path_w0_max_abs_diff_P"])
    if write:
        fn = OUT / "repro_check.json"; fn.write_text(json.dumps(chk, indent=1, default=float))
    return chk


def dry(kind):
    R, game, ctx, ro_opp, real = setup(); rng = np.random.default_rng(1020 if kind == "noise" else 1021); sl = slots(ctx, game)
    bank = make_fake_bank(kind, ctx, game, sl, real, rng); cdir = NNC / f"dry_{kind}"
    res, Sc, Pq = evaluate("dry-" + kind, S, R, game, ctx, ro_opp, real, bank, cdir)
    # reference arm must equal the posthoc fixed-setting number
    ref = json.loads((ROOT / ".scratch/e96/posthoc_x_fixed.json").read_text())["scores"]["F_opp_x"]["score"]; res["REF_vs_posthoc_absdiff"] = abs(res["scores"]["REF"]["score"] - ref)
    d = res["delta_vs_REF"]; cond = {}
    if kind == "noise":
        for nm in ("T2", "T2b", "T3", "T4"): cond[f"{nm}_delta_within_pm_{NOISE_TOL}"] = bool(abs(d[nm]["delta"]) <= NOISE_TOL)
    else:
        for nm in ("T3", "T4"):
            cond[f"{nm}_delta_ge_{SIGNAL_MIN}_and_lo_gt_0"] = bool(d[nm]["delta"] >= SIGNAL_MIN and d[nm]["lo"] > 0); cond[f"{nm}_shuf_lo_le_0"] = bool(d[nm + "_shuf"]["lo"] <= 0)
        for nm in ("T2", "T2b"): cond[f"{nm}_info_delta"] = d[nm]["delta"]
    res["dry_run_conditions"] = cond; res["dry_run_pass"] = bool(all(v for k, v in cond.items() if isinstance(v, bool))) and res["REF_vs_posthoc_absdiff"] < 1e-9; res["mode"] = "dry-" + kind
    (OUT / f"dry_{kind}.json").write_text(json.dumps(res, indent=1, default=float))
    print("DRY", kind, "REF", round(res["scores"]["REF"]["score"], 5), "REF_vs_posthoc", res["REF_vs_posthoc_absdiff"]); print("delta vs REF", {k: (round(v["delta"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in d.items()})
    print("chosen T3", [(c["beta"], c["w"]) for c in res["T3_chosen"]["T3"]], "T4", [(c["K"], c["beta"], c["lam4"]) for c in res["T4_chosen"]["T4"]])
    print("conditions", cond, "PASS" if res["dry_run_pass"] else "FAIL"); print("peak rss GB", rss_gb())


def bench():
    rng = np.random.default_rng(1); B = unit_rows(rng.normal(size=(N_BANK, 128))); Q = unit_rows(rng.normal(size=(400, 128))); log("bank", B.nbytes / 1e6, "MB; rss", rss_gb())
    t = time.time(); knn(Q, B, 50); dt = time.time() - t; per = dt / 400; log("knn k=50: 400 queries %.1fs -> %.1f ms/query; 41,589 queries ~ %.1f min; 4,568 (T4) ~ %.1f min" % (dt, 1e3 * per, 41589 * per / 60, 4568 * per / 60))
    Wn = unit_rows(rng.normal(size=(6039, 128))); t = time.time(); head_top1(B[:100000], Wn); dt = time.time() - t; log("head pass 100k rows %.1fs -> full bank one side ~ %.1f min" % (dt, dt * N_BANK / 1e5 / 60)); log("peak rss GB", rss_gb())


def check_bank_files_only():
    for side in ("own", "opp"):
        for suf in ("", ".json"):
            f = OUT / f"S_bank_{side}_v4.npz{suf}"
            if not f.exists(): raise SystemExit(f"STOP: {f} does not exist - the vector bank export is not finished; nothing computed")


def rehearse():
    """Formal code path on a 300-row REAL-FORMAT bank (the CPU self-test vectors in .scratch/e102/selftest + the first 300 rows of bank_meta), in .scratch/e102/rehearsal.  Never touches the real bank files or result.json."""
    global OUT, NNC, N_BANK
    import shutil; real_out = OUT; R_ = real_out / "rehearsal"; R_.mkdir(parents=True, exist_ok=True); m = np.load(real_out / "bank_meta.npz"); n = 300
    np.savez(R_ / "bank_meta.npz", **{k: m[k][:n] for k in m.files})
    for side in ("own", "opp"):
        for suf in ("", ".json"): shutil.copy(real_out / "selftest" / f"S_{side}_lim300.npz{suf}", R_ / f"S_bank_{side}_v4.npz{suf}")
    for f in (R_ / "result.json",): f.unlink(missing_ok=True)
    shutil.rmtree(R_ / "nn_cache", ignore_errors=True); OUT = R_; NNC = R_ / "nn_cache"; N_BANK = n; run_real()


def rehearse_full():
    """Memory / time rehearsal of the formal path at FULL size: 857,634-row bank files with RANDOM unit vectors (real bank_meta, real head checkpoint sha) written to .scratch/e102/rehearsal_full (~0.9 GB, delete afterwards).  Numbers meaningless."""
    global OUT, NNC
    real_out = OUT; R_ = real_out / "rehearsal_full"; R_.mkdir(parents=True, exist_ok=True); import shutil; shutil.copy(real_out / "bank_meta.npz", R_ / "bank_meta.npz")
    sha = str(np.load(ROOT / ".scratch/e96/head_v4.npz", allow_pickle=True)["ckpt_sha16"]); rid = np.load(real_out / "bank_meta.npz")["rid"]; rng = np.random.default_rng(7)
    for side in ("own", "opp"):
        gv = np.empty((N_BANK, 128), np.float32)
        for a in range(0, N_BANK, 100000): gv[a:a + 100000] = unit_rows(rng.normal(size=(min(100000, N_BANK - a), 128)))
        np.savez(R_ / f"S_bank_{side}_v4.npz", gv_all=gv, ids=np.arange(N_BANK), rid=rid); (R_ / f"S_bank_{side}_v4.npz.json").write_text(json.dumps(dict(ckpt_sha16=sha, FAKE_RANDOM=True))); del gv
    (R_ / "result.json").unlink(missing_ok=True); shutil.rmtree(R_ / "nn_cache", ignore_errors=True); OUT = R_; NNC = R_ / "nn_cache"; run_real()


def run_real():
    fn = OUT / "result.json"
    if fn.exists(): raise SystemExit("STOP: result.json exists - the formal scoring runs ONCE")
    check_bank_files_only()
    R, game, ctx, ro_opp, real = setup(); bank = load_real_bank(S, ctx, real); log("bank files verified; rss GB", rss_gb())
    chk = do_repro(R, game, ctx, ro_opp, real, write=False)
    if not (chk["all_within_1e9"] and chk["slot_path_ok"]): raise SystemExit("STOP: reference arm / slot path does not reproduce - no arm scored")
    res, Sc, Pq = evaluate("run", S, R, game, ctx, ro_opp, real, bank, NNC / "real")
    res.update(mode="run", plan="experiments/E102_rank_vector_bank/PLAN.md", plan_sha16=hashlib.sha256(PLAN.read_bytes()).hexdigest()[:16], script_sha16=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
               repro_check=dict(all_within_1e9=chk["all_within_1e9"], max_absdiff=chk["max_absdiff"], slot_path_max_abs_diff_P=chk["slot_path_w0_max_abs_diff_P"]), n_questions_fold4=int(len(ctx["yq"])),
               setting_fixed=dict(gamma=GM, tau_lo=LO, alpha=AL, lam=LAM, alpha4=AL4, T3_grid=T3_GRID, T4_grid=T4_GRID, K_nn3=K_NN3))
    assert hashlib.sha256((ROOT / ".scratch/e96/result_v4.json").read_bytes()).hexdigest()[:16] == chk["result_v4_sha16"]
    with open(fn, "x") as f: json.dump(res, f, indent=1, default=float)
    print("RESULT written", fn); print("scores", {k: round(v["score"], 4) for k, v in res["scores"].items()}); print("delta vs REF", {k: (round(v["delta"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in res["delta_vs_REF"].items()})
    print("verdict", json.dumps(res["verdict"])); print("peak rss GB", rss_gb())


if __name__ == "__main__":
    mode = sys.argv[1]; S = None
    if mode == "unit": unit_tests()
    elif mode == "bench": bench()
    elif mode == "repro": R, game, ctx, ro_opp, real = setup(); do_repro(R, game, ctx, ro_opp, real)
    elif mode in ("dry-noise", "dry-signal"): dry(mode[4:])
    elif mode == "run": run_real()
    elif mode == "rehearse": rehearse()
    elif mode == "rehearse-full": rehearse_full()
    else: raise SystemExit("modes: unit repro dry-noise dry-signal bench rehearse rehearse-full run")
    log("peak rss GB", rss_gb())
