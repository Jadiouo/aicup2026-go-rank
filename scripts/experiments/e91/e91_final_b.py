"""E91 v10b: the v10a opponent-vote fusion with the library enlarged to the E90 `finh_full` network (994,875 games, 7,365 players incl. the 640 H players; 10x128).
NEW file: it imports (read-only) e91_final.py / e91_score.py and changes nothing in them.  NO score is ever computed on the test side (no labels), and no H player is used for any evaluation:
H players are only classes of the LIBRARY (their ranks come from the training CSVs, see e91_embed_b.py).  The fusion setting (gamma, tau, alpha, lam) is READ from .scratch/e91/final_params.json
(chosen once for v10a on the 4,568 validation questions; not re-selected here: the finh_full library contains the fold-4 validation players, so it cannot be re-fitted).

  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/experiments/e91/e91_final_b.py make [--name rank_v10b_對手票全量庫]
        inputs : .scratch/e91/b/head_b.npz, S_test_own_b.npz, S_test_opp_b.npz   (made by .scratch/e91/b/gpu_job_test_b.sh)
        outputs: data/submissions/<name>.csv  (+ check_submission) and .scratch/e91/b/final_stats_b.json ;  refuses to overwrite either.
  ... make --dry            : MECHANICS dry run.  head = b/dry/head_h0_dryrun.npz (finh_h0, 3,992 classes), vectors = the real finh_h0 vectors of the first K test games (b/dry/S_test_*_h0_dryrun.npz) +
                              FAKE random vectors for the other games (4% of the opponent games planted as a library player).  Writes ONLY .scratch/e91/b/dry/ (dryrun_*), never data/submissions.
  ... make --legacy-A       : CONSISTENCY check.  Feeds the ORIGINAL setting-A outputs (.scratch/e91/head.npz, S_test_{own,opp}.npz) through this very code and compares the answers row by row
                              with data/submissions/rank_v10a_對手票.csv (and pi_hat with pi_test.json).  Writes only .scratch/e91/b/dry/.  Exit code 1 if any row differs.
"""
import os, sys, json, argparse, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / ".scratch/e91"; B = OUT / "b"; DRY = B / "dry"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(ROOT / "scripts/experiments/e90"), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]
import numpy as np
import e91_final as F                                           # read-only reuse: readout, test_meta, thresholds, load_params, TEST_OWN ...

NET_DIR = ROOT / ".scratch/e90/colab/finh_full"; HEAD_B = B / "head_b.npz"; S_OWN_B = B / "S_test_own_b.npz"; S_OPP_B = B / "S_test_opp_b.npz"
HEAD_DRY = DRY / "head_h0_dryrun.npz"; S_OWN_DRY = DRY / "S_test_own_h0_dryrun.npz"; S_OPP_DRY = DRY / "S_test_opp_h0_dryrun.npz"
V10A = ROOT / "data/submissions/rank_v10a_對手票.csv"; V9 = ROOT / "data/submissions/rank_v9_逐局面注意力.csv"
NAME_DEFAULT = "rank_v10b_對手票全量庫"


def sha_file(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def sha16_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------------------------ vector sources
def load_head_file(p):
    h = np.load(p, allow_pickle=True); W = h["W"].astype(np.float32); cr = h["cls_rank"].astype(np.int64)
    assert W.ndim == 2 and W.shape[1] == 128 and len(cr) == len(W) and cr.min() >= 0 and cr.max() <= 9 and (np.bincount(cr, minlength=10) > 0).all(), "bad class table"
    return W, cr, h


def load_real(fo, fp, W):
    """own / opp gv of the 10,000 Rank test games (same checks as e91_final.vectors)."""
    so, sp = np.load(fo), np.load(fp); rid = np.load(F.TEST_OWN)["rid"].astype(np.int64)
    for s in (so, sp): assert (s["rid"].astype(np.int64) == rid).all() and len(s["gv_all"]) == 10000 and np.isfinite(s["gv_all"]).all() and np.abs(np.linalg.norm(s["gv_all"], axis=1) - 1).max() < 1e-3
    cos = (so["gv_all"] * sp["gv_all"]).sum(1); assert np.median(cos) < .9, "own and opponent vectors are (nearly) identical - the side flip did not take effect"
    return so["gv_all"], sp["gv_all"], dict(S_own=str(fo), S_opp=str(fp), cos_own_opp_median=float(np.median(cos)))


def get_source(mode):
    """-> own gv, opp gv, W, cls_rank, info.   mode: 'b' (real v10b) | 'dry' (finh_h0 on the first K games + FAKE rest) | 'A' (original setting A)."""
    if mode == "A":
        W, cr = F.load_head("A"); own, opp, _, _, info = F.vectors("A", False); return own, opp, W, cr, dict(info, source="setting A (E91 original)")
    if mode == "b":
        W, cr, h = load_head_file(HEAD_B); assert str(h["roster"]) == "finh_full" and len(W) == 7365 and int(h["steps"]) == 77740, (str(h["roster"]), len(W), int(h["steps"]))
        assert str(h["ckpt_sha16"]) == sha16_file(NET_DIR / "final_model.pt"), "head_b.npz does not belong to the current finh_full final_model.pt"
        for f in (S_OWN_B, S_OPP_B):
            j = json.loads(Path(str(f) + ".json").read_text()); assert j["ckpt_sha16"] == str(h["ckpt_sha16"]) and j["n_games"] == 10000, f"{f}: made by another checkpoint or not 10,000 games"
        own, opp, info = load_real(S_OWN_B, S_OPP_B, W); return own, opp, W, cr, dict(info, source="finh_full", ckpt_sha16=str(h["ckpt_sha16"]), n_classes=len(W), n_h_classes=int(h["n_h_classes"]))
    W, cr, h = load_head_file(HEAD_DRY); n = 10000; rng = np.random.default_rng(1234); unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    own = unit(rng.normal(size=(n, 128))).astype(np.float32); opp = unit(rng.normal(size=(n, 128))).astype(np.float32); Wn = unit(W)
    for g in np.flatnonzero(rng.random(n) < .04): opp[g] = unit((Wn[rng.integers(len(W))] + .5 * rng.normal(size=128) / np.sqrt(128))[None])[0]
    so, sp = np.load(S_OWN_DRY), np.load(S_OPP_DRY); K = len(so["rid"]); rid = np.load(F.TEST_OWN)["rid"].astype(np.int64)
    assert (so["rid"].astype(np.int64) == rid[:K]).all() and (sp["rid"].astype(np.int64) == rid[:K]).all() and np.isfinite(sp["gv_all"]).all()
    own[:K] = so["gv_all"]; opp[:K] = sp["gv_all"]
    return own, opp, W, cr, dict(source="DRYRUN: finh_h0 real vectors for the first %d games + FAKE random vectors for the rest" % K, FAKE_VECTORS_EXCEPT_FIRST_K=K, n_classes=len(W), ckpt_sha16=str(h["ckpt_sha16"]))


# ------------------------------------------------------------------------------------------------------------------------------ pi_hat (same algorithm as e91_final.pi_test)
def pi_hat(meta, ro_own, ro_opp, thr):
    nq = len(meta["qids"]); res = {}
    for s, nm in ((0, "public"), (1, "private")):
        m = (meta["gsplit"] == s) & (meta["hd"] == 0); q = meta["gq"][m]; t = thr[s][.95]
        a = np.bincount(q, weights=(ro_opp["pmax"][m] >= t), minlength=nq); b = np.bincount(q, weights=(ro_own["pmax"][m] >= t), minlength=nq); n = np.bincount(q, minlength=nq).astype(float)
        qs = np.flatnonzero(n > 0); a, b, n = a[qs], b[qs], n[qs]; pi = a.sum() / n.sum() - b.sum() / n.sum()
        C = np.random.default_rng(91).multinomial(len(qs), np.full(len(qs), 1 / len(qs)), size=2000).astype(np.float64); bs = (C @ a - C @ b) / (C @ n)
        res[nm] = dict(n_games_hd0=int(m.sum()), n_questions=int(len(qs)), thr_Q50_Q80_Q95_own=[thr[s][x] for x in (.5, .8, .95)], share_opp_high=float(a.sum() / n.sum()), share_own_high=float(b.sum() / n.sum()),
                       pi_hat_test=float(pi), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ make
def make(mode, name, dry_like):
    import pandas as pd
    import e91_score as E
    from gogame.constants import RANKS
    from gogame.models import decide_ev
    from gogame.submission import write_rank_submission
    from scripts.experiments.e37.predict_test import sha, K, DATA
    nrm = lambda p: p / p.sum(1, keepdims=True); P = F.load_params(); gm, tq, al, lam = P["gamma"], P["tau_q"], P["alpha"], P["lam"]; assert lam > 0
    tag = {"b": "", "dry": "_dryrun", "A": "_consistencyA"}[mode]; name = name or NAME_DEFAULT
    if mode == "b": out_csv = ROOT / "data/submissions" / f"{name}.csv"; stats_fn = B / "final_stats_b.json"
    else: DRY.mkdir(parents=True, exist_ok=True); out_csv = DRY / f"{'dryrun' if mode == 'dry' else 'consistencyA'}_{name}.csv"; stats_fn = DRY / f"final_stats_b{tag}.json"
    assert mode != "b" or not dry_like
    assert not out_csv.exists() and not stats_fn.exists(), f"{out_csv} / {stats_fn} exists - refuse to overwrite"
    B.mkdir(parents=True, exist_ok=True)
    meta = F.test_meta(); qids = meta["qids"].tolist(); nqn = meta["n"]; rows = meta["rows"]
    # ---- v9 test-side probabilities (same formula as e91_final.make / scripts/oneoff/e79_fuse.py) and the lam = 0 assert
    fw = json.loads((ROOT / "experiments/E79_rank_pos_attn/results/final.json").read_text())["final_weights_all_oof"]["F_E79"]; w, v, u = fw["w_A"], fw["v_M03"], fw["u_A1P"]; assert (w, v, u) == (.9, .5, .75)
    p03, p04 = np.load(ROOT / ".scratch/e66/final/test_m03.npy"), np.load(ROOT / ".scratch/e66/final/test_m04.npy"); s2, sp_ = np.load(ROOT / ".scratch/e66/final/set_A1T2.npy"), np.load(ROOT / ".scratch/e79/final/set_A1P.npy")
    assert p03.shape == p04.shape == (800, 10) and s2.shape == sp_.shape == (5, 3, 800, 10)
    Fp = nrm((1 - w) * nrm(v * p03 + (1 - v) * p04) + w * nrm((1 - u) * nrm(s2.mean((0, 1))) + u * nrm(sp_.mean((0, 1)))))
    v9 = pd.read_csv(V9); assert list(v9["question_id"]) == qids; r9 = np.array([RANKS.index(x) for x in v9["pred_rank"]])
    assert (decide_ev(Fp) == r9).all(), "decode of the rebuilt v9 test probabilities != rank_v9 csv: not the v9 probabilities - STOP"
    # ---- votes
    own, opp, W, cr, info = get_source(mode); ro_own, ro_opp = F.readout(own, W, cr), F.readout(opp, W, cr); thr = F.thresholds(ro_own, meta)
    V = np.full((800, 10), .1); has = np.zeros(800, bool); rid = np.arange(10000)
    for s in (0, 1):
        lo = 0. if tq is None else thr[s][tq]; qm = np.flatnonzero(meta["qsplit"] == s)
        Vs, hs = E.question_votes(rows[qm], rid, meta["hd"], ro_opp, gm, lo, 9.); V[qm] = Vs; has[qm] = hs
    assert (np.abs(V.sum(1) - 1) < 1e-9).all()
    F0 = E.fuse(Fp, V, 0., al); assert (decide_ev(F0) == r9).all(), "lam = 0 does not reproduce v9 answers question by question"
    print("ASSERT OK: lam=0 reproduces v9 (800/800 answers identical)", flush=True)
    Pf = E.fuse(Fp, V, lam, al); pred = decide_ev(Pf); assert pred.shape == (800,) and pred.min() >= 0 and pred.max() < K
    write_rank_submission(qids, pred, out_csv)
    import check_submission
    rc = check_submission.main([str(out_csv), "--task", "rank", "--data-dir", str(DATA)]); assert rc == 0, "check_submission failed"
    sp_q = meta["qsplit"]; same = pred == r9
    v10a = pd.read_csv(V10A); assert list(v10a["question_id"]) == qids; r10a = np.array([RANKS.index(x) for x in v10a["pred_rank"]]); same_a = pred == r10a
    pi = pi_hat(meta, ro_own, ro_opp, thr); lib_prior = (np.bincount(cr, minlength=10) / len(cr)).tolist()
    qv = lambda x: [float(t) for t in np.quantile(x, [.1, .5, .9, .99])]
    m0 = meta["hd"] == 0; vote_max = V[has].max(1)
    st = dict(mode=mode, DRYRUN=(mode == "dry"), consistency_check_A=(mode == "A"), file=str(out_csv), sha256=sha(out_csv), check_submission_rc=rc, params=dict(gamma=gm, tau_q=tq, alpha=al, lam=lam, read_from=str(OUT / "final_params.json")),
              vectors=info, library=dict(n_classes=int(len(W)), rank_counts=np.bincount(cr, minlength=10).tolist(), rank_prior=lib_prior), lam0_reproduces_v9="800/800", n_questions=800,
              same_answer_as_v9=float(same.mean()), n_changed_vs_v9=int((~same).sum()), same_answer_as_v10a=float(same_a.mean()), n_changed_vs_v10a=int((~same_a).sum()), frac_questions_with_vote=float(has.mean()),
              thresholds_own_pmax_by_split={"public": thr[0], "private": thr[1]}, pmax_quantiles_10_50_90_99=dict(opp_hd0=qv(ro_opp["pmax"][m0]), own_hd0=qv(ro_own["pmax"][m0])),
              vote_sharpness=dict(mean_max_vote_prob=float(vote_max.mean()), median_max_vote_prob=float(np.median(vote_max))),
              pred_share_v9=np.bincount(r9, minlength=K).tolist(), pred_share_v10a=np.bincount(r10a, minlength=K).tolist(), pred_share_new=np.bincount(pred, minlength=K).tolist(), mean_rank_shift_vs_v9=float((pred - r9).mean()),
              per_split={nm: dict(n=int((sp_q == s).sum()), same_as_v9=float(same[sp_q == s].mean()), n_changed_vs_v9=int((~same[sp_q == s]).sum()), same_as_v10a=float(same_a[sp_q == s].mean()), n_changed_vs_v10a=int((~same_a[sp_q == s]).sum()),
                                  frac_with_vote=float(has[sp_q == s].mean()), pred_share_v9=np.bincount(r9[sp_q == s], minlength=K).tolist(), pred_share_new=np.bincount(pred[sp_q == s], minlength=K).tolist()) for s, nm in ((0, "public"), (1, "private"))},
              per_N_same_as_v9={int(n): float(same[nqn == n].mean()) for n in (5, 10, 15, 20)}, pi_hat_test=pi, v9_sha256=sha(V9), v10a_sha256=sha(V10A))
    try: st["pi_hat_v10a_for_reference"] = {k: json.loads((OUT / "pi_test.json").read_text())[k] for k in ("public", "private")}; st["validation_gate0_setting_A"] = {k: v for k, v in json.loads((OUT / "result.json").read_text())["gate0"].items() if k in ("pi_hat", "lo", "hi")}
    except Exception as e: st["pi_hat_v10a_for_reference"] = str(e)
    if mode == "A":
        st["consistency"] = dict(rows_identical_to_v10a=int(same_a.sum()), of=800, ALL_IDENTICAL=bool(same_a.all()), csv_bytes_identical=bool(out_csv.read_bytes() == V10A.read_bytes()),
                                 sha256_equals_v10a=bool(sha(out_csv) == sha(V10A)), pi_hat_equals_pi_test_json=bool(all(abs(pi[k][f] - json.loads((OUT / "pi_test.json").read_text())[k][f]) < 1e-9 for k in ("public", "private") for f in ("pi_hat_test", "lo", "hi"))))
    stats_fn.write_text(json.dumps(st, indent=1, default=float)); print(json.dumps(st, indent=1, default=float))
    if mode == "A":
        ok = st["consistency"]["ALL_IDENTICAL"] and st["consistency"]["pi_hat_equals_pi_test_json"]; print("CONSISTENCY", "PASS" if ok else "FAIL", st["consistency"]); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["make"]); ap.add_argument("--name"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--legacy-A", action="store_true")
    a = ap.parse_args(); assert not (a.dry and a.legacy_A)
    make("dry" if a.dry else ("A" if a.legacy_A else "b"), a.name, a.dry or a.legacy_A)
