"""E66 test side, D120 legs: M03T2 (E63: M03 + 7 SE + tk48 + lk48 bag aggregates) and M04T (E57 C1: CORAL on 522 HSL aggregate + 240 tk-PCA aggregate = 762 cols), both trained once on all non-H players through the v6 final pipeline
(scripts/experiments/e37/dse_final.py -> worktree e21-se-pipeline HslLgbmRank.fit(method="ens")).  Differences to e63_final.py: P.fit_coral is wrapped so the M04 matrix gets the tk bag aggregate (final PCA tk, 5 stats, bag_n/is_white dropped) to the right of the HSL block, on the SAME training bags
(captured from the se_feats call that precedes fit_coral in fit()).  Fixed M03 rounds = median of the 5 E63 OOF best iterations (63).
  train   --out DIR [--threads 4]            -> DIR/rank_hsl_lgbm_se.pkl (bundle: m03 = M03T2, m04 = M04T), DIR/train_info.json
  predict --model-dir DIR --out-prefix PFX   -> PFX_m03.npy, PFX_m04.npy (800,10 each; M03T2 and M04T separately, not averaged)
Never touches H labels (non-H players only) / h_eval_ledger."""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]; S = ROOT / ".scratch/e55"; S63 = ROOT / ".scratch/e63"; sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import scripts.experiments.e37.dse_final as DF
PN = tuple(f"tk{j:02d}" for j in range(48)) + tuple(f"lk{j:02d}" for j in range(48))
def project(G, pca, chunk=100_000):
    out = np.zeros((len(G), 48), np.float32); mu, sd, C = pca["mu"], pca["sd"], pca["comps"]
    for a in range(0, len(G), chunk):
        z = (np.asarray(G[a:a + chunk], np.float32) - mu) / sd; bad = ~np.isfinite(z).all(1); z[bad] = 0; out[a:a + chunk] = z @ C.T
    return out
CAP = {}
def patch():
    P = DF._import_predict(); orig = P.se_feats; orig_fc = P.fit_coral
    from gogame.aggregate import L0Store, aggregate_many, make_spec
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); uid = aux["uid"].astype(str).tolist(); tu = [str(u) for u in np.load(ROOT / ".scratch/e29/test/test_bags.npz", allow_pickle=True)["uids"]]
    pca = dict(np.load(S / "pca/final.npz")); dense = np.vstack([project(np.load(S / "G_full.npy", mmap_mode="r"), pca), project(np.load(S / "G_test.npy"), pca)]); pl = dict(np.load(S63 / "pca/final.npz"))
    late = np.vstack([project(np.load(S63 / "L_full.npy", mmap_mode="r"), pl), project(np.load(S63 / "L_test.npy"), pl)])
    store_tk = L0Store(uid + tu, dense, None, PN[:48]); spec_tk = make_spec(PN[:48], np.zeros(0, np.int32))                    # M04T: tk only (E57 C1)
    store = L0Store(uid + tu, np.hstack([dense, late]), None, PN); spec = make_spec(PN, np.zeros(0, np.int32))                  # M03T2: tk + lk (E63)
    def tk_agg(bags): return aggregate_many(bags, store_tk, spec_tk)[:, :-2]
    def se_feats_T(bags, sm, ss): CAP["bags"] = bags; return np.hstack([orig(bags, sm, ss), aggregate_many(bags, store, spec)[:, :-2]])
    def fit_coral_T(Xh, y, w, *a, **k):
        bags = CAP["bags"]; assert len(bags) == len(Xh); Xt = np.hstack([Xh, tk_agg(bags)]); assert Xt.shape[1] == Xh.shape[1] + 240 == 762, Xt.shape; print("M04T matrix", Xt.shape, flush=True); return orig_fc(Xt, y, w, *a, **k)
    P.se_feats = se_feats_T; P.fit_coral = fit_coral_T; return P, tk_agg
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--out", required=True); t.add_argument("--threads", type=int, default=4)
    q = sp.add_parser("predict"); q.add_argument("--model-dir", required=True); q.add_argument("--out-prefix", required=True); q.add_argument("--hsl", default=str(ROOT / ".scratch/e37/v6final/test/hsl_test120_games.parquet"))
    a = ap.parse_args(); P, tk_agg = patch()
    if a.cmd == "train":
        DF.OOF_W120 = S63 / "dse_out/M03TL"; DF.cmd_train(SimpleNamespace(mode="w120", out=a.out, threads=a.threads))
    else:
        import os; os.environ["GOGAME_COLOR_SEMANTICS"] = "actual"
        TQ = P.load_questions(DF.DATA / P.FILES["rank_pub"], "pub", "rank") + P.load_questions(DF.DATA / P.FILES["rank_priv"], "priv", "rank")
        m = P.HslEnsembleSeRank.load(Path(a.model_dir), n_jobs=2); assert m.meta.get("decode") == "ev" and m.meta.get("method") == "ens" and m.meta.get("features") == "l0+hsl+se", m.meta
        m.set_features(Path(a.hsl)); m.set_se_features(DF.SE_TEST); Xh, bags, have = m._hsl_features(TQ); X03 = m._m03_matrix(Xh, TQ, bags, have)
        p03 = np.asarray(m.bundle.m03.predict_proba(X03), np.float64); p04 = np.asarray(m.bundle.m04.predict_proba(np.hstack([Xh, tk_agg(bags)])), np.float64)
        assert p03.shape == p04.shape == (800, 10) and np.isfinite(p03).all() and np.isfinite(p04).all(), (X03.shape,)
        np.save(a.out_prefix + "_m03.npy", p03); np.save(a.out_prefix + "_m04.npy", p04); print("saved", a.out_prefix, "m03 cols", X03.shape[1], "m04 cols", Xh.shape[1] + 240)
