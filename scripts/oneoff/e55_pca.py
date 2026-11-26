"""E55 step 3a: assemble per-game trunk vectors (blk*_gv/_gid from e55_prep/e55_trunk) into G_full.npy (n_bundle x 768 fp16, NaN rows = games the encoder skipped) and G_test.npy (10,000 x 768);
fit per-fold PCA-48 (standardised; fit games = bundle games of players with player_fold != k, <=150k random games, seed 55+k) and the final PCA (all bundle games) via eigh of the covariance.
Writes .scratch/e55/{G_full,G_test}.npy, .scratch/e55/pca/{fold0..4,final}.npz (mu, sd, comps[48,768], evr, n_fit), and removes the block files after the assembly checks.
Usage: .venv/bin/python scripts/oneoff/e55_pca.py [--keep-blocks]"""
import sys, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; S = ROOT / ".scratch/e55"; NPC = 48
def project(G, pca, chunk=100_000):
    """float32 [n, NPC]; NaN rows (skipped games) -> 0 (= the fit-sample mean after standardisation)."""
    out = np.zeros((len(G), NPC), np.float32); mu, sd, C = pca["mu"], pca["sd"], pca["comps"]
    for a in range(0, len(G), chunk):
        z = (np.asarray(G[a:a + chunk], np.float32) - mu) / sd; bad = ~np.isfinite(z).all(1); z[bad] = 0; out[a:a + chunk] = z @ C.T
    return out
def fit_pca(G, rows, seed):
    rng = np.random.default_rng(seed); rows = np.sort(rng.permutation(rows)[:150_000]); X = np.asarray(G[rows], np.float32); X = X[np.isfinite(X).all(1)]
    mu = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - mu) / sd; cov = (Z.T @ Z) / (len(Z) - 1); w, V = np.linalg.eigh(cov.astype(np.float64)); o = np.argsort(w)[::-1][:NPC]
    return {"mu": mu, "sd": sd, "comps": V[:, o].T.astype(np.float32), "evr": (w[o] / w.sum()).astype(np.float64), "n_fit": np.array(len(Z))}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--keep-blocks", action="store_true"); a = ap.parse_args()
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); n = len(aux["uid"]); gp = aux["game_player"]; pf = aux["player_fold"]
    assert (S / "PREP_DONE").exists() and not list(S.glob("*blk*_bits.npy")), "prep/trunk not finished"
    blocks = sorted(S.glob("blk*_gv.npy"), key=lambda p: int(p.name[3:].split("_")[0]))
    G = np.lib.format.open_memmap(str(S / "G_full.tmp.npy"), mode="w+", dtype=np.float16, shape=(n, 768)); G[:] = np.nan; seen = np.zeros(n, bool)
    for p in blocks:
        gid = np.load(str(p).replace("_gv", "_gid")); gv = np.load(p); assert len(gid) == len(gv) and not seen[gid].any(); G[gid] = gv; seen[gid] = True
    G.flush(); del G; (S / "G_full.tmp.npy").rename(S / "G_full.npy")
    t = np.load(S / "tblk0_gv.npy"); tg = np.load(S / "tblk0_gid.npy"); GT = np.full((10000, 768), np.nan, np.float16); GT[tg] = t; np.save(S / "G_test.npy", GT)
    G = np.load(S / "G_full.npy", mmap_mode="r"); info = {"n_games": n, "n_with_vec": int(seen.sum()), "n_missing": int((~seen).sum()), "test_with_vec": int(len(tg)), "n_blocks": len(blocks)}
    print(info, flush=True); assert np.isfinite(np.asarray(G[:2000], np.float32)[seen[:2000]]).all()
    (S / "pca").mkdir(exist_ok=True); evr = {}
    for k in list(range(5)) + ["final"]:
        rows = np.flatnonzero(seen & ((pf[gp] != k) if k != "final" else np.ones(n, bool))); P = fit_pca(G, rows, 55 + (k if k != "final" else 9)); np.savez(S / "pca" / f"fold{k}.npz" if k != "final" else S / "pca/final.npz", **P)
        evr[str(k)] = [float(P["evr"][:48].sum()), int(P["n_fit"])]; print("pca", k, evr[str(k)], flush=True)
    info["evr48_nfit"] = evr; (S / "assemble_info.json").write_text(json.dumps(info, indent=1))
    if not a.keep_blocks:
        for p in list(S.glob("*blk*_gv.npy")) + list(S.glob("*blk*_gid.npy")) + list(S.glob("*blk*_glob.npy")) + list(S.glob("*blk*_loc.npy")) + list(S.glob("*blk*_rows.npy")) + list(S.glob("*blk*_color.npy")): p.unlink()
