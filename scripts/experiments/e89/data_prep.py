"""E89 S line: rebuild the E87 step-0 split (verbatim logic, seed 87) and dump every split game as compact move arrays.

Output .scratch/e89/data/s_games.npz  (moves int16 concatenated, off int64[G+1], eff int8, hd int8, cls int16, role int8 0/1/2,
rid int32, player int32, pcount int16, plus library info) and s_games_stats.json.
Only library players (player_fold in 0..3, >= 8 bundle games) appear -> no H player, no fold-4 player (asserted below).
Games come from data/training_v1.1 (carries HD[n]; v1.0 SGFs differ only by that property, see scripts/oneoff/e76_hd.py).
  PYTHONPATH=src:. .venv/bin/python scripts/experiments/e89/data_prep.py
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "katago")]
import numpy as np, pandas as pd
import kgio

OUT = ROOT / ".scratch/e89/data"; OUT.mkdir(parents=True, exist_ok=True)
EXPECT = {"train": 459_916, "val": 56_772, "test": 56_772, "classes": 5_700}   # .scratch/e87/step0.log


def e87_split():
    """Verbatim from scripts/oneoff/e87_step0.py main(): returns aux arrays, library classes and role per bundle row (-1 not in library)."""
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=True)
    gp = aux["game_player"].astype(np.int64); gc = aux["game_color"].astype(np.int64); pf = aux["player_fold"]
    hd = np.load(ROOT / ".scratch/e76/HD_full.npy"); n_p = np.bincount(gp, minlength=len(pf))
    libp = np.flatnonzero((pf <= 3) & (n_p >= 8)); cls_of = np.full(len(pf), -1, np.int64); cls_of[libp] = np.arange(len(libp)); C = len(libp)
    lib_games = np.flatnonzero(cls_of[gp] >= 0)
    rng = np.random.default_rng(87); order = rng.permutation(lib_games); gpl = gp[order]; srt = np.argsort(gpl, kind="stable"); order = order[srt]; gpl = gpl[srt]
    start = np.r_[0, np.flatnonzero(np.diff(gpl)) + 1]; cnt = np.diff(np.r_[start, len(gpl)]); role = np.zeros(len(order), np.int8)
    for s, c in zip(start, cnt):
        g = order[s:s + c]; el = np.flatnonzero(hd[g] <= 1); k = max(1, int(round(.1 * c)))
        if len(el) < 2 * k: k = len(el) // 2
        if k >= 1: role[s + el[:k]] = 1; role[s + el[k:2 * k]] = 2
    rid = np.full(len(gp), -1, np.int8); rid[order] = role
    return aux, gp, gc, pf, hd, n_p, libp, cls_of, C, rid


def main():
    t0 = time.time()
    aux, gp, gc, pf, hd, n_p, libp, cls_of, C, rid = e87_split()
    uid = aux["uid"].astype(str)
    n_tr, n_va, n_te = (int((rid == x).sum()) for x in (0, 1, 2))
    print("split", {"train": n_tr, "val": n_va, "test": n_te, "classes": C}, "expected", EXPECT, flush=True)
    assert (n_tr, n_va, n_te, C) == (EXPECT["train"], EXPECT["val"], EXPECT["test"], EXPECT["classes"]), "SPLIT MISMATCH vs E87 log"
    # no H player, no fold-4 player among the games we keep
    keep = np.flatnonzero(rid >= 0)
    assert (pf[gp[keep]] <= 3).all() and (pf[gp[keep]] >= 0).all(), "fold-4 (or unknown fold) player in split"
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet"); u2p = dict(zip(meta.uid, meta.player_id))
    sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"])
    pid_keep = {u2p[u] for u in uid[keep]}; assert not (pid_keep & H), "H player in split"
    print("H check ok: split players", len(pid_keep), "H players", len(H), flush=True)
    row_of = pd.Series(np.arange(len(uid), dtype=np.int64), index=uid)
    want = np.zeros(len(uid), bool); want[keep] = True
    moves, lens, rows, effs, hds, skips = {}, {}, [], {}, {}, {}
    skip_cnt = {}; trunc = 0; n_seen = 0
    for f in sorted((ROOT / "data/training_v1.1/training").glob("train_*.csv")):
        for ch in pd.read_csv(f, usecols=["game_id", "rank", "color", "sgf_content"], chunksize=10_000):
            ch["uid"] = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str)
            ch = ch[ch["uid"].isin(row_of.index)]
            if len(ch) == 0: continue
            r_arr = row_of.reindex(ch["uid"]).to_numpy()
            for r, col, s in zip(r_arr, ch["color"], ch["sgf_content"]):
                r = int(r)
                if not want[r]: continue
                n_seen += 1
                assert (0 if col == "B" else 1) == int(gc[r]), "csv colour != aux colour"
                ng = kgio.normalize_game(s, int(gc[r]))
                if ng.skip: skip_cnt[ng.skip] = skip_cnt.get(ng.skip, 0) + 1; skips[r] = ng.skip; continue
                assert ng.hd == min(int(hd[r]), 9), (r, ng.hd, hd[r])
                trunc += int(ng.truncated)
                mv = np.array([(361 if rc is None else rc[0] * 19 + rc[1]) + (362 if c == "W" else 0) for c, rc in ng.moves], np.int16)
                own = int(((mv >= 362).astype(np.int8) == ng.eff_color).sum())
                if own == 0: skip_cnt["no_own_move"] = skip_cnt.get("no_own_move", 0) + 1; skips[r] = "no_own_move"; continue
                moves[r] = mv; effs[r] = ng.eff_color; hds[r] = ng.hd
        print(f.name, "seen", n_seen, "kept", len(moves), flush=True)
    assert n_seen == len(keep), (n_seen, len(keep))
    ok_rows = np.array(sorted(moves), dtype=np.int64)
    off = np.zeros(len(ok_rows) + 1, np.int64); off[1:] = np.cumsum([len(moves[r]) for r in ok_rows])
    mv_all = np.concatenate([moves[r] for r in ok_rows]).astype(np.int16)
    role = rid[ok_rows].astype(np.int8)
    stats = {"split_expected": EXPECT, "split_before_skip": {"train": n_tr, "val": n_va, "test": n_te, "classes": C},
             "skipped_by_reason": skip_cnt, "skipped_by_role": {str(k): int(sum(1 for r in skips if rid[r] == k)) for k in (0, 1, 2)},
             "truncated_at_illegal": trunc, "kept_by_role": {str(k): int((role == k).sum()) for k in (0, 1, 2)},
             "hd_counts_kept": np.bincount([hds[r] for r in ok_rows], minlength=10).tolist(), "n_moves_total": int(len(mv_all)),
             "mean_moves_per_game": float(len(mv_all) / len(ok_rows)), "players_in_split": len(pid_keep),
             "h_players_in_file": 0, "fold4_players_in_file": 0, "seconds": round(time.time() - t0)}
    np.savez(OUT / "s_games.npz", moves=mv_all, off=off, eff=np.array([effs[r] for r in ok_rows], np.int8), hd=np.array([hds[r] for r in ok_rows], np.int8),
             cls=cls_of[gp[ok_rows]].astype(np.int16), role=role, rid=ok_rows.astype(np.int32), player=gp[ok_rows].astype(np.int32),
             pcount=n_p[gp[ok_rows]].astype(np.int16), n_classes=np.int64(C))
    (OUT / "s_games_stats.json").write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
