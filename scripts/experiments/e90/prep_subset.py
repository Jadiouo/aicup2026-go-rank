"""E90 stage 0: S-network input for an arbitrary list of bundle rows (copy of the logic of scripts/experiments/e89/data_prep.py main(),
without the E87-split count asserts and without the library-player restriction).

Output arrays have the same structure as .scratch/e89/data/s_games.npz:
  moves int16 (concatenated), off int64[G+1], eff int8, hd int8, cls int16, role int8, rid int32 (bundle row), player int32 (aux game_player),
  pcount int16, n_classes int64.   Rows are written in ascending bundle-row order (as data_prep.py does).
cls/role come from the verbatim E87 split (e87_split below); rows outside the E87 library get cls = -1, role = -1.

  PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/experiments/e90/prep_subset.py a1      # check A1 (300 games of s_games.npz, seed 0)
  PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/experiments/e90/prep_subset.py target  # the 309 fold-4 eval players' games
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "katago")]
import numpy as np, pandas as pd
import kgio

OUT = ROOT / ".scratch/e90/stage0"; OUT.mkdir(parents=True, exist_ok=True)


def e87_split():
    """Verbatim from scripts/experiments/e89/data_prep.py e87_split() (= scripts/oneoff/e87_step0.py main())."""
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


def prep(rows, log=print):
    """rows: bundle row numbers (any order, unique).  Returns dict of arrays (see module doc) + dict of stats."""
    t0 = time.time()
    aux, gp, gc, pf, hd, n_p, libp, cls_of, C, rid = e87_split()
    rows = np.unique(np.asarray(rows, dtype=np.int64)); uid = aux["uid"].astype(str)
    row_of = pd.Series(np.arange(len(uid), dtype=np.int64), index=uid)
    want = np.zeros(len(uid), bool); want[rows] = True
    moves, effs, hds, skips, skip_cnt, trunc, n_seen = {}, {}, {}, {}, {}, 0, 0
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
        log(f.name, "seen", n_seen, "kept", len(moves), flush=True)
    assert n_seen == len(rows), (n_seen, len(rows))
    ok_rows = np.array(sorted(moves), dtype=np.int64)
    off = np.zeros(len(ok_rows) + 1, np.int64); off[1:] = np.cumsum([len(moves[r]) for r in ok_rows])
    mv_all = np.concatenate([moves[r] for r in ok_rows]).astype(np.int16)
    d = dict(moves=mv_all, off=off, eff=np.array([effs[r] for r in ok_rows], np.int8), hd=np.array([hds[r] for r in ok_rows], np.int8),
             cls=cls_of[gp[ok_rows]].astype(np.int16), role=rid[ok_rows].astype(np.int8), rid=ok_rows.astype(np.int32), player=gp[ok_rows].astype(np.int32),
             pcount=n_p[gp[ok_rows]].astype(np.int16), n_classes=np.int64(C))
    stats = dict(n_requested=int(len(rows)), n_kept=int(len(ok_rows)), skipped_by_reason=skip_cnt, truncated_at_illegal=int(trunc), seconds=round(time.time() - t0))
    return d, stats


def check_a1():
    S = np.load(ROOT / ".scratch/e89/data/s_games.npz"); rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(S["rid"]), 300, replace=False))
    d, stats = prep(S["rid"][pick].astype(np.int64))
    res = {}
    for k in S.files:
        if k in ("moves", "off"): continue
        res[k] = bool(np.array_equal(d[k], S[k][pick])) if k != "n_classes" else bool(d[k] == S[k])
    mv_ok = all(np.array_equal(d["moves"][d["off"][i]:d["off"][i + 1]], S["moves"][S["off"][g]:S["off"][g + 1]]) for i, g in enumerate(pick))
    res["moves"] = bool(mv_ok); res["off_lengths"] = bool(np.array_equal(np.diff(d["off"]), np.diff(S["off"])[pick]))
    out = dict(check="A1", n=300, seed=0, arrays=res, all_identical=all(res.values()), stats=stats, pick_index_first5=pick[:5].tolist())
    (OUT / "check_A1.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    assert out["all_identical"], "A1 FAILED"


def target_rows():
    """309 fold-4 sim-eval players' games among the games flagged sim_eval in .scratch/e80/meta.npz.  Returns (ev_rows in sim_eval order, ev_gid, info)."""
    m = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet")
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=True); z = np.load(ROOT / ".scratch/e80/meta.npz")
    pid = aux["player_ids"]; idx = {p: i for i, p in enumerate(pid)}; gp = aux["game_player"].astype(np.int64)
    ep = m.loc[m.grp == "eval", "player_id"].unique(); assert len(ep) == 1481
    epi = np.array([idx[p] for p in ep]); f4 = epi[aux["player_fold"][epi] == 4]; assert len(f4) == 309
    # no H player: aux contains only non-H players (7,367); H check via split parquet as in data_prep
    sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"])
    assert not (set(pid[f4]) & H), "H player among targets"
    rows, gid, se = z["sim_rows"], z["sim_gid"], z["sim_eval"]
    evrows = rows[se]; evgid = gid[se]
    assert (m.grp.values[evgid] == "eval").all() and (m.player_id.values[evgid] == pid[gp[evrows]]).all()
    sel = np.isin(gp[evrows], f4)
    S = np.load(ROOT / ".scratch/e89/data/s_games.npz")
    assert not np.isin(f4, np.unique(S["player"])).any(), "a target player appears in s_games.npz"
    assert not np.isin(evrows[sel], S["rid"]).any(), "a target game appears in s_games.npz"
    cnt = np.bincount(gp[evrows[sel]])[f4]
    info = dict(n_players=int(len(f4)), n_games=int(sel.sum()), games_per_player_set=sorted(set(cnt.tolist())), n_h_players=0, in_s_games_players=0)
    return evrows[sel], evgid[sel], f4, info


def main_target():
    ev_rows, ev_gid, f4, info = target_rows(); print(info)
    d, stats = prep(ev_rows)
    assert stats["n_kept"] == info["n_games"], ("some target games skipped", stats)
    np.savez(OUT / "target_games.npz", **d)
    np.savez(OUT / "target_order.npz", ev_rows=ev_rows, ev_gid=ev_gid, f4_players=f4)
    out = dict(info=info, stats=stats); (OUT / "target_info.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"a1": check_a1, "target": main_target}[sys.argv[1]]()
