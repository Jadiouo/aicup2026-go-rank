"""E90 stages 1-3: data files for the S-network training (PLAN experiments/E90_s_channel/PLAN.md s2).  Output dir .scratch/e90/data/.

  u_games.npz   union of the six roster lists = fin_full (707,820 non-H games, bundle rows ascending).  s_games.npz-compatible keys
                (moves int16, off int64[G+1], eff int8, hd int8, cls int16, role int8, rid int32 = bundle row, player int32 = aux game_player,
                pcount int16, n_classes int64).  cls = -1 and role = 0 everywhere: the roster decides the classes (train_e90.py).
  rosters.npz   six bool masks (len = number of u_games rows) sim_full sim_h0 sim_h1 fin_full fin_h0 fin_h1, eval_players, counts.
  t_games.npz   test games (read only, never labelled, never trained on): rows 0..7099 Player pub (Z_pub order), 7100..14199 Player priv (Z_priv order),
                14200..24199 Rank test (test_bags uids order).  Same keys; rid = 10M+split*100k+pos (Player) / 30M+pos (Rank); player = -1, cls = -1.
  t_games_uids.npz  uid strings of the above.
  export_spec.npz / export_spec.json   which games each network embeds and how the output file is assembled (assemble_pe.py).
  data_stats.json   asserts and counters.

  PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/experiments/e90/make_data.py union|test|meta   (union takes ~25 min, 1 thread)
Everything stops (assert) on a skipped or truncated game; nothing is skipped silently.
"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "katago"), str(HERE)]
import numpy as np, pandas as pd
import kgio
import prep_subset as ps

OUT = ROOT / ".scratch/e90/data"; OUT.mkdir(parents=True, exist_ok=True)
EXPECT_GAMES = dict(sim_full=355_947, sim_h0=177_882, sim_h1=178_065, fin_full=707_820, fin_h0=344_529, fin_h1=363_291)       # PLAN s2 table
EXPECT_STEPS = dict(sim_full=27_820, sim_h0=13_900, sim_h1=13_920, fin_full=55_300, fin_h0=26_920, fin_h1=28_400)   # PLAN s2 table says 28,380, but ceil(363,291/256)=1,420 steps/epoch x 20 = 28,400 (E89 recipe: ceil); PLAN figure is an arithmetic slip, reported
NAMES = list(EXPECT_GAMES)
N_EVAL, N_SIM_EMBED, N_FIN_EMBED, N_SIM_OUT, N_FIN_OUT, N_RANKVAL = 1481, 110_448, 109_560, 223_693, 206_536, 66_496


def log(*a, **k): print(time.strftime("%H:%M:%S"), *a, flush=True)


def stats_path(): return OUT / "data_stats.json"


def stats_update(**kw):
    d = json.loads(stats_path().read_text()) if stats_path().exists() else {}
    d.update(kw); stats_path().write_text(json.dumps(d, indent=1))


def bundle():
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=True)
    z = np.load(ROOT / ".scratch/e80/meta.npz")
    return aux, z


# ---------------------------------------------------------------- union
def make_union():
    aux, z = bundle()
    rows = np.flatnonzero(z["fin_full"]); assert len(rows) == EXPECT_GAMES["fin_full"]
    for k in NAMES: assert (z[k] <= z["fin_full"]).all(), f"{k} not inside fin_full"
    # no H player among the rows (games_meta uid -> player_id, same check as e89/data_prep.py)
    sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"])
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet"); u2p = dict(zip(meta.uid, meta.player_id)); uid = aux["uid"].astype(str)
    pid_rows = {u2p[u] for u in uid[rows]}; assert not (pid_rows & H), "H player among the union rows"
    assert not (set(aux["player_ids"].astype(str)) & H), "aux contains an H player"
    log("union rows", len(rows), "players", len(pid_rows), "H players", len(H), "overlap 0")
    d, st = ps.prep(rows, log=log)
    assert st["n_kept"] == len(rows) and st["skipped_by_reason"] == {} and st["truncated_at_illegal"] == 0, ("skipped or truncated games, STOP", st)
    assert np.array_equal(d["rid"].astype(np.int64), rows)
    gp = aux["game_player"].astype(np.int64); assert np.array_equal(d["player"].astype(np.int64), gp[rows])
    d["cls"] = np.full(len(rows), -1, np.int16); d["role"] = np.zeros(len(rows), np.int8); d["n_classes"] = np.int64(0)
    np.savez(OUT / "u_games.npz", **d)
    stats_update(union=dict(n_games=int(len(rows)), n_moves=int(len(d["moves"])), mean_moves=float(len(d["moves"]) / len(rows)), prep=st, h_players=0,
                            hd_counts=np.bincount(d["hd"], minlength=10).tolist()))
    log("union written", OUT / "u_games.npz")


# ---------------------------------------------------------------- test games
def _conv(sgf, col_char):
    ng = kgio.normalize_game(sgf, 0 if col_char == "B" else 1)
    assert not ng.skip, ("test game skipped, STOP", ng.skip)
    assert not ng.truncated, "test game truncated, STOP"
    mv = np.array([(361 if rc is None else rc[0] * 19 + rc[1]) + (362 if c == "W" else 0) for c, rc in ng.moves], np.int16)
    own = int(((mv >= 362).astype(np.int8) == ng.eff_color).sum()); assert own > 0, "test game without own move, STOP"
    return mv, ng.eff_color, ng.hd


def make_test():
    """READ ONLY: test-set SGFs are converted to move arrays; nothing is labelled or trained on."""
    t2 = ROOT / ".scratch/p1111/t2"; items = []; rids = []
    for si, s in enumerate(("pub", "priv")):
        z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); zu = z["uids"].astype(str); assert len(zu) == 7100
        got = {}; w = "public" if s == "pub" else "private"
        c = pd.read_csv(t2 / f"player_identification_candidates_{w}.csv", dtype=str)
        for r in c.itertuples(): got[f"cand_{s}:{r.game_id}"] = (r.sgf_content, r.color)
        q = pd.read_csv(t2 / f"player_identification_test_{w}.csv", dtype=str)
        for r in q.itertuples(index=False):
            d = dict(zip(q.columns, r))
            for k in range(1, int(d["num_games"]) + 1): got[f"{d['question_id']}#{k}"] = (d[f"sgf_{k}"], d[f"color_{k}"])
        assert set(got) == set(zu), (s, len(got), len(zu))
        for p, u in enumerate(zu): items.append((u,) + got[u]); rids.append(10_000_000 + si * 100_000 + p)
    tb = np.load(ROOT / ".scratch/e29/test/test_bags.npz", allow_pickle=False); ru = tb["uids"].astype(str); assert len(ru) == 10_000
    got = {}
    for w in ("public", "private"):
        q = pd.read_csv(ROOT / f"data/contest_1104/rank_prediction_test_{w}.csv", dtype=str)
        for r in q.itertuples(index=False):
            d = dict(zip(q.columns, r))
            for k in range(1, int(d["num_games"]) + 1): got[f"{d['question_id']}#{k}"] = (d[f"sgf_{k}"], d[f"color_{k}"])
    assert set(got) == set(ru), (len(got), len(ru))
    for p, u in enumerate(ru): items.append((u,) + got[u]); rids.append(30_000_000 + p)
    assert len(items) == 24_200
    mvs, eff, hd = [], [], []
    for u, sgf, col in items:
        mv, e, h = _conv(sgf, col); mvs.append(mv); eff.append(e); hd.append(h)
    off = np.zeros(len(mvs) + 1, np.int64); off[1:] = np.cumsum([len(m) for m in mvs])
    n = len(mvs)
    np.savez(OUT / "t_games.npz", moves=np.concatenate(mvs).astype(np.int16), off=off, eff=np.array(eff, np.int8), hd=np.array(hd, np.int8), cls=np.full(n, -1, np.int16),
             role=np.full(n, -1, np.int8), rid=np.array(rids, np.int32), player=np.full(n, -1, np.int32), pcount=np.zeros(n, np.int16), n_classes=np.int64(0))
    np.savez(OUT / "t_games_uids.npz", uids=np.array([i[0] for i in items]))
    stats_update(test=dict(n_games=n, player_pub=7100, player_priv=7100, rank=10_000, skipped=0, truncated=0, hd_counts=np.bincount(hd, minlength=10).tolist(),
                           mean_moves=float(off[-1] / n)))
    log("test games written", n)


# ---------------------------------------------------------------- rosters, export spec, asserts
def gh(mv, strip):
    a = np.asarray(mv, np.int16)
    if strip: a = np.where(a >= 362, a - 362, a).astype(np.int16)
    return hashlib.blake2b(a.tobytes(), digest_size=8).digest()


def make_meta():
    aux, z = bundle(); U = np.load(OUT / "u_games.npz"); U = {k: U[k] for k in U.files}; T = np.load(OUT / "t_games.npz"); T = {k: T[k] for k in T.files}
    rid = U["rid"].astype(np.int64); G = len(rid); assert np.array_equal(rid, np.flatnonzero(z["fin_full"]))
    player = U["player"].astype(np.int64); gp = aux["game_player"].astype(np.int64)
    masks = {k: z[k][rid] for k in NAMES}
    cnt = {}
    for k in NAMES:
        assert int(masks[k].sum()) == EXPECT_GAMES[k], (k, int(masks[k].sum()), EXPECT_GAMES[k])
        cnt[k] = dict(games=int(masks[k].sum()), players=int(len(np.unique(player[masks[k]]))), min_games_per_player=int(np.bincount(player[masks[k]])[np.unique(player[masks[k]])].min()),
                      steps=int(np.ceil(masks[k].sum() / 256) * 20))
        assert cnt[k]["steps"] == EXPECT_STEPS[k], (k, cnt[k]["steps"])
    ml = json.loads((ROOT / ".scratch/e80/meta_log.json").read_text())
    for k in NAMES: assert cnt[k]["players"] == ml[k]["players"] and cnt[k]["games"] == ml[k]["games"], (k, cnt[k], ml[k])
    assert not (masks["sim_h0"] & masks["sim_h1"]).any() and not (masks["fin_h0"] & masks["fin_h1"]).any()
    assert np.array_equal(masks["sim_h0"] | masks["sim_h1"], masks["sim_full"]) and np.array_equal(masks["fin_h0"] | masks["fin_h1"], masks["fin_full"]) and masks["fin_full"].all()
    # simulation-evaluation players: none of the three sim lists may contain one
    sm = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet"); ev = sm.loc[sm.grp == "eval", "player_id"].unique(); assert len(ev) == N_EVAL
    pidx = {p: i for i, p in enumerate(aux["player_ids"].astype(str))}; evc = np.array(sorted(pidx[p] for p in ev)); assert len(evc) == N_EVAL
    is_ev = np.isin(player, evc)
    for k in ("sim_full", "sim_h0", "sim_h1"): assert not (masks[k] & is_ev).any(), f"{k} contains a sim eval player"
    assert int(is_ev.sum()) == ml["sim_eval_games_excluded"] == 351_873                # E80 meta_log: sim_eval_games_excluded
    # no H player in any list (aux has only non-H players; re-check through the split file)
    sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"])
    assert not (set(aux["player_ids"].astype(str)[np.unique(player)]) & H), "H player in union"
    assert (player >= 0).all()
    # ---- embedding sets (E80 e80_enc.py: sim_rows/fin_rows of meta.npz; test games of the final full encoder)
    sim_u = np.searchsorted(rid, z["sim_rows"]); fin_u = np.searchsorted(rid, z["fin_rows"])
    assert np.array_equal(rid[sim_u], z["sim_rows"]) and np.array_equal(rid[fin_u], z["fin_rows"])
    assert len(sim_u) == N_SIM_EMBED and len(fin_u) == N_FIN_EMBED
    assert len(np.unique(z["sim_gid"])) == N_SIM_EMBED and z["sim_gid"].max() < N_SIM_OUT and len(np.unique(z["fin_gid"])) == N_FIN_EMBED and int(z["fin_nrows"]) == N_FIN_OUT and z["fin_gid"].max() < N_FIN_OUT
    assert int(z["sim_eval"].sum()) == 35_544 and is_ev[sim_u][z["sim_eval"]].all() and not is_ev[sim_u][~z["sim_eval"]].any()      # eval games = exactly the sim_eval flag
    # Rank validation questions of the 1,481 sim eval players (optional Rank leg): rows in val_rows_f{k} whose val_player is a sim eval player
    rv = []
    for k in range(5):
        vr = aux[f"val_rows_f{k}"]; vp = aux[f"val_player_f{k}"].astype(np.int64); sel = np.isin(vp, evc)
        rr = vr[sel]; rr = rr[rr >= 0]; assert (gp[rr] == np.repeat(vp[sel], (vr[sel] >= 0).sum(1))).all(); rv.append(rr)
    rv = np.unique(np.concatenate(rv)).astype(np.int64); rank_u = np.searchsorted(rid, rv); assert np.array_equal(rid[rank_u], rv), "rank val game not in union"
    log("rank-val games of sim eval players", len(rv), "(interface notes: 66,496)")
    # ---- test games: hash overlap with the training games (read only; informational)
    hu = {}; hus = {}
    for i in range(G):
        m = U["moves"][U["off"][i]:U["off"][i + 1]]; hu.setdefault(gh(m, False), []).append(i); hus.setdefault(gh(m, True), []).append(i)
    ov = {"player_test_full": 0, "player_test_stripped": 0, "rank_test_full": 0, "rank_test_stripped": 0}; ex = {}
    for j in range(len(T["rid"])):
        m = T["moves"][T["off"][j]:T["off"][j + 1]]; kind = "player" if j < 14_200 else "rank"
        if gh(m, False) in hu: ov[f"{kind}_test_full"] += 1; ex.setdefault(kind, []).append(j)
        if gh(m, True) in hus: ov[f"{kind}_test_stripped"] += 1
    log("test/train move-sequence overlap", ov)
    np.savez(OUT / "rosters.npz", **masks, eval_players=evc, u_rid=rid)
    np.savez(OUT / "export_spec.npz", sim_embed_u=sim_u, sim_gid=z["sim_gid"].astype(np.int64), fin_embed_u=fin_u, fin_gid=z["fin_gid"].astype(np.int64),
             sim_eval_in_embed=z["sim_eval"], test_player_t=np.arange(14_200), test_rank_t=np.arange(14_200, 24_200), rankval_u=rank_u)
    spec = {
        "n_out": {"sim": N_SIM_OUT, "fin": N_FIN_OUT, "test_each": 7100},
        "exports": {
            "sim_h0": [dict(name="sim_h0", games="u", rows="sim_embed_u", scatter="sim_gid", n_out=N_SIM_OUT, out=["pe_sim_h0.npy"], required=True)],
            "sim_h1": [dict(name="sim_h1", games="u", rows="sim_embed_u", scatter="sim_gid", n_out=N_SIM_OUT, out=["pe_sim_h1.npy"], required=True)],
            "sim_full": [dict(name="sim_full", games="u", rows="sim_embed_u", scatter="sim_gid", n_out=N_SIM_OUT, out=["pe_sim_full.npy"], required=True),
                         dict(name="sim_full_rankval", games="u", rows="rankval_u", out=[], required=False, note="optional Rank leg: sim eval players' Rank validation games (PLAN s2, Rank measure not pinned yet)")],
            "fin_h0": [dict(name="fin_h0", games="u", rows="fin_embed_u", scatter="fin_gid", n_out=N_FIN_OUT, out=["pe_final_h0.npy"], required=True)],
            "fin_h1": [dict(name="fin_h1", games="u", rows="fin_embed_u", scatter="fin_gid", n_out=N_FIN_OUT, out=["pe_final_h1.npy"], required=True)],
            "fin_full": [dict(name="fin_full_test", games="t", rows="test_player_t", out=["pe_test_pub.npy", "pe_test_priv.npy"], required=True),
                         dict(name="fin_full_ranktest", games="t", rows="test_rank_t", out=[], required=False, note="optional Rank leg: Rank test games (10,000, test_bags uid order)")],
        }}
    (OUT / "export_spec.json").write_text(json.dumps(spec, indent=1))
    stats_update(rosters=cnt, sim_eval_players=int(len(evc)), sim_eval_games_in_union=int(is_ev.sum()), h_players_in_any_list=0, rankval_games=int(len(rv)),
                 embed=dict(sim_embed=int(len(sim_u)), sim_eval_embed=int(z["sim_eval"].sum()), fin_embed=int(len(fin_u))), test_train_overlap=ov)
    log("meta done", json.dumps(cnt))


if __name__ == "__main__":
    {"union": make_union, "test": make_test, "meta": make_meta}[sys.argv[1]]()
