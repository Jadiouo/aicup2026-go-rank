"""E90 "about 1M games" variant: add every game of the 640 holdout (H) players to the E90 TRAINING data (user decision 2026-11-17: H may enter training,
never any score).  Nothing in here reads an H label or a score; only the SGFs of the H players' games are converted.  Output dir .scratch/e90/data/.

  h_games.npz        only the H games (same keys as u_games.npz; cls = -1, role = 0, n_classes = 0).  rid = 20,000,000 + running number (sorted by uid),
                     player = len(aux player_ids) + rank of the player_id among the H players (sorted).  Conversion = prep_subset.prep() logic verbatim
                     (kgio.normalize_game, own-move check, same skip / truncate rules) except that the hd / colour checks against the bundle (HD_full, aux)
                     are replaced by the games_meta colour check (H games are not in the bundle).  Skipped games are counted, truncated games are kept and counted.
  uh_games.npz       u_games.npz followed by h_games.npz (off shifted).  Deterministic file bytes (fixed zip timestamps) so the sha256 can be compared after
                     re-concatenating on the Colab VM.
  rosters_h.npz      the six original masks (padded with False to len(uh)) + finh_full / finh_h0 / finh_h1 (= fin_* | all H games); eval_players; u_rid = rid of uh.
  export_spec_h.json / export_spec_h.npz   export_spec.json + the three finh_* keys (same entries as fin_*); export_spec_h.npz is byte-identical to
                     export_spec.npz (train_e90.py looks for <spec>.npz next to the json; H games come last so no index changes).
  data_stats_h.json  counters and asserts.

  PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/experiments/e90/make_data_h.py build      (H conversion, ~10-15 min, 1 thread; then concat)
  python make_data_h.py concat [--dir /content/e90]                                                                 (numpy only; local or Colab VM)
"""
import sys, json, time, math, hashlib, zipfile, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
OUT = ROOT / ".scratch/e90/data"
N_U = 707_820; N_H_PLAYERS = 640; H_RID0 = 20_000_000
NEW = {"finh_full": "fin_full", "finh_h0": "fin_h0", "finh_h1": "fin_h1"}
KEYS = ["moves", "off", "eff", "hd", "cls", "role", "rid", "player", "pcount", "n_classes"]


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_npz(path, arrays):
    """np.savez equivalent with fixed zip timestamps -> identical bytes for identical arrays (np.load reads it normally)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for k, v in arrays.items():
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_STORED
            with z.open(zi, "w", force_zip64=True) as f: np.lib.format.write_array(f, np.asanyarray(v), allow_pickle=False)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def load(p):
    d = np.load(p); return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- concat (numpy only)
def concat(dirp):
    dirp = Path(dirp); U = load(dirp / "u_games.npz"); H = load(dirp / "h_games.npz")
    assert list(U) == KEYS and list(H) == KEYS, (list(U), list(H))
    assert len(U["rid"]) == N_U, len(U["rid"])
    n_u, n_h = len(U["rid"]), len(H["rid"]); assert H["off"][0] == 0 and U["off"][0] == 0
    out = {}
    for k in KEYS:
        if k == "off": out[k] = np.concatenate([U["off"], H["off"][1:] + U["off"][-1]]).astype(np.int64)
        elif k == "n_classes": out[k] = U[k]
        else:
            assert U[k].dtype == H[k].dtype, (k, U[k].dtype, H[k].dtype); out[k] = np.concatenate([U[k], H[k]])
    assert len(out["rid"]) == n_u + n_h and out["off"][-1] == len(out["moves"]) and len(out["off"]) == n_u + n_h + 1
    assert all(np.array_equal(out[k][:n_u], U[k]) for k in KEYS if k not in ("off", "n_classes", "moves")), "prefix mismatch"
    assert np.array_equal(out["off"][:n_u + 1], U["off"]) and np.array_equal(out["moves"][:len(U["moves"])], U["moves"]), "prefix mismatch (moves/off)"
    assert np.array_equal(out["moves"][len(U["moves"]):], H["moves"]) and np.array_equal(np.diff(out["off"][n_u:]), np.diff(H["off"]))
    write_npz(dirp / "uh_games.npz", out)
    sh = sha256_file(dirp / "uh_games.npz"); print("uh_games.npz games", n_u + n_h, "moves", len(out["moves"]), "sha256", sh, flush=True)
    return sh, n_u, n_h


# ---------------------------------------------------------------- build (H conversion, rosters, spec, stats)
def gh(mv, strip):
    a = np.asarray(mv, np.int16)
    if strip: a = np.where(a >= 362, a - 362, a).astype(np.int16)
    return hashlib.blake2b(a.tobytes(), digest_size=8).digest()


def build():
    sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "katago"), str(HERE)]
    import pandas as pd, kgio
    t0 = time.time()
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=True); n_aux_players = len(aux["player_ids"]); assert n_aux_players == 7367
    sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); Hset = sorted(sp.loc[sp["is_holdout"].astype(bool), "player_id"].astype(str))
    log("H players (is_holdout)", len(Hset))
    assert len(Hset) == N_H_PLAYERS, ("H player count is not 640", len(Hset))
    assert not (set(aux["player_ids"].astype(str)) & set(Hset)), "aux contains an H player"
    code_of = {p: n_aux_players + i for i, p in enumerate(Hset)}
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet"); assert meta.uid.is_unique
    mh = meta[meta.player_id.astype(str).isin(code_of)].copy().sort_values("uid").reset_index(drop=True)
    assert not set(mh.uid) & set(aux["uid"].astype(str)), "an H game is in the bundle"
    log("H games in games_meta", len(mh), "players", mh.player_id.nunique())
    want = {u: i for i, u in enumerate(mh.uid)}; mcol = mh["color"].to_numpy().astype(np.int64)
    moves, effs, hds, skips, skip_cnt, trunc, seen = {}, {}, {}, {}, {}, 0, set()
    for f in sorted((ROOT / "data/training_v1.1/training").glob("train_*.csv")):
        for ch in pd.read_csv(f, usecols=["game_id", "rank", "color", "sgf_content"], chunksize=10_000):
            ch["uid"] = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str)
            ch = ch[ch["uid"].isin(want)]
            for u, col, s in zip(ch["uid"], ch["color"], ch["sgf_content"]):
                i = want[u]; assert i not in seen, ("duplicate uid in csv", u); seen.add(i)
                assert (0 if col == "B" else 1) == int(mcol[i]), "csv colour != games_meta colour"
                ng = kgio.normalize_game(s, int(mcol[i]))
                if ng.skip: skip_cnt[ng.skip] = skip_cnt.get(ng.skip, 0) + 1; skips[i] = ng.skip; continue
                trunc += int(ng.truncated)
                mv = np.array([(361 if rc is None else rc[0] * 19 + rc[1]) + (362 if c == "W" else 0) for c, rc in ng.moves], np.int16)
                own = int(((mv >= 362).astype(np.int8) == ng.eff_color).sum())
                if own == 0: skip_cnt["no_own_move"] = skip_cnt.get("no_own_move", 0) + 1; skips[i] = "no_own_move"; continue
                moves[i] = mv; effs[i] = ng.eff_color; hds[i] = ng.hd
        log(f.name, "seen", len(seen), "kept", len(moves))
    missing = len(mh) - len(seen)
    log("H games meta", len(mh), "found in csv", len(seen), "missing", missing, "skipped", skip_cnt, "truncated(kept)", trunc)
    assert missing == 0, ("H games not found in the raw csv - STOP and report", missing)
    ok = np.array(sorted(moves), dtype=np.int64); n_h = len(ok)
    off = np.zeros(n_h + 1, np.int64); off[1:] = np.cumsum([len(moves[i]) for i in ok])
    pl = mh.player_id.astype(str).map(code_of).to_numpy().astype(np.int64)
    n_p_all = np.bincount(pl - n_aux_players, minlength=N_H_PLAYERS)                       # games per H player in the data (before skipping), as prep's pcount (= n_p of all bundle games)
    rid = (H_RID0 + np.arange(n_h)).astype(np.int32)                                        # consecutive, uid-sorted order
    H = dict(moves=np.concatenate([moves[i] for i in ok]).astype(np.int16), off=off, eff=np.array([effs[i] for i in ok], np.int8), hd=np.array([hds[i] for i in ok], np.int8),
             cls=np.full(n_h, -1, np.int16), role=np.zeros(n_h, np.int8), rid=rid, player=pl[ok].astype(np.int32),
             pcount=n_p_all[pl[ok] - n_aux_players].astype(np.int16), n_classes=np.int64(0))
    assert list(H) == KEYS
    h_players = np.unique(H["player"]); h_with_games = len(h_players)
    assert h_players.min() >= n_aux_players and h_players.max() < n_aux_players + N_H_PLAYERS
    assert len(np.unique(H["rid"])) == n_h and H["rid"].min() >= H_RID0 and H["rid"].max() < 30_000_000
    write_npz(OUT / "h_games.npz", H); log("h_games.npz written", n_h, "games", len(H["moves"]), "moves")
    # ---------------- concat + rosters
    sh_uh, n_u, n_h2 = concat(OUT); assert n_u == N_U and n_h2 == n_h
    U = load(OUT / "u_games.npz"); Z = load(OUT / "rosters.npz"); G = n_u + n_h
    assert int(U["rid"].max()) < 10_000_000 and not np.isin(H["rid"], U["rid"]).any()
    assert not np.isin(h_players, np.unique(U["player"])).any(), "H player code collides with a non-H player code"
    UH = load(OUT / "uh_games.npz")
    assert np.array_equal(UH["rid"][:n_u], U["rid"]) and np.array_equal(UH["player"][:n_u], U["player"]) and np.array_equal(UH["moves"][:len(U["moves"])], U["moves"])
    assert set(UH["player"][n_u:].tolist()) == set(h_players.tolist()) and (UH["player"][n_u:] >= n_aux_players).all() and (UH["player"][:n_u] < n_aux_players).all()
    assert (UH["player"][:n_u] >= 0).all()
    Tg = load(OUT / "t_games.npz"); assert not np.isin(UH["rid"], Tg["rid"]).any(), "rid collides with a test game rid"
    old = [k for k in Z if k not in ("eval_players", "u_rid")]; assert old == ["sim_full", "sim_h0", "sim_h1", "fin_full", "fin_h0", "fin_h1"], old
    R = {}
    for k in old: assert len(Z[k]) == n_u; R[k] = np.concatenate([Z[k].astype(bool), np.zeros(n_h, bool)])
    isH = np.zeros(G, bool); isH[n_u:] = True
    for new, base in NEW.items(): R[new] = R[base] | isH
    R["eval_players"] = Z["eval_players"]; R["u_rid"] = UH["rid"]
    cnt = {}
    for k in NEW:
        b = NEW[k]; assert int(R[k].sum()) == int(R[b].sum()) + n_h and (R[k] & ~isH == R[b]).all() and R[k][n_u:].all()
        pls = np.unique(UH["player"][R[k]]); assert len(pls) == len(np.unique(UH["player"][R[b]])) + h_with_games
        cnt[k] = dict(games=int(R[k].sum()), players=int(len(pls)), steps=int(math.ceil(R[k].sum() / 256) * 20), base=b, base_games=int(R[b].sum()))
    assert not (R["finh_h0"] & R["finh_h1"] & ~isH).any() and (R["finh_h0"] | R["finh_h1"] == R["finh_full"]).all() and R["finh_full"].all()
    write_npz(OUT / "rosters_h.npz", R)
    # ---------------- export spec
    spec = json.loads((OUT / "export_spec.json").read_text())
    for new, base in NEW.items(): spec["exports"][new] = spec["exports"][base]
    (OUT / "export_spec_h.json").write_text(json.dumps(spec, indent=1))
    ea = np.load(OUT / "export_spec.npz"); mx_u = max(int(ea[k].max()) for k in ("sim_embed_u", "fin_embed_u", "rankval_u")); assert mx_u < n_u, "export index into the H part"
    assert int(ea["test_rank_t"].max()) < len(Tg["rid"])
    (OUT / "export_spec_h.npz").write_bytes((OUT / "export_spec.npz").read_bytes())
    assert sha256_file(OUT / "export_spec_h.npz") == sha256_file(OUT / "export_spec.npz")
    # ---------------- test-game overlap (informational only)
    hH, hHs = set(), set()
    for i in range(n_h):
        m = H["moves"][H["off"][i]:H["off"][i + 1]]; hH.add(gh(m, False)); hHs.add(gh(m, True))
    ov = {"player_test_full": 0, "player_test_stripped": 0, "rank_test_full": 0, "rank_test_stripped": 0}
    for j in range(len(Tg["rid"])):
        m = Tg["moves"][Tg["off"][j]:Tg["off"][j + 1]]; kind = "player" if j < 14_200 else "rank"
        if gh(m, False) in hH: ov[f"{kind}_test_full"] += 1
        if gh(m, True) in hHs: ov[f"{kind}_test_stripped"] += 1
    log("H / test move-sequence overlap (record only)", ov)
    # ---------------- stats
    st = dict(h_games=int(n_h), h_games_in_meta=int(len(mh)), h_players_split=len(Hset), h_players_with_games=int(h_with_games), skipped_by_reason=skip_cnt, skipped=int(sum(skip_cnt.values())),
              truncated_at_illegal_kept=int(trunc), n_moves=int(len(H["moves"])), mean_moves=float(len(H["moves"]) / n_h), hd_counts=np.bincount(H["hd"], minlength=10).tolist(),
              eff_counts=np.bincount(H["eff"], minlength=2).tolist(), rid_range=[int(H["rid"].min()), int(H["rid"].max())], player_code_range=[int(h_players.min()), int(h_players.max())],
              u_games=n_u, uh_games=int(G), rosters=cnt, test_overlap=ov, uh_games_sha256=sh_uh, h_games_sha256=sha256_file(OUT / "h_games.npz"), rosters_h_sha256=sha256_file(OUT / "rosters_h.npz"),
              seconds=round(time.time() - t0), player_code_map={p: c for p, c in code_of.items()})
    (OUT / "data_stats_h.json").write_text(json.dumps(st, indent=1)); log("done", json.dumps({k: v for k, v in st.items() if k != "player_code_map"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "concat"]); ap.add_argument("--dir", default=str(OUT)); a = ap.parse_args()
    build() if a.cmd == "build" else concat(a.dir)
