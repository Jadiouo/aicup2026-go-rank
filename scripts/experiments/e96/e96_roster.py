"""E96 roster `finh_v4` = `finh_full` minus EVERY game of every fold-4 player (validation version of the finh_full identity network).

Definition (nothing else changes):
  finh_full (rosters_h.npz) = fin_full (707,820 non-H games of players with >= the E80 minimum number of games) + all 287,055 H games.
  finh_v4 = finh_full & (game's player is NOT a fold-4 player).  H players (fold -1, 640 players) stay in (library / training class only, never scored).
  Classes = dense rank of the player codes of the roster's games (train_e90h.py does this itself: `ups = np.unique(player)`, `cls = searchsorted`), so classes <-> players is one-to-one.
  The minimum-games rule is inherited: whole players are removed, no player loses a game (asserted: every kept player has the same number of games as in finh_full).
  Probe (in-sample monitoring, observation only): same rule as finh_full, executed by the training code (512 games drawn with default_rng(12345) from the roster's games, every 4 epochs).

Fold source: data/splits/split.parquet (`fold`, `is_holdout`); cross-checked against aux.npz `player_fold` / `player_rank` (must be identical) and the player-code map of the H players.
Also asserts: the 41,589 fold-4 validation games (r0_games.npz, the games E91/E96 score) are not in the roster (rid and full move sequence incl. colour: 0 duplicates),
and their 1,420 players are fold 4 and not classes.

  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/experiments/e96/e96_roster.py
Writes (deterministic, re-runnable):  .scratch/e96/rosters_v4.npz  (finh_v4 mask, u_rid, fold4_codes, exp_games/exp_classes/exp_steps scalars),
                                      .scratch/e96/fold4_pids.json, .scratch/e96/roster_v4_stats.json
"""
import json, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D90 = ROOT / ".scratch/e90/data"; OUT = ROOT / ".scratch/e96"
AUX = ROOT / ".scratch/e29/bundle/aux.npz"; SPLIT = ROOT / "data/splits/split.parquet"; R0 = ROOT / ".scratch/e90/stage0r/r0_games.npz"
GAMES_PER_STEP, EPOCHS = 256, 20
N_AUX, N_H = 7367, 640


def seq_hash(a):
    return hashlib.blake2b(np.asarray(a, np.int16).tobytes(), digest_size=8).digest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aux = np.load(AUX, allow_pickle=True); aux_ids = aux["player_ids"].astype(str); aux_fold = aux["player_fold"].astype(int); aux_rank = aux["player_rank"].astype(int)
    sp = pd.read_parquet(SPLIT).set_index("player_id"); assert len(aux_ids) == N_AUX
    # ---- fold source cross-check: split.parquet == aux.npz for all 7,367 aux players
    assert (sp.loc[aux_ids, "fold"].to_numpy() == aux_fold).all(), "split.parquet fold != aux.npz player_fold"
    assert (sp.loc[aux_ids, "rank_id"].to_numpy() == aux_rank).all(), "split.parquet rank_id != aux.npz player_rank"
    assert not sp.loc[aux_ids, "is_holdout"].any()
    cmap = json.loads((D90 / "data_stats_h.json").read_text())["player_code_map"]; assert len(cmap) == N_H
    h_pid = np.array([None] * (N_AUX + N_H), dtype=object)
    for pid, c in cmap.items(): h_pid[int(c)] = pid
    assert all(h_pid[c] is not None for c in range(N_AUX, N_AUX + N_H)) and sp.loc[[h_pid[c] for c in range(N_AUX, N_AUX + N_H)], "is_holdout"].all(), "H code map != split is_holdout"
    pid_all = np.concatenate([aux_ids, np.array([h_pid[c] for c in range(N_AUX, N_AUX + N_H)])]); fold_all = np.concatenate([aux_fold, np.full(N_H, -1)])
    rank_all = sp.loc[pid_all, "rank_id"].to_numpy().astype(int)
    assert (sp.loc[pid_all, "fold"].to_numpy() == fold_all).all()
    fold4_codes = np.flatnonzero(fold_all == 4)
    print("fold-4 players (split.parquet == aux.npz):", len(fold4_codes), "| fold sizes", {k: int((fold_all == k).sum()) for k in (-1, 0, 1, 2, 3, 4)})

    # ---- finh_full and the games
    U = np.load(D90 / "uh_games.npz"); R = np.load(D90 / "rosters_h.npz")
    player = U["player"].astype(np.int64); rid = U["rid"]; hd = U["hd"]; off = U["off"]; n_games = len(player)
    full = R["finh_full"].astype(bool); assert full.all() and len(full) == n_games == 994_875 and (R["u_rid"] == rid).all()
    is_f4 = np.isin(player, fold4_codes)
    mask = full & ~is_f4
    print(f"finh_full games {full.sum()}  fold-4 games dropped {int((full & is_f4).sum())}  finh_v4 games {int(mask.sum())}")

    # ---- assertions: (1) fold 4 has 0 games  (2) classes <-> players one-to-one  (3) inherited rules
    kept_pl = player[mask]; ups = np.unique(kept_pl); n_cls = len(ups)
    assert not np.isin(kept_pl, fold4_codes).any(), "a fold-4 player has a game in finh_v4"
    assert (fold_all[ups] != 4).all()
    cls = np.searchsorted(ups, kept_pl); assert (ups[cls] == kept_pl).all() and len(np.unique(cls)) == n_cls == len(ups), "classes and players are not one-to-one"
    ful_cnt = np.bincount(player[full], minlength=N_AUX + N_H); v4_cnt = np.bincount(kept_pl, minlength=N_AUX + N_H)
    keepers = np.flatnonzero(fold_all != 4); assert np.array_equal(ful_cnt[keepers], v4_cnt[keepers]), "a kept player lost games"
    assert (v4_cnt[fold4_codes] == 0).all() and v4_cnt[ups].min() == ful_cnt[ups].min() > 0
    assert set(np.flatnonzero(fold_all == -1)) <= set(ups.tolist()) and (fold_all[ups] == -1).sum() == N_H, "an H player is missing"
    assert (full & ~mask).sum() == int(is_f4.sum())
    # ---- validation games (the ones scored): not in the roster, by rid and by content; their players are fold 4 and not classes
    r0 = np.load(R0); r0_player = r0["player"].astype(np.int64); assert np.isin(r0_player, fold4_codes).all(), "a validation player is not fold 4"
    assert not np.isin(r0_player, ups).any(), "a validation player is a class"
    assert not np.isin(r0["rid"], rid[mask]).any(), "a validation game (rid) is in finh_v4"
    r0m, r0o = r0["moves"], r0["off"]; hv = {seq_hash(r0m[r0o[i]:r0o[i + 1]]) for i in range(len(r0o) - 1)}
    moves = U["moves"]; ndup = sum(seq_hash(moves[off[g]:off[g + 1]]) in hv for g in np.flatnonzero(mask))
    assert ndup == 0, f"{ndup} finh_v4 games have the same move sequence (incl. colour) as a validation game"
    print(f"validation games {len(r0['rid'])}, players {len(np.unique(r0_player))}: not in roster by rid; same-sequence duplicates against the {int(mask.sum())} roster games: {ndup}")

    # ---- numbers the training code asserts (hard-coded there, copied from this output)
    steps_per_epoch = math.ceil(int(mask.sum()) / GAMES_PER_STEP); exp = dict(games=int(mask.sum()), classes=int(n_cls), steps=steps_per_epoch * EPOCHS)
    rk = rank_all[ups]; by_rank = np.bincount(rk, minlength=10).tolist(); by_rank_nonH = np.bincount(rank_all[ups[fold_all[ups] != -1]], minlength=10).tolist()
    st = dict(roster="finh_v4", definition="finh_full minus all games of fold-4 players (split.parquet fold == 4); H kept", **exp, steps_per_epoch=steps_per_epoch,
              finh_full=dict(games=int(full.sum()), classes=int(len(np.unique(player)))), dropped_games=int(is_f4.sum()), dropped_players_with_games=int((ful_cnt[fold4_codes] > 0).sum()),
              fold4_players_total=int(len(fold4_codes)), h_classes=int((fold_all[ups] == -1).sum()), classes_by_fold={str(k): int((fold_all[ups] == k).sum()) for k in (-1, 0, 1, 2, 3)},
              players_per_rank_D_C_B_A_1D_to_6D=by_rank, players_per_rank_non_H=by_rank_nonH, min_games_per_class=int(v4_cnt[ups].min()), games_by_hd=np.bincount(hd[mask], minlength=10).tolist(),
              validation_games=int(len(r0["rid"])), validation_players=int(len(np.unique(r0_player))), validation_dup_sequences=ndup)
    print(json.dumps(st, indent=1))
    np.savez(OUT / "rosters_v4.npz", finh_v4=mask, u_rid=rid, fold4_codes=fold4_codes.astype(np.int64), exp_games=np.int64(exp["games"]), exp_classes=np.int64(exp["classes"]), exp_steps=np.int64(exp["steps"]))
    (OUT / "fold4_pids.json").write_text(json.dumps(dict(pid=pid_all[fold4_codes].tolist(), code=fold4_codes.tolist())))
    (OUT / "roster_v4_stats.json").write_text(json.dumps(st, indent=1))
    z = np.load(OUT / "rosters_v4.npz"); assert z["finh_v4"].sum() == exp["games"] and not np.isin(player[z["finh_v4"]], z["fold4_codes"]).any()
    print("ALL ASSERTIONS PASSED; wrote", OUT / "rosters_v4.npz")


if __name__ == "__main__":
    main()
