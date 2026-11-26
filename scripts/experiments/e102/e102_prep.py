"""E102 step 1 (CPU): game files of the `finh_v4` roster (857,634 training games) for the vector bank, own side and opponent side + row table.
Same recipe as E91 (scripts/experiments/e91/e91_prep.py flip_side): the own side file is the roster subset of .scratch/e90/data/uh_games.npz (eff = colour of the FILE OWNER = the annotated player, exactly what
train_e96.py trains on); the opponent side file differs ONLY in eff -> 1-eff and a key `side` (=1).  moves/off/hd/rid/player/pcount identical, so row i is the same game in both files, in the same order as the roster.
Outputs (refuses to overwrite):
  .scratch/e102/bank_own_games.npz   keys as r0_games.npz (moves,off,eff,hd,cls,role,rid,player,pcount,n_classes[=6039])
  .scratch/e102/bank_opp_games.npz   same + side (int8, all 1), eff flipped
  .scratch/e102/bank_meta.npz        rid, owner_cls (row of head_v4.npz: W / cls_rank / cls_code / cls_pid), owner_code (player code), owner_rank (0-9), hd, eff (owner colour), n_moves
  /usr/bin/python3 scripts/experiments/e102/e102_prep.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / ".scratch/e102"; OUT.mkdir(parents=True, exist_ok=True)
UH = ROOT / ".scratch/e90/data/uh_games.npz"; ROS = ROOT / ".scratch/e96/rosters_v4.npz"; HEAD = ROOT / ".scratch/e96/head_v4.npz"
F4 = ROOT / ".scratch/e96/fold4_pids.json"; R0 = ROOT / ".scratch/e90/stage0r/r0_games.npz"
N_GAMES, N_CLS = 857_634, 6039
P_OWN, P_OPP, P_META = OUT / "bank_own_games.npz", OUT / "bank_opp_games.npz", OUT / "bank_meta.npz"


def main():
    for p in (P_OWN, P_OPP, P_META): assert not p.exists(), f"{p} exists - refuse to overwrite"
    U = np.load(UH); R = np.load(ROS); H = np.load(HEAD, allow_pickle=True); f4 = json.loads(F4.read_text())
    mask = R["finh_v4"].astype(bool); assert mask.sum() == N_GAMES == int(R["exp_games"]) and (R["u_rid"] == U["rid"]).all() and len(mask) == len(U["eff"])
    idx = np.flatnonzero(mask); player = U["player"][idx].astype(np.int64); rid = U["rid"][idx]; eff = U["eff"][idx]; hd = U["hd"][idx]
    assert len(np.unique(rid)) == N_GAMES and (np.diff(rid) > 0).all(), "rid not unique/ascending"
    assert set(np.unique(eff).tolist()) <= {0, 1}
    # fold 4 guard
    assert not np.isin(player, np.array(f4["code"])).any() and not np.isin(player, R["fold4_codes"]).any(), "a fold-4 player owns a bank game - STOP"
    # classes <-> head rows
    cls_code = H["cls_code"].astype(np.int64); assert int(H["n_classes"]) == N_CLS == len(cls_code) and (np.diff(cls_code) > 0).all()
    assert np.array_equal(np.unique(player), cls_code), "players of the roster != classes of the head"
    owner_cls = np.searchsorted(cls_code, player); assert (cls_code[owner_cls] == player).all()
    owner_rank = H["cls_rank"].astype(np.int8)[owner_cls]; assert owner_rank.min() >= 0 and owner_rank.max() <= 9
    # zero overlap with the validation games
    r0rid = np.load(R0)["rid"]; assert len(np.intersect1d(rid, r0rid)) == 0, "bank rid overlaps r0_games rid - STOP"
    # moves of the roster games (block-wise gather, int16)
    off_u = U["off"]; lens = np.diff(off_u)[idx]; off = np.zeros(N_GAMES + 1, np.int64); off[1:] = np.cumsum(lens)
    mv_u = U["moves"]; moves = np.empty(off[-1], np.int16)
    for a in range(0, N_GAMES, 100_000):
        b = min(a + 100_000, N_GAMES); ln = lens[a:b]; tot = int(ln.sum())
        src = np.repeat(off_u[idx[a:b]] - (off[a:b] - off[a]), ln) + np.arange(tot); moves[off[a]:off[b]] = mv_u[src]
    for k in (0, 1, N_GAMES // 2, N_GAMES - 1):      # spot check of the gather
        assert np.array_equal(moves[off[k]:off[k + 1]], mv_u[off_u[idx[k]]:off_u[idx[k] + 1]])
    del mv_u
    base = dict(moves=moves, off=off, eff=eff.astype(np.int8), hd=hd.astype(np.int8), cls=U["cls"][idx], role=U["role"][idx], rid=rid.astype(np.int32), player=player.astype(np.int32),
                pcount=U["pcount"][idx], n_classes=np.int64(N_CLS))
    opp = dict(base); opp["eff"] = (1 - base["eff"]).astype(np.int8); opp["side"] = np.ones(N_GAMES, np.int8)
    assert np.array_equal(1 - opp["eff"], base["eff"]) and opp["moves"] is base["moves"] and (opp["eff"] != base["eff"]).all()
    meta = dict(rid=base["rid"], owner_cls=owner_cls.astype(np.int16), owner_code=base["player"], owner_rank=owner_rank, hd=base["hd"], eff=base["eff"], n_moves=lens.astype(np.int16))
    np.savez(P_OWN, **base); np.savez(P_OPP, **opp); np.savez(P_META, **meta)
    st = dict(n_games=N_GAMES, n_classes=N_CLS, n_moves_total=int(off[-1]), hd_counts={int(k): int(v) for k, v in zip(*np.unique(hd, return_counts=True))}, rank_counts_games=np.bincount(owner_rank, minlength=10).tolist(),
              n_h_games=int((player >= 7367).sum()), eff_counts=np.bincount(eff, minlength=2).tolist())
    (OUT / "bank_stats.json").write_text(json.dumps(st, indent=1)); print(json.dumps(st, indent=1)); print("OK: 0 fold-4 players, 0 rid overlap with r0_games, classes == head rows")


if __name__ == "__main__":
    main()
