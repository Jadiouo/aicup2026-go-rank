"""E103 C2 vector bank, TEST version (CPU): opponent-side game file + row table of ALL 994,875 training games (the `finh_full` roster = every game of uh_games.npz, fold-4 and H players included;
H only as library) for the small identity network `finh_full` (10x128, 7,365 classes, .scratch/e90/colab/finh_full).
Copy of scripts/experiments/e102/e102_prep.py (stays untouched) with: source = whole uh_games.npz (roster finh_full is all-True, asserted), no fold-4 guard (finh_full legitimately contains them), class table = head_b.npz
(.scratch/e91/b/head_b.npz, the head v10b/v10d use), only the opponent side file (T3 needs no own side).  Export range is the same as E102 (all hd; hd >= 2 is filtered at scoring time).
Same recipe as E91/E102 flip_side: opp file differs from the own-side file ONLY in eff -> 1-eff (moves shared) plus key `side` (all 1).  eff = colour of the FILE OWNER (annotated player).
Outputs (refuses to overwrite), in .scratch/e103/bank/:
  bank_opp_games_full.npz   keys as E102's bank_opp_games.npz (moves,off,eff,hd,cls,role,rid,player,pcount,n_classes,side)
  bank_meta_full.npz        rid int32, owner_cls int16 (row of head_b.npz), owner_code int32, owner_rank int8 (0-9), hd int8, eff int8 (owner colour), n_moves int16   (same names/dtypes as .scratch/e102/bank_meta.npz)
  bank_stats_full.json
  /usr/bin/python3 scripts/experiments/e103/e103_bank_prep.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / ".scratch/e103/bank"; OUT.mkdir(parents=True, exist_ok=True)
UH = ROOT / ".scratch/e90/data/uh_games.npz"; ROS = ROOT / ".scratch/e90/data/rosters_h.npz"; HEAD = ROOT / ".scratch/e91/b/head_b.npz"
N_GAMES, N_CLS = 994_875, 7365
P_OPP, P_META, P_ST = OUT / "bank_opp_games_full.npz", OUT / "bank_meta_full.npz", OUT / "bank_stats_full.json"


def main():
    for p in (P_OPP, P_META, P_ST): assert not p.exists(), f"{p} exists - refuse to overwrite"
    U = np.load(UH); R = np.load(ROS); H = np.load(HEAD, allow_pickle=True)
    assert R["finh_full"].all() and len(R["finh_full"]) == N_GAMES == len(U["eff"]) and (R["u_rid"] == U["rid"]).all(), "roster finh_full is not 'all games of uh_games'"
    rid = U["rid"]; eff = U["eff"]; hd = U["hd"]; player = U["player"].astype(np.int64)
    assert len(np.unique(rid)) == N_GAMES and (np.diff(rid) > 0).all(), "rid not unique/ascending"
    assert set(np.unique(eff).tolist()) <= {0, 1}
    cls_code = H["cls_code"].astype(np.int64); assert int(H["n_classes"]) == N_CLS == len(cls_code) and (np.diff(cls_code) > 0).all() and str(H["roster"]) == "finh_full" and int(H["steps"]) == 77740
    assert np.array_equal(np.unique(player), cls_code), "players of the games != classes of head_b"
    owner_cls = np.searchsorted(cls_code, player); assert (cls_code[owner_cls] == player).all()
    owner_rank = H["cls_rank"].astype(np.int8)[owner_cls]; assert owner_rank.min() >= 0 and owner_rank.max() <= 9
    off = U["off"]; assert len(off) == N_GAMES + 1 and off[0] == 0 and (np.diff(off) > 0).all(); lens = np.diff(off)
    moves = U["moves"]; assert len(moves) == off[-1]
    opp = dict(moves=moves, off=off, eff=(1 - eff).astype(np.int8), hd=hd.astype(np.int8), cls=U["cls"], role=U["role"], rid=rid.astype(np.int32), player=player.astype(np.int32), pcount=U["pcount"],
               n_classes=np.int64(N_CLS), side=np.ones(N_GAMES, np.int8))
    assert (opp["eff"] != eff).all() and np.array_equal(1 - opp["eff"], eff)
    # cross-check of the flip against the E91 helper on a slice
    sys.path.insert(0, str(ROOT / "scripts/experiments/e91")); import e91_prep as P
    fl = P.flip_side(dict(eff=eff[:1000])); assert np.array_equal(fl["eff"], opp["eff"][:1000]) and (fl["side"] == 1).all()
    meta = dict(rid=opp["rid"], owner_cls=owner_cls.astype(np.int16), owner_code=opp["player"], owner_rank=owner_rank, hd=opp["hd"], eff=eff.astype(np.int8), n_moves=lens.astype(np.int16))
    np.savez(P_OPP, **opp); np.savez(P_META, **meta)
    st = dict(n_games=N_GAMES, n_classes=N_CLS, n_moves_total=int(off[-1]), hd_counts={int(k): int(v) for k, v in zip(*np.unique(hd, return_counts=True))}, rank_counts_games=np.bincount(owner_rank, minlength=10).tolist(),
              n_h_games=int((player >= 7367).sum()), eff_counts=np.bincount(eff, minlength=2).tolist())
    P_ST.write_text(json.dumps(st, indent=1)); print(json.dumps(st, indent=1)); print("OK: roster finh_full == all games, classes == head_b rows, flip == e91_prep.flip_side")


if __name__ == "__main__":
    main()
