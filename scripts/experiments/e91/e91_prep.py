"""E91 step A: opponent-side game file for the fold-4 validation games.

The S network (E89 setting A) reads the positions of ONE side of a game: the side `eff` (0 = black, 1 = white, in the normalised frame of
scripts/katago/kgio.normalize_game).  The moves of both sides are already in `moves` (colour bit: e >= 362 = white), so the opponent side is
obtained by flipping `eff` only -- no move is changed, no new SGF parsing.  Handicap games (hd >= 2) are normalised by kgio into a frame where
the handicap stones belong to black and white moves first; flipping `eff` there also yields the other player.
Everything else (rid, player, cls, role, hd, off) is kept, so row i of the output is row i of the own-side file (embed outputs line up 1:1).
`player` / `cls` keep describing the QUESTIONED player (the annotated side), not the opponent.  A new key `side` (int8, all 1) marks the file.

  /usr/bin/python3 e91_prep.py full   [--src .scratch/e90/stage0r/r0_games.npz] [--out .scratch/e91/opp_games.npz]
  /usr/bin/python3 e91_prep.py sample N [--seed 0]     # N random games of the source -> .scratch/e91/{opp,own}_games_s<N>.npz (small CPU tests; rid ascending)
Refuses to overwrite.  Also writes a stats json (positions per side = cost estimate for the embedding) and runs the file-level checks
(double flip == source, every game has an opponent position).  No score of any kind is computed here.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / ".scratch/e91"; OUT.mkdir(parents=True, exist_ok=True)
SRC = ROOT / ".scratch/e90/stage0r/r0_games.npz"


def load(p):
    d = np.load(p); return {k: d[k] for k in d.files}


def flip_side(d):
    """Return a copy of game dict d with the side flipped (eff -> 1 - eff); `side` toggled 0/1."""
    o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    assert set(np.unique(o["eff"])) <= {0, 1}
    o["eff"] = (1 - o["eff"]).astype(np.int8)
    o["side"] = (1 - d.get("side", np.zeros(len(d["eff"]), np.int8))).astype(np.int8)
    return o


def own_positions(d, side_eff):
    """# of non-pass moves of colour side_eff[g] in every game (= number of positions the S network embeds)."""
    mv, off = d["moves"], d["off"]; col = (mv >= 362).astype(np.int8); pos = mv - 362 * col.astype(np.int16)
    ok = (pos != 361).astype(np.int64)
    gid = np.repeat(np.arange(len(off) - 1), np.diff(off))
    return np.bincount(gid, weights=ok * (col == side_eff[gid]), minlength=len(off) - 1).astype(np.int64)


def subset(d, idx):
    """subset of games (idx ascending) with moves/off rebuilt; per-game arrays sliced."""
    idx = np.asarray(idx); n = len(d["eff"]); o = {}
    for k, v in d.items():
        if k == "moves" or k == "off": continue
        o[k] = v[idx] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n else v
    lens = np.diff(d["off"])[idx]; off = np.zeros(len(idx) + 1, np.int64); off[1:] = np.cumsum(lens)
    o["off"] = off; o["moves"] = np.concatenate([d["moves"][d["off"][g]:d["off"][g + 1]] for g in idx]).astype(np.int16)
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["full", "sample"]); ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--src", default=str(SRC)); ap.add_argument("--out", default=str(OUT / "opp_games.npz")); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); d = load(a.src)
    assert (np.diff(d["rid"]) > 0).all() and "side" not in d
    if a.mode == "sample":
        rng = np.random.default_rng(a.seed); idx = np.sort(rng.choice(len(d["rid"]), a.n, replace=False)); d = subset(d, idx)
        outs = {OUT / f"opp_games_s{a.n}.npz": flip_side(d), OUT / f"own_games_s{a.n}.npz": dict(d, side=np.zeros(len(d["eff"]), np.int8))}
    else:
        outs = {Path(a.out): flip_side(d)}
    for p, o in outs.items():
        assert not p.exists(), f"{p} exists - refuse to overwrite"
        # file-level checks (no scores)
        back = flip_side(o)      # opp file: flipping again must give the source; own file (side 0): it is the source itself
        for k in d:
            assert np.array_equal(back[k] if int(o["side"][0]) == 1 else o[k], d[k]), f"double flip / own file differs from source in {k}"
        assert np.array_equal(o["moves"], d["moves"]) and np.array_equal(o["off"], d["off"])
        npos = own_positions(o, o["eff"]); npos_other = own_positions(o, 1 - o["eff"])
        assert (npos > 0).all(), "a game has no position for this side"
        st = dict(file=str(p), n_games=int(len(o["eff"])), side=int(o["side"][0]), n_positions_this_side=int(npos.sum()), n_positions_other_side=int(npos_other.sum()),
                  n_games_hd0=int((o["hd"] == 0).sum()), n_games_hd1=int((o["hd"] == 1).sum()), n_games_hd2plus=int((o["hd"] >= 2).sum()),
                  n_positions_hd0_this_side=int(npos[o["hd"] == 0].sum()), min_positions=int(npos.min()), double_flip_equals_source=True)
        np.savez(p, **o); Path(str(p) + ".json").write_text(json.dumps(st, indent=1)); print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
