"""Generates the synthetic golden fixtures (NOT official data): 50 random 19x19 games (5 per rank, ids are hashes of dummy strings),
the independent sgfmill replay golden (golden_board.json) and a regression snapshot of the L0 features (golden_l0_snapshot.json).

  python tests/tools/make_synthetic_games.py tests/fixtures/golden

Needs sgfmill (requirements.txt).  The snapshot is produced by this repo's own compute_l0, so it guards against regressions; the independent
check of the board engine is golden_board.json (sgfmill only).
"""
import csv, hashlib, json, random, sys
from pathlib import Path
from sgfmill import boards

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]

def coord(r_top, c):   # sgf: column letter then row letter (row 0 = top)
    return "abcdefghijklmnopqrs"[c] + "abcdefghijklmnopqrs"[r_top]

def random_game(rng, n_moves):
    b = boards.Board(19)
    moves, color = [], "b"
    star = [(3, 3), (3, 15), (15, 3), (15, 15), (3, 9), (9, 3), (9, 15), (15, 9), (9, 9)]
    for i in range(n_moves):
        for _ in range(200):
            if i < 8 and rng.random() < .7:
                r, c = rng.choice(star)
            else:
                r, c = rng.randrange(19), rng.randrange(19)
            if b.get(18 - r, c) is None:
                break
        else:
            break
        try:
            b.play(18 - r, c, color)
        except ValueError:
            continue
        moves.append((color, r, c)); color = "w" if color == "b" else "b"
    return "(;SZ[19]" + "".join(f";{col.upper()}[{coord(r, c)}]" for col, r, c in moves) + ")"

def make(path):
    rng = random.Random(20261126)
    rows = []
    for k in range(50):
        pid = hashlib.sha256(f"synthetic-player-{k}".encode()).hexdigest()[:16]
        rows.append(dict(player_id=pid, game_id=f"g_{1000000 + k * 7919:07d}", rank=RANKS[k // 5],
                         color="B" if k % 5 < 3 else "W", sgf_content=random_game(rng, rng.randrange(100, 321))))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["player_id", "game_id", "rank", "color", "sgf_content"]); w.writeheader(); w.writerows(rows)

def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    make(out / "games_50.csv")
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here.parents[1] / "src")]
    from golden_from_sgfmill import build_golden
    json.dump(build_golden(str(out / "games_50.csv")), open(out / "golden_board.json", "w"), sort_keys=True, separators=(",", ":"))
    import numpy as np, pandas as pd
    from gogame.features_l0 import compute_l0
    from gogame.sgf import parse_sgf
    df = pd.read_csv(out / "games_50.csv", dtype=str, keep_default_na=False)
    snap = {}
    for k in (0, 17, 3):
        color = 0 if df["color"][k] == "B" else 1
        r = compute_l0(parse_sgf(df["sgf_content"][k]), color)
        snap[str(k)] = dict(color=df["color"][k], dense=[None if np.isnan(x) else x for x in r.dense.tolist()], sp_idx=r.sp_idx.tolist(), sp_val=r.sp_val.tolist())
    json.dump(snap, open(out / "golden_l0_snapshot.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
