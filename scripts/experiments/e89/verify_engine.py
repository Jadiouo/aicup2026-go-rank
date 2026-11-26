"""E89 engine check (PLAN section 1): 1,000 random games (>=100 each of hd=0 / hd=1 / hd>=2, taken from s_games.npz) replayed by
the numba engine and by src/gogame/board.py; every ply's stones/captures and ko point must match, every own position's grid, previous
moves, ko and move must match, the own-liberty plane must equal planes.py layer 6 and the opponent-liberty plane liberty_map(g, 2).
Also times single-core throughput.   PYTHONPATH=src:. .venv/bin/python scripts/experiments/e89/verify_engine.py
Writes experiments/E89_full_data/engine_check.json"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "katago"), str(Path(__file__).resolve().parent)]
import numpy as np
import fastboard as fb
import kgio
from gogame.board import Board, iter_own_positions, liberty_map, replay
from gogame.planes import encode_position
from gogame.sgf import GameFlags, ParsedGame

assert {k: [(ord(s[1]) - 97) * 19 + ord(s[0]) - 97 for s in v] for k, v in fb.HANDICAP_STONES.items()} == \
       {k: [(ord(s[1]) - 97) * 19 + ord(s[0]) - 97 for s in v] for k, v in kgio.HANDICAP_STONES.items()}, "handicap table differs from kgio"
d = np.load(ROOT / ".scratch/e89/data/s_games.npz")
mvs, off, eff, hd = d["moves"], d["off"], d["eff"], d["hd"]
rng = np.random.default_rng(891)
picks = []
for name, mask in (("hd0", hd == 0), ("hd1", hd == 1), ("hd>=2", hd >= 2), ("any", hd >= 0)):
    pool = np.flatnonzero(mask)
    n = 300 if name != "any" else 100
    picks += rng.choice(pool, n, replace=False).tolist()
picks = sorted(set(picks))
while len(picks) < 1000:
    c = int(rng.integers(len(hd)))
    if c not in picks: picks.append(c)
picks = np.array(sorted(picks)[:1000])
print("games", len(picks), "hd0/1/>=2", int((hd[picks] == 0).sum()), int((hd[picks] == 1).sum()), int((hd[picks] >= 2).sum()), flush=True)
FL = GameFlags(False, False, False, 0, False, 0)
stats = dict(games=int(len(picks)), plies=0, own_positions=0, ply_state_mismatch=0, ko_mismatch=0, pos_grid_mismatch=0, pos_meta_mismatch=0,
             plane6_mismatch=0, opp_lib_mismatch=0, planes0_5_8_mismatch=0)
for g in picks:
    mv = mvs[off[g]:off[g + 1]]; e = int(eff[g]); h = int(hd[g])
    col = (mv >= 362).astype(np.int16); pos = mv - 362 * col
    arr = np.stack([col, np.where(pos == 361, -1, pos // 19), np.where(pos == 361, -1, pos % 19)], 1).astype(np.int16)
    setup = np.array([(0, p // 19, p % 19) for p in fb.HD_PTS[h].tolist()], np.int16).reshape(-1, 3)
    game = ParsedGame(arr, setup, FL)
    states, kos = fb.replay_states(mv, fb.HD_PTS[h], fb.NBR)
    for ply, b, res in replay(game):
        stats["plies"] += 1
        if not np.array_equal(b.grid.reshape(-1), states[ply]): stats["ply_state_mismatch"] += 1
        kp = -1 if b.ko_point is None else b.ko_point[0] * 19 + b.ko_point[1]
        if kp != kos[ply]: stats["ko_mismatch"] += 1
    sel = fb.valid_ordinals(mv, e)
    grid, lib, po, pw, ko, move, ply, oi = fb.encode(mv, e, h, sel)
    X = fb.build_planes_np(grid, lib, po, pw, ko, move, ply, oi, e)
    ref = {p.own_idx: p for p in iter_own_positions(game, e)}
    for k in range(len(sel)):
        p = ref[int(oi[k])]; stats["own_positions"] += 1
        if not np.array_equal(p.grid_own.reshape(-1), grid[k]): stats["pos_grid_mismatch"] += 1
        f = lambda t: -1 if t is None else t[0] * 19 + t[1]
        if (f(p.prev_opp), f(p.prev_own), f(p.ko_point), f(p.move), p.ply) != (po[k], pw[k], ko[k], move[k], ply[k]): stats["pos_meta_mismatch"] += 1
        R = encode_position(p, include_move=True).astype(np.float32)
        l6 = (X[k, 3] * 1 + X[k, 4] * 2 + X[k, 5] * 3 + X[k, 6] * 4)
        if not np.array_equal(R[6], l6): stats["plane6_mismatch"] += 1
        lo = np.clip(liberty_map(p.grid_own, 2), 0, 4).astype(np.float32)
        if not np.array_equal(lo, X[k, 7] * 1 + X[k, 8] * 2 + X[k, 9] * 3 + X[k, 10] * 4): stats["opp_lib_mismatch"] += 1
        if not (np.array_equal(R[0], X[k, 0]) and np.array_equal(R[1], X[k, 1]) and np.array_equal(R[2], X[k, 2]) and np.array_equal(R[3], X[k, 11])
                and np.array_equal(R[4], X[k, 12]) and np.array_equal(R[5], X[k, 13]) and np.array_equal(R[7], X[k, 15]) and np.array_equal(R[8], X[k, 14])):
            stats["planes0_5_8_mismatch"] += 1
stats["pass"] = all(stats[k] == 0 for k in stats if k.endswith("mismatch"))
# throughput, single core, one thread
sample = rng.choice(len(hd), 2000, replace=False)
fb.encode(mvs[off[sample[0]]:off[sample[0] + 1]], int(eff[sample[0]]), int(hd[sample[0]]), np.array([0], np.int32))   # compile
t = time.time(); npos = 0
for g in sample:
    mv = mvs[off[g]:off[g + 1]]; vo = fb.valid_ordinals(mv, int(eff[g])); s = np.sort(rng.choice(vo, min(16, len(vo)), replace=False)); fb.encode(mv, int(eff[g]), int(hd[g]), s); npos += len(s)
dt = time.time() - t; stats["speed_train_style_16_per_game_positions_per_s"] = round(npos / dt); stats["speed_games_per_s"] = round(len(sample) / dt)
t = time.time(); npos = 0
for g in sample[:500]:
    mv = mvs[off[g]:off[g + 1]]; vo = fb.valid_ordinals(mv, int(eff[g])); fb.encode(mv, int(eff[g]), int(hd[g]), vo); npos += len(vo)
stats["speed_all_positions_per_s"] = round(npos / (time.time() - t))
print(json.dumps(stats, indent=1))
(ROOT / "experiments/E89_full_data/engine_check.json").write_text(json.dumps(stats, indent=1))
sys.exit(0 if stats["pass"] else 1)
