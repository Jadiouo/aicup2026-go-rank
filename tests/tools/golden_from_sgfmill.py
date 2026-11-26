"""Independent golden board generator: replays games with sgfmill ONLY (no gogame code involved).

Output per uid: n_plies, captures_per_ply (opponent stones removed), self_captures_per_ply,
final_black, final_white, final_grid_hash (sha256 of 361 bytes, row-major, row 0 = TOP row,
0 empty / 1 black / 2 white -- the same orientation as gogame.board.Board.grid).

Usage: python tests/tools/golden_from_sgfmill.py tests/fixtures/golden/games_50.csv tests/fixtures/golden/golden_board.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys


def replay_with_sgfmill(sgf_text: str) -> dict:
    from sgfmill import boards, sgf

    game = sgf.Sgf_game.from_bytes(sgf_text.encode("ascii"))
    size = game.get_size()
    board = boards.Board(size)
    caps, selfcaps = [], []

    def count(colour):
        return sum(1 for r in range(size) for c in range(size) if board.get(r, c) == colour)

    for node in game.get_main_sequence():
        colour, move = node.get_move()
        if colour is None:
            continue
        if move is None:
            caps.append(0)
            selfcaps.append(0)
            continue
        opp = "w" if colour == "b" else "b"
        opp_before, own_before = count(opp), count(colour)
        board.play(move[0], move[1], colour)   # raises ValueError on occupied points
        caps.append(opp_before - count(opp))
        selfcaps.append(own_before + 1 - count(colour))
    grid = bytearray()
    for top_row in range(size):
        for c in range(size):
            v = board.get(size - 1 - top_row, c)   # sgfmill row 0 is the bottom row
            grid.append(0 if v is None else (1 if v == "b" else 2))
    return {
        "n_plies": len(caps),
        "captures_per_ply": caps,
        "self_captures_per_ply": selfcaps,
        "final_black": sum(1 for v in grid if v == 1),
        "final_white": sum(1 for v in grid if v == 2),
        "final_grid_hash": hashlib.sha256(bytes(grid)).hexdigest(),
    }


def build_golden(csv_path: str) -> dict:
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = f"{row['rank']}_{row['game_id']}"
            out[uid] = replay_with_sgfmill(row["sgf_content"])
    return out


if __name__ == "__main__":
    data = build_golden(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, separators=(",", ":"))
    print(f"wrote {len(data)} games to {sys.argv[2]}")
