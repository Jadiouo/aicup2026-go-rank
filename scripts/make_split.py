"""Build the official player-grouped split + hold-out H, validation questions and a report.

  .venv/bin/python scripts/make_split.py [--data-dir data/training] [--out-dir data/splits] [--folds 5]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gogame.constants import RANKS  # noqa: E402
from gogame.io import cross_rank_players, read_train_csv  # noqa: E402
from gogame.questions import build_player_index, build_val_questions, to_questions_frame  # noqa: E402
from gogame.split import assert_no_player_leak, make_split_table, read_split, write_split  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "training"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-holdout", type=int, default=640)
    ap.add_argument("--min-black", type=int, default=120)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = [read_train_csv(Path(a.data_dir) / f"train_{r}.csv", usecols=["player_id", "game_id", "rank", "color"]) for r in RANKS]
    cross = cross_rank_players(frames)
    if cross:
        raise SystemExit(f"{len(cross)} players appear under several ranks, e.g. {sorted(cross)[:3]}")
    meta = pd.concat([f[["uid", "player_id", "rank_id", "color_id"]] for f in frames], ignore_index=True)
    pm = meta.groupby("player_id").agg(rank_id=("rank_id", "first"), n_black=("color_id", lambda s: int((s == 0).sum()))).reset_index()

    tab = make_split_table(pm, n_folds=a.folds, min_black=a.min_black, n_holdout=a.n_holdout)
    write_split(tab, out / "split.parquet")
    tab = read_split(out / "split.parquet")                     # re-read: validates the file that was written

    groups = {f"fold{k}": set(tab.loc[tab["fold"] == k, "player_id"]) for k in range(a.folds)}
    groups["H"] = set(tab.loc[tab["is_holdout"], "player_id"])
    for (na, sa), (nb, sb) in itertools.combinations(groups.items(), 2):
        assert_no_player_leak(sa, sb)                           # raises on any shared player

    index = build_player_index(meta)
    rank_name = dict(enumerate(RANKS))
    report: dict = {"n_players": len(tab), "n_holdout": len(groups["H"]), "n_folds": a.folds, "groups": {}, "questions": {}}
    allq = []
    for name, players in groups.items():
        sub = tab[tab["player_id"].isin(players)]
        qs = build_val_questions(index, players)
        to_questions_frame(qs).to_parquet(out / f"questions_{name}.parquet", index=False)
        allq += qs
        qdf = pd.DataFrame({"n": [q.n for q in qs], "rank_id": [q.rank_id for q in qs]})
        report["groups"][name] = {
            "n_players": len(sub),
            "players_per_rank": {rank_name[r]: int(c) for r, c in sub.groupby("rank_id").size().items()},
            "n_questions": len(qs),
            "players_per_n": {int(n): int(c) for n, c in qdf.groupby("n").size().items()},
        }
    qall = pd.DataFrame({"n": [q.n for q in allq], "rank_id": [q.rank_id for q in allq]})
    report["questions"]["players_per_n_by_rank"] = {
        int(n): {rank_name[r]: int(c) for r, c in g.groupby("rank_id").size().items()} for n, g in qall.groupby("n")}
    h = tab[tab["is_holdout"]]
    report["holdout"] = {"n_ge_min_black": int((h["n_black"] >= a.min_black).sum()), "min_n_black": int(h["n_black"].min()),
                         "n_heavy_candidates": int((tab["n_black"] >= a.min_black).sum())}
    report["disjoint_checked_pairs"] = len(list(itertools.combinations(groups, 2)))
    (out / "split_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
