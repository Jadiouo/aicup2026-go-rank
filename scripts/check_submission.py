"""Check a submission CSV against the contest test files. Exit 0 if valid, 1 otherwise.

  .venv/bin/python scripts/check_submission.py --task rank --data-dir data/contest submission_rank.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gogame.io import normalize_qid, read_candidates_csv, read_questions_csv  # noqa: E402
from gogame.submission import check_player_submission, check_rank_submission  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--task", choices=["rank", "player"], required=True)
    ap.add_argument("--data-dir", required=True, help="directory holding the official test / candidate CSVs")
    ap.add_argument("--expected-rows", type=int, default=None)
    a = ap.parse_args(argv)
    d = Path(a.data_dir)
    stem = "rank_prediction_test" if a.task == "rank" else "player_identification_test"
    pub = [normalize_qid(q, "pub") for q in read_questions_csv(d / f"{stem}_public.csv", "pub")["question_id"]]
    prv = [normalize_qid(q, "priv") for q in read_questions_csv(d / f"{stem}_private.csv", "priv")["question_id"]]
    if a.task == "rank":
        issues = check_rank_submission(a.file, pub, prv, a.expected_rows)
    else:
        cp = set(read_candidates_csv(d / "player_identification_candidates_public.csv", "pub")["player_id"])
        cv = set(read_candidates_csv(d / "player_identification_candidates_private.csv", "priv")["player_id"])
        issues = check_player_submission(a.file, pub, prv, cp, cv, a.expected_rows)
    for i in issues:
        print(f"{i.code}\trow={i.row}\t{i.detail}")
    print("OK" if not issues else f"FAILED: {sorted({i.code for i in issues})}")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
