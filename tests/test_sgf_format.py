"""Format hardening (FAQ Q13/Q14, 2026-10-29, not final): missing SZ, HD handicap games, color_semantics, format_report."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "katago"))
sys.path.insert(0, str(ROOT / "scripts"))

import kgio  # noqa: E402
from gogame.board import iter_own_positions  # noqa: E402
from gogame.errors import DataFormatError  # noqa: E402
from gogame.sgf import (HANDICAP_STONES, coord_to_rc, format_gate, format_report, normalize_sgf,  # noqa: E402
                        parse_sgf, root_props)

Q13 = "(;CA[UTF-8]HD[2]RE[W+R];B[pp];W[dd];B[qp];W[cc])"
STAR = {2: "pd dp", 3: "pd dp dd", 4: "pd dp dd pp", 5: "pd dp dd pp jj", 6: "pd dp dd pp dj pj", 7: "pd dp dd pp dj pj jj",
        8: "pd dp dd pp dj pj jd jp", 9: "pd dp dd pp dj pj jd jp jj"}      # transcribed from FAQ Q13


def test_q13_example_becomes_ha2_ab_pl_w():
    ng = normalize_sgf(Q13, 0, "as_annotated")
    assert ng.hd == 2 and ng.komi == 0.0 and ng.skip is None
    assert sorted(map(tuple, ng.parsed.setup.tolist())) == sorted((0, *coord_to_rc(x)) for x in ("pd", "dp"))
    assert ng.parsed.moves[:, 0].tolist() == [1, 0, 1, 0]          # W[pp];B[dd];W[pd];B[dp]  (PL[W])
    assert ng.parsed.moves[0].tolist() == [1, *coord_to_rc("pp")]
    k = kgio.normalize_game(Q13, 0, "as_annotated")
    assert k.initial_player == "W" and k.komi == 0.0 and k.moves[0] == ("W", coord_to_rc("pp")) and k.moves[1][0] == "B"
    assert [c for c, _ in k.initial_stones] == ["B", "B"]


@pytest.mark.parametrize("n", range(2, 10))
def test_star_points(n):
    assert " ".join(HANDICAP_STONES[n]) == STAR[n]
    assert " ".join(kgio.HANDICAP_STONES[n]) == STAR[n]
    sgf = f"(;CA[UTF-8]HD[{n}]RE[W+0.5];B[pp];W[dd])"
    ng = normalize_sgf(sgf, 1)
    assert ng.parsed.setup.shape == (n, 3) and (ng.parsed.setup[:, 0] == 0).all()
    k = kgio.normalize_game(sgf, 1)
    assert len(k.initial_stones) == n
    assert sorted((r, c) for _, r, c in ng.parsed.setup.tolist()) == sorted(rc for _, rc in k.initial_stones)


def test_hd1_komi0_no_swap_no_stones():
    sgf = "(;CA[UTF-8]HD[1]RE[W+0.5];B[pp];W[dd])"
    ng = normalize_sgf(sgf, 0)
    assert ng.komi == 0.0 and ng.parsed.setup.shape == (0, 3) and ng.parsed.moves[:, 0].tolist() == [0, 1] and ng.eff_color == 0
    k = kgio.normalize_game(sgf, 0)
    assert k.komi == 0.0 and not k.initial_stones and k.moves[0][0] == "B" and k.eff_color == 0


def test_hd0_and_missing_hd_like_before():
    for sgf in ("(;SZ[19];B[pd];W[dp])", "(;CA[UTF-8]HD[0]RE[B+R];B[pd];W[dp])", "(;RE[B+R];B[pd];W[dp])"):
        ng, k = normalize_sgf(sgf, 0), kgio.normalize_game(sgf, 0)
        assert ng.komi == 6.5 and k.komi is None and ng.skip is None and k.skip is None
        assert ng.parsed.moves.tolist() == parse_sgf(sgf).moves.tolist() and k.moves == kgio.parse_sgf_moves(sgf)
        assert ng.parsed.setup.shape == (0, 3) and not k.initial_stones


def test_missing_sz_ok_other_sz_and_setup_skipped_with_reason():
    assert normalize_sgf("(;CA[UTF-8]RE[B+1.5];B[pd])", 0).skip is None
    assert kgio.normalize_game("(;CA[UTF-8]RE[B+1.5];B[pd])", 0).skip is None
    for sgf, why in (("(;SZ[13];B[dd])", "bad_size"), ("(;SZ[19]AB[dd];W[pp])", "has_setup"), ("(;HD[x];B[pd])", "bad_hd"),
                     ("(;HD[12];B[pd])", "bad_hd"), ("", "empty_sgf"), ("garbage", "no_nodes")):
        assert normalize_sgf(sgf, 0).skip == why == kgio.normalize_game(sgf, 0).skip, sgf
    assert normalize_sgf("(;SZ[19:19];B[pd])", 0).skip is None


def test_color_semantics_both():
    for cs, eff_for_black in (("as_annotated", 1), ("actual", 0)):
        ng, k = normalize_sgf(Q13, 0, cs), kgio.normalize_game(Q13, 0, cs)
        assert ng.eff_color == k.eff_color == eff_for_black
        ng, k = normalize_sgf(Q13, 1, cs), kgio.normalize_game(Q13, 1, cs)
        assert ng.eff_color == k.eff_color == 1 - eff_for_black
    # even game: never changes
    for cs in ("as_annotated", "actual"):
        assert normalize_sgf("(;SZ[19];B[pd];W[dp])", 1, cs).eff_color == 1
    with pytest.raises(ValueError):
        normalize_sgf(Q13, 0, "nonsense")


def test_color_semantics_env_default(monkeypatch):
    monkeypatch.delenv("GOGAME_COLOR_SEMANTICS", raising=False)
    assert normalize_sgf(Q13, 0).eff_color == 1                      # default as_annotated
    monkeypatch.setenv("GOGAME_COLOR_SEMANTICS", "actual")
    assert normalize_sgf(Q13, 0).eff_color == 0 and kgio.normalize_game(Q13, 0).eff_color == 0


def test_replay_starts_from_handicap_stones():
    ng = normalize_sgf(Q13, 0, "as_annotated")                       # annotated B == real white
    pos = list(iter_own_positions(ng.parsed, ng.eff_color))
    assert len(pos) == 2 and pos[0].move == coord_to_rc("pp")
    g = pos[0].grid_own                                              # own = white; the 2 handicap stones are the enemy (2)
    assert int((g == 2).sum()) == 2 and int((g == 1).sum()) == 0
    assert g[coord_to_rc("pd")] == 2 and g[coord_to_rc("dp")] == 2


def test_mk_query_extra_fields_only_for_handicap():
    mv = [("B", (3, 3))]
    plain = kgio.mk_query("a", mv, [0], 1, 6.5, "japanese", profile="rank_1d", policy=True, nsym=1)
    assert "initialStones" not in plain and "initialPlayer" not in plain and plain["komi"] == 6.5
    k = kgio.normalize_game(Q13, 0)
    q = kgio.mk_query("b", k.moves, [0], 1, k.komi, "japanese", initial_stones=k.initial_stones, initial_player=k.initial_player)
    assert q["komi"] == 0.0 and q["initialPlayer"] == "W" and q["initialStones"] == [["B", "Q16"], ["B", "D4"]]
    assert q["moves"][0] == ["W", "Q4"]


def test_runner_process_chunk_handicap_and_semantics(monkeypatch):
    import pandas as pd
    import scan_runner
    pins = json.loads((ROOT / "scripts/katago/scan_pins.json").read_text())

    class Cap:
        def __init__(self): self.qs = []
        def run(self, qs):
            self.qs += qs
            return [{"id": q["id"], "turnNumber": t, "humanPolicy": [1 / 362] * 362} for q in qs for t in q["analyzeTurns"]], 0.0
    for cs, first_own_ply in (("as_annotated", 0), ("actual", 1)):
        monkeypatch.setenv("GOGAME_COLOR_SEMANTICS", cs)
        cap = Cap()
        sub = pd.DataFrame({"uid": ["h", "e", "z"], "color": [0, 0, 0],
                            "sgf": [Q13, "(;SZ[19];B[pd];W[dp])", "(;SZ[9];B[dd])"]})
        df, n_eval, _, skipped = scan_runner.process_chunk(cap, sub, pins)
        assert skipped == [("z", "bad_size")]
        qh = [q for q in cap.qs if q["id"].startswith("h|")][0]
        qe = [q for q in cap.qs if q["id"].startswith("e|")][0]
        assert qh["komi"] == 0.0 and "initialStones" in qh and qh["analyzeTurns"][0] == first_own_ply
        assert qe["komi"] == pins["query"]["komi"] and "initialStones" not in qe


def test_format_report_counts_and_warnings():
    games = ["(;SZ[19];B[pd];W[dp])"] * 97 + [Q13, "(;CA[UTF-8]HD[1]RE[W+0.5];B[pd];W[dp])", "(;CA[UTF-8]RE[B+R];B[pd];W[dp])"]
    rep = format_report(games, [0] * 100)
    assert rep["n"] == 100 and rep["SZ"] == {"19": 97, "absent": 3} and rep["HD"]["2"] == 1 and rep["HD"]["1"] == 1
    assert rep["n_skipped"] == 0 and rep["n_handicap_ge2"] == 1 and rep["RE"]["B+"] == 1 and rep["RE"]["W+"] == 2
    assert not [w for w in rep["warnings"] if w[0] == "WARN"]
    bad = format_report(games + ["(;SZ[19]KM[6.5]PB[x];B[pd])", "(;SZ[13];B[dd])", "(;SZ[19]AB[dd];W[pp])"], [0] * 103)
    assert bad["other_props"] == {"KM": 1, "PB": 1} and bad["skip_reasons"] == {"bad_size": 1, "has_setup": 1}
    assert bad["n_AB"] == 1 and any(w[0] == "WARN" for w in bad["warnings"])
    format_gate(bad, False, log=lambda *_: None)                      # warns only
    with pytest.raises(DataFormatError):
        format_gate(bad, True, log=lambda *_: None)
    rep = format_report(["(;SZ[19];B[pd])", "", "junk", "(;SZ[19])"], [0, 0, 0, 0])
    assert rep["unparseable"] == 2 and rep["skip_reasons"] == {"empty_sgf": 1, "no_nodes": 1, "no_own_moves": 1}
    assert rep["skip_frac"] > 0.01 and any("skip rate" in m for _, m in rep["warnings"])


def test_root_props_ignores_moves_and_escapes():
    assert root_props("(;CA[UTF-8]HD[2]RE[W+R];B[pp])") == {"CA": ["UTF-8"], "HD": ["2"], "RE": ["W+R"]}
    assert root_props("(;SZ[19];HD[3];B[pp])") == {"SZ": ["19"]}
    t = "(;C[a\\];x]SZ[19];B[pp])"           # escaped ']' and ';' inside a value stay inside it
    assert kgio.root_props(t) == root_props(t) == {"C": ["a\\];x"], "SZ": ["19"]}


def test_expected_own_moves_matches_runner():
    import predict_hsl as ph
    assert ph.expected_own_moves(Q13, 0) == 2 == ph.expected_own_moves(Q13, 1)
    assert ph.expected_own_moves("(;SZ[13];B[dd])", 0) == 0
    assert ph.predicted_skip_reason("(;SZ[13];B[dd])", 0) == "bad_size" and ph.predicted_skip_reason("(;SZ[19];B[dd])", 1) == "no_own_moves"


SNAP = ROOT / ".scratch" / "fmt_regress" / "snap_before.json"


@pytest.mark.skipif(not SNAP.exists(), reason="pre-change baseline snapshot (.scratch/fmt_regress) not present")
def test_current_format_bitwise_identical_to_baseline(tmp_path):
    """200 real training games + the 1104 fake test games: parse / L0 features / KataGo query JSON hashes == pre-change baseline."""
    r = subprocess.run([sys.executable, str(ROOT / ".scratch/fmt_regress/snap.py"), "pytest"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    a = json.loads(SNAP.read_text())
    b = json.loads((ROOT / ".scratch/fmt_regress/snap_pytest.json").read_text())
    for k in ("uids", "parse", "l0_dense", "l0_sparse", "query_ids", "query_json", "moves_df", "n_eval", "expected_own"):
        assert a[k] == b[k], k
    assert [u for u, _ in a["skipped"]] == [u for u, _ in b["skipped"]]


# ----------------------------------------------------------------------------- illegal moves / engine errors / semantics cross-check
import io  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402


class _Stdin:
    def __init__(self, q): self.q = q
    def write(self, line):
        qr = json.loads(line)
        if "bad" in qr["id"]:
            self.q.put(json.dumps({"id": qr["id"], "error": "Illegal move"}) + "\n")      # ONE line for the whole query
        else:
            for t in qr["analyzeTurns"]:
                self.q.put(json.dumps({"id": qr["id"], "turnNumber": t, "humanPolicy": [1 / 362] * 362}) + "\n")
    def flush(self): pass


class _Stdout:
    def __init__(self, q): self.q = q
    def readline(self):
        try:
            return self.q.get(timeout=5)
        except queue.Empty:
            return ""                                                                      # = engine died -> test fails, never hangs


def _fake_engine():
    q = queue.Queue()
    e = kgio.Engine.__new__(kgio.Engine)
    e.p = type("P", (), {"stdin": _Stdin(q), "stdout": _Stdout(q)})()
    return e


def test_engine_run_error_does_not_hang_and_good_query_is_intact():
    e = _fake_engine()
    qs = [kgio.mk_query("bad|p", [("B", (3, 3))], [0, 1, 2], 1), kgio.mk_query("ok|p", [("B", (3, 3)), ("W", (15, 15))], [0, 1], 1)]
    res, _ = e.run(qs)
    assert [(r["id"], r["turnNumber"]) for r in res] == [("ok|p", 0), ("ok|p", 1)] and set(e.last_errors) == {"bad|p"}


def test_engine_error_without_known_id_raises():
    e = _fake_engine()
    e.p.stdin.q.put(json.dumps({"error": "boom"}) + "\n")
    with pytest.raises(RuntimeError):
        e.run([kgio.mk_query("ok|p", [("B", (3, 3))], [0], 1)])


def test_first_illegal_occupied_suicide_capture():
    assert kgio.first_illegal([("B", (3, 3)), ("W", (3, 3))]) == 1                     # occupied
    # suicide: W at corner (0,0) surrounded by B at (0,1),(1,0)
    assert kgio.first_illegal([("B", (0, 1)), ("B", (1, 0)), ("W", (0, 0))]) == 2
    # capture first makes it legal: W (0,1) captured by B (0,2)/(1,1)... then B plays (0,0)? simple capture then replay on freed point
    assert kgio.first_illegal([("B", (0, 1)), ("W", (0, 0)), ("B", (1, 0)), ("W", (5, 5)), ("B", (0, 0))]) is None
    assert kgio.first_illegal([("B", (3, 3))], stones=[("B", (3, 3))]) == 0            # handicap stone occupies the point
    assert kgio.first_illegal([("B", None), ("W", (2, 2))]) is None


def test_truncate_at_first_illegal_and_runner_survives(monkeypatch):
    import pandas as pd
    import scan_runner
    pins = json.loads((ROOT / "scripts/katago/scan_pins.json").read_text())
    illegal = "(;SZ[19];B[dd];W[pp];B[dd];W[dp];B[pd])"           # 3rd move re-plays dd
    ng = kgio.normalize_game(illegal, 0)
    assert ng.truncated and len(ng.moves) == 2 and ng.skip is None
    hd = "(;CA[UTF-8]HD[2]RE[W+R];B[pp];W[pd])"                   # annotated W[pd] -> actual B on a handicap stone
    nh = kgio.normalize_game(hd, 0)
    assert nh.truncated and len(nh.moves) == 1

    class Cap:                                                     # engine that rejects one profile query of "g2" like KataGo would
        last_errors = {}
        def run(self, qs):
            self.last_errors = {q["id"]: "x" for q in qs if q["id"].startswith("g2|rank_1d")}
            return [{"id": q["id"], "turnNumber": t, "humanPolicy": [1 / 362] * 362} for q in qs if q["id"] not in self.last_errors
                    for t in q["analyzeTurns"]], 0.0
    sub = pd.DataFrame({"uid": ["g1", "g2", "g3"], "color": [0, 0, 0],
                        "sgf": [illegal, "(;SZ[19];B[dd];W[pp])", "(;SZ[19];B[dd];W[pp];B[pd])"]})
    df, n_eval, _, skipped = scan_runner.process_chunk(Cap(), sub, pins)
    assert skipped == [("g2", "engine_error")]
    assert set(df.uid) == {"g1", "g3"} and (df[df.uid == "g1"].own_move_idx.tolist() == [0])


def test_color_semantics_cross_check(tmp_path, monkeypatch):
    import predict
    for k, (theirs, mine, ok) in enumerate([("as_annotated", "as_annotated", True), ("actual", "as_annotated", False),
                                              ("as_annotated", "actual", False)]):
        d = tmp_path / str(k); d.mkdir()
        (d / "features_manifest.json").write_text(json.dumps({"color_semantics": theirs}))
        monkeypatch.setenv("GOGAME_COLOR_SEMANTICS", mine)
        if ok:
            assert predict.check_hsl_color_semantics(d / "h.parquet") == theirs
        else:
            with pytest.raises(DataFormatError, match="color_semantics mismatch"):
                predict.check_hsl_color_semantics(d / "h.parquet")
    d = tmp_path / "pm"; d.mkdir()
    (d / "prepare_meta.json").write_text(json.dumps({"color_semantics": "actual"}))
    monkeypatch.setenv("GOGAME_COLOR_SEMANTICS", "as_annotated")
    with pytest.raises(DataFormatError):
        predict.check_hsl_color_semantics(d / "h.parquet")
    d = tmp_path / "none"; d.mkdir()
    msgs = []
    assert predict.check_hsl_color_semantics(d / "h.parquet", log=msgs.append) is None and msgs
