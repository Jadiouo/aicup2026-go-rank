"""Test-day strength-estimator (SE) extraction for the optional Rank variant `hsl_lgbm_se` (E21).  NOT part of the default flow.

  extract      --data-dir DIR [--out-dir data/test_scan] [--model se_go.pt] [--device cuda|cpu] [--half]
                 official rank_prediction_test_{public,private}.csv -> <out-dir>/se_test_games.parquet (uid, se_mean, se_std, se_n)
                 + se_manifest.json (color_semantics, model sha256, settings).  One row per test game; a game that cannot be
                 scored (HD>=2 with setup stones, parse error) gets NaN / se_n 0, never a missing row.
  consistency  --se-table .scratch/e21/se_uid.parquet [--n 30] [--device cpu] [--repo-root DIR]
                 re-scores training games with THIS extractor and compares with the table that Dse was trained on (max |diff| <= 0.01).

Settings are those of scripts/experiments/e21/extract.py (the training table): 32 own-colour positions evenly spread from first to last
own move, board BEFORE the move, rotation 0, `score` head, per-game mean and std (ddof 0) in float64, fp16 on GPU (use --half on cuda
to match the table; the E21 fp16-vs-fp32 check gave max |diff| 0.0115 per game, so fp32/CPU is within the 0.01-0.02 band, not identical).
Test uid = '<pub|priv>_<qid>#<slot>' (predict.py explode_question), the key hsl_lgbm_se joins on.

torch is not in .venv: put a torch install on PYTHONPATH, or set GOGAME_TORCH_PATH=<dir containing torch> (appended to sys.path).
Use a GPU (Colab or any CUDA machine) for speed; CPU works but is slow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "experiments" / "e21"))
if os.environ.get("GOGAME_TORCH_PATH"):
    sys.path.append(os.environ["GOGAME_TORCH_PATH"])

import numpy as np
import pandas as pd

from gogame.se_feats import SE_N_POSITIONS, is_handicap

DEFAULT_OUT = ROOT / "data" / "test_scan"
DEFAULT_MODEL = ROOT / "data" / "external" / "se" / "extracted" / "se_go.pt"
RANK_FILES = ("rank_pub", "rank_priv")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def collect_games(data_dir: Path) -> pd.DataFrame:
    """uid, color (0=B,1=W), sgf of every Rank test game (same order and uid as predict_hsl.collect_test_games)."""
    import predict as P
    rows = []
    for key in RANK_FILES:
        path = Path(data_dir) / P.FILES[key]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        for q in P.load_questions(path, key.split("_")[1], "rank"):
            rows += [(g.uid, int(g.color), g.sgf) for g in q.games]
    df = pd.DataFrame(rows, columns=["uid", "color", "sgf"])
    if df["uid"].duplicated().any():
        raise ValueError("duplicate test game uid")
    return df


# ------------------------------------------------------------------------------------------------------ scoring
def score_games(model, df: pd.DataFrame, device: str = "cpu", half: bool = False, n_max: int = SE_N_POSITIONS, workers: int = 0,
                gb: int = 48, sub: int = 512):
    """(mean, std, n) float64/float64/int arrays aligned with df[uid,color,sgf].  Mirrors extract.score_df (same encoder call, same
    sub-batching, float64 mean and ddof-0 std per game); only the device is a parameter.  A game whose encoding raises -> NaN, n 0."""
    import torch
    import se_encode as se

    class DS(torch.utils.data.Dataset):
        def __init__(self):
            self.col, self.sgf = df["color"].astype(int).tolist(), df["sgf"].tolist()

        def __len__(self):
            return len(self.col)

        def __getitem__(self, i):
            try:
                if is_handicap(self.sgf[i]):
                    raise ValueError("HD>=2: no SE (not in the E21 training range)")
                x = se.encode_game(self.sgf[i], self.col[i], "pre", n_max)
            except Exception:
                x = np.zeros((0, 18, 19, 19), np.float32)
            return i, torch.from_numpy(x)

    def collate(b):
        return [i for i, _ in b], [len(x) for _, x in b], torch.cat([x for _, x in b])

    dl = torch.utils.data.DataLoader(DS(), batch_size=gb, num_workers=workers, collate_fn=collate,
                                     prefetch_factor=4 if workers else None)
    mean = np.full(len(df), np.nan)
    std = np.full(len(df), np.nan)
    cnt = np.zeros(len(df), int)
    with torch.inference_mode():
        for idx, ns, x in dl:
            outs = []
            for k in range(0, len(x), sub):
                xb = x[k:k + sub].to(device)
                if half:
                    xb = xb.half()
                outs.append(model(xb)["score"].float().squeeze(1).cpu())
            s = torch.cat(outs).numpy().astype(np.float64) if outs else np.zeros(0)
            o = 0
            for i, n in zip(idx, ns):
                if n > 0:
                    v = s[o:o + n]
                    mean[i], std[i], cnt[i] = v.mean(), v.std(), n
                o += n
    return mean, std, cnt


def load_model(path, device: str, half: bool):
    import torch
    m = torch.jit.load(str(path), map_location=device).eval()
    return m.half() if (half and device != "cpu") else m


def _setup(a):
    import torch
    torch.set_num_threads(max(1, a.threads))
    if a.half and a.device == "cpu":
        print("WARNING: --half has no effect on cpu (fp32 is used); the E21 table was made with fp16 on GPU", file=sys.stderr)
    return load_model(a.model, a.device, a.half)


# ------------------------------------------------------------------------------------------------------ extract
def cmd_extract(a) -> int:
    from gogame.sgf import get_color_semantics
    import torch
    if a.games_parquet:  # prebuilt (uid, color, sgf) table, made locally by collect_games (a GPU host without lightgbm); no input csv hashes then
        df = pd.read_parquet(a.games_parquet)[["uid", "color", "sgf"]]
    else:
        df = collect_games(Path(a.data_dir))
    if a.limit:
        df = df.iloc[:a.limit].reset_index(drop=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = _setup(a)
    t0 = time.time()
    m, s, n = score_games(model, df, a.device, a.half, SE_N_POSITIONS, a.workers)
    res = pd.DataFrame({"uid": df["uid"].values, "se_mean": m, "se_std": s, "se_n": n.astype(np.int16)})
    p = out / "se_test_games.parquet"
    res.to_parquet(str(p) + ".tmp", index=False)
    os.replace(str(p) + ".tmp", p)
    from gogame.sgf import parse_sgf
    hd2 = np.array([is_handicap(sg) for sg in df["sgf"]], bool)
    isnan = res["se_mean"].isna().to_numpy()
    n_setup = 0
    for sg in df["sgf"][isnan]:
        try:
            n_setup += bool(parse_sgf(sg).has_setup)
        except Exception:
            pass
    n_bad = int((isnan & ~hd2).sum())
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "color_semantics": get_color_semantics(), "n_games": int(len(df)),
           "n_nan": int(res["se_mean"].isna().sum()), "n_nan_with_setup_stones": int(n_setup), "n_hd_ge2": int(hd2.sum()),
           "n_se_n_lt32": int((res["se_n"] < SE_N_POSITIONS).sum()), "n_nan_not_hd_ge2": n_bad, "n_positions": SE_N_POSITIONS,
           "device": a.device, "half": bool(a.half and a.device != "cpu"), "model": str(a.model), "model_sha256": sha256_file(a.model),
           "torch": torch.__version__, "seconds": round(time.time() - t0), "se_sha256": sha256_file(p),
           "inputs": (None if a.games_parquet else {k: sha256_file(Path(a.data_dir) / __import__("predict").FILES[k]) for k in RANK_FILES}),
           "games_parquet_sha256": sha256_file(a.games_parquet) if a.games_parquet else None}
    (out / "se_manifest.json").write_text(json.dumps(man, indent=1))
    if len(df) and n_bad / len(df) > a.max_bad_nan_frac:
        print(f"ERROR: {n_bad} of {len(df)} games ({n_bad / len(df):.2%}) are NaN although not HD>=2 (> {a.max_bad_nan_frac:.2%}); "
              f"extraction is broken (manifest written for diagnosis)", file=sys.stderr)
        rc_bad = 1
    else:
        rc_bad = 0
    print(f"wrote {p}: {man['n_games']} games, {man['n_nan']} NaN ({n_setup} with setup stones), {man['seconds']}s, "
          f"color_semantics={man['color_semantics']}, HD>=2 {man['n_hd_ge2']}, NaN-not-HD>=2 {n_bad}")
    return rc_bad


# ------------------------------------------------------------------------------------------------------ consistency
def load_training_games(uids, repo_root: Path) -> pd.DataFrame:
    """uid, color (from the csv `color` column, as extract.py), sgf for training uids '<rank>_<game_id>'."""
    by = {}
    for u in uids:
        r, g = u.split("_", 1)
        by.setdefault(r, set()).add(g)
    parts = []
    for r, ids in by.items():
        f = repo_root / "data" / "training_v1.1" / "training" / f"train_{r}.csv"
        for ch in pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["game_id", "color", "sgf_content"], chunksize=20000):
            ch = ch[ch["game_id"].isin(ids)]
            if len(ch):
                parts.append(ch.assign(uid=r + "_" + ch["game_id"]))
    d = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"uid": d["uid"], "color": (d["color"] == "W").astype(int), "sgf": d["sgf_content"]})


def cmd_consistency(a) -> int:
    repo = Path(a.repo_root)
    tab = pd.read_parquet(a.se_table)
    sp = pd.read_parquet(repo / "data" / "splits" / "split.parquet")
    meta = pd.read_parquet(repo / "data" / "cache" / "games_meta.parquet", columns=["uid", "player_id", "color"])
    H = set(sp.loc[sp["is_holdout"], "player_id"])
    pl = meta.set_index("uid")["player_id"]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(tab["uid"].to_numpy(), size=min(a.n, len(tab)), replace=False)
    assert not (set(pl.reindex(pick).dropna()) & H), "sample contains H players"
    df = load_training_games(list(pick), repo)
    mc = meta.set_index("uid")["color"].reindex(df["uid"]).to_numpy()
    assert (mc == df["color"].to_numpy()).all(), "csv color != games_meta color"
    model = _setup(a)
    t0 = time.time()
    m, s, n = score_games(model, df, a.device, a.half, SE_N_POSITIONS, a.workers)
    ref = tab.set_index("uid").loc[df["uid"]]
    dm = np.abs(m - ref["se_mean"].to_numpy())
    ds = np.abs(s - ref["se_std"].to_numpy())
    rep = {"n_games": int(len(df)), "device": a.device, "half": bool(a.half and a.device != "cpu"), "max_abs_diff_se_mean": float(np.nanmax(dm)),
           "mean_abs_diff_se_mean": float(np.nanmean(dm)), "max_abs_diff_se_std": float(np.nanmax(ds)), "n_nan_new": int(np.isnan(m).sum()),
           "n_differs_se_n": int((n != ref["se_n"].to_numpy()).sum()), "max_allowed": a.max_diff, "seconds": round(time.time() - t0)}
    rep["ok"] = bool(rep["max_abs_diff_se_mean"] <= a.max_diff and rep["n_nan_new"] == 0 and rep["n_differs_se_n"] == 0)
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "consistency"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=str(DEFAULT_MODEL))
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        p.add_argument("--half", action="store_true", help="fp16 (cuda only); the training table was made with fp16")
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--color-semantics", choices=["as_annotated", "actual"], default=None)
        if name == "extract":
            p.add_argument("--data-dir", default=None)
            p.add_argument("--games-parquet", default=None, help="prebuilt uid,color,sgf table instead of --data-dir (used on a remote GPU host)")
            p.add_argument("--out-dir", default=str(DEFAULT_OUT))
            p.add_argument("--max-bad-nan-frac", type=float, default=0.01, help="error if more than this fraction of games is NaN although not HD>=2")
            p.add_argument("--limit", type=int, default=0, help="only the first N games (smoke test)")
        else:
            p.add_argument("--se-table", required=True)
            p.add_argument("--n", type=int, default=30)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--max-diff", type=float, default=0.01)
            p.add_argument("--repo-root", default=str(ROOT), help="repo holding data/ (a worktree has none)")
            p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.color_semantics:
        os.environ["GOGAME_COLOR_SEMANTICS"] = a.color_semantics
    from gogame.sgf import get_color_semantics
    get_color_semantics()
    return cmd_extract(a) if a.cmd == "extract" else cmd_consistency(a)


if __name__ == "__main__":
    sys.exit(main())
