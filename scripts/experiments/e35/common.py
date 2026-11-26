"""E35 shared constants/paths.  Plan: experiments/E35_late_hsl_full/PLAN.md."""
from __future__ import annotations
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from gogame.constants import SEED  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402

E29B = ROOT / ".scratch/e29/bundle"          # full non-H bundle (read-only)
E29T = ROOT / ".scratch/e29/test"            # rank_v4 test features (read-only)
E34B = ROOT / ".scratch/e34/bundle"          # E34 pilot bundle (read-only)
S35 = ROOT / ".scratch/e35"                  # scratch of this experiment
BUNDLE = S35 / "bundle"                      # reduced bundle: X60.npy, X120.npy, aux.npz, selection.parquet
PART = S35 / "parts"                         # uid-partitioned per-move rows (old = stored 0..59, new = 60..119)
W35 = ROOT / "data/scan_e35"                 # E35 ledger / sgf shards / results (never data/scan, never data/scan_w2)
E34W2 = ROOT / "data/scan_w2"                # E34 pilot results (read-only)
EXP = ROOT / "experiments/E35_late_hsl_full"
RES = EXP / "results"
KATAGO = ROOT / "scripts/katago"
E34S = ROOT / "scripts/experiments/e34"
PINS_W2 = E34S / "scan_pins_w2.json"         # the very same setting as E34 (config_hash_w2 a9d7ab4b...)
SHARD_GAMES = 4000
POOL = 40
NS = (5, 10, 15, 20)
NFOLD = 5
RANGE = (60, 120)
NPART = 12
SCHOOL_DIR = "gogame_e21/e35"                # relative to $HOME on the school machine
CONFIG_HASH_W2 = "a9d7ab4b2962d41073a91afba69cfc48a71b39bf47c28cdd815b115d9cda67e2"
PARENT_HASH = "d6811b41b13b5d88f4d50171792686c372ae2c91853b63a43ce001de33792165"
SEM_TRAIN, SEM_TEST = "as_annotated", "actual"


def sd(*keys: str) -> int:
    return derive_seed(SEED, "e35", *keys)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()
