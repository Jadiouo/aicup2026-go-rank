"""E36 shared constants/paths.  Plan: experiments/E36_full_late_hsl/PLAN.md.
Same scan setting as E34/E35 (config_hash_w2 a9d7ab4b...); independent ledger/dirs (data/scan_e36/, .scratch/e36/); school dir ~/gogame_e21/e36/."""
from __future__ import annotations
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from gogame.constants import SEED  # noqa: E402
from gogame.seeds import derive_seed  # noqa: E402

E29B = ROOT / ".scratch/e29/bundle"          # full non-H bundle (read-only): aux.npz, X.npy (W60, 139 cols)
S35 = ROOT / ".scratch/e35"                  # E35 scratch (read-only)
S36 = ROOT / ".scratch/e36"                  # scratch of this experiment
BUNDLE = S36 / "bundle"
PART = S36 / "parts"
INBOX = S36 / "sync_inbox"
STATE = S36 / "state"
W36 = ROOT / "data/scan_e36"                 # E36 ledger / sgf shards / results (never data/scan*, never data/scan_e35, data/scan_w2)
W35 = ROOT / "data/scan_e35"                 # read-only
E34W2 = ROOT / "data/scan_w2"                # read-only
EXP = ROOT / "experiments/E36_full_late_hsl"
KATAGO = ROOT / "scripts/katago"
E34S = ROOT / "scripts/experiments/e34"
PINS_W2 = E34S / "scan_pins_w2.json"
SHARD_GAMES = 2000
RANGE = (60, 120)
NPART = 40
CODE_STAGE = S36 / "code_stage"              # byte-identical copy of the school scan code (see PLAN §3)
CODE_FILES = ("analysis.cfg", "kgio.py", "scan_pins_parent.json", "scan_pins_w2.json", "scan_runner.py", "scan_runner_w2.py")
# sha256 of the school (E35) copies, taken 2026-11-10 from ~/gogame_e21/e35/code/ and the repo sources: they must be what every platform runs
CODE_SHA = {
    "analysis.cfg": "8b9cc5f705ea5fcf8bb8028d41fb698bdfda6b32a4521c07aad7f53dfab92e35",
    "kgio.py": "53bf159ea994d53efeb9b1da176cf0500f9a99a9894311d40b39c71fdfd0d79b",
    "scan_pins_parent.json": "b2690438df67d36d07533ad50b016f7fa2c64c15633f4d8c2c2fbf7d9e11b794",
    "scan_pins_w2.json": "cc4f3f656dd1a4dcd7d44e7eda4990e4dc63671d9c1d2632d091315eba780408",
    "scan_runner.py": "8d226019ac99c69969df8dd63f0c5fbeb18e7e82d563da4185c5cbb9fc4e7293",
    "scan_runner_w2.py": "7f05db3dad30d5cee20f898d4674e3add1bc8df73a5ad64c98064779300cfc0c",
}
CONFIG_HASH_W2 = "a9d7ab4b2962d41073a91afba69cfc48a71b39bf47c28cdd815b115d9cda67e2"
PARENT_HASH = "d6811b41b13b5d88f4d50171792686c372ae2c91853b63a43ce001de33792165"
SEM_TRAIN = "as_annotated"
SCHOOL_DIR = "gogame_e21/e36"                # relative to $HOME on the school machine
REMOTE_DIR = "gogame_e36"                    # relative to $HOME on the remote VM
PLATFORMS = ("school", "remote-vm", "colab")


def sd(*keys: str) -> int:
    return derive_seed(SEED, "e36", *keys)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()
