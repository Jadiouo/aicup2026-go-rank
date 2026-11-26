from __future__ import annotations

import hashlib


def derive_seed(base: int, *keys: str) -> int:
    digest = hashlib.sha256(f"{base}:{'/'.join(keys)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**63
