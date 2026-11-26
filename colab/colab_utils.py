"""Helpers for the Colab notebooks (standard library only; works on any machine with Python 3.8+).

  fetch_weights(...)       download OUR trained weights from the GitHub Release and verify sha256        (category "weights" in assets_manifest.json)
  fetch_intermediate(...)  copy the official-data-derived intermediate files from a private folder       (category "intermediate"; NOT public)
  fetch_external(...)      download the third-party pretrained files (KataGo binary and networks, HumanSL checkpoint, Strength Estimator) and verify sha256
  sha256_file(...)

`RELEASE_URL` is the base URL of the GitHub Release that holds our weights, e.g. https://github.com/<user>/<repo>/releases/download/<tag>
(value to be filled in once the Release exists).  A Release has no folders, so an asset named `rank/val_preds.npz` in ASSETS.md would be stored as
`rank__val_preds.npz`; the manifest's `asset` field is already the flat name.

`INTERMEDIATE_DIR` is a folder that holds the intermediate files derived from the official match data.  They are NOT in this repository and NOT in the
Release; they are handed to the organizers on request through a private link (see README).  fetch_intermediate looks for each file in that folder
under its flat `asset` name first, then under its `dest` path (the same `.scratch/...` layout the scripts expect).
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

PLACEHOLDER = "待補"

# third-party files (not redistributed here).  bytes / sha256 measured 2026-11-26; see README "外部資源".
EXTERNAL = {
    "katago_zip": dict(
        url="https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip",
        bytes=51350638, sha256="2cb197fc9bc2050f29112abb5b83cdbad63936cb4605d6e199772aedc9806c02",
        dest="data/external/katago/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip", license="MIT-style (KataGo)"),
    "katago_main_net": dict(
        url="https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
        bytes=97898094, sha256="9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d",
        dest="data/external/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz", license="KataGo Neural Network License (MIT-style)"),
    "katago_human_net": dict(
        url="https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.bin.gz",
        bytes=99066230, sha256="637746e44f0efe00ad1245a50aa9bbf0716efe364c43965ead97bd6835d84ab5",
        dest="data/external/katago/b18c384nbt-humanv0.bin.gz", license="KataGo Neural Network License (MIT-style)"),
    "humansl_ckpt": dict(
        url="https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.ckpt",
        bytes=323752318, sha256="84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27",
        dest="data/external/katago/b18c384nbt-humanv0.ckpt", license="KataGo Neural Network License (MIT-style)"),
    "se_tar": dict(
        url="https://rlg.iis.sinica.edu.tw/papers/strength-estimator/assets/models/go.tar.gz",
        bytes=448813641, sha256="885bea594a62956958de526f6a2073f48ecd181535b4d9e72d2c71a339172caf",
        dest="data/external/se/download/go.tar.gz", license="library GPL-3.0; weights: no explicit license found, downloaded from the rights holder's public page only, never redistributed by us"),
}
SE_GO_PT_SHA256 = "95fc295137b1fd09a0ec4d5cc79ed888f8c12d0ba6a77fea58b24efc3f418126"     # se_go.pt after extraction


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _download(url, dest, retries=3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for k in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aicup2026-go-colab"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dest)
            return
        except Exception as e:          # network hiccup: retry
            print(f"  download failed ({e}); attempt {k}/{retries}", file=sys.stderr)
            time.sleep(5 * k)
    raise RuntimeError(f"could not download {url}")


def get_file(url_or_none, dest, sha256, size=None, local_src=None):
    """Make `dest` exist with the expected sha256: reuse it if already correct, else copy from local_src or download from url."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == (size if size is not None else dest.stat().st_size) and sha256_file(dest) == sha256:
        return "cached"
    if local_src is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_src, dest)
        how = "copied"
    else:
        if not url_or_none or PLACEHOLDER in url_or_none:
            raise RuntimeError(f"no download URL for {dest.name}: set RELEASE_URL (it is still '{PLACEHOLDER}') or put the file at {dest}")
        _download(url_or_none, dest)
        how = "downloaded"
    got = sha256_file(dest)
    if got != sha256:
        dest.unlink()
        raise RuntimeError(f"sha256 mismatch for {dest.name}: got {got}, expected {sha256}")
    return how


def _manifest(repo_dir, manifest=None):
    return json.loads(Path(manifest or Path(repo_dir) / "colab" / "assets_manifest.json").read_text(encoding="utf-8"))


def fetch_weights(repo_dir, release_url, final_only=False, manifest=None, local_mirror=None):
    """Download our own trained weights (manifest category "weights") from the Release into <repo_dir>/<dest> and verify sha256.
    final_only=True fetches only those that the last step needs.  local_mirror (offline tests): folder with the flat asset names."""
    repo_dir = Path(repo_dir)
    items = [m for m in _manifest(repo_dir, manifest) if m["category"] == "weights" and (m["final"] or not final_only)]
    total = sum(m["bytes"] for m in items)
    print(f"{len(items)} weight files, {total / 1e6:.1f} MB")
    for m in items:
        src = None
        if local_mirror is not None:
            src = Path(local_mirror) / m["asset"]
            if not src.exists():
                raise FileNotFoundError(src)
        how = get_file(f"{release_url.rstrip('/')}/{m['asset']}", repo_dir / m["dest"], m["sha256"], m["bytes"], src)
        print(f"  {how:10s} {m['dest']}  ({m['bytes']:,} bytes, sha256 {m['sha256'][:12]})")
    return len(items)


def find_intermediate(intermediate_dir, m):
    d = Path(intermediate_dir)
    for cand in (d / m["asset"], d / m["dest"]):
        if cand.exists():
            return cand
    return None


def fetch_intermediate(repo_dir, intermediate_dir, final_only=True, manifest=None):
    """Copy the intermediate files (manifest category "intermediate") from INTERMEDIATE_DIR into <repo_dir>/<dest> and verify sha256.
    These files are derived from the official data and are only given to the organizers (README); final_only=True takes the ones the last step needs."""
    repo_dir = Path(repo_dir)
    if not intermediate_dir or PLACEHOLDER in str(intermediate_dir) or not Path(intermediate_dir).is_dir():
        raise RuntimeError(f"INTERMEDIATE_DIR ('{intermediate_dir}') is not an existing folder.  The intermediate files are only provided to the organizers "
                           "(private link); put them in a folder (e.g. a mounted Google Drive) and point INTERMEDIATE_DIR at it.  See README.")
    items = [m for m in _manifest(repo_dir, manifest) if m["category"] == "intermediate" and (m["final"] or not final_only)]
    print(f"{len(items)} intermediate files, {sum(m['bytes'] for m in items) / 1e6:.1f} MB")
    for m in items:
        src = find_intermediate(intermediate_dir, m)
        if src is None:
            raise FileNotFoundError(f"{m['asset']} (or {m['dest']}) not found under {intermediate_dir}")
        how = get_file(None, repo_dir / m["dest"], m["sha256"], m["bytes"], src)
        print(f"  {how:10s} {m['dest']}  ({m['bytes']:,} bytes, sha256 {m['sha256'][:12]})")
    return len(items)


def fetch_external(repo_dir, names=("katago_zip", "katago_main_net", "katago_human_net", "humansl_ckpt", "se_tar"), extract_se=True):
    """Download third-party pretrained files into <repo_dir>/data/external/... and verify sha256 (sizes/hashes pinned in EXTERNAL)."""
    repo_dir = Path(repo_dir)
    for n in names:
        e = EXTERNAL[n]
        how = get_file(e["url"], repo_dir / e["dest"], e["sha256"], e["bytes"])
        print(f"  {how:10s} {n}: {e['dest']}  [{e['license']}]")
    if extract_se and "se_tar" in names:
        out = repo_dir / "data/external/se/extracted"
        pt = out / "se_go.pt"
        if not (pt.exists() and sha256_file(pt) == SE_GO_PT_SHA256):
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(repo_dir / EXTERNAL["se_tar"]["dest"]) as t:
                for member in t.getmembers():                      # the archive holds 5 flat .pt files; only se_go.pt is used
                    if member.isfile() and member.name == "se_go.pt":
                        t.extract(member, out)
            got = sha256_file(pt)
            assert got == SE_GO_PT_SHA256, f"se_go.pt sha256 {got} != expected {SE_GO_PT_SHA256}"
        print(f"  se_go.pt extracted and verified: {pt.relative_to(repo_dir)}")
