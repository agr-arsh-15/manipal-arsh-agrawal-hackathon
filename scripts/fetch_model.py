"""
Downloads the fine-tuned risk-engine checkpoint published as a GitHub Release asset.

    python run.py fetch-model            (or: python -m scripts.fetch_model [--force])

The archive is verified against model.release_sha256 in config/engine.yaml and extracted to
model.checkpoint_dir. Safe to re-run: it does nothing when the checkpoint is already present.
"""
import argparse
import hashlib
import os
import sys
import zipfile

import requests
import yaml
from tqdm import tqdm

CONFIG = "config/engine.yaml"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str) -> str:
    h = hashlib.sha256()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="risk_engine") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                h.update(chunk)
                bar.update(len(chunk))
    return h.hexdigest()


def extract(archive: str, parent: str) -> None:
    parent_abs = os.path.abspath(parent)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = os.path.abspath(os.path.join(parent, member))
            if not target.startswith(parent_abs + os.sep):
                raise RuntimeError(f"unsafe path in archive: {member}")
        zf.extractall(parent)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the fine-tuned risk-engine checkpoint.")
    parser.add_argument("--force", action="store_true", help="re-download even if the checkpoint exists")
    parser.add_argument("--url", help="override model.release_url (a URL or a local .zip path)")
    args = parser.parse_args()

    with open(CONFIG, "r", encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)["model"]
    target = mcfg["checkpoint_dir"]
    if os.path.exists(os.path.join(target, "heads.pt")) and not args.force:
        print(f"Checkpoint already present in {target}; nothing to do (use --force to re-download).")
        return 0

    url = args.url or mcfg["release_url"]
    expected = (mcfg.get("release_sha256") or "").lower()
    parent = os.path.dirname(os.path.normpath(target)) or "."
    os.makedirs(parent, exist_ok=True)

    local = os.path.isfile(url)
    if local:
        archive, digest = url, sha256_of(url)
    else:
        archive = os.path.join(parent, os.path.basename(url) + ".part")
        print(f"Downloading {url}")
        try:
            digest = download(url, archive)
        except requests.RequestException as e:
            if os.path.exists(archive):
                os.remove(archive)
            print(f"[error] Download failed: {e}\n"
                  "        The engine still works with the committed TF-IDF baseline. Retry when online, or\n"
                  "        download the zip from the repository's Releases page and run:\n"
                  "        python run.py fetch-model --url path/to/risk_engine-v1.0.0.zip",
                  file=sys.stderr)
            return 1

    if expected and digest != expected:
        if not local:
            os.remove(archive)
        print(f"[error] SHA256 mismatch: expected {expected}, got {digest}", file=sys.stderr)
        return 1
    extract(archive, parent)
    if not local:
        os.remove(archive)
    if not os.path.exists(os.path.join(target, "heads.pt")):
        print(f"[error] Archive did not contain {target}/heads.pt", file=sys.stderr)
        return 1
    print(f"Checkpoint ready in {target} (sha256 {'verified' if expected else 'not pinned'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
