"""
Packages the fine-tuned checkpoint for a GitHub Release and records its SHA256.

    python -m scripts.package_model

Writes dist/risk_engine-<version>.zip and sets model.release_sha256 in config/engine.yaml, so
scripts/fetch_model.py can verify the download.
"""
import hashlib
import os
import re
import zipfile

import yaml

CONFIG = "config/engine.yaml"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    src = cfg["model"]["checkpoint_dir"]
    if not os.path.exists(os.path.join(src, "heads.pt")):
        raise SystemExit(f"No checkpoint in {src}; run `python run.py train` first.")

    os.makedirs("dist", exist_ok=True)
    out = os.path.join("dist", os.path.basename(cfg["model"]["release_url"]))
    root = os.path.basename(os.path.normpath(src))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, files in os.walk(src):
            for name in sorted(files):
                path = os.path.join(dirpath, name)
                arcname = os.path.join(root, os.path.relpath(path, src)).replace(os.sep, "/")
                zf.write(path, arcname)

    digest = sha256_of(out)
    with open(CONFIG, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r'release_sha256: ".*"', f'release_sha256: "{digest}"', text)
    with open(CONFIG, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)\nsha256 {digest} (saved to {CONFIG})")


if __name__ == "__main__":
    main()
