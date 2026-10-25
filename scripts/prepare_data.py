"""Get higgs.zip (use a local copy, or download it) and convert it once to a float32 .npy file.

Usage:
  python scripts/prepare_data.py                       # uses data/raw/higgs.zip, downloading if missing
  python scripts/prepare_data.py --zip ~/path/higgs.zip  # use a copy you already have (nothing is copied)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs.data import ZIP_MEMBER, csv_to_array  # noqa: E402


def check_zip(path: Path) -> bool:
    """True if `path` is a readable zip containing HIGGS.csv.gz with a correct CRC."""
    try:
        with zipfile.ZipFile(path) as zf:
            return ZIP_MEMBER in zf.namelist() and zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def download(url: str, dest: Path, attempts: int = 3) -> None:
    """Download to a .part file and keep it only if the zip verifies.

    The UCI server does not support byte ranges, so a dropped connection means starting over.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        print(f"Downloading {url} (~2.8 GB), attempt {attempt}/{attempts} ...", flush=True)
        part.unlink(missing_ok=True)
        if subprocess.run(["curl", "-L", "--fail", "-o", str(part), url]).returncode == 0 and check_zip(part):
            part.rename(dest)
            return
        print("  download incomplete or corrupt; retrying", flush=True)
    raise SystemExit(f"could not download {url} after {attempts} attempts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="existing higgs.zip to read instead of data/raw/higgs.zip")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["data"]
    array_path = ROOT / cfg["array_path"]
    if array_path.exists():
        print(f"{array_path} already exists; nothing to do.")
        return
    raw = args.zip.expanduser().resolve() if args.zip else ROOT / cfg["raw_path"]
    if not raw.exists():
        if args.zip:
            raise SystemExit(f"{raw} not found")
        download(cfg["url"], raw)
    print(f"Verifying {raw} ...", flush=True)
    if not check_zip(raw):
        raise SystemExit(f"{raw} is not a valid higgs.zip (expected member {ZIP_MEMBER} with a good CRC)")
    print(f"Converting {raw.name} -> {cfg['array_path']} ...", flush=True)
    csv_to_array(raw, array_path)
    print(f"Done: {array_path}")


if __name__ == "__main__":
    main()
