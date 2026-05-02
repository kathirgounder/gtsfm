"""Rename phototourism images to sequential `imgNNNN.JPG` and write a manifest.

Crowdsourced photo dumps often have filenames with quotes, spaces, special chars
that break shells, COLMAP's SQLite, dask serialization, etc. This script:

  1. Renames all images in `--images-dir` to `imgNNNN.<ext>` (zero-padded).
  2. Writes a `rename_manifest.csv` next to the images dir mapping old → new
     plus EXIF capture time + camera model (handy for later clustering).
  3. Idempotent: skips files already in the imgNNNN form.

Sort order: alphanumeric on the original filename, so renaming is reproducible.

Usage:
    python scripts/clean_dataset_filenames.py --images-dir benchmarks/thanjavur/images
    python scripts/clean_dataset_filenames.py --images-dir benchmarks/thanjavur/images --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


SEQUENTIAL_PATTERN = re.compile(r"^img\d{4}\.[A-Za-z]+$")


def get_exif_summary(path: Path) -> tuple[str, str]:
    """Return (capture_time, camera_model) — empty strings on failure."""
    try:
        with Image.open(path) as im:
            exif = im._getexif() or {}
            tag_map = {TAGS.get(k, k): v for k, v in exif.items()}
            t = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime") or ""
            make = (tag_map.get("Make") or "").strip()
            model = (tag_map.get("Model") or "").strip()
            cam = f"{make} {model}".strip() if (make or model) else ""
            return str(t), cam
    except Exception:
        return "", ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True, type=Path,
                        help="Directory containing the images to rename.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest CSV path (default: <images_dir>/../rename_manifest.csv).")
    parser.add_argument("--prefix", default="img",
                        help="Filename prefix (default: 'img').")
    parser.add_argument("--digits", type=int, default=4,
                        help="Zero-padding digit count (default: 4 → up to 9999 images).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be renamed without touching disk.")
    args = parser.parse_args()

    images_dir: Path = args.images_dir.resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"images_dir not found: {images_dir}")

    manifest_path: Path = args.manifest or (images_dir.parent / "rename_manifest.csv")

    files = sorted(images_dir.iterdir(), key=lambda p: p.name.lower())
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    files = [f for f in files if f.is_file() and f.suffix.lower() in image_exts]
    if not files:
        raise SystemExit(f"No image files found in {images_dir}")

    n_seq = sum(1 for f in files if SEQUENTIAL_PATTERN.match(f.name))
    print(f"Found {len(files)} images in {images_dir}")
    print(f"  Already in imgNNNN form: {n_seq}")
    print(f"  To rename: {len(files) - n_seq}")
    if args.dry_run:
        print("\n--dry-run: no changes will be made.")

    rows: list[dict] = []
    rename_count = 0
    counter = 1
    for f in files:
        capture_time, camera = get_exif_summary(f)
        # Find the next free sequential index that isn't taken by another file we
        # haven't touched yet — handles partial-rename resume correctly.
        if SEQUENTIAL_PATTERN.match(f.name):
            new_name = f.name
        else:
            ext = f.suffix.lower().lstrip(".")
            while True:
                candidate = f"{args.prefix}{counter:0{args.digits}d}.{ext}"
                target = images_dir / candidate
                if not target.exists() or target.resolve() == f.resolve():
                    new_name = candidate
                    counter += 1
                    break
                counter += 1
            if not args.dry_run:
                f.rename(images_dir / new_name)
            rename_count += 1
        rows.append({
            "new_name": new_name,
            "original_name": f.name,
            "capture_time": capture_time,
            "camera": camera,
        })

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["new_name", "original_name", "capture_time", "camera"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRenamed {rename_count} files. Wrote manifest: {manifest_path}")
    else:
        print(f"\nWould rename {rename_count} files. Would write manifest: {manifest_path}")
        for row in rows[:5]:
            print(f"  {row['original_name']!r:60s} → {row['new_name']}")
        if len(rows) > 5:
            print(f"  ... ({len(rows) - 5} more)")


if __name__ == "__main__":
    main()
