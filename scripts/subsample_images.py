"""Random-subsample an image directory via symlinks (no disk cost, originals untouched).

Use when a phototourism set is too big for laptop reconstruction. Symlink-based
so the original images stay put — pipeline runs against the subset directory.

Usage:
    python scripts/subsample_images.py --src benchmarks/thanjavur/images --n 300
    python scripts/subsample_images.py --src benchmarks/thanjavur/images --n 300 --seed 7
    python scripts/subsample_images.py --src benchmarks/thanjavur/images --n 200 --dst benchmarks/thanjavur/images_200
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="Source images directory.")
    parser.add_argument("--n", type=int, required=True, help="Number of images to sample.")
    parser.add_argument("--dst", type=Path, default=None,
                        help="Destination dir (default: <src>_<n>, e.g. images → images_300).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    src = args.src.resolve()
    if not src.is_dir():
        raise SystemExit(f"src not found: {src}")

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG"}
    all_imgs = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {e.lower() for e in image_exts})
    if len(all_imgs) < args.n:
        raise SystemExit(f"Only {len(all_imgs)} images in src; can't sample {args.n}.")

    dst = args.dst or src.parent / f"{src.name}_{args.n}"
    dst.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    chosen = rng.sample(all_imgs, args.n)

    n_linked, n_skipped = 0, 0
    for p in chosen:
        target = dst / p.name
        if target.exists() or target.is_symlink():
            n_skipped += 1
            continue
        target.symlink_to(p)
        n_linked += 1

    print(f"src: {src} ({len(all_imgs)} images)")
    print(f"dst: {dst} ({len(list(dst.iterdir()))} symlinks now)")
    print(f"created {n_linked} new symlinks, skipped {n_skipped} pre-existing")
    print(f"seed={args.seed}, sample size={args.n}")


if __name__ == "__main__":
    main()
