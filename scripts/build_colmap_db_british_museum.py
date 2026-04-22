"""Build a COLMAP database for British Museum via pycolmap.

Produces a fresh COLMAP SQLite DB with:
  - SIFT features extracted at max_num_features=8192 (matches our current gtsfm config)
  - Exhaustive matching across all image pairs
  - Two-view geometry verification (F + E + inlier masks in two_view_geometries table)

The DB is consumed by:
  1. GTSFM via ColmapCorrespondenceGenerator (feeds matches into our pipeline)
  2. GLOMAP via `glomap mapper --database_path DB ...`

Usage:
    python scripts/build_colmap_db_british_museum.py [--db-path PATH]

Default output path: ./benchmarks/british_museum/colmap_shared.db
"""

import argparse
from pathlib import Path

import pycolmap


DEFAULT_IMAGE_DIR = Path("/Users/kathir/Desktop/gtsfm/benchmarks/british_museum/images")
DEFAULT_DB_PATH = Path("/Users/kathir/Desktop/gtsfm/benchmarks/british_museum/colmap_shared.db")


def build_db(db_path: Path, image_dir: Path, max_num_features: int = 8192) -> None:
    if db_path.exists():
        raise FileExistsError(f"DB already exists at {db_path}; delete it first for a clean build.")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # SIFT extraction — parameters chosen to match our current gtsfm pipeline.
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.max_num_features = max_num_features
    sift_options.estimate_affine_shape = False
    sift_options.domain_size_pooling = False

    print(f"[1/3] Extracting SIFT features into {db_path}")
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO,
        sift_options=sift_options,
    )

    print(f"[2/3] Running exhaustive matching")
    matching_options = pycolmap.ExhaustiveMatchingOptions()
    pycolmap.match_exhaustive(
        database_path=str(db_path),
        matching_options=matching_options,
    )

    print(f"[3/3] Verification is performed inline during matching; two_view_geometries populated")

    db = pycolmap.Database(str(db_path))
    print(
        f"Done: {db.num_images} images, {db.num_keypoints} keypoints, "
        f"{db.num_matched_image_pairs} matched pairs, {db.num_verified_image_pairs} verified pairs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--max-features", type=int, default=8192)
    args = parser.parse_args()

    build_db(args.db_path, args.image_dir, args.max_features)


if __name__ == "__main__":
    main()
