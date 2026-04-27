"""Build a shared COLMAP database for a phototourism dataset via pycolmap.

Generalization of build_colmap_db_british_museum.py. Produces a COLMAP SQLite DB
with SIFT features + exhaustive matching + two-view geometry verification that
both GTSFM (via ColmapCorrespondenceGenerator) and GLOMAP (via `glomap mapper
--database_path ...`) can consume, so their backends compare on identical inputs.

Default dataset-to-path mapping follows our benchmarks/ layout:
  british_museum        -> benchmarks/british_museum/{images,colmap_shared.db}
  grand_place_brussels  -> benchmarks/grand_place_brussels/{images,colmap_shared.db}
  pantheon_exterior     -> benchmarks/pantheon_exterior/{images,colmap_shared.db}
  sacre_coeur           -> benchmarks/sacre_coeur/{images,colmap_shared.db}

Usage:
    python scripts/build_colmap_db_phototourism.py --dataset grand_place_brussels
"""

import argparse
from pathlib import Path

import pycolmap


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = REPO_ROOT / "benchmarks"

DATASETS = {
    "british_museum":       BENCHMARKS / "british_museum",
    "grand_place_brussels": BENCHMARKS / "grand_place_brussels",
    "pantheon_exterior":    BENCHMARKS / "pantheon_exterior",
    "sacre_coeur":          BENCHMARKS / "sacre_coeur",
}


def build_db(db_path: Path, image_dir: Path, max_num_features: int = 8192) -> None:
    if db_path.exists():
        raise FileExistsError(f"DB already exists at {db_path}; delete it first for a clean build.")

    db_path.parent.mkdir(parents=True, exist_ok=True)

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

    print("[2/3] Running exhaustive matching")
    matching_options = pycolmap.ExhaustiveMatchingOptions()
    pycolmap.match_exhaustive(
        database_path=str(db_path),
        matching_options=matching_options,
    )

    print("[3/3] Verification is performed inline during matching; two_view_geometries populated")

    db = pycolmap.Database(str(db_path))
    print(
        f"Done: {db.num_images} images, {db.num_keypoints} keypoints, "
        f"{db.num_matched_image_pairs} matched pairs, {db.num_verified_image_pairs} verified pairs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="Dataset name (determines benchmarks/<dataset>/{images,colmap_shared.db} paths).",
    )
    parser.add_argument("--max-features", type=int, default=8192)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override default db path (defaults to benchmarks/<dataset>/colmap_shared.db).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Override default image dir (defaults to benchmarks/<dataset>/images).",
    )
    args = parser.parse_args()

    ds_root = DATASETS[args.dataset]
    db_path = args.db_path or (ds_root / "colmap_shared.db")
    image_dir = args.image_dir or (ds_root / "images")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    build_db(db_path, image_dir, args.max_features)


if __name__ == "__main__":
    main()
