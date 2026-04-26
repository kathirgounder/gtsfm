"""Run GLOMAP on a MegaLoc-restricted viewgraph (causal experiment).

Filters the canonical COLMAP shared DB down to a (num_matched, min_score)
retrieval pair set, then invokes GLOMAP on the filtered DB. This isolates
the *pair-selection* variable: same features, same matcher, same per-pair
two-view geometry as GLOMAP's exhaustive baseline — only the pair set
differs. Lets us answer: "if GLOMAP saw the sparser viewgraph our pipeline
ran on, how does its AUC compare?"

Usage:
    python scripts/run_glomap_with_retrieval.py --dataset brussels --num-matched 100 --min-score 0.20
    python scripts/run_glomap_with_retrieval.py --dataset brussels --num-matched 50  --min-score 0.25 --mode ba_no_retri

Default mode is `ba_no_retri` (matches the GLOMAP-no-retri 0.760 baseline).

Output layout:
    benchmark_results/GLOMAP_RETRIEVAL/<canonical_ds>/nm<N>_ms<M>/<mode>/0/{cameras,images,points3D}.bin
    benchmarks/<canonical_ds>/colmap_shared_nm<N>_ms<M>.db   (the filtered DB)

After the run, evaluate with the existing eval script (the command is printed at end).

Authors: Kathir Gounder
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# Reuse helpers from the deep dive (canonical paths, retrieval logic, image listing).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from megaloc_deep_dive import (  # noqa: E402
    REPO,
    compute_kept_pairs,
    get_image_fnames,
    load_similarity_matrix,
    resolve_dataset,
)


DEFAULT_GLOMAP_BIN = "/Users/kathir/Desktop/glomap/build/glomap/glomap"
COLMAP_PAIR_ID_OFFSET = 2147483647  # COLMAP's pair_id encoding: id1 * OFFSET + id2.


def encode_pair_id(id1: int, id2: int) -> int:
    """COLMAP's pair_id = min(a,b) * 2147483647 + max(a,b)."""
    a, b = (id1, id2) if id1 < id2 else (id2, id1)
    return a * COLMAP_PAIR_ID_OFFSET + b


def build_db_image_id_map(colmap_db: Path, image_fnames: list[str]) -> dict[int, int]:
    """Returns {loader_idx → db_image_id} mapping built from the COLMAP DB."""
    fname_to_loader = {f: i for i, f in enumerate(image_fnames)}
    conn = sqlite3.connect(str(colmap_db))
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    loader_to_db: dict[int, int] = {}
    for db_id, name in cur.fetchall():
        basename = Path(name).name
        if basename in fname_to_loader:
            loader_to_db[fname_to_loader[basename]] = int(db_id)
    conn.close()
    if len(loader_to_db) != len(image_fnames):
        print(
            f"  ⚠️  loader→DB mapping covers {len(loader_to_db)}/{len(image_fnames)} images; "
            f"some images are missing from the COLMAP DB."
        )
    return loader_to_db


def filter_db_to_pairs(filtered_db: Path, allowed_pair_ids: set[int]) -> tuple[int, int, int, int]:
    """Delete rows from `matches` and `two_view_geometries` for any pair_id NOT in
    `allowed_pair_ids`. Returns (matches_before, matches_after, tvg_before, tvg_after).

    Strategy: we use a temp table for the allowed IDs (avoids SQLite's variable-count
    limit on parameterized IN clauses, which caps at ~999 by default).
    """
    conn = sqlite3.connect(str(filtered_db))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM matches")
    matches_before = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM two_view_geometries")
    tvg_before = cur.fetchone()[0]

    cur.execute("CREATE TEMP TABLE allowed_pairs (pair_id INTEGER PRIMARY KEY)")
    cur.executemany("INSERT INTO allowed_pairs VALUES (?)", [(pid,) for pid in allowed_pair_ids])
    conn.commit()

    cur.execute("DELETE FROM matches WHERE pair_id NOT IN (SELECT pair_id FROM allowed_pairs)")
    cur.execute("DELETE FROM two_view_geometries WHERE pair_id NOT IN (SELECT pair_id FROM allowed_pairs)")
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM matches")
    matches_after = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM two_view_geometries")
    tvg_after = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM two_view_geometries WHERE config != 0")
    tvg_verified_after = cur.fetchone()[0]

    cur.execute("DROP TABLE allowed_pairs")
    conn.execute("VACUUM")  # reclaim space, optional but tidy.
    conn.close()
    return matches_before, matches_after, tvg_before, tvg_after, tvg_verified_after


def glomap_skip_flags(mode: str) -> list[str]:
    if mode == "full":
        return []
    if mode == "ba_no_retri":
        return ["--skip_retriangulation", "1"]
    if mode == "gp_only":
        return ["--skip_bundle_adjustment", "1", "--skip_retriangulation", "1"]
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. brussels, british_museum).")
    parser.add_argument("--num-matched", type=int, required=True, help="Top-K per-image cap (e.g. 100).")
    parser.add_argument("--min-score", type=float, required=True, help="MegaLoc score threshold (e.g. 0.15).")
    parser.add_argument("--mode", choices=["full", "ba_no_retri", "gp_only"], default="ba_no_retri",
                        help="GLOMAP mode (default ba_no_retri to match the GLOMAP no-retri baseline).")
    parser.add_argument("--config", default="F-megaloc_sift_gp_single_pt",
                        help="Config dir under benchmark_results/ (where the similarity matrix lives).")
    parser.add_argument("--glomap-bin", default=DEFAULT_GLOMAP_BIN, help="Path to the glomap binary.")
    parser.add_argument("--dry-run", action="store_true", help="Build/filter the DB but skip the GLOMAP run.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing filtered DB / output dir.")
    args = parser.parse_args()

    canonical = resolve_dataset(args.dataset)
    nm, ms = args.num_matched, args.min_score
    suffix = f"nm{nm}_ms{ms:.2f}"

    canonical_db = REPO / "benchmarks" / canonical / "colmap_shared.db"
    similarity_txt = REPO / "benchmark_results" / args.config / args.dataset / "results" / "plots" / "similarity_matrix.txt"
    images_dir = REPO / "benchmarks" / canonical / "images"
    filtered_db = REPO / "benchmarks" / canonical / f"colmap_shared_{suffix}.db"
    output_dir = REPO / "benchmark_results" / "GLOMAP_RETRIEVAL" / canonical / suffix / args.mode

    print(f"Dataset: {args.dataset} (canonical: {canonical})")
    print(f"Retrieval: num_matched={nm}, min_score={ms}, mode={args.mode}")
    print(f"  similarity matrix: {similarity_txt}")
    print(f"  canonical DB:      {canonical_db}")
    print(f"  images dir:        {images_dir}")
    print(f"  filtered DB →      {filtered_db}")
    print(f"  output dir →       {output_dir}")
    for required in (canonical_db, similarity_txt, images_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required input missing: {required}")

    # Step 1: similarity matrix → kept pairs (loader-idx pairs).
    print(f"\nLoading similarity matrix ...")
    W = load_similarity_matrix(similarity_txt)
    image_fnames = get_image_fnames(images_dir)
    if W.shape[0] != len(image_fnames):
        raise ValueError(
            f"Matrix shape {W.shape} ≠ image count {len(image_fnames)}. "
            f"Loader/matrix mismatch — re-run F config or check images dir."
        )
    print(f"  matrix: {W.shape}, images: {len(image_fnames)}")

    print(f"Computing retrieval pair set at (num_matched={nm}, min_score={ms}) ...")
    kept_pairs_loader = compute_kept_pairs(W, nm, ms)
    print(f"  {len(kept_pairs_loader)} loader-idx pairs from MegaLoc retrieval")

    # Step 2: map loader-idx → DB image_id → pair_id.
    print(f"\nBuilding loader→DB image_id map ...")
    loader_to_db = build_db_image_id_map(canonical_db, image_fnames)
    allowed_pair_ids: set[int] = set()
    skipped = 0
    for i, j in kept_pairs_loader:
        if i in loader_to_db and j in loader_to_db:
            allowed_pair_ids.add(encode_pair_id(loader_to_db[i], loader_to_db[j]))
        else:
            skipped += 1
    if skipped:
        print(f"  ⚠️  Skipped {skipped} retrieval pairs (one or both images not in DB)")
    print(f"  {len(allowed_pair_ids)} pair_ids to keep in filtered DB")

    # Step 3: copy + filter the DB.
    if filtered_db.exists() and not args.force:
        print(f"\nFiltered DB already exists at {filtered_db.relative_to(REPO)} (use --force to overwrite). Skipping copy/filter.")
    else:
        if filtered_db.exists():
            filtered_db.unlink()
        print(f"\nCopying canonical DB to filtered DB ...")
        t0 = time.time()
        shutil.copy2(canonical_db, filtered_db)
        print(f"  copied in {time.time() - t0:.1f}s")

        print(f"Filtering filtered DB to allowed pair_ids ...")
        t0 = time.time()
        m_before, m_after, t_before, t_after, v_after = filter_db_to_pairs(filtered_db, allowed_pair_ids)
        print(f"  filtered in {time.time() - t0:.1f}s")
        print(f"  matches:              {m_before:>6d} → {m_after:>6d}  ({100 * m_after / max(m_before, 1):.1f}% kept)")
        print(f"  two_view_geometries:  {t_before:>6d} → {t_after:>6d}  ({100 * t_after / max(t_before, 1):.1f}% kept)")
        print(f"  └─ verified (config!=0):           {v_after:>6d}  ← what GLOMAP actually uses")
        if v_after < t_after:
            print(f"  → {t_after - v_after} retrieval pairs are present but unverified "
                  f"(COLMAP rejected them at two-view geometry).")

    # Step 4: invoke GLOMAP.
    if args.dry_run:
        print(f"\n--dry-run: skipping GLOMAP. To run manually:")
    else:
        print(f"\nRunning GLOMAP ({args.mode}) ...")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.glomap_bin, "mapper",
        "--database_path", str(filtered_db),
        "--image_path", str(images_dir),
        "--output_path", str(output_dir),
        *glomap_skip_flags(args.mode),
    ]
    print(f"  $ {' '.join(cmd)}")

    if not args.dry_run:
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=False)
        print(f"\nGLOMAP finished in {time.time() - t0:.1f}s (exit code {result.returncode})")
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Step 5: print evaluation hint.
    gt_dir = REPO / "benchmarks" / canonical / "sfm_updated"
    if not gt_dir.exists():
        gt_dir = REPO / "benchmarks" / canonical / "sfm"
    eval_output_json = output_dir / "glomap_metrics.json"
    print(f"\nNext: evaluate with")
    print(
        f"  python scripts/eval_glomap_reconstruction.py "
        f"--recon {output_dir.relative_to(REPO)} "
        f"--gt {gt_dir.relative_to(REPO)} "
        f"--output {eval_output_json.relative_to(REPO)}"
    )


if __name__ == "__main__":
    main()
