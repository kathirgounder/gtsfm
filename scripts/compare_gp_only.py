"""Compare post-GP (pre-BA) reconstructions from GTSFM vs GLOMAP against GT.

Prints a text table of Sim(3)-aligned absolute pose errors and AUCs for:
  - GTSFM D (pre-BA state, from `ba_input/`)
  - GLOMAP (run with `--skip_bundle_adjustment 1 --skip_retriangulation 1`)

Both reconstructions are compared against a COLMAP-format GT via
`gtsfm/evaluation/compare_colmap_outputs.py` (Sim(3) alignment + AUC metrics).

Prerequisite: a GLOMAP GP-only run, e.g.
  glomap mapper --database_path DB --image_path IMG --output_path OUT \
    --skip_bundle_adjustment 1 --skip_retriangulation 1

Example (British Museum, with existing artifacts):
  python scripts/compare_gp_only.py \
      --glomap_gp_recon benchmark_results/glomap_bm_shared_db_gp_only/0 \
      --gtsfm_ba_input  benchmark_results/F-megaloc_sift_gp_single_pt/british_museum/results/ba_input \
      --gtsfm_ba_output benchmark_results/F-megaloc_sift_gp_single_pt/british_museum/results/ba_output \
      --gt              benchmarks/british_museum/sfm_updated
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_SCRIPT = REPO_ROOT / "gtsfm" / "evaluation" / "compare_colmap_outputs.py"


def remap_ba_input_filenames(ba_input: Path, ba_output: Path, out_dir: Path) -> None:
    """Rewrite ba_input/images.txt NAME column using ba_output's id→name mapping."""
    id_to_name = {}
    with open(ba_output / "images.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10:
                id_to_name[parts[0]] = parts[9]
    out_dir.mkdir(parents=True, exist_ok=True)
    for copy_name in ["cameras.txt", "points3D.txt"]:
        (out_dir / copy_name).write_text((ba_input / copy_name).read_text())
    with open(ba_input / "images.txt") as src, open(out_dir / "images.txt", "w") as dst:
        for line in src:
            if line.startswith("#") or not line.strip():
                dst.write(line)
                continue
            parts = line.split()
            if len(parts) >= 10 and parts[0] in id_to_name:
                parts[9] = id_to_name[parts[0]]
                dst.write(" ".join(parts) + "\n")
            else:
                dst.write(line)


def count_gt_overlap(recon_images_txt: Path, gt_dir: Path) -> tuple[int, int]:
    """Return (num_overlap_with_gt, num_recon_images). GT is binary or text.

    COLMAP images.txt has two lines per image (pose + POINTS2D). Alternate lines.
    """
    recon_names = set()
    expect_pose = True
    with open(recon_images_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if expect_pose:
                parts = line.split()
                if len(parts) >= 10:
                    recon_names.add(parts[9])
            expect_pose = not expect_pose
    # Use pycolmap to read GT (handles both .bin and .txt)
    import pycolmap
    gt = pycolmap.Reconstruction(str(gt_dir))
    gt_names = {img.name for img in gt.images.values()}
    return len(recon_names & gt_names), len(recon_names)


def run_eval(baseline: Path, current: Path, out: Path) -> Path:
    """Invoke compare_colmap_outputs.py; return path to the metrics json."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(COMPARE_SCRIPT),
        "--baseline", str(baseline),
        "--current", str(current),
        "--output", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[warn] compare_colmap_outputs.py exited {result.returncode}")
        print(result.stderr[-1500:])
    return out / "ba_pose_error_metrics.json"


def parse_metrics(metrics_path: Path) -> dict:
    """Extract the summary fields we care about from the metrics json."""
    with open(metrics_path) as f:
        bd = json.load(f)["ba_pose_error_metrics"]
    rot = bd["rotation_angle_error_deg"]["summary"]
    tra = bd["translation_angle_error_deg"]["summary"]
    registered = None
    if rot.get("len") is not None and rot.get("invalid") is not None:
        registered = rot["len"] - rot["invalid"]
    return {
        "registered": registered,
        "rot_med": rot.get("median"),
        "trans_med": tra.get("median"),
        "auc_1": bd.get("pose_auc_@1.0_deg"),
        "auc_3": bd.get("pose_auc_@3.0_deg"),
        "auc_5": bd.get("pose_auc_@5.0_deg"),
        "auc_10": bd.get("pose_auc_@10.0_deg"),
        "auc_20": bd.get("pose_auc_@20.0_deg"),
    }


def fmt(x, fmt_spec: str) -> str:
    return format(x, fmt_spec) if x is not None else "N/A"


def print_row(label: str, m: dict) -> None:
    print(
        f"{label:<14}"
        f"{fmt(m['registered'], '>4d' if m['registered'] is not None else '>4')}   "
        f"{fmt(m['rot_med'], '>6.3f')}   "
        f"{fmt(m['trans_med'], '>7.3f')}   "
        f"{fmt(m['auc_3'], '>6.4f')}  "
        f"{fmt(m['auc_5'], '>6.4f')}  "
        f"{fmt(m['auc_10'], '>6.4f')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glomap_gp_recon", type=Path, required=True,
                        help="GLOMAP post-GP reconstruction dir (COLMAP binary)")
    parser.add_argument("--gtsfm_ba_input", type=Path, required=True,
                        help="GTSFM pre-BA state dir (COLMAP text, placeholder filenames)")
    parser.add_argument("--gtsfm_ba_output", type=Path, required=True,
                        help="GTSFM post-BA state dir — used for id→filename map")
    parser.add_argument("--gt", type=Path, required=True,
                        help="Ground-truth COLMAP reconstruction dir (binary or text)")
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/gp_compare"))
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    # Step 1: remap GTSFM filenames
    gtsfm_remapped = args.workdir / "gtsfm_gp_remapped"
    print(f"[1/3] Remapping GTSFM filenames → {gtsfm_remapped}")
    remap_ba_input_filenames(args.gtsfm_ba_input, args.gtsfm_ba_output, gtsfm_remapped)

    # Sanity: overlap with GT
    overlap, n_recon = count_gt_overlap(gtsfm_remapped / "images.txt", args.gt)
    if n_recon > 0 and overlap / n_recon < 0.9:
        print(f"[warn] GTSFM↔GT filename overlap is only {overlap}/{n_recon} = "
              f"{100 * overlap / n_recon:.1f}% — remap may have failed.")

    # Step 2: run eval for each
    print("[2/3] Evaluating GTSFM vs GT")
    gtsfm_metrics = parse_metrics(run_eval(args.gt, gtsfm_remapped, args.workdir / "eval_gtsfm"))
    print("[2/3] Evaluating GLOMAP vs GT")
    glomap_metrics = parse_metrics(run_eval(args.gt, args.glomap_gp_recon, args.workdir / "eval_glomap"))

    # Step 3: print table
    print()
    print("=== Post-GP comparison (Sim(3)-aligned absolute pose errors vs GT) ===")
    header = (
        f"{'System':<14}"
        f"{'Reg':>4}   "
        f"{'RotMed°':>6}   "
        f"{'TraMed°':>7}   "
        f"{'AUC@3':>6}  "
        f"{'AUC@5':>6}  "
        f"{'AUC@10':>6}"
    )
    print(header)
    print("-" * len(header))
    print_row("GTSFM (D)", gtsfm_metrics)
    print_row("GLOMAP", glomap_metrics)
    print()
    print("Notes:")
    print("  - Reg = number of reconstruction cameras with a matching GT image (by filename).")
    print("  - Rot/Trans medians are absolute (Sim(3)-aligned) angular errors in degrees.")
    print("  - AUCs are `pose_auc_@N_deg` keys from compare_colmap_outputs.py.")
    print(f"  - Detailed metrics JSON under {args.workdir}/eval_gtsfm and {args.workdir}/eval_glomap.")


if __name__ == "__main__":
    main()
