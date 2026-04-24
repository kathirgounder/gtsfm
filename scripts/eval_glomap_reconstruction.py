"""Evaluate a GLOMAP reconstruction against a ground-truth COLMAP model.

Reads:
  --recon DIR: GLOMAP output directory containing cameras.bin, images.bin, points3D.bin
  --gt   DIR: GT COLMAP model directory with the same three files (typically sfm_updated/)

Computes Sim(3) alignment of GLOMAP cameras to GT, then emits rotation angle error,
translation angle error, translation distance, and pose AUC @ {3°, 5°, 10°} in both
full-graph and constructed-only flavors.

Output: --output PATH (JSON). Matches the schema emitted by GTSFM's BA module, so the
parity table script can read GTSFM and GLOMAP metrics the same way.

Usage:
    python scripts/eval_glomap_reconstruction.py \\
        --recon benchmark_results/GLOMAP/british_museum/full \\
        --gt    benchmarks/british_museum/sfm_updated \\
        --output benchmark_results/GLOMAP/british_museum/full/glomap_metrics.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pycolmap
from gtsam import Pose3, Rot3

from gtsfm.utils.align import sim3_from_Pose3_maps_robust
from gtsfm.utils.metrics import compute_ba_pose_metrics


def _img_to_pose(img) -> Pose3:
    """pycolmap Image -> gtsam world-from-camera Pose3."""
    cTw = img.cam_from_world
    R_c_w = cTw.matrix()[:3, :3]
    t_c_w = np.asarray(cTw.translation).reshape(3)
    return Pose3(Rot3(R_c_w), t_c_w).inverse()


def _match_recons(
    gt: pycolmap.Reconstruction, comp: pycolmap.Reconstruction
) -> tuple[Dict[int, Pose3], Dict[int, Pose3], list[str]]:
    """Match registered images between two reconstructions, tolerant of naming schemes.

    Tries two strategies in order:
    1. Match by image name (GLOMAP vs GT — both use real filenames)
    2. Match by image_id (GTSFM's ba_input uses synthetic 'image_000001.jpg' names
       but preserves image_id from the COLMAP loader's ordering)
    """
    gt_by_name = {gt.image(i).name: _img_to_pose(gt.image(i)) for i in gt.reg_image_ids()}
    comp_by_name = {comp.image(i).name: _img_to_pose(comp.image(i)) for i in comp.reg_image_ids()}
    common_names = sorted(set(gt_by_name) & set(comp_by_name))
    if len(common_names) >= 3:
        gt_wTi = {i: gt_by_name[n] for i, n in enumerate(common_names)}
        comp_wTi = {i: comp_by_name[n] for i, n in enumerate(common_names)}
        return gt_wTi, comp_wTi, common_names

    # Fall back to image_id match.
    gt_by_id = {i: _img_to_pose(gt.image(i)) for i in gt.reg_image_ids()}
    comp_by_id = {i: _img_to_pose(comp.image(i)) for i in comp.reg_image_ids()}
    common_ids = sorted(set(gt_by_id) & set(comp_by_id))
    if len(common_ids) >= 3:
        gt_wTi = {i: gt_by_id[i] for i in common_ids}
        comp_wTi = {i: comp_by_id[i] for i in common_ids}
        ordered_labels = [f"image_id={i}" for i in common_ids]
        return gt_wTi, comp_wTi, ordered_labels

    raise RuntimeError(
        f"Too few matched poses: by name {len(common_names)}, by image_id {len(common_ids)}. "
        f"GT {len(gt_by_name)} cams, computed {len(comp_by_name)} cams."
    )


def _resolve_recon_dir(recon_dir: Path) -> Path:
    """GLOMAP writes reconstructions to <output>/<model_id>/ (e.g. <output>/0/).
    Auto-descend if cameras.bin isn't directly under recon_dir.
    """
    if (recon_dir / "cameras.bin").exists() or (recon_dir / "cameras.txt").exists():
        return recon_dir
    # Look for a subdir containing cameras.bin; prefer "0" if present.
    candidates = sorted(
        [d for d in recon_dir.iterdir() if d.is_dir() and (d / "cameras.bin").exists()],
        key=lambda p: (p.name != "0", p.name),  # "0" first
    )
    if not candidates:
        raise FileNotFoundError(
            f"No cameras.bin under {recon_dir} or its immediate subdirs"
        )
    return candidates[0]


def evaluate(recon_dir: Path, gt_dir: Path) -> dict:
    recon_dir = _resolve_recon_dir(recon_dir)
    gt_dir = _resolve_recon_dir(gt_dir)

    gt_recon = pycolmap.Reconstruction(str(gt_dir))
    computed_recon = pycolmap.Reconstruction(str(recon_dir))

    gt_wTi, comp_wTi, names = _match_recons(gt_recon, computed_recon)

    # Sim(3) alignment: aTi (target=GT) and bTi (input=computed) -> aSb takes computed to GT frame.
    aSb = sim3_from_Pose3_maps_robust(gt_wTi, comp_wTi)
    aligned_comp: Dict[int, Pose3] = {i: aSb.transformFrom(comp_wTi[i]) for i in comp_wTi}

    # Include missing cameras as None in computed so AUC integrates over all GT cams.
    computed_for_metrics: Dict[int, Pose3] = {i: aligned_comp.get(i) for i in gt_wTi}
    metrics_group = compute_ba_pose_metrics(
        gt_wTi=gt_wTi,
        computed_wTi=computed_for_metrics,
        metric_constructed_only=True,
    )

    # GtsfmMetricsGroup -> plain dict. summary values (median) match GTSFM's display format.
    metrics_dict = {}
    for m in metrics_group.metrics:
        if hasattr(m, "summary") and m.summary:
            metrics_dict[m.name] = {"summary": m.summary}
        else:
            try:
                metrics_dict[m.name] = float(m.data)
            except (TypeError, ValueError):
                metrics_dict[m.name] = m.data if isinstance(m.data, (int, float, str)) else None

    # Also include GT/computed registered counts and matched count for table context.
    metrics_dict["number_gt_cameras"] = gt_recon.num_reg_images()
    metrics_dict["number_computed_cameras"] = computed_recon.num_reg_images()
    metrics_dict["number_matched_cameras"] = len(names)

    return {"glomap_metrics": metrics_dict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recon", type=Path, required=True, help="GLOMAP output dir (cameras/images/points3D.bin)")
    parser.add_argument("--gt", type=Path, required=True, help="GT COLMAP model dir")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    args = parser.parse_args()

    if not args.recon.exists():
        raise FileNotFoundError(f"Recon dir not found: {args.recon}")
    if not args.gt.exists():
        raise FileNotFoundError(f"GT dir not found: {args.gt}")

    result = evaluate(args.recon, args.gt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Brief stdout summary.
    m = result["glomap_metrics"]
    def _get(key):
        v = m.get(key)
        if isinstance(v, dict):
            return v.get("summary", {}).get("median")
        return v
    print(f"Wrote {args.output}")
    print(
        f"  cams matched {m['number_matched_cameras']}/{m['number_gt_cameras']}  "
        f"rot {_get('rotation_angle_error_deg')}  "
        f"trans {_get('translation_angle_error_deg')}  "
        f"AUC@5 {_get('pose_auc_@5.0_deg')}  "
        f"AUC@10 {_get('pose_auc_@10.0_deg')}"
    )


if __name__ == "__main__":
    main()
