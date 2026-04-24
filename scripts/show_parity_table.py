#!/usr/bin/env python3
"""Render a GTSFM vs GLOMAP backend-parity table for phototourism datasets.

For each dataset, prints a 4-row block showing GLOMAP (GP-only / BA-no-retri / full)
plus the GTSFM D/F config's numbers, in the same format as the user's British Museum
table pasted in the plan.

GTSFM metrics: read from benchmark_results/F-megaloc_sift_gp_single_pt/<dataset>/
    results/metrics/{bundle_adjustment,merging}_metrics.json.

GLOMAP metrics: read from benchmark_results/GLOMAP/<dataset>/{gp_only,ba_no_retri,full}/
    glomap_metrics.json (produced by scripts/eval_glomap_reconstruction.py).

Usage:
    python scripts/show_parity_table.py                    # all datasets
    python scripts/show_parity_table.py --dataset british_museum   # one
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pycolmap
from gtsam import Pose3, Rot3

# Reuse the evaluator's Sim(3) alignment + pose-metric computation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_glomap_reconstruction import evaluate as evaluate_recon  # noqa: E402

from gtsfm.utils.align import sim3_from_Pose3_maps_robust  # noqa: E402
from gtsfm.utils.metrics import compute_ba_pose_metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["british_museum", "brussels", "pantheon_exterior", "sacre_coeur"]

# Map our dataset labels to benchmark_results dataset dir names.
# GTSFM writes results under benchmarks/<name>, which for brussels is keyed as "brussels".
# GLOMAP eval writes under benchmark_results/GLOMAP/<name>.
GLOMAP_MODES = [
    ("gp_only", "GLOMAP (GP only)"),
    ("ba_no_retri", "GLOMAP (BA, skip retri)"),
    ("full", "GLOMAP (full: BA + retri)"),
]

GTSFM_F_DIR = REPO_ROOT / "benchmark_results" / "F-megaloc_sift_gp_single_pt"
GLOMAP_DIR = REPO_ROOT / "benchmark_results" / "GLOMAP"


def _get_median(d, key):
    """Extract median from a GtsfmMetric-style {"summary": {"median": ...}} blob, or scalar."""
    if d is None:
        return None
    v = d.get(key)
    if isinstance(v, dict):
        return v.get("summary", {}).get("median")
    return v


def load_gtsfm_row(dataset: str) -> dict | None:
    """Build a GTSFM row with both full-graph AUC and constructed-only AUC columns.

    - cams, rotation/translation error: from merging_metrics.json (post-merge stats)
    - AUC + CO AUC: from bundle_adjustment_metrics.json
    """
    base = GTSFM_F_DIR / dataset / "results" / "metrics"
    if not base.exists():
        return None
    merging_path = base / "merging_metrics.json"
    ba_path = base / "bundle_adjustment_metrics.json"
    merging = json.load(open(merging_path)) if merging_path.exists() else None
    ba = json.load(open(ba_path)) if ba_path.exists() else None

    md = merging.get("merging_metrics") if merging else None
    bd = ba.get("bundle_adjustment_metrics") if ba else None

    if md is None and bd is None:
        return None

    cams = None
    if md and md.get("number_cameras_merged", 0) > 0:
        cams = md.get("number_cameras_merged")
    elif bd:
        cams = _get_median(bd, "number_cameras_filtered")

    pose_src = md if (md and md.get("number_cameras_merged", 0) > 0) else bd

    # Matches show_results.py semantics:
    #   AUC column = merging_metrics' pose_auc (full graph, penalizes dropped cams)
    #   CO column  = BA metrics' pose_auc_constructed_only (only recovered cams)
    # Falls back between sources if one is missing.
    auc_src = md if md is not None else bd
    co_src = bd if bd is not None else md

    return {
        "label": "GTSFM (Config: sift_gp_single)",
        "cams": cams,
        "rot": _get_median(pose_src, "rotation_angle_error_deg"),
        "trans": _get_median(pose_src, "translation_angle_error_deg"),
        "auc3":  auc_src.get("pose_auc_@3.0_deg"),
        "auc5":  auc_src.get("pose_auc_@5.0_deg"),
        "auc10": auc_src.get("pose_auc_@10.0_deg"),
        "co3":   co_src.get("pose_auc_constructed_only_@3.0_deg") or co_src.get("pose_auc_@3.0_deg"),
        "co5":   co_src.get("pose_auc_constructed_only_@5.0_deg") or co_src.get("pose_auc_@5.0_deg"),
        "co10":  co_src.get("pose_auc_constructed_only_@10.0_deg") or co_src.get("pose_auc_@10.0_deg"),
    }


_GTSFM_GP_CACHE: dict[str, dict] = {}


def _img_to_pose(img) -> Pose3:
    """pycolmap Image -> gtsam world-from-camera Pose3."""
    R = img.cam_from_world.matrix()[:3, :3]
    t = np.asarray(img.cam_from_world.translation).reshape(3)
    return Pose3(Rot3(R), t).inverse()


def load_gtsfm_gp_row(dataset: str) -> dict | None:
    """Evaluate GTSFM's post-GP state (results/ba_input/) against GT.

    ba_input has synthetic image names (`image_000001.jpg`) so we bridge via
    ba_output's real filenames (same image_id scheme). Then match GT by real name.
    """
    if dataset in _GTSFM_GP_CACHE:
        return _GTSFM_GP_CACHE[dataset]

    ds_root = GTSFM_F_DIR / dataset / "results"
    ba_input_dir = ds_root / "ba_input"
    ba_output_dir = ds_root / "ba_output"
    gt_dir = REPO_ROOT / "benchmarks" / _gt_subdir(dataset) / "sfm_updated"
    cache_path = ds_root / "metrics" / "gtsfm_gp_metrics.json"

    # Disk cache: if the cached JSON is newer than ba_input/images.txt, use it.
    if cache_path.exists() and ba_input_dir.exists():
        try:
            if cache_path.stat().st_mtime >= (ba_input_dir / "images.txt").stat().st_mtime:
                cached = json.load(open(cache_path))
                _GTSFM_GP_CACHE[dataset] = cached.get("row")
                return _GTSFM_GP_CACHE[dataset]
        except Exception:
            pass  # fall through to recompute

    required = [(ba_input_dir, "cameras.txt"), (ba_output_dir, "cameras.txt"), (gt_dir, None)]
    for d, f in required:
        if not d.exists():
            _GTSFM_GP_CACHE[dataset] = None
            return None
        if f is not None and not (d / f).exists():
            _GTSFM_GP_CACHE[dataset] = None
            return None

    try:
        ba_input = pycolmap.Reconstruction(str(ba_input_dir))
        ba_output = pycolmap.Reconstruction(str(ba_output_dir))
        gt = pycolmap.Reconstruction(str(gt_dir))

        # image_id -> real name via ba_output
        id_to_real_name = {i: ba_output.image(i).name for i in ba_output.reg_image_ids()}
        # real name -> GT pose via gt
        gt_by_name = {gt.image(i).name: _img_to_pose(gt.image(i)) for i in gt.reg_image_ids()}

        # Build matched pose dicts keyed by a shared index.
        gt_wTi: dict[int, Pose3] = {}
        comp_wTi: dict[int, Pose3] = {}
        idx = 0
        for image_id in ba_input.reg_image_ids():
            real_name = id_to_real_name.get(image_id)
            if real_name is None or real_name not in gt_by_name:
                continue
            gt_wTi[idx] = gt_by_name[real_name]
            comp_wTi[idx] = _img_to_pose(ba_input.image(image_id))
            idx += 1

        if len(gt_wTi) < 3:
            raise RuntimeError(f"Too few matched poses: {len(gt_wTi)}")

        aSb = sim3_from_Pose3_maps_robust(gt_wTi, comp_wTi)
        aligned = {i: aSb.transformFrom(comp_wTi[i]) for i in comp_wTi}
        metrics_group = compute_ba_pose_metrics(
            gt_wTi=gt_wTi,
            computed_wTi={i: aligned.get(i) for i in gt_wTi},
            metric_constructed_only=True,
        )

        blob = {}
        for m in metrics_group.metrics:
            if hasattr(m, "summary") and m.summary:
                blob[m.name] = {"summary": m.summary}
            else:
                try:
                    blob[m.name] = float(m.data)
                except (TypeError, ValueError):
                    blob[m.name] = m.data if isinstance(m.data, (int, float, str)) else None
    except Exception as e:
        print(f"[warn] GTSFM GP eval failed for {dataset}: {e}", file=sys.stderr)
        _GTSFM_GP_CACHE[dataset] = None
        return None

    row = {
        "label": "GTSFM (GP only)",
        "cams": len(gt_wTi),
        "rot": _get_median(blob, "rotation_angle_error_deg"),
        "trans": _get_median(blob, "translation_angle_error_deg"),
        "auc3":  blob.get("pose_auc_@3.0_deg"),
        "auc5":  blob.get("pose_auc_@5.0_deg"),
        "auc10": blob.get("pose_auc_@10.0_deg"),
        "co3":   blob.get("pose_auc_constructed_only_@3.0_deg") or blob.get("pose_auc_@3.0_deg"),
        "co5":   blob.get("pose_auc_constructed_only_@5.0_deg") or blob.get("pose_auc_@5.0_deg"),
        "co10":  blob.get("pose_auc_constructed_only_@10.0_deg") or blob.get("pose_auc_@10.0_deg"),
    }

    # Persist to disk so subsequent runs don't recompute.
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"row": row, "raw_metrics": blob}, f, indent=2, default=str)
    except Exception as e:
        print(f"[warn] couldn't cache GTSFM GP row for {dataset}: {e}", file=sys.stderr)

    _GTSFM_GP_CACHE[dataset] = row
    return row


def _gt_subdir(dataset: str) -> str:
    """Translate parity-table dataset names to benchmarks/ subdir names."""
    return {
        "british_museum": "british_museum",
        "brussels": "grand_place_brussels",
        "pantheon_exterior": "pantheon_exterior",
        "sacre_coeur": "sacre_coeur",
    }.get(dataset, dataset)


def load_glomap_row(dataset: str, mode_dir: str, label: str) -> dict | None:
    """GLOMAP registers all GT cams, so CO AUC == full AUC. We still emit both
    columns so the table stays a clean grid — the paired values will be identical.
    """
    path = GLOMAP_DIR / _gt_subdir(dataset) / mode_dir / "glomap_metrics.json"
    if not path.exists():
        return None
    blob = json.load(open(path)).get("glomap_metrics", {})
    return {
        "label": label,
        "cams": blob.get("number_matched_cameras") or blob.get("number_gt_cameras"),
        "rot": _get_median(blob, "rotation_angle_error_deg"),
        "trans": _get_median(blob, "translation_angle_error_deg"),
        "auc3":  blob.get("pose_auc_@3.0_deg"),
        "auc5":  blob.get("pose_auc_@5.0_deg"),
        "auc10": blob.get("pose_auc_@10.0_deg"),
        "co3":   blob.get("pose_auc_constructed_only_@3.0_deg") or blob.get("pose_auc_@3.0_deg"),
        "co5":   blob.get("pose_auc_constructed_only_@5.0_deg") or blob.get("pose_auc_@5.0_deg"),
        "co10":  blob.get("pose_auc_constructed_only_@10.0_deg") or blob.get("pose_auc_@10.0_deg"),
    }


def fmt_row(row: dict, suffix: str = "") -> str:
    def _f(v, w, d):
        if v is None:
            return " " * (w - 1) + "-"
        return f"{v:>{w}.{d}f}"
    def _fi(v, w):
        if v is None:
            return " " * (w - 1) + "-"
        return f"{int(v):>{w}d}"

    label = row["label"]
    return (
        f"{label:<30}"
        f"{_fi(row['cams'], 5)}  "
        f"{_f(row['rot'], 5, 2)}  "
        f"{_f(row['trans'], 7, 2)}  "
        f"{_f(row['auc3'], 6, 3)}  "
        f"{_f(row['auc5'], 6, 3)}  "
        f"{_f(row['auc10'], 6, 3)}  "
        f"{_f(row.get('co3'), 6, 3)}  "
        f"{_f(row.get('co5'), 6, 3)}  "
        f"{_f(row.get('co10'), 6, 3)}"
        f"{suffix}"
    )


def render_block(dataset: str) -> None:
    header = (
        f"{'':<30}"
        f"{'Cams':>5}  {'Rot°':>5}  {'Trans°':>7}  "
        f"{'AUC@3':>6}  {'AUC@5':>6}  {'AUC@10':>6}  "
        f"{'CO@3':>6}  {'CO@5':>6}  {'CO@10':>6}"
    )
    rule = "─" * len(header)

    print(f"\n=== {dataset} ===")
    print(header)
    print(rule)

    for mode_dir, label in GLOMAP_MODES:
        row = load_glomap_row(dataset, mode_dir, label)
        if row is None:
            row = {
                "label": label, "cams": None, "rot": None, "trans": None,
                "auc3": None, "auc5": None, "auc10": None,
                "co3": None, "co5": None, "co10": None,
            }
        print(fmt_row(row))
        # Interleave GTSFM (GP only) directly after GLOMAP's GP-only row for direct compare.
        if mode_dir == "gp_only":
            gp_row = load_gtsfm_gp_row(dataset)
            if gp_row is not None:
                print(fmt_row(gp_row))

    print(rule)

    gt = load_gtsfm_row(dataset)
    if gt is None:
        print(f"{'GTSFM (Config: sift_gp_single)':<30}  (no F results for {dataset})")
    else:
        print(fmt_row(gt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Render only the given dataset (default: all four).",
    )
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else DATASETS
    for ds in targets:
        render_block(ds)


if __name__ == "__main__":
    main()
