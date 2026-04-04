"""Visualize global positioner convergence from random initialization.

Creates a frame-by-frame animation showing cameras (red) and 3D points (black)
converging from random positions to a structured scene over LM iterations.

Two-phase workflow:
  1. Run the GTSFM pipeline normally (saves results to results/ directory)
  2. Run this script to load the saved results, replay GP with per-iteration
     capture, and render frames.

Usage:
    # First, run the pipeline:
    ./run --config_name sift_global_positioner --loader olsson \
          --dataset_dir tests/data/set1_lund_door --num_workers 1

    # Then, render the convergence animation:
    python scripts/visualize_gp_convergence.py \
        --results_dir results/ba_input \
        --max_iterations 50 \
        --output_dir /tmp/gp_frames

    # Stitch to video:
    ffmpeg -framerate 5 -i /tmp/gp_frames/frame_%04d.png \
           -c:v libx264 -pix_fmt yuv420p convergence.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import gtsam
from gtsam import (
    BinaryMeasurementUnit3,
    BinaryMeasurementsUnit3,
    Point3,
    Pose3,
    Rot3,
    Symbol,
    TranslationRecovery,
    Unit3,
    Values,
)
from gtsam.symbol_shorthand import A as C  # Camera keys
from gtsam.symbol_shorthand import B as L  # Landmark keys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.io as io_utils
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.global_positioner.global_positioner import compute_world_bearings, filter_tracks


def load_rotations_and_tracks_from_colmap(
    colmap_dir: str,
    images_dir: Optional[str] = None,
) -> Tuple[List[Optional[Rot3]], List[Optional[gtsfm_types.CALIBRATION_TYPE]], List[SfmTrack2d], int, Optional[np.ndarray]]:
    """Load rotations, intrinsics, 2D tracks, and per-track RGB colors from COLMAP-format output.

    Returns:
        wRi_list, intrinsics, tracks_2d, num_images, track_colors (N,3) float [0,1] or None
    """
    wTi_list, img_fnames, calibrations, point_cloud, rgb, img_dims = io_utils.read_scene_data_from_colmap_format(
        colmap_dir
    )

    num_images = len(wTi_list)

    # Extract rotations
    wRi_list: List[Optional[Rot3]] = []
    for wTi in wTi_list:
        if wTi is not None:
            wRi_list.append(wTi.rotation())
        else:
            wRi_list.append(None)

    # Build intrinsics list
    intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]] = list(calibrations)

    # Build 2D tracks from the COLMAP points3D data
    import pycolmap
    from gtsfm.common.sfm_track import SfmMeasurement
    from PIL import Image

    reconstruction = pycolmap.Reconstruction(colmap_dir)

    # Try to load images for color sampling
    loaded_images = {}
    if images_dir is None:
        # Try common relative paths
        for candidate in [os.path.join(os.path.dirname(colmap_dir), "images"),
                          os.path.join(colmap_dir, "..", "images")]:
            if os.path.isdir(candidate):
                images_dir = candidate
                break

    if images_dir and os.path.isdir(images_dir):
        # Try matching by COLMAP name first, then fall back to sorted directory listing
        for img_id, image in reconstruction.images.items():
            img_path = os.path.join(images_dir, image.name)
            if os.path.exists(img_path):
                try:
                    loaded_images[img_id] = np.array(Image.open(img_path))
                except Exception:
                    pass

        # If no matches by name, load by sorted order (GTSFM renames images internally)
        if not loaded_images:
            sorted_img_ids = sorted(reconstruction.images.keys())
            img_files = sorted([f for f in os.listdir(images_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
            for img_id, fname in zip(sorted_img_ids, img_files):
                try:
                    loaded_images[img_id] = np.array(Image.open(os.path.join(images_dir, fname)))
                except Exception:
                    pass

        print(f"Loaded {len(loaded_images)} images for color sampling")

    tracks_2d = []
    track_colors = []

    for point3D_id, point3D in reconstruction.points3D.items():
        measurements = []
        sampled_color = None

        for track_elem in point3D.track.elements:
            img_id = track_elem.image_id
            point2D_idx = track_elem.point2D_idx
            image = reconstruction.images[img_id]
            uv = image.points2D[point2D_idx].xy
            cam_idx = img_id - 1 if img_id > 0 else img_id
            measurements.append(SfmMeasurement(i=cam_idx, uv=np.array(uv)))

            # Sample color from first available image
            if sampled_color is None and img_id in loaded_images:
                img_arr = loaded_images[img_id]
                u, v = int(round(uv[0])), int(round(uv[1]))
                if 0 <= v < img_arr.shape[0] and 0 <= u < img_arr.shape[1]:
                    pixel = img_arr[v, u]
                    if len(pixel) >= 3:
                        sampled_color = pixel[:3] / 255.0

        if len(measurements) >= 2:
            tracks_2d.append(SfmTrack2d(measurements=measurements))
            track_colors.append(sampled_color if sampled_color is not None else [0.3, 0.3, 0.3])

    track_colors_arr = np.array(track_colors) if track_colors else None
    print(f"Loaded {num_images} cameras, {len(tracks_2d)} tracks from {colmap_dir}")
    if track_colors_arr is not None and len(loaded_images) > 0:
        colored = np.sum(np.any(track_colors_arr != 0.3, axis=1))
        print(f"Sampled RGB color for {colored}/{len(tracks_2d)} tracks")

    return wRi_list, intrinsics, tracks_2d, num_images, track_colors_arr


def build_graph_and_iterate(
    wRi_list: List[Optional[Rot3]],
    intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
    tracks_2d: List[SfmTrack2d],
    num_images: int,
    max_iterations: int = 50,
    min_track_measurements: int = 3,
) -> Tuple[List[Values], List[float], Set[int], Set[int]]:
    """Build the global positioner graph and capture values at each iteration."""
    valid_cameras = {i for i, wRi in enumerate(wRi_list) if wRi is not None}
    filtered_tracks = filter_tracks(tracks_2d, valid_cameras, min_track_measurements)
    observations = compute_world_bearings(filtered_tracks, intrinsics, wRi_list, valid_cameras)

    print(f"Building graph: {len(observations)} observations, {len(valid_cameras)} cameras, {len(filtered_tracks)} tracks")

    # Build measurements
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 0.01)
    huber = gtsam.noiseModel.mEstimator.Huber.Create(0.1)
    robust_noise = gtsam.noiseModel.Robust.Create(huber, noise_model)

    measurements = BinaryMeasurementsUnit3()
    camera_indices: Set[int] = set()
    track_indices: Set[int] = set()

    for track_idx, cam_idx, bearing in observations:
        measurements.append(BinaryMeasurementUnit3(
            C(cam_idx), L(track_idx), Unit3(Point3(*bearing)), robust_noise,
        ))
        camera_indices.add(cam_idx)
        track_indices.add(track_idx)

    # Build graph via TranslationRecovery (uses native C++ BilinearAngleTranslationFactor)
    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(max_iterations)
    lm_params.setVerbosityLM("SILENT")
    tr = TranslationRecovery(lm_params, use_bilinear_translation_factor=True)

    graph = tr.buildGraph(measurements)
    tr.addPrior(measurements, 1.0, gtsam.BinaryMeasurementsPoint3(), graph)

    # Initialize randomly (matching TranslationRecovery C++ code: uniform(-1, 1))
    rng = np.random.RandomState(42)
    initial = Values()
    for cam_idx in camera_indices:
        initial.insert(C(cam_idx), Point3(*rng.uniform(-1, 1, 3)))
    for track_idx in track_indices:
        initial.insert(L(track_idx), Point3(*rng.uniform(-1, 1, 3)))
    # Scale variables (one per observation, initialized to 1.0)
    for i in range(len(observations)):
        initial.insert(Symbol('S', i).key(), np.array([[1.0]]))

    initial_error = graph.error(initial)
    print(f"Initial error: {initial_error:.2f}")

    # Iterate and capture values at each step
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, lm_params)
    values_trace = [initial]
    error_trace = [initial_error]

    for iteration in range(1, max_iterations + 1):
        t0 = time.time()
        optimizer.iterate()
        dt = time.time() - t0
        error = optimizer.error()
        values_trace.append(optimizer.values())
        error_trace.append(error)

        if iteration <= 5 or iteration % 5 == 0:
            print(f"  iter {iteration:3d}: error={error:.2f}  ({dt:.2f}s)")

    print(f"Final error: {error_trace[-1]:.4f} ({len(values_trace)} frames captured)")
    return values_trace, error_trace, camera_indices, track_indices


def compute_auto_view_angle(cam_positions: np.ndarray) -> Tuple[float, float]:
    """Compute elev/azim that looks along the normal to the camera plane.

    Uses PCA to find the plane that best fits the camera positions.
    The view direction is set along the least-variance axis (normal to the
    camera arrangement), giving a natural front-on or overhead view depending
    on the scene geometry.
    """
    centered = cam_positions - cam_positions.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    # 3rd principal component = normal to camera plane (least variance direction)
    normal = Vt[2]

    # Ensure we look from the "positive" side (arbitrary but consistent)
    if normal[2] < 0:
        normal = -normal

    # Convert to matplotlib elev/azim (degrees)
    elev = np.degrees(np.arcsin(np.clip(normal[2], -1, 1)))
    azim = np.degrees(np.arctan2(normal[1], normal[0]))

    return elev, azim


def extract_positions(
    values: Values,
    wRi_list: List[Optional[Rot3]],
    camera_indices: Set[int],
    track_indices: Set[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract camera positions, forward directions, and landmark positions."""
    cam_positions = []
    cam_forwards = []
    for cam_idx in sorted(camera_indices):
        pos = values.atPoint3(C(cam_idx))
        cam_positions.append(pos)
        wRi = wRi_list[cam_idx]
        if wRi is not None:
            fwd = wRi.rotate(Point3(0, 0, 1))
            cam_forwards.append(np.array([fwd[0], fwd[1], fwd[2]]))
        else:
            cam_forwards.append(np.array([0, 0, 1]))

    landmarks = []
    for track_idx in sorted(track_indices):
        try:
            pt = values.atPoint3(L(track_idx))
            landmarks.append(pt)
        except RuntimeError:
            continue

    return np.array(cam_positions), np.array(cam_forwards), np.array(landmarks)


def render_frame(
    cam_positions: np.ndarray,
    cam_forwards: np.ndarray,
    landmarks: np.ndarray,
    iteration: int,
    error: float,
    initial_error: float,
    output_path: str,
    view_center: np.ndarray,
    view_range: float,
    point_colors: Optional[np.ndarray] = None,
    elev: float = 30,
    azim: float = -60,
):
    """Render a single frame with cameras (red) and points (colored or black)."""
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Points
    if len(landmarks) > 0:
        # Subsample for rendering speed
        max_pts = 5000
        # Match color array size to landmarks (trace may have more tracks than COLMAP output)
        if point_colors is not None and len(point_colors) < len(landmarks):
            point_colors = None  # size mismatch, fall back to no color

        if len(landmarks) > max_pts:
            idx = np.random.RandomState(0).choice(len(landmarks), max_pts, replace=False)
            lm_plot = landmarks[idx]
            colors_plot = point_colors[idx] if point_colors is not None else None
        else:
            lm_plot = landmarks
            colors_plot = point_colors

        if colors_plot is not None:
            ax.scatter(lm_plot[:, 0], lm_plot[:, 1], lm_plot[:, 2],
                       c=colors_plot, s=0.8, alpha=0.7, depthshade=False)
        else:
            ax.scatter(lm_plot[:, 0], lm_plot[:, 1], lm_plot[:, 2],
                       c='black', s=0.5, alpha=0.4, depthshade=True)

    # Cameras (red triangles with forward direction)
    if len(cam_positions) > 0:
        ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
                   c='red', s=60, marker='^', alpha=0.9, depthshade=True,
                   edgecolors='darkred', linewidths=0.3)

        # Draw forward direction as short lines
        scale = view_range * 0.03
        for pos, fwd in zip(cam_positions, cam_forwards):
            end = pos + fwd * scale
            ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                    c='red', alpha=0.6, linewidth=1)

    # Consistent view bounds
    ax.set_xlim(view_center[0] - view_range, view_center[0] + view_range)
    ax.set_ylim(view_center[1] - view_range, view_center[1] + view_range)
    ax.set_zlim(view_center[2] - view_range, view_center[2] + view_range)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # Compute error reduction percentage
    reduction = (1.0 - error / initial_error) * 100 if initial_error > 0 else 0

    ax.set_title(
        f"Global Positioner Convergence  —  GLOMAP-style BATA via GTSAM\n"
        f"Iteration {iteration}  |  Cost: {error:,.0f}  |  "
        f"Reduction: {reduction:.1f}%  |  "
        f"{len(cam_positions)} cameras, {len(landmarks)} points",
        fontsize=13, fontweight='bold', pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize global positioner convergence")
    parser.add_argument("--trace_file", type=str, default="results/gp_convergence_trace.pkl",
                        help="Path to GP convergence trace pickle from a pipeline run with save_iteration_visualization=True")
    parser.add_argument("--results_dir", type=str, default="results/ba_input",
                        help="Fallback: Path to COLMAP-format results (used if trace file not found)")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Path to images directory for RGB color sampling")
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--min_track_measurements", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="/tmp/gp_convergence_frames")
    parser.add_argument("--interp_steps", type=int, default=5, help="Interpolated sub-frames between each iteration")
    parser.add_argument("--elev", type=float, default=None, help="Camera elevation angle (auto-computed if not set)")
    parser.add_argument("--azim", type=float, default=None, help="Camera azimuth angle (auto-computed if not set)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Try loading pre-saved trace from pipeline run (preferred)
    if os.path.exists(args.trace_file):
        print("=" * 60)
        print(f"Loading convergence trace from {args.trace_file}...")
        print("=" * 60)
        import pickle
        with open(args.trace_file, "rb") as f:
            trace_data = pickle.load(f)
        values_trace = trace_data["values_trace"]
        error_trace = trace_data["error_trace"]
        camera_indices = trace_data["camera_indices"]
        track_indices = trace_data["track_indices"]
        # Reconstruct wRi_list from saved data
        wRi_list_sparse = trace_data["wRi_list"]
        wRi_indices = trace_data["wRi_indices"]
        wRi_list = [None] * (max(wRi_indices) + 1)
        for idx, wRi in zip(wRi_indices, wRi_list_sparse):
            wRi_list[idx] = wRi
        track_colors = trace_data.get("track_colors", None)
        filtered_tracks = trace_data.get("filtered_tracks", None)
        print(f"Loaded {len(values_trace)} frames, {len(camera_indices)} cameras, {len(track_indices)} tracks")

        # Sample colors from images using the saved 2D tracks
        if track_colors is None and filtered_tracks is not None and args.images_dir:
            print("Sampling colors from images using saved tracks...")
            from PIL import Image as PILImage
            from gtsfm.common.sfm_track import SfmMeasurement

            # Load images by sorted filename
            img_files = sorted([f for f in os.listdir(args.images_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
            loaded_images = {}
            for i, fname in enumerate(img_files):
                try:
                    loaded_images[i] = np.array(PILImage.open(os.path.join(args.images_dir, fname)))
                except Exception:
                    pass
            print(f"  Loaded {len(loaded_images)} images")

            # Compute UV scale factor: pipeline may have downscaled images
            sample_img = next(iter(loaded_images.values()))
            img_h, img_w = sample_img.shape[:2]
            max_uv_u, max_uv_v = 0, 0
            for t in filtered_tracks[:200]:
                for m_idx in range(t.number_measurements()):
                    m = t.measurement(m_idx)
                    max_uv_u = max(max_uv_u, m.uv[0])
                    max_uv_v = max(max_uv_v, m.uv[1])
            # If UVs are much smaller than image dims, features were detected at lower res
            scale_u = img_w / max(max_uv_u, 1) if max_uv_u < img_w * 0.8 else 1.0
            scale_v = img_h / max(max_uv_v, 1) if max_uv_v < img_h * 0.8 else 1.0
            if scale_u > 1.5 or scale_v > 1.5:
                print(f"  UV scale correction: {scale_u:.2f}x, {scale_v:.2f}x (pipeline used downscaled images)")
            else:
                scale_u, scale_v = 1.0, 1.0

            colors_list = []
            for track in filtered_tracks:
                sampled = None
                # Collect colors from all observing cameras, take median
                all_colors = []
                for m_idx in range(track.number_measurements()):
                    m = track.measurement(m_idx)
                    if m.i in loaded_images:
                        img = loaded_images[m.i]
                        u = int(round(m.uv[0] * scale_u))
                        v = int(round(m.uv[1] * scale_v))
                        if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                            pixel = img[v, u]
                            if img.ndim == 3 and img.shape[2] >= 3:
                                all_colors.append(pixel[:3] / 255.0)
                            elif img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                                gray = float(pixel if img.ndim == 2 else pixel[0]) / 255.0
                                all_colors.append([gray, gray, gray])
                if all_colors:
                    sampled = np.median(all_colors, axis=0)
                colors_list.append(sampled if sampled is not None else [0.3, 0.3, 0.3])
            track_colors = np.array(colors_list)
            colored = np.sum(np.any(track_colors != 0.3, axis=1))
            print(f"  Sampled color for {colored}/{len(filtered_tracks)} tracks")
    else:
        # Fallback: load from COLMAP results and replay
        print("=" * 60)
        print(f"No trace file found at {args.trace_file}")
        print("Loading from COLMAP results and replaying optimization...")
        print("=" * 60)
        wRi_list, intrinsics, tracks_2d, num_images, track_colors = load_rotations_and_tracks_from_colmap(
            args.results_dir, getattr(args, 'images_dir', None)
        )

        values_trace, error_trace, camera_indices, track_indices = build_graph_and_iterate(
            wRi_list, intrinsics, tracks_2d, num_images,
            max_iterations=args.max_iterations,
            min_track_measurements=args.min_track_measurements,
        )

    # Step 3: Compute view bounds from final converged result
    print("\n" + "=" * 60)
    print(f"Step 3: Rendering {len(values_trace)} frames...")
    print("=" * 60)
    final_cams, _, final_lmks = extract_positions(
        values_trace[-1], wRi_list, camera_indices, track_indices
    )

    all_pts = np.vstack([final_cams, final_lmks]) if len(final_lmks) > 0 else final_cams
    view_center = np.median(all_pts, axis=0)
    dists = np.linalg.norm(all_pts - view_center, axis=1)
    view_range = np.percentile(dists, 90) * 1.05

    # Auto-compute view angle from camera geometry (or use manual override)
    if args.elev is None or args.azim is None:
        auto_elev, auto_azim = compute_auto_view_angle(final_cams)
        # Cap elevation so it's never too top-down — keep between 20° and 45°
        auto_elev = np.clip(auto_elev, 20.0, 45.0)
        elev = auto_elev if args.elev is None else args.elev
        azim = auto_azim if args.azim is None else args.azim
        print(f"Auto view angle: elev={elev:.1f}°, azim={azim:.1f}°")
    else:
        elev, azim = args.elev, args.azim
        print(f"Manual view angle: elev={elev:.1f}°, azim={azim:.1f}°")

    # Render frames with interpolation between iterations + slow orbit
    initial_error = error_trace[0]
    num_iters = len(values_trace)
    interp_steps = args.interp_steps  # sub-frames between each iteration
    total_frames = (num_iters - 1) * interp_steps + 1
    azim_sweep = 120  # total degrees to sweep

    frame_idx = 0
    for i in range(num_iters):
        cam_pos_i, cam_fwd_i, lmk_pos_i = extract_positions(values_trace[i], wRi_list, camera_indices, track_indices)

        if i < num_iters - 1:
            cam_pos_next, cam_fwd_next, lmk_pos_next = extract_positions(
                values_trace[i + 1], wRi_list, camera_indices, track_indices
            )
            steps = interp_steps
        else:
            steps = 1  # last iteration, no interpolation

        for k in range(steps):
            t = k / interp_steps if steps > 1 else 0.0
            # Lerp positions
            cam_pos = cam_pos_i * (1 - t) + cam_pos_next * t if i < num_iters - 1 else cam_pos_i
            lmk_pos = lmk_pos_i * (1 - t) + lmk_pos_next * t if i < num_iters - 1 else lmk_pos_i
            error = error_trace[i] * (1 - t) + error_trace[min(i + 1, num_iters - 1)] * t

            frame_azim = azim + (frame_idx / max(total_frames - 1, 1)) * azim_sweep
            frame_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.png")
            render_frame(
                cam_pos, cam_fwd_i, lmk_pos, i, error, initial_error,
                frame_path, view_center, view_range, track_colors, elev, frame_azim,
            )

            if frame_idx % 20 == 0 or frame_idx == total_frames - 1:
                print(f"  Rendered frame {frame_idx}/{total_frames - 1}  (iter {i}, error: {error:,.0f})")
            frame_idx += 1

    print(f"\n{'=' * 60}")
    print(f"{frame_idx} frames saved to {args.output_dir}")
    print(f"\nTo create video:")
    print(f"  ffmpeg -y -framerate 10 -i {args.output_dir}/frame_%04d.png \\")
    print(f"         -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" \\")
    print(f"         -c:v libx264 -pix_fmt yuv420p convergence.mp4")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
