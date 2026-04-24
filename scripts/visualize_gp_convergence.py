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
from gtsfm.global_positioner.global_positioner import compute_world_directions, filter_tracks


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
    observations = compute_world_directions(filtered_tracks, intrinsics, wRi_list, valid_cameras)

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


def compute_up_alignment(
    cam_positions: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    cam_rotations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a rotation matrix that aligns the scene's up-axis with +Z.

    After applying this rotation to all points, matplotlib's azimuth parameter
    will orbit around the scene's true up axis — no more "going under" the
    reconstruction during orbit.

    Strategy:
    1. Primary: average the world-space "up" direction of every camera
       (negative Y in camera frame, since image Y points down). For upright
       tourist cameras this gives the scene's true up axis directly.
    2. Fallback: PCA least-variance axis when rotations are unavailable or
       the averaged camera-up is inconsistent (|sum| small).

    Always oriented so that landmarks (when available) sit above cameras.
    """
    scene_up = None

    chosen_source = None

    if cam_rotations is not None and len(cam_rotations) > 0:
        # Camera image-up in world: wRi @ [0, -1, 0] — the vector that points
        # from the bottom of the image toward the top, expressed in world.
        image_up = np.array([0.0, -1.0, 0.0])
        per_cam_up = cam_rotations @ image_up  # (N, 3)
        avg_up = per_cam_up.mean(axis=0)
        norm = np.linalg.norm(avg_up)
        print(f"  [up-alignment] camera image-up consistency: |avg|={norm:.3f} (1.0 = perfect agreement)")
        # Only trust it if cameras mostly agree (avg magnitude > 0.5 = majority aligned).
        if norm > 0.5:
            scene_up = avg_up / norm
            chosen_source = "camera image-up"

    if scene_up is None:
        # Fallback: PCA least-variance axis.
        pts = cam_positions if landmarks is None or len(landmarks) == 0 else np.vstack([cam_positions, landmarks])
        centered = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        scene_up = Vt[2]
        chosen_source = "PCA least-variance"
    print(f"  [up-alignment] source: {chosen_source}, scene_up={scene_up}")

    # Orient: landmarks should sit above cameras (heuristic tiebreaker).
    # Applied for both rotation-based and PCA paths — ceiling-dominated scenes
    # like the British Museum Great Court can have cameras tilted back, so the
    # averaged image-up vector points away from the actual scene up. The
    # landmarks-above-cameras check catches this and flips when wrong.
    if landmarks is not None and len(landmarks) > 0:
        cam_center = cam_positions.mean(axis=0)
        lmk_center = landmarks.mean(axis=0)
        dot = np.dot(lmk_center - cam_center, scene_up)
        print(f"  [up-alignment] landmarks-above-cameras dot={dot:.3f}")
        if dot < 0:
            print("  [up-alignment] flipping: landmarks were below cameras under proposed up")
            scene_up = -scene_up
    elif scene_up[2] < 0:
        scene_up = -scene_up

    # Build orthonormal frame: make scene_up → +Z.
    # Use a standard basis vector as a reference to build the frame.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, scene_up)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(scene_up, ref)
    x_axis /= (np.linalg.norm(x_axis) + 1e-12)
    y_axis = np.cross(scene_up, x_axis)

    # Rotation matrix: rows are the new basis expressed in old coords.
    # After applying R, old scene_up becomes new +Z, old x_axis becomes new +X.
    R = np.stack([x_axis, y_axis, scene_up], axis=0)
    return R


def compute_auto_view_angle(
    cam_positions: np.ndarray, landmarks: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Compute elev/azim for a 3/4 view — assumes scene is already up-aligned.

    Since we pre-rotate the scene so +Z is up, elevation is a straightforward
    30° and azimuth is chosen so we look along the main horizontal spread
    direction (giving a wide-view shot of the reconstruction).
    """
    pts = cam_positions if landmarks is None or len(landmarks) == 0 else np.vstack([cam_positions, landmarks])
    centered = pts - pts.mean(axis=0)

    # After up-alignment, PCA in the XY plane tells us the main horizontal
    # direction of the scene. We want to view perpendicular to that (face-on).
    xy = centered[:, :2]
    _, _, Vt = np.linalg.svd(xy, full_matrices=False)
    main_dir = Vt[0]  # main horizontal spread axis

    # Look perpendicular to main_dir: azim points along the normal direction.
    # In matplotlib, azim=0 looks along +X toward origin; azim=90 looks along +Y.
    perp = np.array([-main_dir[1], main_dir[0]])  # 90° rotation in XY
    azim = np.degrees(np.arctan2(perp[1], perp[0]))

    elev = 30.0  # 3/4 view — cameras + structure both visible
    return elev, azim


def _rot_to_matrix(wRi: Optional[Rot3]) -> np.ndarray:
    """Get 3x3 world-from-camera rotation matrix, or identity if rotation is None."""
    if wRi is None:
        return np.eye(3)
    try:
        return np.array(wRi.matrix())
    except Exception:
        return np.eye(3)


def extract_positions_from_dict(
    pos_dict: dict,
    wRi_list: List[Optional[Rot3]],
    camera_indices: Set[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract positions from pre-extracted numpy dict format (new trace format)."""
    cam_positions = []
    cam_forwards = []
    cam_rotations = []
    for cam_idx in sorted(camera_indices):
        pos = pos_dict["cameras"].get(cam_idx, np.zeros(3))
        cam_positions.append(pos)
        wRi = wRi_list[cam_idx] if cam_idx < len(wRi_list) else None
        if wRi is not None:
            fwd = wRi.rotate(Point3(0, 0, 1))
            cam_forwards.append(np.array([fwd[0], fwd[1], fwd[2]]))
        else:
            cam_forwards.append(np.array([0, 0, 1]))
        cam_rotations.append(_rot_to_matrix(wRi))

    landmarks = list(pos_dict["landmarks"].values())
    return (
        np.array(cam_positions),
        np.array(cam_forwards),
        np.array(landmarks) if landmarks else np.empty((0, 3)),
        np.array(cam_rotations) if cam_rotations else np.empty((0, 3, 3)),
    )


def extract_positions(
    values: Values,
    wRi_list: List[Optional[Rot3]],
    camera_indices: Set[int],
    track_indices: Set[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract camera positions, forward directions, and landmark positions."""
    cam_positions = []
    cam_forwards = []
    cam_rotations = []
    for cam_idx in sorted(camera_indices):
        pos = values.atPoint3(C(cam_idx))
        cam_positions.append(pos)
        wRi = wRi_list[cam_idx]
        if wRi is not None:
            fwd = wRi.rotate(Point3(0, 0, 1))
            cam_forwards.append(np.array([fwd[0], fwd[1], fwd[2]]))
        else:
            cam_forwards.append(np.array([0, 0, 1]))
        cam_rotations.append(_rot_to_matrix(wRi))

    landmarks = []
    for track_idx in sorted(track_indices):
        try:
            pt = values.atPoint3(L(track_idx))
            landmarks.append(pt)
        except RuntimeError:
            continue

    return (
        np.array(cam_positions),
        np.array(cam_forwards),
        np.array(landmarks),
        np.array(cam_rotations) if cam_rotations else np.empty((0, 3, 3)),
    )


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
    cam_rotations: Optional[np.ndarray] = None,
):
    """Render a single frame with cameras (red frustums) and points (colored or black)."""
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

    # Cameras as proper wireframe frustums.
    if len(cam_positions) > 0:
        scale = view_range * 0.03  # size of frustum along viewing direction
        half = scale * 0.5  # half-width of image plane

        # Local-frame frustum corners (camera looks down +Z, X right, Y down).
        local_corners = np.array([
            [-half, -half, scale],
            [ half, -half, scale],
            [ half,  half, scale],
            [-half,  half, scale],
        ])

        # Batched: if rotations available, use them; else use forward vectors.
        if cam_rotations is not None and len(cam_rotations) == len(cam_positions):
            # (N, 4, 3) world-frame corners.
            rotated = np.einsum("nij,kj->nki", cam_rotations, local_corners)
            world_corners = rotated + cam_positions[:, np.newaxis, :]
        else:
            # Fallback: align frustum Z with forward vector (less accurate, just for viz).
            world_corners = np.zeros((len(cam_positions), 4, 3))
            for n, (pos, fwd) in enumerate(zip(cam_positions, cam_forwards)):
                # Build orthonormal frame: z=fwd, x=right, y=up
                z = fwd / (np.linalg.norm(fwd) + 1e-12)
                up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
                x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-12)
                y = np.cross(z, x)
                R = np.column_stack([x, y, z])
                world_corners[n] = (R @ local_corners.T).T + pos

        # Collect all line segments in two arrays for a single call.
        # Each camera contributes 8 segments: 4 apex-to-corner + 4 corner-to-corner.
        n_cams = len(cam_positions)
        # Apex → corner lines: 4 per camera.
        apex_lines_start = np.repeat(cam_positions[:, np.newaxis, :], 4, axis=1).reshape(-1, 3)
        apex_lines_end = world_corners.reshape(-1, 3)
        # Corner quad closing lines: 4 per camera.
        quad_start = world_corners  # (N, 4, 3)
        quad_end = world_corners[:, [1, 2, 3, 0], :]  # cyclic shift
        quad_lines_start = quad_start.reshape(-1, 3)
        quad_lines_end = quad_end.reshape(-1, 3)

        # Draw with a single Line3DCollection for speed.
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        apex_segments = np.stack([apex_lines_start, apex_lines_end], axis=1)
        quad_segments = np.stack([quad_lines_start, quad_lines_end], axis=1)
        all_segments = np.concatenate([apex_segments, quad_segments], axis=0)
        lc = Line3DCollection(all_segments, colors='red', linewidths=0.7, alpha=0.85)
        ax.add_collection3d(lc)

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
    parser.add_argument("--orbit_degrees", type=float, default=60.0, help="Total azimuth sweep across the animation (0 = fixed view)")
    parser.add_argument("--flip_up", action="store_true", help="Invert the auto-detected up axis (use if render is upside down)")
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
        error_trace = trace_data["error_trace"]
        camera_indices = trace_data["camera_indices"]
        track_indices = trace_data["track_indices"]
        track_colors = trace_data.get("track_colors", None)
        filtered_tracks = trace_data.get("filtered_tracks", None)

        # Support both old format (values_trace with GTSAM Values) and
        # new format (positions_trace with pure numpy dicts)
        if "positions_trace" in trace_data:
            positions_trace = trace_data["positions_trace"]
            wRi_matrices = trace_data["wRi_matrices"]
            # Build wRi_list from rotation matrices
            max_idx = max(camera_indices) if camera_indices else 0
            wRi_list = [None] * (max_idx + 1)
            for ci, mat in wRi_matrices.items():
                wRi_list[ci] = Rot3(mat)
            use_positions_trace = True
        else:
            values_trace = trace_data["values_trace"]
            wRi_list_sparse = trace_data["wRi_list"]
            wRi_indices = trace_data["wRi_indices"]
            wRi_list = [None] * (max(wRi_indices) + 1)
            for idx, wRi in zip(wRi_indices, wRi_list_sparse):
                wRi_list[idx] = wRi
            use_positions_trace = False

        num_frames = len(positions_trace) if use_positions_trace else len(values_trace)
        print(f"Loaded {num_frames} frames, {len(camera_indices)} cameras, {len(track_indices)} tracks")

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
        use_positions_trace = False

    # Raw position extractors (scene-frame).
    def get_frame_positions_raw(frame_idx_local):
        if use_positions_trace:
            return extract_positions_from_dict(positions_trace[frame_idx_local], wRi_list, camera_indices)
        else:
            return extract_positions(values_trace[frame_idx_local], wRi_list, camera_indices, track_indices)

    # Step 3: Compute world-up alignment — find the scene's "up" axis via PCA
    # over the final reconstruction, then rotate so that axis aligns with
    # matplotlib's +Z. This makes matplotlib's azimuth orbit around the scene's
    # actual up axis (instead of the arbitrary reconstruction Z axis).
    num_iters = len(positions_trace) if use_positions_trace else len(values_trace)
    print("\n" + "=" * 60)
    print(f"Step 3: Rendering {num_iters} frames...")
    print("=" * 60)
    final_cams_raw, _, final_lmks_raw, final_rots_raw = get_frame_positions_raw(num_iters - 1)

    align_R = compute_up_alignment(final_cams_raw, final_lmks_raw, final_rots_raw)
    if args.flip_up:
        # Flip scene upside-down: rotate 180° around X so +Z → -Z.
        flip = np.diag([1.0, -1.0, -1.0])
        align_R = flip @ align_R
        print("  [up-alignment] --flip_up applied (render was upside down)")

    # Apply alignment transform to every frame.
    def get_frame_positions(frame_idx_local):
        cams, fwds, lmks, rots = get_frame_positions_raw(frame_idx_local)
        if len(cams) > 0:
            cams = cams @ align_R.T
            fwds = fwds @ align_R.T
            # Rotate per-camera rotation matrices: R_world_new = align_R @ R_world_old.
            # Einsum "ij,njk->nik" computes align_R @ rots[n] for each n.
            if len(rots) > 0:
                rots = np.einsum("ij,njk->nik", align_R, rots)
        if len(lmks) > 0:
            lmks = lmks @ align_R.T
        return cams, fwds, lmks, rots

    final_cams, _, final_lmks, _ = get_frame_positions(num_iters - 1)

    # Size view bounds to the final (converged) scene. Initial random points
    # may fly off-frame during the first few iterations, which is visually
    # dramatic and matches the original animation's feel.
    all_pts = np.vstack([final_cams, final_lmks]) if len(final_lmks) > 0 else final_cams
    view_center = np.median(all_pts, axis=0)
    dists = np.linalg.norm(all_pts - view_center, axis=1)
    view_range = np.percentile(dists, 90) * 1.05

    # Auto-compute view angle from camera geometry (or use manual override)
    if args.elev is None or args.azim is None:
        auto_elev, auto_azim = compute_auto_view_angle(final_cams, final_lmks)
        auto_elev = np.clip(auto_elev, 20.0, 45.0)
        elev = auto_elev if args.elev is None else args.elev
        azim = auto_azim if args.azim is None else args.azim
        print(f"Auto view angle: elev={elev:.1f}°, azim={azim:.1f}°")
    else:
        elev, azim = args.elev, args.azim
        print(f"Manual view angle: elev={elev:.1f}°, azim={azim:.1f}°")

    # Render frames with interpolation between iterations (fixed view by default).
    initial_error = error_trace[0]
    interp_steps = args.interp_steps
    total_frames = (num_iters - 1) * interp_steps + 1
    azim_sweep = args.orbit_degrees

    frame_idx = 0
    for i in range(num_iters):
        cam_pos_i, cam_fwd_i, lmk_pos_i, cam_rot_i = get_frame_positions(i)

        if i < num_iters - 1:
            cam_pos_next, cam_fwd_next, lmk_pos_next, _ = get_frame_positions(i + 1)
            steps = interp_steps
        else:
            steps = 1  # last iteration, no interpolation

        for k in range(steps):
            t = k / interp_steps if steps > 1 else 0.0
            # Lerp positions
            cam_pos = cam_pos_i * (1 - t) + cam_pos_next * t if i < num_iters - 1 else cam_pos_i
            lmk_pos = lmk_pos_i * (1 - t) + lmk_pos_next * t if i < num_iters - 1 else lmk_pos_i
            error = error_trace[i] * (1 - t) + error_trace[min(i + 1, num_iters - 1)] * t

            # Center orbit on the auto-computed best angle (sweep from -half to +half).
            frame_azim = azim - azim_sweep / 2 + (frame_idx / max(total_frames - 1, 1)) * azim_sweep
            frame_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.png")
            render_frame(
                cam_pos, cam_fwd_i, lmk_pos, i, error, initial_error,
                frame_path, view_center, view_range, track_colors, elev, frame_azim,
                cam_rotations=cam_rot_i,
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
