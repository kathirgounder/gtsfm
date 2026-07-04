"""Unit tests for global-track-ID merge anchoring (gid-index sidecar).

Two clusters that triangulated the SAME global tracks from DISJOINT camera sets must be matchable by
global identity (the shared-camera matcher finds nothing there), and the MERGE_GUARD must drop a large
child whose ID-matched correspondences are too few to constrain its 7-DoF Sim3 seat.

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmTrack

from gtsfm.cluster_merging import (
    _SCENE_GID_INDEX_ATTR,
    _lookup_gid,
    _select_gid_point_correspondences,
    _track_gid,
    _union_gid_indices,
    annotate_scene_with_metadata,
    build_measurement_gid_arrays,
    merge_scenes_with_sim3_nonlinear,
    slice_gid_index,
)
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d

NUM_IMAGES = 64
PARENT_CAMS = [0, 1, 2]
CHILD_CAMS = [10, 11, 12]  # DISJOINT from parent — no shared cameras anywhere.


def _uv(gid: int, cam: int) -> np.ndarray:
    """Deterministic per-(track, camera) pixel; same values used for global tracks and scene tracks."""
    return np.array([13.0 * gid + cam + 0.25, 7.0 * gid + 2 * cam + 0.75])


def _global_tracks(num: int) -> list[SfmTrack2d]:
    """Global 2D tracks spanning BOTH camera sets (the cross-cluster edges made them)."""
    tracks = []
    for gid in range(num):
        meas = [SfmMeasurement(c, _uv(gid, c)) for c in PARENT_CAMS + CHILD_CAMS]
        tracks.append(SfmTrack2d(measurements=meas))
    return tracks


def _scene(cams: list[int], gids: list[int], point_offset: float) -> GtsfmData:
    """A reconstruction over ``cams`` containing one 3D track per gid (measurements = the global uvs)."""
    data = GtsfmData(number_images=NUM_IMAGES)
    for k, c in enumerate(cams):
        pose = Pose3(Rot3(), np.array([k, 0.0, 0.0]))
        data.add_camera(c, PinholeCameraCal3Bundler(pose, Cal3Bundler()))
    for gid in gids:
        track = SfmTrack(Point3(float(gid), point_offset, 0.0))
        for c in cams:
            track.addMeasurement(c, _uv(gid, c))
        assert data.add_track(track)
    return data


class TestGidIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.tracks_2d = _global_tracks(60)
        self.arrays = build_measurement_gid_arrays(self.tracks_2d)

    def test_pack_slice_lookup_roundtrip(self) -> None:
        """Every measurement resolves to its gid through a camera-restricted slice."""
        parent_index = slice_gid_index(*self.arrays, set(PARENT_CAMS))
        child_index = slice_gid_index(*self.arrays, set(CHILD_CAMS))
        self.assertIsNotNone(parent_index)
        for gid in (0, 7, 59):
            self.assertEqual(_lookup_gid(parent_index, 0, _uv(gid, 0)), gid)
            self.assertEqual(_lookup_gid(child_index, 11, _uv(gid, 11)), gid)
        # A camera outside the slice must not resolve.
        self.assertEqual(_lookup_gid(parent_index, 10, _uv(3, 10)), -1)

    def test_correspondences_across_disjoint_cameras(self) -> None:
        """Same global tracks triangulated from disjoint camera sets match by identity."""
        parent = _scene(PARENT_CAMS, list(range(60)), point_offset=0.0)
        child = _scene(CHILD_CAMS, list(range(60)), point_offset=100.0)
        parent_index = slice_gid_index(*self.arrays, set(PARENT_CAMS))
        child_index = slice_gid_index(*self.arrays, set(CHILD_CAMS))

        pairs = _select_gid_point_correspondences(parent, child, parent_index, child_index, max_correspondences=100)
        self.assertEqual(len(pairs), 60)
        # Pairs must join the SAME physical point: x-coords encode the gid on both sides.
        for p_pt, c_pt in pairs:
            self.assertAlmostEqual(p_pt[0], c_pt[0])
            self.assertAlmostEqual(p_pt[1] + 100.0, c_pt[1])

    def test_track_gid_and_union(self) -> None:
        parent_index = slice_gid_index(*self.arrays, set(PARENT_CAMS))
        child_index = slice_gid_index(*self.arrays, set(CHILD_CAMS))
        union = _union_gid_indices(parent_index, child_index)
        parent = _scene(PARENT_CAMS, [5], point_offset=0.0)
        child = _scene(CHILD_CAMS, [5], point_offset=100.0)
        # Both scenes' copies of global track 5 resolve to the same gid through the union index.
        self.assertEqual(_track_gid(parent.get_track(0), union), 5)
        self.assertEqual(_track_gid(child.get_track(0), union), 5)

    def test_merge_guard_drops_large_weak_child(self) -> None:
        """A >=30-camera child with too few ID correspondences is dropped (returns parent unchanged)."""
        parent = _scene(PARENT_CAMS, list(range(60)), point_offset=0.0)
        big_cams = list(range(20, 55))  # 35 cameras, disjoint from parent
        # Child shares only 5 global tracks with the parent -> 5 ID correspondences << 50 floor.
        child = GtsfmData(number_images=NUM_IMAGES)
        for k, c in enumerate(big_cams):
            child.add_camera(c, PinholeCameraCal3Bundler(Pose3(Rot3(), np.array([k, 0.0, 0.0])), Cal3Bundler()))
        for gid in range(5):
            track = SfmTrack(Point3(float(gid), 100.0, 0.0))
            for c in big_cams[:3]:
                track.addMeasurement(c, _uv(gid, c))
            self.assertTrue(child.add_track(track))

        # Global tracks for the child's cameras exist too (so its index resolves).
        tracks_2d = _global_tracks(60)
        for gid in range(5):
            for c in big_cams[:3]:
                tracks_2d[gid].measurements.append(SfmMeasurement(c, _uv(gid, c)))
        arrays = build_measurement_gid_arrays(tracks_2d)
        annotate_scene_with_metadata(parent, None, None, slice_gid_index(*arrays, set(PARENT_CAMS)))
        annotate_scene_with_metadata(child, None, None, slice_gid_index(*arrays, set(big_cams)))

        merged = merge_scenes_with_sim3_nonlinear(parent, [child])
        # Guard dropped the only child -> parent returned as-is.
        self.assertEqual(merged.number_tracks(), parent.number_tracks())
        self.assertEqual(set(merged.get_valid_camera_indices()), set(PARENT_CAMS))

    def test_no_sidecar_runs_legacy_path(self) -> None:
        """Without gid sidecars (enable_gid_merge_anchoring=False), the merge is the R3-baseline legacy
        path: shared-camera matching, no corr-floor guard — a large low-corr child is KEPT, not dropped."""
        shared = list(range(30, 46))
        parent = _scene(PARENT_CAMS + shared, list(range(30)), point_offset=0.0)
        child_cams = shared + list(range(46, 62))  # 32 cams, low legacy correspondences
        child = _scene(child_cams, list(range(60)), point_offset=0.0)
        # No annotate_scene_with_metadata -> no sidecars -> id_mode False everywhere.
        merged = merge_scenes_with_sim3_nonlinear(parent, [child])
        self.assertTrue(set(child_cams).issubset(set(merged.get_valid_camera_indices())))

    def test_structureless_child_dropped_any_size(self) -> None:
        """A 0-track child is never merged, even with shared cameras (nothing can anchor or verify it)."""
        parent = _scene(PARENT_CAMS, list(range(20)), point_offset=0.0)
        child = GtsfmData(number_images=NUM_IMAGES)
        for k, c in enumerate(PARENT_CAMS + [20, 21]):  # shares ALL parent cameras, but zero tracks
            child.add_camera(c, PinholeCameraCal3Bundler(Pose3(Rot3(), np.array([k, 0.0, 0.0])), Cal3Bundler()))
        merged = merge_scenes_with_sim3_nonlinear(parent, [child])
        self.assertEqual(set(merged.get_valid_camera_indices()), set(PARENT_CAMS))

    def test_overlap_escape_keeps_large_child_with_strong_camera_anchor(self) -> None:
        """A large child with >=15 shared cameras is KEPT despite low ID-corr (measured: 21-shared seats fine).

        Reproduces the gid-run regression where a 50-cam child with 21 shared cameras and 25 correspondences
        was dropped, costing Nc with no accuracy gain.
        """
        shared = list(range(30, 46))  # 16 shared cameras
        parent = _scene(PARENT_CAMS + shared, list(range(30)), point_offset=0.0)
        child_cams = shared + list(range(46, 62))  # 32 cams total, 16 shared
        child = _scene(child_cams, list(range(5)), point_offset=0.0)  # only 5 MATCHING tracks (< 50 corr floor)
        # Give the child real structure (>=50 tracks for the escape's structure floor) whose measurements
        # do NOT resolve in the gid index (offset uvs) -> correspondences stay at 5.
        for extra in range(60):
            track = SfmTrack(Point3(float(extra), -50.0, 5.0))
            for c in child_cams[:4]:
                track.addMeasurement(c, _uv(extra, c) + np.array([5000.0, 5000.0]))
            self.assertTrue(child.add_track(track))

        # Global tracks must SPAN this test's cameras so both slices resolve (id_mode active).
        tracks_2d = []
        for gid in range(30):
            meas = [SfmMeasurement(c, _uv(gid, c)) for c in PARENT_CAMS + shared + list(range(46, 62))]
            tracks_2d.append(SfmTrack2d(measurements=meas))
        arrays = build_measurement_gid_arrays(tracks_2d)
        annotate_scene_with_metadata(parent, None, None, slice_gid_index(*arrays, set(PARENT_CAMS + shared)))
        annotate_scene_with_metadata(child, None, None, slice_gid_index(*arrays, set(child_cams)))

        merged = merge_scenes_with_sim3_nonlinear(parent, [child])
        # Escape clause: child kept (its cameras present in the merged scene).
        self.assertTrue(set(child_cams).issubset(set(merged.get_valid_camera_indices())))


if __name__ == "__main__":
    unittest.main()
