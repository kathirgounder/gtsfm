"""Recursive triangulation per COLMAP paper Section 4.3.

A single 2D track produced by union-find may contain measurements of multiple
independent 3D points due to *false-merge* scenarios — most pronounced on
mirror-symmetric scenes (Brussels' plaza) and repetitive geometry (BM's
colonnade, Pantheon's columns) where two-view verification incorrectly groups
features that look identical but project from physically distinct points.

The recursive triangulation algorithm (Schönberger et al. 2016 §4.3) handles
this by treating each input track as a *set of measurements possibly from
multiple 3D points*:

    1. Run RANSAC consensus on the current measurement set
    2. If a valid consensus is found, emit it as one 3D point
    3. Remove consensus measurements from the set
    4. Recurse on the remainder
    5. Stop when remainder has fewer than min_consensus_size measurements

Output: a list of independent 3D points (gtsam.SfmTrack) recovered from the
input. For clean tracks (no false merges) this returns a single track. For
falsely-merged tracks it returns 2+ tracks — each cleanly triangulated.

The implementation reuses Point3dInitializer.triangulate() (which already runs
RANSAC + cheirality + reprojection-error check) as the per-recursion-step
solver. We just wrap it in the strip-and-recurse loop.

Authors: Kathir Gounder
"""

from __future__ import annotations

from typing import List

import gtsam

import gtsfm.utils.logger as logger_utils
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.data_association.point3d_initializer import Point3dInitializer

logger = logger_utils.get_logger()


def triangulate_track_recursive(
    track_2d: SfmTrack2d,
    initializer: Point3dInitializer,
    min_consensus_size: int = 3,
    max_recursion_depth: int = 5,
) -> List[gtsam.SfmTrack]:
    """Recursive RANSAC triangulation: split falsely-merged tracks into multiple 3D points.

    Args:
        track_2d: Input 2D measurement track (potentially from union-find).
        initializer: Configured Point3dInitializer (RANSAC mode + thresholds).
        min_consensus_size: Minimum measurements in a single 3D point's consensus set.
            Tracks below this are dropped. GLOMAP/Bundler default is 3.
        max_recursion_depth: Cap on number of recursive splits per input track.
            Default 5 — handles "4 falsely-merged tracks" + safety margin from the paper.

    Returns:
        List of `gtsam.SfmTrack` objects (one per independent 3D point recovered).
        Empty list if no valid consensus could be found at all.
    """
    output: List[gtsam.SfmTrack] = []
    remaining = track_2d
    depth = 0
    while remaining.number_measurements() >= min_consensus_size and depth < max_recursion_depth:
        new_track, _, _ = initializer.triangulate(remaining)
        if new_track is None or new_track.numberMeasurements() < min_consensus_size:
            break
        output.append(new_track)

        # Identify which input measurements are now in the consensus by camera index.
        # Within a single SfmTrack2d, each camera contributes at most one measurement
        # (per union-find's construction), so m.i alone is unique.
        consensus_cam_ids = {
            new_track.measurement(k)[0] for k in range(new_track.numberMeasurements())
        }
        remaining_measurements = [m for m in remaining.measurements if m.i not in consensus_cam_ids]

        # Avoid infinite loop if triangulate consumed nothing (defensive).
        if len(remaining_measurements) == len(remaining.measurements):
            break

        remaining = SfmTrack2d(measurements=remaining_measurements)
        depth += 1

    return output
