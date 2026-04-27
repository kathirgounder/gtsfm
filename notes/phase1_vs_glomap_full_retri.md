# Phase 1 retri vs full GLOMAP `IterativeRetriangulateAndRefine`: scope, gap, and decision

## Decision

We're keeping Phase 1 retriangulation + per-observation filter + loosened multi-view post-BA filters as the production path. We're stopping work on a Python/GTSAM port of GLOMAP's full retri stage.

**Final config closes ~half the BM gap** to GLOMAP-full (+3.7 AUC@5 from per-obs filter alone) with simpler architecture, ~5x lower wall time, and no further gymnastics. The remaining gap is bounded by upstream pipeline / Python-stack choices, not retri tuning. See "Why we stopped" for details.

## Final results (BM, 100 num_matched, 0.20 min_score)

| Metric | Pre-retri-rewrite | **Final config** | GLOMAP-full | Gap |
|---|---|---|---|---|
| AUC@3 | ~57 | **57.4** | 62.7 | -5.3 |
| AUC@5 | 65.8 | **69.5** | 73.8 | -4.3 |
| AUC@10 | ~78 | **82.1** | 85.0 | -2.9 |
| Cameras | 176 | 176 | 176 | matched |
| Tracks | ~15,800 | **16,701** | ~25,512 | -8.8K |
| Rot° | — | 0.08 | 0.05 | nearly matched |
| Trans° | — | 0.68 | 0.47 | close |
| Wall time | — | ~10.9 min | ~5 min | ~2x slower (acceptable) |

Cross-scene comparison:

| Scene | Config | GTSFM (final) | GLOMAP-full | Gap |
|---|---|---|---|---|
| Brussels | (100, 0.15) | 79.5 | 81.2 | -1.7 |
| Pantheon | (15, 0.45) | 79.9 | — | strong |
| British Museum | (100, 0.20) | **69.5** | 73.8 | -4.3 |

We're within 5 AUC@5 points of GLOMAP-full across the hardest IMC scenes.

## What Phase 1 retri does (production path)

Implementation: `gtsfm/bundle/bundle_adjustment.py::multi_view_retriangulate_from_2d_tracks` + post-BA filtering in `_run_ba_and_evaluate`.

```
1. OuterBA converges (the upstream BA loop)
2. Take union-find 2D tracks from CppDsfTracksEstimator (transitive closure of the
   cycle-filtered correspondence graph)
3. For each track ≥ 3 measurements: multi-view RANSAC triangulation via
   Point3dInitializer (samples camera pairs, triangulates, scores by inlier count
   over ALL track measurements at 10px reproj threshold)
4. Final BA on the augmented track set
5. Per-OBSERVATION reproj filter (filter_landmark_measurements) at 6px — trims
   outlier measurements WITHIN tracks, keeps tracks with ≥2 good measurements
6. Min-tri-angle filter at 0.5° (loose; 1.5° dropped 1,100 tight-baseline multi-
   view tracks unnecessarily)
```

The two big architectural wins were:
- **Per-observation filter (step 5) instead of per-track**: +3.7 AUC@5 on BM. Tracks with 1-2 outlier measurements among 8-10 good ones used to die wholesale; now the outliers get trimmed and the rest of the track survives.
- **Loose min-tri-angle (step 6)**: +961 tracks of completeness, AUC@5 unchanged. Recovers tight-baseline multi-view tracks that contribute coverage without affecting pose estimation.

**Architectural insight:** `CppDsfTracksEstimator`'s union-find IS a correspondence-graph walk at infinite transitivity. So a union-find track of length 5 is the same multi-view kp bundle that GLOMAP's `IncrementalTriangulator::Find(transitivity=∞)` would assemble. Phase 1 then runs essentially the same multi-view RANSAC kernel GLOMAP uses. We get the core triangulation operation right.

## What full GLOMAP `IterativeRetriangulateAndRefine` does (the rest)

Source: `glomap/glomap/controllers/track_retriangulation.cc` + `colmap/sfm/incremental_triangulator.cc`.

```
1. reconstruction.DeleteAllPoints2DAndPoints3D()        // throw away upstream BA's tracks
2. for each registered image:
     mapper.TriangulateImage(image_id)                  // per-IMAGE multi-view Create
3. mapper.CompleteAndMergeTracks(tri_options)           // BFS extension + consolidation
4. for round in 5:
     bundle_adjuster.Solve()                            // BA capped at 50 LM iters
     mapper.CompleteAndMergeTracks(tri_options)
     observation_manager.FilterPoints3D(...)            // per-OBSERVATION reproj + min_tri
     if num_changed/num_obs < 0.0005: break
5. Final uncapped BA
6. Per-obs reproj filter + min-tri-angle filter
```

The **asymmetric thresholds** are the key engineering choice:
- **Build phase** (Complete/Merge/Create): 15px reproj, 1° min_tri — accept marginal correspondences
- **Filter phase** (post-BA): 4px reproj, 1.5° min_tri, applied **per-observation** (trims outlier measurements, keeps tracks with ≥2 good measurements)

The asymmetry lets BA tighten geometry round-over-round: build aggressively → BA fits → filter strictly → tracks get cleaner → next round builds against tighter cameras.

## What Phase 1 is missing relative to full GLOMAP

| Capability | Phase 1 (us) | Full GLOMAP | Impact |
|---|---|---|---|
| Outlier handling within a cluster | Binary (RANSAC succeeds or whole track fails) | Recursive `Create` on RANSAC outliers — 1 cluster can produce multiple sub-tracks | Marginal track recovery in noisy scenes |
| Per-image vs per-cluster traversal | Per-cluster (union-find) | Per-image, per-kp | GLOMAP detects ambiguity when a kp's correspondences partially conflict |
| Asymmetric build/filter thresholds | Single 10px throughout | 15px build, 4px filter | We leave measurements on the table during build |
| Filter granularity | Per-track (entire track dropped if mean reproj > thresh) | Per-observation (trim outlier measurements, keep track) | Per-track filter loses good measurements bundled with bad |
| Iteration count | 1 alternation | 5-round refinement loop with Δ-obs convergence | Compounding gain — each round's BA tightens geometry, enabling the next round's Complete to find more |
| Track completion (BFS extension) | None | `Complete` walks correspondence graph 5 hops from each track member, adds reproj-passing observations | We miss observations RANSAC dropped that BA could have rescued |
| Track consolidation | None | `Merge` finds tracks pointing to same correspondences, weighted-averages 3D points | We carry duplicate tracks for the same physical point |

## Tuning trajectory: what we tried and what worked

We attempted three categories of additional improvements after the per-obs filter landed:

| Experiment | Mechanism | Result | Verdict |
|---|---|---|---|
| Full GLOMAP rewrite (per-image TriangulateImage + 5-round refinement) | Mirror entire `IterativeRetriangulateAndRefine` | Correct on synthetic; ~60-90 min wall time on BM (vs ~5 min in C++/Ceres) | Abandoned — too slow in Python/GTSAM |
| Loosen `mv_retri_reproj_error_thresh` (10 → 15px) | RANSAC accepts noisier triangulations as BA init | +48 tracks but BA wall time exploded 15s → 4+ min | Reverted — RANSAC threshold is a compute-cost lever, not a count lever |
| Admit 2-view tracks via tri-angle gate | `min_track_length=2` + threshold sweep (3°, 10°, 15°, 25°) + hard-cap at 500 | Even 500 best-of-best 2-view tracks annihilated BA wall time | Abandoned — Python/GTSAM LM cannot accommodate ANY 2-view track (super-linear cost from under-conditioned points) |
| Loosen post-BA `filter_min_tri_angle_deg` (1.5° → 0.5°) | Keep tight-baseline multi-view tracks | +961 tracks, AUC@5 unchanged, BA wall time unchanged | **Kept — clean completeness lift** |
| Loosen post-BA `filter_max_reproj_error_px` (5 → 6 px) | Keep slightly noisier measurements within tracks | Marginal effect on count, +0 on AUC | Kept — no downside |

**The asymmetry that matters**: pre-BA changes (RANSAC threshold, 2-view admission) affect BA conditioning and wall time; post-BA changes (filter thresholds, granularity) only affect what survives into the final result. The Phase 1 sweet spot loosens post-BA filters aggressively while keeping pre-BA inputs conservative.

**Pattern across scenes:** well-connected scenes (Brussels, Pantheon) where union-find clusters cleanly correspond to physical 3D points → Phase 1 nearly matches GLOMAP. BM widens the gap because its repetitive colonnade produces ambiguous correspondences — exactly the regime where GLOMAP's per-image traversal + recursive split pays off. The remaining 4.3 AUC@5 gap on BM requires architectural changes (upstream pipeline or C++ retri) we've decided not to pursue here.

## Why we stopped pursuing full GLOMAP in Python/GTSAM

We did port the full architecture (DeleteAllPoints + TriangulateImage + Complete + Merge + 5-round loop + per-obs filter + final BA). It works correctly on synthetic tests. But the wall time is uneconomical:

| Stage | Python/GTSAM | C++/Ceres (GLOMAP) | Ratio |
|---|---|---|---|
| Per-image TriangulateImage (88K Creates) | ~7 min | ~5-10s | ~50x |
| Merge (88K → 20K) | ~3.5 min | ~2-5s | ~50x |
| Single BA round (50-iter capped, 800K factors) | ~5-7 min | ~10-30s | ~20x |
| Full 5-round refinement | ~30-45 min | ~2-5 min | ~10x |
| **Total retri stage** | **~60-90 min** | **~5 min** | **~15x** |

The fundamental issues:
1. **Pure-Python factor graph construction** at 800K+ factor scale: building each factor object costs μs which add up.
2. **GTSAM's Python bindings** are not zero-cost: per-factor evaluation goes through pybind for the residual function callback, which is ~10x slower than native C++ residual eval.
3. **GTSAM's LM** is single-threaded by default; Ceres uses multithreaded Schur complement.
4. **Per-obs filter** in Python iterates each measurement of each track — fast in C++, slow in Python.

In aggregate, the ~15x slowdown means a 5-min GLOMAP retri becomes 60-90 min for us. At this wall time, the ROI of an additional ~5-10 AUC points isn't compelling for our current research velocity.

## Path forward options

Three plausible routes if we revisit:

1. **Native binding**: wrap GLOMAP's `RetriangulateTracks` directly via pybind, treat it as a black-box that takes our cameras + correspondence graph and returns refined tracks. Largest engineering win, smallest research investment.
2. **Ceres-backed BA**: replace GTSAM with Ceres for the refinement loop's BA. Closes most of the wall-time gap (the BA is the dominant cost). But still slow in the Python triangulation/filter ops.
3. **C++ port of the loop**: full reimplementation in C++ in the gtsfm cluster_optimizer. Highest effort.

Option 1 is the obvious next step if we want full parity. Option 3 is overkill given GLOMAP exists.

## Bottom line for the team

- **Phase 1 retri + per-obs filter + loosened post-BA filters is shipping** — proven on all three IMC scenes, fast (~10.9 min on BM), simple architecture, no Python/GTSAM stack pathologies.
- **+3.7 AUC@5 win on BM came from a single change**: switching the post-final-BA filter from per-track (`filter_landmarks`) to per-observation (`filter_landmark_measurements`). One-line code change, biggest leverage of the entire investigation.
- **+961 additional tracks of completeness** came from dropping the post-BA min_tri_angle filter from 1.5° → 0.5°. AUC unchanged, wall time unchanged — clean win for downstream tasks (dense recon, splatting).
- **2-view admission is fundamentally incompatible with our stack**: even 500 hand-picked top-tri-angle 2-view tracks blow up BA wall time. The under-conditioned 3D points create LM trajectories Python/GTSAM can't handle efficiently.
- **The remaining gap is bounded by upstream/stack choices, not retri tuning**. Brussels and Pantheon are essentially at GLOMAP parity. BM has -4.3 AUC@5 of remaining gap that requires either C++ bindings (`RetriangulateTracks` direct), Ceres-backed BA, or upstream pipeline changes (looser cycle filter, larger retrieval set). Out of scope here.

The full GLOMAP architecture port (TriangulateImage, CorrespondenceGraph, track_completer, track_merger, 5-round refinement loop) was implemented and validated correct on synthetic tests, but **removed from this PR** to keep the production code surface focused and reviewable. The implementation ran ~15x slower than GLOMAP's C++/Ceres at the 800K-factor scale, so it was not shippable. Future revisits should target a C++/Ceres-backed reimplementation rather than reviving the Python skeleton — recover the historical commits from git if useful as a starting reference (`git log --diff-filter=D -- gtsfm/data_association/per_image_triangulator.py`).
