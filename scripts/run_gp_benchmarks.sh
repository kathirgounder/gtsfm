#!/bin/bash
# =============================================================================
# Local benchmark harness for Akshay's comparison table.
#
# MegaLoc + SIFT front-end, GP vs Baseline, Single vs METIS partitioning.
#
# Building datasets (gerrard-100, south-128, palace-281):
#   A: megaloc_sift_baseline_single  — Baseline + Single
#   B: megaloc_sift_gp_single        — GP + Single
#   C: megaloc_sift_baseline_metis   — Baseline + METIS
#   D: megaloc_sift_gp_metis         — GP + METIS
#
# Phototourism datasets (british_museum, brussels):
#   E: megaloc_sift_baseline_single_pt — Baseline + Single (phototourism)
#   F: megaloc_sift_gp_single_pt       — GP + Single (phototourism)
#   G: megaloc_sift_baseline_metis_pt  — Baseline + METIS (phototourism)
#   H: megaloc_sift_gp_metis_pt        — GP + METIS (phototourism)
#
# Usage:
#   bash scripts/run_gp_benchmarks.sh                  # all configs, all datasets
#   bash scripts/run_gp_benchmarks.sh gerrard-100      # all configs, one dataset
#   bash scripts/run_gp_benchmarks.sh gerrard-100 A    # one config, one dataset
# =============================================================================

OUTPUT_BASE="benchmark_results"
FILTER_DATASET="${1:-}"
FILTER_CONFIG="${2:-}"

run_one() {
    local config_key="$1"
    local config_label="$2"
    local config_name="$3"
    local dataset_name="$4"
    local loader="$5"
    local dataset_dir="$6"
    local images_dir="${7:-}"
    local max_res="${8:-}"
    local share_intrinsics="${9:-true}"

    local output_dir="$OUTPUT_BASE/$config_key-$config_name/$dataset_name"

    echo ""
    echo "=================================================================="
    echo "  Config $config_key: $config_label"
    echo "  Dataset: $dataset_name"
    echo "  Output:  $output_dir"
    echo "=================================================================="

    if [ -f "$output_dir/results/metrics/bundle_adjustment_metrics.json" ]; then
        echo "  -> Results exist, skipping. Delete $output_dir to re-run."
        return 0
    fi

    local cmd="./run --output_root $output_dir --num_workers 2"
    if [ "$share_intrinsics" = "true" ]; then
        cmd="$cmd --share_intrinsics"
    fi
    cmd="$cmd --config_name $config_name"
    cmd="$cmd --loader $loader --dataset_dir $dataset_dir"

    if [ -n "$images_dir" ]; then
        cmd="$cmd --images_dir $images_dir"
    fi

    if [ -n "$max_res" ]; then
        cmd="$cmd --max_resolution $max_res"
    fi

    echo "  -> $cmd"
    eval $cmd || echo "  !! FAILED: $config_key / $dataset_name"
}

run_all_configs() {
    local dataset_name="$1"
    local loader="$2"
    local dataset_dir="$3"
    local images_dir="${4:-}"
    local max_res="${5:-}"

    # A: Baseline + Single
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "A" ]; then
        run_one "A" "Baseline + Single" "megaloc_sift_baseline_single" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res"
    fi

    # B: GP + Single
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "B" ]; then
        run_one "B" "GP + Single" "megaloc_sift_gp_single" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res"
    fi

    # C: Baseline + METIS
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "C" ]; then
        run_one "C" "Baseline + METIS" "megaloc_sift_baseline_metis" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res"
    fi

    # D: GP + METIS
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "D" ]; then
        run_one "D" "GP + METIS" "megaloc_sift_gp_metis" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res"
    fi
}

run_all_configs_pt() {
    local dataset_name="$1"
    local loader="$2"
    local dataset_dir="$3"
    local images_dir="${4:-}"
    local max_res="${5:-}"

    # E: Baseline + Single (phototourism) — no shared intrinsics
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "E" ]; then
        run_one "E" "Baseline + Single (PT)" "megaloc_sift_baseline_single_pt" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res" "false"
    fi

    # F: GP + Single (phototourism)
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "F" ]; then
        run_one "F" "GP + Single (PT)" "megaloc_sift_gp_single_pt" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res" "false"
    fi

    # G: Baseline + METIS (phototourism)
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "G" ]; then
        run_one "G" "Baseline + METIS (PT)" "megaloc_sift_baseline_metis_pt" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res" "false"
    fi

    # H: GP + METIS (phototourism)
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "H" ]; then
        run_one "H" "GP + METIS (PT)" "megaloc_sift_gp_metis_pt" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res" "false"
    fi

    # I: Baseline + Single (SuperPoint+LightGlue phototourism)
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "I" ]; then
        run_one "I" "Baseline + Single (SP)" "megaloc_splg_baseline_single_pt" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res" "false"
    fi

    # J: GP + Single (SuperPoint+LightGlue phototourism)
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "J" ]; then
        run_one "J" "GP + Single (SP)" "megaloc_splg_gp_single_pt" \
                "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res" "false"
    fi

    # K: GP + SharedDB (PT) — reads keypoints+matches from a precomputed COLMAP SQLite DB.
    # Only runs when the per-dataset DB exists at benchmarks/<dataset>/colmap_shared.db.
    if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "K" ]; then
        local shared_db_path="benchmarks/$dataset_name/colmap_shared.db"
        if [ -f "$shared_db_path" ]; then
            run_one "K" "GP + SharedDB (PT)" "megaloc_sift_gp_single_pt_shared_db" \
                    "$dataset_name" "$loader" "$dataset_dir" "$images_dir" "$max_res" "false"
        else
            echo "  -> Skipping K for $dataset_name: no $shared_db_path. Build with scripts/build_colmap_db_british_museum.py"
        fi
    fi
}

# Dataset: gerrard-100
if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "gerrard-100" ]; then
    run_all_configs "gerrard-100" "colmap" "benchmarks/gerrard-hall/sparse" "benchmarks/gerrard-hall/images"
fi

# Dataset: south-128
if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "south-128" ]; then
    run_all_configs "south-128" "colmap" "benchmarks/south-building/sparse" "benchmarks/south-building/images"
fi

# Dataset: palace-281
if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "palace-281" ]; then
    run_all_configs "palace-281" "olsson" "benchmarks/palace-fine-arts-281" "" "320"
fi

# Dataset: british_museum (phototourism, 176 images)
if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "british_museum" ]; then
    run_all_configs_pt "british_museum" "colmap" "benchmarks/british_museum/sfm_updated" "benchmarks/british_museum/images"
fi

# Dataset: brussels (phototourism, 236 images)
if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "brussels" ]; then
    run_all_configs_pt "brussels" "colmap" "benchmarks/grand_place_brussels/sfm_updated" "benchmarks/grand_place_brussels/images"
fi

# Dataset: pantheon_exterior (phototourism, 321 images)
if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "pantheon_exterior" ]; then
    run_all_configs_pt "pantheon_exterior" "colmap" "benchmarks/pantheon_exterior/sfm_updated" "benchmarks/pantheon_exterior/images"
fi

# Dataset: sacre_coeur (phototourism)
if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "sacre_coeur" ]; then
    run_all_configs_pt "sacre_coeur" "colmap" "benchmarks/sacre_coeur/sfm_updated" "benchmarks/sacre_coeur/images"
fi

# Dataset: taj_mahal (phototourism, 399 images)
if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "taj_mahal" ]; then
    run_all_configs_pt "taj_mahal" "colmap" "benchmarks/taj_mahal/sfm_updated" "benchmarks/taj_mahal/images"
fi

echo ""
echo "=================================================================="
echo "All benchmarks complete. Results in $OUTPUT_BASE/"
echo "=================================================================="
