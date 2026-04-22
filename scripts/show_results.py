#!/usr/bin/env python3
"""Parse and display benchmark results from benchmark_results/ directory."""

import json
import os
import sys

configs = [
    ('A-megaloc_sift_baseline_single', 'Baseline+Single'),
    ('B-megaloc_sift_gp_single', 'GP+Single'),
    ('C-megaloc_sift_baseline_metis', 'Baseline+METIS'),
    ('D-megaloc_sift_gp_metis', 'GP+METIS'),
    ('E-megaloc_sift_baseline_single_pt', 'Baseline+Single(PT)'),
    ('F-megaloc_sift_gp_single_pt', 'GP+Single(PT)'),
    ('G-megaloc_sift_baseline_metis_pt', 'Baseline+METIS(PT)'),
    ('H-megaloc_sift_gp_metis_pt', 'GP+METIS(PT)'),
    ('I-megaloc_splg_baseline_single_pt', 'Baseline+Single(SP)'),
    ('J-megaloc_splg_gp_single_pt', 'GP+Single(SP)'),
    ('K-megaloc_sift_gp_single_pt_shared_db', 'GP+SharedDB(PT)'),
]

datasets = ['gerrard-100', 'south-128', 'palace-281', 'british_museum', 'brussels',
            'pantheon_exterior', 'sacre_coeur', 'taj_mahal']


def get_val(data, key):
    v = data.get(key)
    if isinstance(v, dict):
        return v.get('summary', {}).get('median')
    return v


def load_metrics(base_path):
    merge_path = os.path.join(base_path, 'merging_metrics.json')
    ba_path = os.path.join(base_path, 'bundle_adjustment_metrics.json')
    if os.path.exists(merge_path):
        with open(merge_path) as f:
            md = json.load(f).get('merging_metrics', {})
        if md.get('number_cameras_merged', 0) > 0:
            result = {
                'cams': md.get('number_cameras_merged'),
                'tracks': md.get('number_tracks_merged'),
                'reproj': get_val(md, 'reprojection_errors_merged_px'),
                'rot': get_val(md, 'rotation_angle_error_deg'),
                'trans': get_val(md, 'translation_angle_error_deg'),
                'auc1': md.get('pose_auc_@1.0_deg'),
                'auc25': md.get('pose_auc_@2.5_deg'),
                'auc3': md.get('pose_auc_@3.0_deg'),
                'auc5': md.get('pose_auc_@5.0_deg'),
                'auc10': md.get('pose_auc_@10.0_deg'),
                'auc20': md.get('pose_auc_@20.0_deg'),
                'co_auc1': md.get('pose_auc_constructed_only_@1.0_deg'),
                'co_auc25': md.get('pose_auc_constructed_only_@2.5_deg'),
                'co_auc3': md.get('pose_auc_constructed_only_@3.0_deg'),
                'co_auc5': md.get('pose_auc_constructed_only_@5.0_deg'),
                'co_auc10': md.get('pose_auc_constructed_only_@10.0_deg'),
                'source': 'merged',
            }
            # Fall back to BA metrics for constructed-only if not in merging.
            if result['co_auc5'] is None and os.path.exists(ba_path):
                with open(ba_path) as f:
                    bd = json.load(f).get('bundle_adjustment_metrics', {})
                result['co_auc1'] = bd.get('pose_auc_constructed_only_@1.0_deg')
                result['co_auc25'] = bd.get('pose_auc_constructed_only_@2.5_deg')
                result['co_auc3'] = bd.get('pose_auc_constructed_only_@3.0_deg')
                result['co_auc5'] = bd.get('pose_auc_constructed_only_@5.0_deg')
                result['co_auc10'] = bd.get('pose_auc_constructed_only_@10.0_deg')
            return result
    if os.path.exists(ba_path):
        with open(ba_path) as f:
            bd = json.load(f).get('bundle_adjustment_metrics', {})
        return {
            'cams': get_val(bd, 'number_cameras_filtered'),
            'tracks': get_val(bd, 'number_tracks_filtered'),
            'reproj': get_val(bd, 'reprojection_errors_filtered_px'),
            'rot': get_val(bd, 'rotation_angle_error_deg'),
            'trans': get_val(bd, 'translation_angle_error_deg'),
            'auc1': get_val(bd, 'pose_auc_@1.0_deg'),
            'auc25': get_val(bd, 'pose_auc_@2.5_deg'),
            'auc5': get_val(bd, 'pose_auc_@5.0_deg'),
            'auc10': get_val(bd, 'pose_auc_@10.0_deg'),
            'auc20': get_val(bd, 'pose_auc_@20.0_deg'),
            'co_auc1': bd.get('pose_auc_constructed_only_@1.0_deg'),
            'co_auc25': bd.get('pose_auc_constructed_only_@2.5_deg'),
                'co_auc3': bd.get('pose_auc_constructed_only_@3.0_deg'),
            'co_auc5': bd.get('pose_auc_constructed_only_@5.0_deg'),
            'co_auc10': bd.get('pose_auc_constructed_only_@10.0_deg'),
            'source': 'ba',
        }
    return None


def get_total_time(base_path):
    p = os.path.join(base_path, 'total_summary_metrics.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f).get('total_summary_metrics', {}).get('total_runtime_sec')
    return None


def fmt(v, w=7, d=3):
    return f'{v:>{w}.{d}f}' if v is not None else ' ' * (w - 1) + '-'


def fmti(v, w=7):
    return f'{v:>{w}.0f}' if v is not None else ' ' * (w - 1) + '-'


# Filter dataset from CLI arg
filter_ds = sys.argv[1] if len(sys.argv) > 1 else None

for ds in datasets:
    if filter_ds and filter_ds != ds:
        continue
    has_data = any(os.path.exists(f'benchmark_results/{c[0]}/{ds}/results/metrics') for c in configs)
    if not has_data:
        continue
    print(f'\n=== {ds} ===')
    print(f'{"Config":<22} {"Cams":>5} {"Tracks":>7} {"Reproj":>8} {"Rot":>7} {"Trans":>7} {"AUC@3":>7} {"AUC@5":>7} {"AUC@10":>7} {"CO@3":>7} {"CO@5":>7} {"CO@10":>7} {"Total(s)":>9} {"Src":>6}')
    print('-' * 120)
    for cfg_dir, label in configs:
        base = f'benchmark_results/{cfg_dir}/{ds}/results/metrics'
        m = load_metrics(base)
        total = get_total_time(base)
        if m is None:
            continue
        print(f'{label:<22} {fmti(m["cams"],5)} {fmti(m["tracks"],7)} {fmt(m["reproj"],8,4)} {fmt(m["rot"])} {fmt(m["trans"])} {fmt(m.get("auc3"))} {fmt(m["auc5"])} {fmt(m["auc10"])} {fmt(m.get("co_auc3"))} {fmt(m.get("co_auc5"))} {fmt(m.get("co_auc10"))} {fmt(total,9,1)} {m["source"]:>6}')
