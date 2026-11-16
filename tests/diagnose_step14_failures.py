#!/usr/bin/env python3
"""Diagnostic script for STEP 14: Diagnose Weak Scenarios (< 0.8000) under v6."""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_flight_handover_benchmark import get_all_frozen_scenarios, compute_boundary_timing_error
from tests.test_held_out_handover_suite import get_all_held_out_scenarios
from tests.test_sensitivity_and_counterexamples import (
    build_counterexample_1_skimmed_header,
    build_counterexample_2_one_two_intercepted,
    build_counterexample_3_clearance_out_of_bounds,
    build_counterexample_4_aerial_50_50_contest,
    build_counterexample_5_double_deflection,
)
from tests.run_v5_v6_comparative_audit import run_pipeline_v6
from phantom11.control_evidence import compute_control_evidence
from phantom11.movement_features import compute_ball_kinematics
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm

def diagnose():
    all_scenarios = []
    for name, b, p, t, gt in get_all_frozen_scenarios():
        all_scenarios.append(("Frozen: " + name, b, p, t, gt))
    for name, b, p, t, gt in get_all_held_out_scenarios():
        all_scenarios.append(("HeldOut: " + name, b, p, t, gt))

    ce_builders = [
        build_counterexample_1_skimmed_header,
        build_counterexample_2_one_two_intercepted,
        build_counterexample_3_clearance_out_of_bounds,
        build_counterexample_4_aerial_50_50_contest,
        build_counterexample_5_double_deflection,
    ]
    for builder in ce_builders:
        cename, b, p, t, gt = builder()
        all_scenarios.append(("Adversarial: " + cename, b, p, t, gt))

    print(f"Total scenarios tested: {len(all_scenarios)}")
    print("=" * 105)
    print(f"{'Scenario Name':<55} | {'v6 spellF1':<10} | {'Status'}")
    print("-" * 105)

    weak_scenarios = []
    for name, b, p, t, gt in all_scenarios:
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)
        p6 = run_pipeline_v6(b, p, t, [])
        p6_codes = pm.label_codes(p6)
        res = pm.spell_scores(gt_codes, p6_codes, runs)
        f1 = res["mean_f1"]
        status = "WEAK (< 0.80)" if f1 < 0.8000 else "STRONG (>= 0.80)"
        print(f"{name:<55} | {f1:<10.4f} | {status}")
        if f1 < 0.8000:
            weak_scenarios.append((name, f1, b, p, t, gt, p6, res))

    print("\n" + "=" * 105)
    print("DETAILED FAILURE BREAKDOWN FOR WEAK SCENARIOS")
    print("=" * 105)

    for name, f1, b, p, t, gt, p6, res in weak_scenarios:
        print(f"\n>>> Scenario: {name}")
        print(f"    v6 Mean spellF1: {f1:.4f}")
        gt_codes = pm.label_codes(gt)
        p6_codes = pm.label_codes(p6)
        runs = [(0, len(gt))]
        gt_spells = pm.possession_spells(gt_codes, runs)
        p6_spells = pm.possession_spells(p6_codes, runs)
        print(f"    GT Spells   (start, stop, code): {gt_spells}")
        print(f"    Pred Spells (start, stop, code): {p6_spells}")
        
        # Overlaps
        overlaps = pm.spell_overlaps(gt_spells, p6_spells)
        print("    Pairwise Overlaps (IoU, gt_idx, pred_idx):")
        for iou, gi, pi in overlaps:
            g_sp = gt_spells[gi]
            p_sp = p6_spells[pi]
            print(f"      GT[{gi}] {g_sp} vs Pred[{pi}] {p_sp} -> IoU = {iou:.4f}")

        # Metrics per IoU
        print("    Official Metric by IoU Threshold:")
        for iou_thresh in (0.3, 0.4, 0.5, 0.6, 0.7):
            d = res["per_iou"][iou_thresh]
            prec = d["precision"]
            rec = d["recall"]
            f = d["f1"]
            tp = d["n_matched"]
            fp = d["n_spurious"]
            fn = d["n_missed"]
            print(f"      IoU >= {iou_thresh:.1f} : P={prec:.3f}, R={rec:.3f}, F1={f:.3f} | TP={tp}, FP={fp}, FN={fn}")

        # Frame level inspection of discrepancies
        diff_idx = [i for i in range(len(gt)) if gt[i] != p6[i]]
        print(f"    Differing frames count: {len(diff_idx)} / {len(gt)}")
        if diff_idx:
            # group into contiguous intervals
            intervals = []
            curr_start = diff_idx[0]
            curr_end = diff_idx[0]
            for idx in diff_idx[1:]:
                if idx == curr_end + 1:
                    curr_end = idx
                else:
                    intervals.append((curr_start, curr_end + 1))
                    curr_start = idx
                    curr_end = idx
            intervals.append((curr_start, curr_end + 1))
            print("    Differing Frame Intervals [start, end):")
            for start, end in intervals:
                gt_lbls = set(gt[start:end])
                p6_lbls = set(p6[start:end])
                print(f"      frames [{start:3d}, {end:3d}) dur={end-start:2d}f | GT={gt_lbls} vs Pred={p6_lbls}")

        # Compute kinematics & evidence around discrepancy to identify mechanism
        ev = compute_control_evidence(b, p, t, [], [], fps=25.0)
        kin = compute_ball_kinematics(b, fps=25.0)
        print("    Kinematics & Control Evidence in Diff Regions:")
        for start, end in intervals[:3]: # inspect first few
            for f in range(start, min(end, start + 5)):
                b_spd = kin["speed"][f]
                dh = ev["d_h1"][f]
                da = ev["d_a1"][f]
                ch = ev["is_carrier_h"][f]
                ca = ev["is_carrier_a"][f]
                print(f"      frame {f:3d}: ball_spd={b_spd:.2f} m/s, d_h1={dh:.2f}m, d_a1={da:.2f}m, carr_h={ch}, carr_a={ca} | GT={gt[f]} P6={p6[f]}")

if __name__ == "__main__":
    diagnose()
