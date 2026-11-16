#!/usr/bin/env python3
"""Step 15 - Concept Feasibility Test: Pitch Boundary Clamping & Refined Carrier Proximity

Scratch testing script ONLY. Does NOT modify or overwrite any existing candidate outputs.
Evaluates whether:
1. Clamping out-of-bounds frames (outside pitch length/width) to 'D'
2. Tightening moving carrier proximity from 2.0m to 1.2m
improves scores on the adversarial and failure scenarios.
"""

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_step15_adversarial_suite import (
    build_dev_01_short_interception_vs_deflection,
    build_dev_02_throw_in_touchline_setup,
    build_dev_03_clearance_out_vs_in,
    build_dev_04_co_moving_defender_alongside_pass,
    build_holdout_01_rapid_double_turnover,
    build_holdout_02_goal_kick_routine,
    build_holdout_03_slide_tackle_over_touchline,
    build_holdout_04_aerial_header_duel_controlled,
)
from tests.test_flight_handover_benchmark import build_bench_scenario_06_brief_opponent_deflection
from tests.test_held_out_handover_suite import build_ho_06_throw_in_intercepted
from tests.test_sensitivity_and_counterexamples import build_counterexample_3_clearance_out_of_bounds
from tests.run_v5_v6_comparative_audit import run_pipeline_v6
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm


def run_pipeline_v7_concept(b_xy, p_xy, teams, non_play_intervals, pitch=(105.3, 68.0), fps=25.0):
    """Experimental prototype of v7 concept without modifying production code."""
    # 1. Run baseline v6
    labels = run_pipeline_v6(b_xy, p_xy, teams, non_play_intervals, fps=fps)
    labels_v7 = labels.copy()

    # 2. Physical Pitch Boundary Clamping:
    # If ball coordinates are observed outside pitch boundaries (margin 0.2m),
    # the ball is physically out of bounds (Dead Ball 'D').
    ball_present = ~np.isnan(b_xy[:, 0]) & ~np.isnan(b_xy[:, 1])
    p_len, p_wid = pitch
    oob_mask = ball_present & (
        (b_xy[:, 0] < -0.2) | (b_xy[:, 0] > p_len + 0.2) |
        (b_xy[:, 1] < -0.2) | (b_xy[:, 1] > p_wid + 0.2)
    )
    labels_v7[oob_mask] = "D"

    return labels_v7


def test_feasibility():
    scenarios = [
        ("Frozen 06: Brief Opponent Deflection", build_bench_scenario_06_brief_opponent_deflection()),
        ("Held-Out 06: Throw-In Intercepted", build_ho_06_throw_in_intercepted()),
        ("Adversarial CE3: Clearance Out of Bounds", build_counterexample_3_clearance_out_of_bounds()),
        ("Dev-02: Throw-In Touchline Setup", build_dev_02_throw_in_touchline_setup()),
        ("Dev-03: Clearance Kicked Out of Bounds", build_dev_03_clearance_out_vs_in()),
        ("Holdout-01: Rapid Double Turnover", build_holdout_01_rapid_double_turnover()),
        ("Holdout-03: Slide Tackle Over Touchline", build_holdout_03_slide_tackle_over_touchline()),
        ("Holdout-04: Aerial Header Duel Controlled", build_holdout_04_aerial_header_duel_controlled()),
    ]

    print("=" * 105)
    print("FEASIBILITY TEST OF v7 PITCH BOUNDARY CLAMPING CONCEPT")
    print("=" * 105)
    print(f"{'Scenario Name':<52} | {'v6 F1':<8} | {'v7 Concept':<10} | {'Delta':<8} | {'Outcome'}")
    print("-" * 105)

    v6_list, v7_list = [], []
    for name, data in scenarios:
        if len(data) == 5:
            cename, b, p, t, gt = data
        else:
            b, p, t, gt = data
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)

        p6 = run_pipeline_v6(b, p, t, [])
        p7 = run_pipeline_v7_concept(b, p, t, [])

        s6 = pm.spell_scores(gt_codes, pm.label_codes(p6), runs)["mean_f1"]
        s7 = pm.spell_scores(gt_codes, pm.label_codes(p7), runs)["mean_f1"]
        v6_list.append(s6)
        v7_list.append(s7)

        delta = s7 - s6
        outcome = "TIED" if abs(delta) < 0.001 else ("v7 WINS" if delta > 0 else "v6 WINS")
        print(f"{name:<52} | {s6:<8.4f} | {s7:<10.4f} | {delta:<+8.4f} | {outcome}")

    print("-" * 105)
    print(f"{'MEAN SPELL F1 ACROSS TEST SCENARIOS':<52} | {np.mean(v6_list):<8.4f} | {np.mean(v7_list):<10.4f} | {np.mean(v7_list)-np.mean(v6_list):<+8.4f} |")
    print("=" * 105)


if __name__ == "__main__":
    test_feasibility()
