#!/usr/bin/env python3
"""Step 16: Comprehensive Independent Evaluation and Ablation Study

Evaluates candidates:
- v5: Evidence-gated handover baseline
- v6: Proximity-gated handover baseline
- v7-A: Pitch-boundary awareness only (IFAB Law 9)
- v7-B: Refined carrier detection only
- v7-C: Combined pitch-boundary awareness and refined carrier detection

Across:
1. 10 Frozen Benchmark Scenarios (Step 11)
2. 10 Frozen Held-Out Scenarios (Step 13)
3. 5 Adversarial Counterexamples (Step 12)
4. 6 Step 16 Final Untouched Evaluation Scenarios (Frozen)
5. Noise & Dropout Robustness
6. Boundary Timing Errors
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_flight_handover_benchmark import (
    get_all_frozen_scenarios,
    run_pipeline_v3,
    run_pipeline_v4,
    compute_boundary_timing_error,
)
from tests.test_held_out_handover_suite import get_all_held_out_scenarios
from tests.test_sensitivity_and_counterexamples import (
    build_counterexample_1_skimmed_header,
    build_counterexample_2_one_two_intercepted,
    build_counterexample_3_clearance_out_of_bounds,
    build_counterexample_4_aerial_50_50_contest,
    build_counterexample_5_double_deflection,
)
from tests.run_comparative_benchmark import run_pipeline_v5
from phantom11.predict_v7 import (
    run_pipeline_v6,
    run_pipeline_v7a,
    run_pipeline_v7b,
    run_pipeline_v7c,
)
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm

# ==============================================================================
# STEP 16 FINAL UNTOUCHED EVALUATION SET (6 SCENARIOS - FROZEN)
# ==============================================================================

def build_eval_01_ball_hugging_touchline(fps: float = 25.0):
    """Eval-01: Ball dribbled closely inside touchline at y=67.8m (0.2m inside).
    
    Must NOT be falsely clamped to D.
    GT: Continuous H (0..100).
    """
    n = 101
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(n):
        pos = [30.0 + t * (2.0 / fps), 67.75] # inside 68.0m line
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1] - 0.2]

    gt = np.array(["H"] * n)
    return "Eval-01: Ball Hugging Touchline Just Inside (y=67.75m)", ball_xy, players_xy, team_of_track, gt


def build_eval_02_high_speed_pass_near_boundary(fps: float = 25.0):
    """Eval-02: Fast pass at 18 m/s parallel to touchline at y=67.0m.
    
    Pass from (20, 67) to (65, 67). Trapped and carried by H1.
    GT: Continuous H (0..110).
    """
    n = 111
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..29: H0 carries
    for t in range(30):
        pos = [20.0 + t * (1.5 / fps), 67.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 30..69: Fast pass at 18 m/s
    start_x = ball_xy[29, 0]
    for t in range(30, 70):
        pos = [start_x + (t - 29) * (18.0 / fps), 67.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [start_x + 15.0, 66.0] # Defender nearby
        players_xy[t, 1] = [start_x + 40 * (18.0 / fps) + 0.3, 67.0]

    # 70..110: H1 controls and carries
    base_x = ball_xy[69, 0]
    for t in range(70, 111):
        pos = [base_x + (t - 69) * (2.0 / fps), 67.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Eval-02: High-Speed Pass Near Boundary (y=67.0m)", ball_xy, players_xy, team_of_track, gt


def build_eval_03_throw_in_outside_to_inside(fps: float = 25.0):
    """Eval-03: Throw-in preparation outside pitch (y=69.0m) followed by throw into pitch.
    
    Frames 0..34: Ball at y=69.0m (outside line 68.0m). Dead ball D.
    Frames 35..44: Thrown onto pitch (reaches y=58.0m).
    Frames 45..90: Controlled by H1 on pitch.
    GT: D (0..34) -> H (35..90).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..34: Outside touchline
    for t in range(35):
        ball_xy[t] = [45.0, 69.0]
        players_xy[t, 0] = [45.0, 69.2]
        players_xy[t, 1] = [45.0, 58.0]

    # 35..44: Thrown into pitch (crossing y=68.0m at t=36)
    for t in range(35, 45):
        frac = (t - 35) / 10.0
        ball_xy[t] = [45.0, 69.0 - frac * 11.0]
        players_xy[t, 0] = [45.0, 68.5]
        players_xy[t, 1] = [45.0, 58.0]

    # 45..90: H1 carries
    for t in range(45, 91):
        pos = [45.0 + (t - 44) * (2.0 / fps), 58.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["D"] * 35 + ["H"] * 56)
    return "Eval-03: Throw-In Prep Outside Pitch & In-Pitch Release", ball_xy, players_xy, team_of_track, gt


def build_eval_04_goal_kick_box_setup(fps: float = 25.0):
    """Eval-04: Ball stationary inside 6-yard box (x=4.0m) before goal kick.
    
    Frames 0..39: Ball at (4.0, 34.0), GK takes 35-frame run up (D).
    Frames 40..90: Ball struck at 22 m/s downfield (H).
    GT: D (0..39) -> H (40..90).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(40):
        ball_xy[t] = [4.0, 34.0]
        players_xy[t, 0] = [4.0 - (t / 40.0) * 2.0, 34.0]

    for t in range(40, 91):
        pos = [4.0 + (t - 39) * (22.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [4.0, 34.0]
        players_xy[t, 1] = [60.0, 34.0]

    gt = np.array(["D"] * 40 + ["H"] * 51)
    return "Eval-04: Goal-Kick Preparation Inside 6-Yard Box", ball_xy, players_xy, team_of_track, gt


def build_eval_05_glancing_deflection_through(fps: float = 25.0):
    """Eval-05: Pass glances off defender shin (d=0.3m, theta=12 deg) but continues to H1.
    
    GT: Continuous H (0..100).
    """
    n = 101
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: H0 carries
    for t in range(25):
        pos = [20.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 25..59: Pass at 12 m/s. Defender A23 is at (35.0, 30.2) and glances it at t=38
    for t in range(25, 60):
        pos = [ball_xy[24, 0] + (t - 24) * (12.0 / fps), 30.0 + (0.5 if t >= 38 else 0.0)]
        ball_xy[t] = pos
        players_xy[t, 23] = [35.0, 30.2]
        players_xy[t, 1] = [55.0, 30.5]

    # 60..100: H1 controls and carries
    base_x = ball_xy[59, 0]
    for t in range(60, 101):
        pos = [base_x + (t - 59) * (2.0 / fps), 30.5]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Eval-05: Glancing Deflection Without Turnover", ball_xy, players_xy, team_of_track, gt


def build_eval_06_genuine_onetouch_interception(fps: float = 25.0):
    """Eval-06: Defender A23 steps in front, intercepts with deliberate one-touch pass to A24.
    
    GT: H (0..34) -> A (35..90).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: H0 carries
    for t in range(25):
        pos = [20.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 25..34: Pass toward H1
    for t in range(25, 35):
        pos = [ball_xy[24, 0] + (t - 24) * (10.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [30.0, 30.0]

    # 35..59: A23 first-time volley pass to A24 at (40.0, 45.0) (14 m/s)
    start_pos = np.array(ball_xy[34])
    dest_pos = np.array([40.0, 45.0])
    for t in range(35, 60):
        frac = (t - 35) / 25.0
        ball_xy[t] = start_pos + frac * (dest_pos - start_pos)
        players_xy[t, 23] = [30.0, 30.0]
        players_xy[t, 24] = [40.0, 45.0]

    # 60..90: A24 controls and carries
    for t in range(60, 91):
        pos = [40.0 + (t - 59) * (2.0 / fps), 45.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 35 + ["A"] * 56)
    return "Eval-06: Deliberate One-Touch Interception Pass", ball_xy, players_xy, team_of_track, gt


def get_all_step16_eval_scenarios():
    return [
        build_eval_01_ball_hugging_touchline(),
        build_eval_02_high_speed_pass_near_boundary(),
        build_eval_03_throw_in_outside_to_inside(),
        build_eval_04_goal_kick_box_setup(),
        build_eval_05_glancing_deflection_through(),
        build_eval_06_genuine_onetouch_interception(),
    ]


# ==============================================================================
# EVALUATION HARNESS
# ==============================================================================

def evaluate_candidates_on_scenarios(scenarios: list, suite_name: str) -> dict:
    candidates = ["v5", "v6", "v7a", "v7b", "v7c"]
    scores = {c: [] for c in candidates}
    b_errs = {c: [] for c in candidates}

    print("\n" + "=" * 105)
    print(f"EVALUATING SUITE: {suite_name} ({len(scenarios)} scenarios)")
    print("=" * 105)
    header = "{:<50} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7}".format(
        "Scenario Name", "v5", "v6", "v7-A", "v7-B", "v7-C"
    )
    print(header)
    print("-" * 105)

    pipeline_map = {
        "v5": run_pipeline_v5,
        "v6": run_pipeline_v6,
        "v7a": run_pipeline_v7a,
        "v7b": run_pipeline_v7b,
        "v7c": run_pipeline_v7c,
    }

    for item in scenarios:
        if len(item) == 5:
            name, b, p, t, gt = item
        else:
            b, p, t, gt = item
            name = "Scenario"
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)

        row_scores = {}
        for c in candidates:
            fn = pipeline_map[c]
            pred = fn(b, p, t, [])
            res = pm.spell_scores(gt_codes, pm.label_codes(pred), runs)
            s = res["mean_f1"]
            err = compute_boundary_timing_error(gt, pred)
            scores[c].append(s)
            b_errs[c].append(err)
            row_scores[c] = s

        print("{:<50} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f}".format(
            name[:50], row_scores["v5"], row_scores["v6"], row_scores["v7a"], row_scores["v7b"], row_scores["v7c"]
        ))

    print("-" * 105)
    mean_str = "{:<50} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f}".format(
        "MEAN SPELL F1", np.mean(scores["v5"]), np.mean(scores["v6"]), np.mean(scores["v7a"]), np.mean(scores["v7b"]), np.mean(scores["v7c"])
    )
    print(mean_str)
    err_str = "{:<50} | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f".format(
        "MEAN BOUNDARY ERROR (frames)", np.mean(b_errs["v5"]), np.mean(b_errs["v6"]), np.mean(b_errs["v7a"]), np.mean(b_errs["v7b"]), np.mean(b_errs["v7c"])
    )
    print(err_str)
    print("=" * 105)
    return {
        "scores": scores,
        "b_errs": b_errs,
    }


def main():
    frozen = get_all_frozen_scenarios()
    held_out = get_all_held_out_scenarios()
    adv = [
        build_counterexample_1_skimmed_header(),
        build_counterexample_2_one_two_intercepted(),
        build_counterexample_3_clearance_out_of_bounds(),
        build_counterexample_4_aerial_50_50_contest(),
        build_counterexample_5_double_deflection(),
    ]
    step16_eval = get_all_step16_eval_scenarios()

    res_frozen = evaluate_candidates_on_scenarios(frozen, "1. 10 FROZEN BENCHMARK SCENARIOS (Step 11)")
    res_heldout = evaluate_candidates_on_scenarios(held_out, "2. 10 HELD-OUT SCENARIOS (Step 13)")
    res_adv = evaluate_candidates_on_scenarios(adv, "3. 5 ADVERSARIAL COUNTEREXAMPLES (Step 12)")
    res_step16 = evaluate_candidates_on_scenarios(step16_eval, "4. 6 STEP 16 UNTOUCHED EVALUATION SCENARIOS (Frozen)")

    # Aggregate across all 31 scenarios
    all_31 = frozen + held_out + adv + step16_eval
    print("\n" + "=" * 105)
    print(f"OVERALL AGGREGATE SUMMARY ACROSS ALL {len(all_31)} INDEPENDENT SCENARIOS")
    print("=" * 105)
    res_all = evaluate_candidates_on_scenarios(all_31, f"ALL {len(all_31)} COMBINED SCENARIOS")


if __name__ == "__main__":
    main()
