#!/usr/bin/env python3
"""Step 15 - Task 4: Unbiased Adversarial Test Suite for Possession Handover

Contains:
1. Development Suite (4 scenarios):
   - Dev-01: Genuine Short Interception (1.2s carry) vs Glancing Deflection
   - Dev-02: Throw-In with Touchline Setup and In-Pitch Release
   - Dev-03: Clearance Out of Bounds vs Clearance Collected In Bounds
   - Dev-04: Co-Moving Defender Running Alongside Pass
2. Final Held-Out Evaluation Suite (4 scenarios - FROZEN):
   - Holdout-01: Rapid Double Turnover (two 1.0s genuine possessions)
   - Holdout-02: Goal-Kick Routine (stationary ball in 6-yard box to strike)
   - Holdout-03: Touchline Slide Tackle Poked Out of Bounds
   - Holdout-04: Aerial Header Duel Won and Headed to Teammate

Evaluates v5 and v6 using official spellF1 implementation.
"""

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_flight_handover_benchmark import (
    run_pipeline_v3,
    run_pipeline_v4,
    compute_boundary_timing_error,
)
from tests.run_comparative_benchmark import run_pipeline_v5
from tests.run_v5_v6_comparative_audit import run_pipeline_v6
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm

# ==============================================================================
# 1. DEVELOPMENT SUITE (4 SCENARIOS)
# ==============================================================================

def build_dev_01_short_interception_vs_deflection(fps: float = 25.0):
    """Dev-01: Genuine 1.2s interception (30f) followed by pass to teammate.
    
    H0 passes at t=20. A23 intercepts at t=35, dribbles for 30 frames (1.2s), passes to A24 at t=65.
    GT: H (0..34) -> A (35..100). Genuine turnover that MUST NOT be erased.
    """
    n = 101
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..19: H0 carries
    for t in range(20):
        pos = [20.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 20..34: Pass toward A23
    for t in range(20, 35):
        pos = [ball_xy[19, 0] + (t - 19) * (10.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [28.0, 30.0]

    # 35..64: A23 controls and dribbles forward (1.2s = 30f)
    for t in range(35, 65):
        pos = [28.0 + (t - 34) * (2.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 24] = [50.0, 35.0]

    # 65..100: A23 passes to A24, A24 carries
    for t in range(65, 101):
        pos = [ball_xy[64, 0] + (t - 64) * (2.0 / fps), 30.0 + (t - 64) * (0.3 / fps)]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 35 + ["A"] * 66)
    return "Dev-01: Genuine Short Interception (1.2s Carry)", ball_xy, players_xy, team_of_track, gt


def build_dev_02_throw_in_touchline_setup(fps: float = 25.0):
    """Dev-02: Throw-in preparation outside pitch followed by active release.
    
    H0 stands at touchline y=-0.8m holding ball stationary (0..34, D).
    At t=35, H0 throws ball onto pitch (speed 9 m/s). H1 traps at t=45 and carries.
    GT: D (0..34) -> H (35..90).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..34: H0 holding ball outside touchline (y = -0.8m)
    for t in range(35):
        ball_xy[t] = [30.0, -0.8]
        players_xy[t, 0] = [30.0, -1.0]
        players_xy[t, 1] = [30.0, 8.0] # Teammate on pitch

    # 35..44: Throw in flight from y=-0.8 to y=8.0 (9 m/s)
    for t in range(35, 45):
        frac = (t - 35) / 10.0
        pos = [30.0, -0.8 + frac * 8.8]
        ball_xy[t] = pos
        players_xy[t, 0] = [30.0, -0.5]
        players_xy[t, 1] = [30.0, 8.0]

    # 45..90: H1 controls on pitch and carries
    for t in range(45, 91):
        pos = [30.0 + (t - 44) * (2.0 / fps), 8.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["D"] * 35 + ["H"] * 56)
    return "Dev-02: Throw-In Touchline Setup and Release", ball_xy, players_xy, team_of_track, gt


def build_dev_03_clearance_out_vs_in(fps: float = 25.0):
    """Dev-03: Clearance kicked directly out of bounds vs collected in-bounds.
    
    H0 attacks (0..29). Defender A23 clears ball at t=30 at 22 m/s.
    Ball crosses touchline (y=69.0) at t=48.
    GT: H (0..29) -> A (30..48) -> D (49..80).
    """
    n = 81
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..29: H0 attacks
    for t in range(30):
        pos = [40.0 + t * (2.0 / fps), 50.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 23] = [43.0, 50.0]

    # 30..48: A23 first-time clearance toward touchline y=69m (22 m/s)
    for t in range(30, 49):
        pos = [ball_xy[29, 0] + (t - 29) * (8.0 / fps), 50.0 + (t - 29) * (25.0 / fps)]
        ball_xy[t] = pos
        players_xy[t, 23] = [43.0, 50.0]

    # 49..80: Out of bounds (y > 68.0), ball missing or stopped outside
    for t in range(49, 81):
        ball_xy[t] = [ball_xy[48, 0] + 0.2, 70.0]
        players_xy[t, 23] = [43.0, 50.0]

    gt = np.array(["H"] * 30 + ["A"] * 19 + ["D"] * 32)
    return "Dev-03: Clearance Kicked Out of Bounds", ball_xy, players_xy, team_of_track, gt


def build_dev_04_co_moving_defender_alongside_pass(fps: float = 25.0):
    """Dev-04: Pass traveling at 6 m/s past sprinting defender who does not touch it.
    
    H0 passes to H1 (distance 30m, speed 6.0 m/s). Defender A23 runs parallel at d=1.0m.
    Ball reaches H1 cleanly.
    GT: Continuous H (0..120). Defender is NOT a carrier.
    """
    n = 121
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..29: H0 carries
    for t in range(30):
        pos = [20.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 30..89: Pass to H1 (6 m/s). Defender A23 sprints alongside at d=1.0m
    for t in range(30, 90):
        pos = [ball_xy[29, 0] + (t - 29) * (6.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0], 31.0] # d = 1.0m, identical velocity
        players_xy[t, 1] = [ball_xy[29, 0] + 60 * (6.0 / fps) + 0.3, 30.0]

    # 90..120: H1 controls and carries
    for t in range(90, 121):
        pos = [ball_xy[89, 0] + (t - 89) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Dev-04: Co-Moving Defender Running Alongside Pass", ball_xy, players_xy, team_of_track, gt


# ==============================================================================
# 2. FINAL HELD-OUT EVALUATION SUITE (4 SCENARIOS - FROZEN)
# ==============================================================================

def build_holdout_01_rapid_double_turnover(fps: float = 25.0):
    """Holdout-01: Two genuine consecutive turnovers of 1.0s (25 frames) each.
    
    H0 carries (0..29) -> A23 tackles & carries 1.0s (30..54) -> H1 tackles & carries (55..90).
    GT: H (0..29) -> A (30..54) -> H (55..90). Must preserve both short possessions!
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..29: H0 carries
    for t in range(30):
        pos = [30.0 + t * (2.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 23] = [pos[0] + 1.5, pos[1]]

    # 30..54: A23 tackles and carries forward (25 frames = 1.0s)
    for t in range(30, 55):
        pos = [ball_xy[29, 0] + (t - 29) * (2.0 / fps), 30.0 - (t - 29) * (0.5 / fps)]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.2, pos[1]]
        players_xy[t, 1] = [pos[0] + 1.2, pos[1]]

    # 55..90: H1 counter-tackles and carries away
    for t in range(55, 91):
        pos = [ball_xy[54, 0] + (t - 54) * (2.5 / fps), ball_xy[54, 1] + (t - 54) * (1.0 / fps)]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 30 + ["A"] * 25 + ["H"] * 36)
    return "Holdout-01: Rapid Double Turnover (Two 1.0s Possessions)", ball_xy, players_xy, team_of_track, gt


def build_holdout_02_goal_kick_routine(fps: float = 25.0):
    """Holdout-02: Goalkeeper sets ball in 6-yard box (x=5.0m), steps back, strikes.
    
    Frames 0..39: Ball stationary at (5.0, 34.0). GK takes run-up (D).
    Frames 40..90: Ball struck at 20 m/s downfield (H).
    GT: D (0..39) -> H (40..90).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..39: Stationary ball at (5.0, 34.0), GK steps back from (5.3, 34.0) to (2.5, 34.0)
    for t in range(40):
        ball_xy[t] = [5.0, 34.0]
        # GK steps back for run-up
        players_xy[t, 0] = [5.0 - (t / 40.0) * 2.5, 34.0]

    # 40..90: Strike downfield at 20 m/s
    for t in range(40, 91):
        pos = [5.0 + (t - 39) * (20.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [5.0, 34.0]
        players_xy[t, 1] = [60.0, 34.0]

    gt = np.array(["D"] * 40 + ["H"] * 51)
    return "Holdout-02: Goal-Kick Routine (6-Yard Box Setup)", ball_xy, players_xy, team_of_track, gt


def build_holdout_03_slide_tackle_over_touchline(fps: float = 25.0):
    """Holdout-03: Slide tackle cleanly knocking ball over touchline into stoppage.
    
    H0 dribbling at y=66m (0..34). A23 slide-tackles ball at t=35.
    Ball slides over touchline y=68.5m at t=50. Stoppage D (50..80).
    GT: H (0..34) -> A (35..49) -> D (50..80).
    """
    n = 81
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..34: H0 dribbles
    for t in range(35):
        pos = [50.0 + t * (2.0 / fps), 66.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 23] = [pos[0] + 1.2, 67.0]

    # 35..49: A23 slides in, ball pokes toward touchline (10 m/s)
    for t in range(35, 50):
        pos = [ball_xy[34, 0] + (t - 34) * (2.0 / fps), 66.0 + (t - 34) * (4.0 / fps)]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] - 0.2, 66.5]

    # 50..80: Out of bounds (y > 68.0m)
    for t in range(50, 81):
        ball_xy[t] = [ball_xy[49, 0] + 0.1, 69.5]
        players_xy[t, 23] = [ball_xy[49, 0], 67.0]

    gt = np.array(["H"] * 35 + ["A"] * 15 + ["D"] * 31)
    return "Holdout-03: Slide Tackle Poked Over Touchline", ball_xy, players_xy, team_of_track, gt


def build_holdout_04_aerial_header_duel_controlled(fps: float = 25.0):
    """Holdout-04: Aerial duel where defender A23 wins header and directs it to teammate A24.
    
    H0 launches lofted ball (0..34, H). At t=35, A23 wins header, directing ball to A24 at t=55.
    A24 controls and carries (55..90).
    GT: H (0..34) -> A (35..90).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..34: High lofted ball from H0
    for t in range(35):
        pos = [20.0 + t * (15.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [20.0, 30.0]
        players_xy[t, 23] = [41.0, 30.0] # Defender waiting
        players_xy[t, 1] = [41.0, 30.5]  # Attacker contesting

    # 35..54: A23 wins header, balls arcs toward A24 at (50, 40)
    for t in range(35, 55):
        frac = (t - 35) / 20.0
        pos = [41.0 + frac * 9.0, 30.0 + frac * 10.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [41.0, 30.0]
        players_xy[t, 24] = [50.0, 40.0]

    # 55..90: A24 controls and carries
    for t in range(55, 91):
        pos = [50.0 + (t - 54) * (2.0 / fps), 40.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 35 + ["A"] * 56)
    return "Holdout-04: Aerial Header Won and Headed to Teammate", ball_xy, players_xy, team_of_track, gt


def evaluate_suite(suite_name, scenarios):
    print("=" * 105)
    print(f"EVALUATING SUITE: {suite_name}")
    print("=" * 105)
    print(f"{'Scenario Name':<55} | {'v5 F1':<8} | {'v6 F1':<8} | {'Delta':<8} | {'Outcome'}")
    print("-" * 105)

    v5_scores = []
    v6_scores = []
    b_errs_v5 = []
    b_errs_v6 = []

    for name, b, p, t, gt in scenarios:
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)
        p5 = run_pipeline_v5(b, p, t, [])
        p6 = run_pipeline_v6(b, p, t, [])

        res5 = pm.spell_scores(gt_codes, pm.label_codes(p5), runs)
        res6 = pm.spell_scores(gt_codes, pm.label_codes(p6), runs)

        s5 = res5["mean_f1"]
        s6 = res6["mean_f1"]
        v5_scores.append(s5)
        v6_scores.append(s6)

        e5 = compute_boundary_timing_error(gt, p5)
        e6 = compute_boundary_timing_error(gt, p6)
        b_errs_v5.append(e5)
        b_errs_v6.append(e6)

        delta = s6 - s5
        outcome = "TIED" if abs(delta) < 0.001 else ("v6 WINS" if delta > 0 else "v5 WINS")
        print(f"{name:<55} | {s5:<8.4f} | {s6:<8.4f} | {delta:<+8.4f} | {outcome}")

    mean5 = np.mean(v5_scores)
    mean6 = np.mean(v6_scores)
    err5 = np.mean(b_errs_v5)
    err6 = np.mean(b_errs_v6)
    print("-" * 105)
    print(f"{'MEAN SPELL F1':<55} | {mean5:<8.4f} | {mean6:<8.4f} | {mean6-mean5:<+8.4f} |")
    print(f"{'MEAN BOUNDARY ERROR (frames)':<55} | {err5:<8.2f}f | {err6:<8.2f}f | {err6-err5:<+8.2f}f |")
    print("=" * 105)
    return {
        "v5_scores": v5_scores,
        "v6_scores": v6_scores,
        "mean_v5": mean5,
        "mean_v6": mean6,
        "err_v5": err5,
        "err_v6": err6,
    }


def main():
    dev_scenarios = [
        build_dev_01_short_interception_vs_deflection(),
        build_dev_02_throw_in_touchline_setup(),
        build_dev_03_clearance_out_vs_in(),
        build_dev_04_co_moving_defender_alongside_pass(),
    ]
    evaluate_suite("1. DEVELOPMENT ADVERSARIAL SUITE (4 SCENARIOS)", dev_scenarios)

    holdout_scenarios = [
        build_holdout_01_rapid_double_turnover(),
        build_holdout_02_goal_kick_routine(),
        build_holdout_03_slide_tackle_over_touchline(),
        build_holdout_04_aerial_header_duel_controlled(),
    ]
    evaluate_suite("2. FINAL HELD-OUT EVALUATION SUITE (4 SCENARIOS - FROZEN)", holdout_scenarios)


if __name__ == "__main__":
    main()
