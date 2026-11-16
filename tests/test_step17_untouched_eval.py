#!/usr/bin/env python3
"""Step 17: Genuinely Untouched Independent Evaluation Suite (Eval-17-01 to Eval-17-05).

Defined independently from candidate predictions:
1. Eval-17-01: Rapid 3-touch turnover sequence in midfield (H -> A -> H).
2. Eval-17-02: Ball crosses touchline into advertising board and rebounds back inside.
3. Eval-17-03: Corner kick arc geometry (ball on arc inside, taker steps outside).
4. Eval-17-04: First-time sliding interception clearance to distant winger.
5. Eval-17-05: Goalkeeper parry and gather near goal line (inside pitch).
"""

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom11.predict_v7 import (
    run_pipeline_v5,
    run_pipeline_v5a,
    run_pipeline_v6,
    run_pipeline_v7a,
    run_pipeline_v7c,
)
from tests.test_flight_handover_benchmark import compute_boundary_timing_error
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm


def build_eval17_01_rapid_three_touch(fps=25.0):
    """Eval-17-01: Rapid 3-touch turnover sequence (H 1.2s -> A 1.0s -> H 1.4s).
    
    GT: H (0..29) -> A (30..54) -> H (55..89).
    """
    n = 90
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..29: H0 carries
    for t in range(30):
        pos = [40.0 + t * (2.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.2, pos[1]]

    # 30..54: A23 tackles, controls, and carries
    for t in range(30, 55):
        pos = [ball_xy[29, 0] - (t - 29) * (2.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] - 0.2, pos[1]]

    # 55..89: H1 counter-tackles and carries away
    for t in range(55, 90):
        pos = [ball_xy[54, 0] + (t - 54) * (2.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.2, pos[1]]

    gt = np.array(["H"] * 30 + ["A"] * 25 + ["H"] * 35)
    return "Eval-17-01: Rapid 3-Touch Turnover Sequence (H->A->H)", ball_xy, players_xy, team_of_track, gt


def build_eval17_02_ad_board_rebound(fps=25.0):
    """Eval-17-02: Ball crosses touchline at y=69.0m (1m OOB), hits board, rebounds in.
    
    GT: H (0..34) -> D (35..65) -> H (66..100).
    """
    n = 101
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..34: H0 passes towards sideline
    for t in range(35):
        pos = [50.0, 60.0 + t * (8.0 / fps)]
        ball_xy[t] = pos
        players_xy[t, 0] = [50.0, 58.0]

    # 35..65: Ball crosses touchline (y=68.0 to 69.5) and bounces back
    for t in range(35, 66):
        frac = (t - 35) / 15.0 if t < 50 else (65 - t) / 15.0
        pos = [50.0, 68.0 + frac * 1.5]
        ball_xy[t] = pos
        players_xy[t, 0] = [50.0, 65.0]

    # 66..100: H0 collects and resumes play inside
    for t in range(66, 101):
        pos = [50.0, 66.0 - (t - 65) * (1.5 / fps)]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0], pos[1] - 0.2]

    gt = np.array(["H"] * 35 + ["D"] * 31 + ["H"] * 35)
    return "Eval-17-02: Out-of-Bounds Rebound off Ad Board", ball_xy, players_xy, team_of_track, gt


def build_eval17_03_corner_arc_taker_outside(fps=25.0):
    """Eval-17-03: Ball placed on corner arc (x=0.5m, y=0.5m), taker stands outside pitch.
    
    Taker at (-0.8m, -0.5m) takes run-up. Ball is inside pitch!
    GT: D (0..40, setup) -> H (41..90, corner delivered).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..40: Ball stationary on corner quadrant (0.5m, 0.5m). Taker outside (-0.8, -0.5)
    for t in range(41):
        ball_xy[t] = [0.5, 0.5]
        players_xy[t, 0] = [-0.8 + (t / 40.0) * 1.2, -0.5 + (t / 40.0) * 0.9]

    # 41..90: Ball struck towards box (x=10.0, y=20.0)
    for t in range(41, 91):
        frac = (t - 40) / 50.0
        ball_xy[t] = [0.5 + frac * 10.0, 0.5 + frac * 20.0]
        players_xy[t, 0] = [0.5, 0.5]
        players_xy[t, 1] = [10.5, 20.5]

    gt = np.array(["D"] * 41 + ["H"] * 50)
    return "Eval-17-03: Corner Kick Arc Setup with Taker Outside", ball_xy, players_xy, team_of_track, gt


def build_eval17_04_sliding_interception_clearance(fps=25.0):
    """Eval-17-04: Defender executes sliding interception kicking ball at 16 m/s to winger.
    
    GT: H (0..34) -> A (35..90).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: H0 carries
    for t in range(25):
        pos = [25.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 25..34: Pass forward
    for t in range(25, 35):
        pos = [ball_xy[24, 0] + (t - 24) * (10.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [35.0, 30.0]

    # 35..64: A23 slides in, first-time boots ball to A24 at (65.0, 50.0)
    start_pos = np.array(ball_xy[34])
    end_pos = np.array([65.0, 50.0])
    for t in range(35, 65):
        frac = (t - 35) / 30.0
        ball_xy[t] = start_pos + frac * (end_pos - start_pos)
        players_xy[t, 23] = [35.0, 30.0]
        players_xy[t, 24] = [65.0, 50.0]

    # 65..90: A24 controls and carries
    for t in range(65, 91):
        pos = [65.0 + (t - 64) * (2.0 / fps), 50.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 35 + ["A"] * 56)
    return "Eval-17-04: Sliding Interception Clearance to Winger", ball_xy, players_xy, team_of_track, gt


def build_eval17_05_gk_parry_near_goal_line(fps=25.0):
    """Eval-17-05: Attacker shoots, GK parries at x=0.8m (inside), gathers firmly at x=0.4m.
    
    Ball never crosses goal line (d_oob = 0).
    GT: H (0..34) -> A (35..90, GK possession).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: H0 carries towards box
    for t in range(25):
        pos = [15.0 - t * (2.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] - 0.3, pos[1]]

    # 25..34: H0 shoots at 20 m/s toward goal
    for t in range(25, 35):
        pos = [ball_xy[24, 0] - (t - 24) * (20.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [1.0, 34.0] # GK

    # 35..50: GK parries down, ball rolls slowly to x=0.4m (never crosses line)
    for t in range(35, 51):
        frac = (t - 35) / 15.0
        ball_xy[t] = [0.8 - frac * 0.4, 34.0 + frac * 0.5]
        players_xy[t, 23] = [ball_xy[t, 0] + 0.2, ball_xy[t, 1]]

    # 51..90: GK gathers and holds
    for t in range(51, 91):
        ball_xy[t] = [0.4, 34.5]
        players_xy[t, 23] = [0.4, 34.5]

    gt = np.array(["H"] * 35 + ["A"] * 56)
    return "Eval-17-05: Goalkeeper Parry & Gather Near Goal Line", ball_xy, players_xy, team_of_track, gt


def main():
    scenarios = [
        build_eval17_01_rapid_three_touch(),
        build_eval17_02_ad_board_rebound(),
        build_eval17_03_corner_arc_taker_outside(),
        build_eval17_04_sliding_interception_clearance(),
        build_eval17_05_gk_parry_near_goal_line(),
    ]

    candidates = ["v5", "v6", "v7a", "v7c", "v5a"]
    pipeline_map = {
        "v5": run_pipeline_v5,
        "v6": run_pipeline_v6,
        "v7a": run_pipeline_v7a,
        "v7c": run_pipeline_v7c,
        "v5a": run_pipeline_v5a,
    }

    scores = {c: [] for c in candidates}
    b_errs = {c: [] for c in candidates}

    print("\n" + "=" * 105)
    print("STEP 17: GENUINELY UNTOUCHED INDEPENDENT EVALUATION (5 SCENARIOS)")
    print("=" * 105)
    print("{:<50} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7}".format(
        "Scenario Name", "v5", "v6", "v7-A", "v7-C", "v5-A"
    ))
    print("-" * 105)

    for item in scenarios:
        name, b, p, t, gt = item
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
            name[:50], row_scores["v5"], row_scores["v6"], row_scores["v7a"], row_scores["v7c"], row_scores["v5a"]
        ))

    print("-" * 105)
    mean_str = "{:<50} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f}".format(
        "MEAN SPELL F1", np.mean(scores["v5"]), np.mean(scores["v6"]), np.mean(scores["v7a"]), np.mean(scores["v7c"]), np.mean(scores["v5a"])
    )
    print(mean_str)
    err_str = "{:<50} | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f".format(
        "MEAN BOUNDARY ERROR (frames)", np.mean(b_errs["v5"]), np.mean(b_errs["v6"]), np.mean(b_errs["v7a"]), np.mean(b_errs["v7c"]), np.mean(b_errs["v5a"])
    )
    print(err_str)
    print("=" * 105)


if __name__ == "__main__":
    main()
