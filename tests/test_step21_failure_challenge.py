#!/usr/bin/env python3
"""Step 21: Final Independent Failure Challenge Suite (Challenge-21-01 to Challenge-21-05).

Defined independently with ground-truth labels and acceptance criteria specified BEFORE execution:
1. Challenge-21-01: Midfield Contested Aerial Duel (High punt, two players contest in air, glances to teammate).
2. Challenge-21-02: First-Time One-Touch Interception Pass (Deliberate half-volley strike to winger).
3. Challenge-21-03: Throw-In Stoppage with Inbound Cross (4s out of bounds, restart to teammate).
4. Challenge-21-04: Slide Tackle with Optical Occlusion (12f missing ball during challenge, tackler emerges).
5. Challenge-21-05: High-Speed Clearance Crossing Endline into Stands (18 m/s clearance, 5f visible OOB then camera dropout).
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
from tests.test_flight_handover_benchmark import compute_boundary_timing_error, run_pipeline_v3
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm


def build_challenge21_01_contested_aerial(fps=25.0):
    """Challenge-21-01: Goalkeeper long aerial punt into midfield.
    
    H0 punts ball (0..39, v=16m/s).
    At frames 40..55, ball drops between H1 and A23 (both jump, contested duel within 1.0m, ball glances).
    At frame 56, H2 collects the ball and carries (56..95).
    GT: H (0..95) - unbroken team possession.
    """
    n = 96
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..39: Ball in flight from punt
    for t in range(40):
        frac = t / 40.0
        pos = [20.0 + frac * 35.0, 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [20.0, 34.0]

    # 40..55: Aerial contest between H1 and A23 (both within 1.2m of ball at [55.0, 34.0])
    for t in range(40, 56):
        ball_xy[t] = [55.0 + (t - 40) * 0.1, 34.0]
        players_xy[t, 1] = [55.0, 34.5]   # H1
        players_xy[t, 23] = [55.2, 33.5]  # A23 (closest for 4 frames, but contested duel)

    # 56..95: H2 collects and dribbles forward
    for t in range(56, 96):
        pos = [57.0 + (t - 56) * (2.2 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 2] = [pos[0] + 0.2, pos[1]]

    gt = np.array(["H"] * n)
    return "Challenge-21-01: Midfield Contested Aerial Duel", ball_xy, players_xy, team_of_track, gt


def build_challenge21_02_one_touch_interception(fps=25.0):
    """Challenge-21-02: First-time half-volley interception pass.
    
    H0 carries and passes (0..29).
    At frame 30, A23 executes a first-time volleyed interception strike (speed 14 m/s) to A24.
    Ball travels in air (31..55) to A24 who collects and carries (56..90).
    GT: H (0..29) -> A (30..90).
    Acceptance: Turnover recognized at frame 30 (moment of strike), not delayed until frame 56.
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..29: H0 passes
    for t in range(30):
        pos = [40.0 + t * (4.0 / fps), 25.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] - 0.5, pos[1]]

    # 30: A23 makes immediate first-time volley contact
    ball_xy[30] = [45.0, 25.0]
    players_xy[30, 23] = [45.1, 25.0]

    # 31..55: Ball flies to A24 downfield
    for t in range(31, 56):
        frac = (t - 30) / 25.0
        pos = [45.0 + frac * 25.0, 25.0 + frac * 20.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [45.0, 25.0]
        players_xy[t, 24] = [70.0, 45.0]

    # 56..90: A24 controls and carries
    for t in range(56, 91):
        pos = [70.0 + (t - 56) * (2.0 / fps), 45.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.2, pos[1]]

    gt = np.array(["H"] * 30 + ["A"] * 61)
    return "Challenge-21-02: One-Touch Interception Half-Volley Pass", ball_xy, players_xy, team_of_track, gt


def build_challenge21_03_throw_in_stoppage(fps=25.0):
    """Challenge-21-03: Throw-in preparation followed by inbound throw.
    
    H0 dribbles toward sideline (0..24).
    Ball crosses touchline at frame 25 (y=69.0m, d_oob=1.0m) for throw-in (25..74, 2.0s = 50f).
    At frame 75, H0 throws inbound to H1 (75..84).
    H1 controls and carries inside (85..119).
    GT: H (0..24) -> D (25..74) -> H (75..119).
    """
    n = 120
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: H0 carries near sideline
    for t in range(25):
        pos = [50.0, 60.0 + t * (7.0 / fps)]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0], pos[1] - 0.3]

    # 25..74: Ball outside touchline (y=69.0m, d_oob=1.0m)
    for t in range(25, 75):
        ball_xy[t] = [50.0, 69.0]
        players_xy[t, 0] = [50.0, 68.8]

    # 75..84: Throw-in traveling back inside
    for t in range(75, 85):
        frac = (t - 75) / 10.0
        pos = [50.0 + frac * 8.0, 68.5 - frac * 10.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [50.0, 68.5]
        players_xy[t, 1] = [58.0, 58.5]

    # 85..119: H1 controls and carries
    for t in range(85, 120):
        pos = [58.0 + (t - 85) * (2.0 / fps), 58.5]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.2, pos[1]]

    gt = np.array(["H"] * 25 + ["D"] * 50 + ["H"] * 45)
    return "Challenge-21-03: Throw-In Stoppage with Inbound Throw", ball_xy, players_xy, team_of_track, gt


def build_challenge21_04_tackle_with_optical_occlusion(fps=25.0):
    """Challenge-21-04: Slide tackle in open field with 12-frame tracking occlusion.
    
    H0 carries ball (0..29).
    At frame 30, A23 executes sliding challenge. Ball is occluded (NaN) for 12 frames (30..41).
    At frame 42, A23 emerges with ball and carries (42..75).
    GT: H (0..29) -> A (30..75).
    Acceptance: Clean turnover, no false Dead Ball during 0.48s live tracking occlusion.
    """
    n = 76
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..29: H0 carries
    for t in range(30):
        pos = [50.0 + t * (2.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.2, pos[1]]
        players_xy[t, 23] = [53.0 - t * (1.0 / fps), 34.0]

    # 30..41: Ball occluded by sliding challenge (NaN)
    for t in range(30, 42):
        ball_xy[t] = [np.nan, np.nan]
        players_xy[t, 0] = [52.5, 34.0]
        players_xy[t, 23] = [52.4, 34.0]

    # 42..75: A23 emerges and carries ball
    for t in range(42, 76):
        pos = [52.4 - (t - 42) * (2.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] - 0.2, pos[1]]

    gt = np.array(["H"] * 30 + ["A"] * 46)
    return "Challenge-21-04: Slide Tackle with Optical Occlusion", ball_xy, players_xy, team_of_track, gt


def build_challenge21_05_clearance_crossing_endline(fps=25.0):
    """Challenge-21-05: High-speed defensive clearance crossing endline into stands.
    
    H0 clears ball under press at 18 m/s towards left endline (0..24).
    Ball crosses goal line at frame 25 (x = -0.5m, d_oob = 0.5m).
    Ball tracked for 6 frames outside pitch (25..30, x from -0.5m to -4.5m).
    Camera drops ball at frame 31 (31..60 = NaN).
    GT: H (0..24) -> D (25..60).
    Acceptance: Boundary clamped to D starting at frame 25, zero ghost possession outside pitch.
    """
    n = 61
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: Ball traveling rapidly toward endline
    for t in range(25):
        pos = [18.0 - t * (18.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [18.0, 34.0]

    # 25..30: Ball crossing outside goal line (x < 0)
    for t in range(25, 31):
        ball_xy[t] = [-0.5 - (t - 25) * 0.8, 34.0]
        players_xy[t, 0] = [1.0, 34.0]

    # 31..60: Camera loses ball outside stadium (NaN)
    for t in range(31, 61):
        ball_xy[t] = [np.nan, np.nan]
        players_xy[t, 0] = [1.0, 34.0]

    gt = np.array(["H"] * 25 + ["D"] * 36)
    return "Challenge-21-05: High-Speed Clearance Crossing Endline into Stands", ball_xy, players_xy, team_of_track, gt


def run_challenge_suite():
    challenges = [
        build_challenge21_01_contested_aerial(),
        build_challenge21_02_one_touch_interception(),
        build_challenge21_03_throw_in_stoppage(),
        build_challenge21_04_tackle_with_optical_occlusion(),
        build_challenge21_05_clearance_crossing_endline(),
    ]

    print("=" * 105)
    print("STEP 21: FINAL INDEPENDENT FAILURE CHALLENGE BENCHMARK")
    print("=" * 105)

    candidates = {
        "v3": run_pipeline_v3,
        "v5": run_pipeline_v5,
        "v6": run_pipeline_v6,
        "v7-A": run_pipeline_v7a,
        "v5-A": run_pipeline_v5a,
    }

    scores = {c: [] for c in candidates}
    errors = {c: [] for c in candidates}

    print(f"{'Challenge Scenario':<60} | {'v3':<6} | {'v5':<6} | {'v6':<6} | {'v7-A':<6} | {'v5-A':<6}")
    print("-" * 105)

    for name, ball_xy, players_xy, team_of_track, gt in challenges:
        n = len(gt)
        gt_codes = np.array([pm.LABELS.index(x) for x in gt], dtype=np.int8)
        runs = pm.valid_runs(gt_codes)
        row = f"{name:<60}"

        for c_name, fn in candidates.items():
            if c_name in ("v7-A", "v5-A"):
                pred = fn(ball_xy, players_xy, team_of_track, non_play_intervals=[], pitch=(105.3, 68.0), fps=25.0)
            else:
                pred = fn(ball_xy, players_xy, team_of_track, non_play_intervals=[], fps=25.0)

            pred_codes = np.array([pm.LABELS.index(x) for x in pred], dtype=np.int8)
            f1 = pm.spell_scores(gt_codes, pred_codes, runs)["mean_f1"]
            b_err = compute_boundary_timing_error(gt, pred)

            scores[c_name].append(f1)
            errors[c_name].append(b_err)
            row += f" | {f1:.4f}"

        print(row)

    print("-" * 105)
    mean_row = f"{'MEAN SPELL F1 (ACROSS 5 INDEPENDENT CHALLENGES)':<60}"
    for c_name in candidates:
        mean_row += f" | {np.mean(scores[c_name]):.4f}"
    print(mean_row)

    err_row = f"{'MEAN BOUNDARY ERROR (FRAMES)':<60}"
    for c_name in candidates:
        err_row += f" | {np.mean(errors[c_name]):.2f}f"
    print(err_row)
    print("=" * 105)


if __name__ == "__main__":
    run_challenge_suite()
