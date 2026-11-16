#!/usr/bin/env python3
"""PHANTOM-11: Frozen Independent Benchmark Suite for Flight-Aware Possession Evaluation

Step 11 of More Than A Hack 2027:
- Independently defined synthetic ground-truth possession sequences covering:
  1. Same-team ground pass (opponent 2D near in transit).
  2. Aerial lofted pass flying over opponent in 2D.
  3. Opposing-team interception in flight.
  4. First-time volley / one-touch clearance.
  5. Headed clearance to goalkeeper.
  6. Brief opponent deflection without control.
  7. Rapid consecutive turnovers (H 1.6s -> A 1.2s -> H 1.6s).
  8. Ball-flight occlusion (tracking dropout mid-pass).
  9. Restart after stoppage (goal kick run-up and strike).
  10. Difficult negative case: Opponent cleanly intercepts pass early.
- Evaluates full pipelines using official evaluation/possession_metrics.py (spellF1).
- Evaluates boundary timing error and robustness under Gaussian noise and dropouts.
- FROZEN BENCHMARK: This file must NOT be modified after seeing model results.
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Resolve official repo for possession_metrics
from phantom11.path_resolver import resolve_official_repo
resolve_official_repo()

import possession_metrics as pm

from phantom11.movement_features import compute_ball_kinematics
from phantom11.control_evidence import compute_control_evidence
from phantom11.boundary_evidence import extract_boundary_evidence
from phantom11.sequence_decoder import decode_possession_sequence, DEFAULT_CONFIG
from phantom11.boundary_refinement import refine_possession_boundaries


def run_pipeline_v3(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    fps: float = 25.0,
    st_events: list = None,
) -> np.ndarray:
    """Run full PHANTOM-11 v3 pipeline."""
    n_frames = len(ball_xy)
    st_events = st_events or []
    ball_present = ~np.isnan(ball_xy[:, 0]) & ~np.isnan(ball_xy[:, 1])

    ball_kin = compute_ball_kinematics(ball_xy, fps=fps)
    ev = compute_control_evidence(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=st_events,
        non_play_intervals=non_play_intervals,
        fps=fps,
    )

    d_raw = np.sqrt(np.sum((players_xy - ball_xy[:, np.newaxis, :]) ** 2, axis=2))
    d_clean = np.where(np.isnan(d_raw), 100.0, d_raw)
    nearest_slot = np.argmin(d_clean, axis=1)
    nearest_team = np.where(ball_present, team_of_track[nearest_slot], -1).astype(np.int8)

    v0_labels = np.full(n_frames, "D", dtype="<U1")
    v0_labels[ball_present & (nearest_team == 0)] = "H"
    v0_labels[ball_present & (nearest_team == 1)] = "A"

    v2_labels = v0_labels.copy()
    c_h = ev["is_carrier_h"] & (ev["score_h"] >= 0.50)
    c_a = ev["is_carrier_a"] & (ev["score_a"] >= 0.50)
    v2_labels[c_h] = "H"
    v2_labels[c_a] = "A"

    geom_nearest = np.where(ev["d_h1"] <= ev["d_a1"], 0, 1).astype(np.int8)
    b_ev = extract_boundary_evidence(
        ball_present=ball_present,
        is_kick_or_deflection=ball_kin["is_kick_or_deflection"],
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        nearest_team=geom_nearest,
        smart_tagging_stoppage=np.zeros(n_frames, dtype=np.float32),
        non_play_intervals=non_play_intervals,
    )

    v3_labels = decode_possession_sequence(
        v2_labels=v2_labels,
        score_h=ev["score_h"],
        score_a=ev["score_a"],
        reliability=ev["reliability"],
        ball_present=ball_present,
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        stoppage_support=np.zeros(n_frames, dtype=np.float32),
        turnover_readiness_ha=b_ev["turnover_readiness_ha"],
        turnover_readiness_ah=b_ev["turnover_readiness_ah"],
        stoppage_readiness=b_ev["stoppage_readiness"],
        restart_readiness=b_ev["restart_readiness"],
        in_non_play=b_ev["in_non_play"],
        config=DEFAULT_CONFIG,
    )
    return v3_labels


def run_pipeline_v4(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    fps: float = 25.0,
    st_events: list = None,
) -> np.ndarray:
    """Run full PHANTOM-11 v4 pipeline (v3 + boundary refinement)."""
    n_frames = len(ball_xy)
    st_events = st_events or []
    ball_present = ~np.isnan(ball_xy[:, 0]) & ~np.isnan(ball_xy[:, 1])

    ball_kin = compute_ball_kinematics(ball_xy, fps=fps)
    ev = compute_control_evidence(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=st_events,
        non_play_intervals=non_play_intervals,
        fps=fps,
    )

    v3_labels = run_pipeline_v3(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        non_play_intervals=non_play_intervals,
        fps=fps,
        st_events=st_events,
    )

    v4_labels = refine_possession_boundaries(
        labels=v3_labels,
        ball_present=ball_present,
        ball_xy=ball_xy,
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        d_h1=ev["d_h1"],
        d_a1=ev["d_a1"],
        ball_speed=ball_kin["speed"],
        stoppage_support=np.zeros(n_frames, dtype=np.float32),
        non_play_intervals=non_play_intervals,
        fps=fps,
    )
    return v4_labels


# ---------------------------------------------------------------------------
# 10 Frozen Benchmark Scenarios
# ---------------------------------------------------------------------------

def build_bench_scenario_01_ground_pass_opponent_near(fps: float = 25.0):
    """Scenario 1: Team H ground pass (35f / 1.4s, 18 m/s). Opponent A23 is within

    0.6m of trajectory at midpoint in 2D, but ball is passing cleanly to H1.
    GT: Continuous H (0..125).
    """
    n = 126
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..39: H0 dribbles
    for t in range(40):
        pos = [30.0 + t * (2.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 40..74: Ground pass to H1 at x=55.2 (18 m/s)
    base_x = ball_xy[39, 0]
    for t in range(40, 75):
        pos = [base_x + (t - 39) * (18.0 / fps), 34.0]
        ball_xy[t] = pos
        # A23 stands at midpoint x=43.0, y=34.6 (0.6m from trajectory in 2D plane)
        players_xy[t, 23] = [43.0, 34.6]
        # H1 is waiting at x=55.2
        players_xy[t, 1] = [55.2 + 0.3, 34.0]

    # 75..125: H1 controls and carries forward
    base_rx = ball_xy[74, 0]
    for t in range(75, 126):
        pos = [base_rx + (t - 74) * (2.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Scenario 01: Same-Team Ground Pass (Opponent Near in 2D)", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_02_aerial_lofted_pass(fps: float = 25.0):
    """Scenario 2: High lobbed cross (60f / 2.4s, 15 m/s, 36m). Opponent A23 stands

    directly underneath at (50, 30.2) in 2D (d=0.2m), but ball is airborne.
    GT: Continuous H (0..159).
    """
    n = 160
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..49: H0 carries
    for t in range(50):
        pos = [20.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 50..109: High lofted cross (15 m/s). A23 is stationary at (50.0, 30.2)
    for t in range(50, 110):
        pos = [ball_xy[49, 0] + (t - 49) * (15.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [50.0, 30.2]
        players_xy[t, 1] = [56.0 + 0.3, 30.0]

    # 110..159: H1 controls and carries
    for t in range(110, 160):
        pos = [ball_xy[109, 0] + (t - 109) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Scenario 02: Aerial Lofted Pass Over Opponent", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_03_in_flight_interception(fps: float = 25.0):
    """Scenario 3: H0 passes at t=40 toward H1. At t=60, opposing defender A23 cuts across,

    traps ball cleanly, and dribbles away t=60..110.
    GT: H (0..59) -> A (60..110).
    """
    n = 111
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..39: H0 carries
    for t in range(40):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 40..59: Pass travels toward x=60 at 15 m/s
    for t in range(40, 60):
        pos = [ball_xy[39, 0] + (t - 39) * (15.0 / fps), 30.0]
        ball_xy[t] = pos
        # A23 is running towards the interception point
        players_xy[t, 23] = [pos[0] + (60 - t) * (1.5 / fps), 30.0 + (60 - t) * (1.0 / fps)]

    # 60..110: A23 gains clean control, traps ball, carries forward
    base_x = ball_xy[59, 0]
    for t in range(60, 111):
        pos = [base_x + (t - 59) * (2.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 60 + ["A"] * 51)
    return "Scenario 03: Opposing-Team Interception in Flight", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_04_first_time_clearance(fps: float = 25.0):
    """Scenario 4: H0 crosses toward box t=0..49. At t=50, defender A23 volleys clearance

    first-time (contact 2 frames, speed 20 m/s). Teammate A24 collects at t=75 and carries.
    GT: H (0..49) -> A (50..120).
    """
    n = 121
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..49: H0 carries and crosses
    for t in range(50):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 50..51: A23 strikes clearance at 20 m/s
    base_x = ball_xy[49, 0]
    for t in range(50, 75):
        pos = [base_x - (t - 49) * (20.0 / fps), 30.0]
        ball_xy[t] = pos
        if t in (50, 51):
            players_xy[t, 23] = [pos[0] + 0.2, pos[1]]
        else:
            players_xy[t, 23] = [base_x + 1.0, 30.0]
        # A24 is waiting at x=10.0
        players_xy[t, 24] = [10.0 + 0.3, 30.0]

    # 75..120: A24 controls the cleared ball and dribbles
    base_cx = ball_xy[74, 0]
    for t in range(75, 121):
        pos = [base_cx - (t - 74) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] - 0.3, pos[1]]

    gt = np.array(["H"] * 50 + ["A"] * 71)
    return "Scenario 04: First-Time Volley / One-Touch Clearance", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_05_headed_clearance_to_gk(fps: float = 25.0):
    """Scenario 5: High cross into box t=0..49. At t=50, defender A23 heads ball back

    to goalkeeper A24 (10 m/s). Goalkeeper catches at t=70 and holds t=70..115.
    GT: H (0..49) -> A (50..115).
    """
    n = 116
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..49: H0 crosses high
    for t in range(50):
        pos = [35.0 + t * (2.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 50..69: A23 heads ball back to GK A24 at x=20.0 (10 m/s)
    base_x = ball_xy[49, 0]
    for t in range(50, 70):
        pos = [base_x - (t - 49) * (10.0 / fps), 34.0]
        ball_xy[t] = pos
        if t in (50, 51):
            players_xy[t, 23] = [base_x, 34.0]
        players_xy[t, 24] = [20.0 + 0.3, 34.0]

    # 70..115: GK A24 catches and holds
    for t in range(70, 116):
        pos = [20.0, 34.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [20.1, 34.0]

    gt = np.array(["H"] * 50 + ["A"] * 66)
    return "Scenario 05: Headed Clearance to Goalkeeper", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_06_brief_opponent_deflection(fps: float = 25.0):
    """Scenario 6: Team H pass glances off defender A23 for 14 frames (0.56s).

    Defender does not control it. Teammate H1 receives and carries t=114..193.
    GT: Continuous H (0..193).
    """
    n = 194
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..99: H0 controls
    for t in range(100):
        pos = [20.0 + t * (2.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 100..113: Glances off defender A23 (14 frames = 0.56s). A23 is nearest but no carrier
    for t in range(100, 114):
        pos = [ball_xy[99, 0] + (t - 99) * (5.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.8, pos[1] + 0.5]

    # 114..193: Teammate H1 receives and carries
    for t in range(114, 194):
        pos = [ball_xy[113, 0] + (t - 113) * (3.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Scenario 06: Brief Opponent Deflection Without Control", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_07_rapid_turnovers(fps: float = 25.0):
    """Scenario 7: H carries 1.6s (40f) -> A tackles & carries 1.2s (30f) -> H tackles back (40f).

    GT: H (0..39) -> A (40..69) -> H (70..109).
    """
    n = 110
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(40):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    for t in range(40, 70):
        pos = [ball_xy[39, 0] - (t - 39) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] - 0.3, pos[1]]

    for t in range(70, 110):
        pos = [ball_xy[69, 0] + (t - 69) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 40 + ["A"] * 30 + ["H"] * 40)
    return "Scenario 07: Rapid Consecutive Turnovers", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_08_ball_flight_occlusion(fps: float = 25.0):
    """Scenario 8: H0 passes at t=40. From t=50..79 (30f / 1.2s), ball is NaN in air.

    Ball reappears at receiver H1's feet at t=80, and H1 carries t=80..129.
    GT: Continuous H (0..129).
    """
    n = 130
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(40):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 40..49: ball visible in flight (15 m/s)
    for t in range(40, 50):
        pos = [ball_xy[39, 0] + (t - 39) * (15.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [55.0 + 0.3, 30.0]

    # 50..79: ball tracking missing (NaN)
    for t in range(50, 80):
        players_xy[t, 1] = [55.0 + 0.3, 30.0]

    # 80..129: ball reappears at H1's feet at x=55.0 and H1 dribbles forward
    for t in range(80, 130):
        pos = [55.0 + (t - 80) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Scenario 08: Ball-Flight Occlusion During Pass", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_09_goal_kick_restart(fps: float = 25.0):
    """Scenario 9: Dead ball D t=0..50 at (10, 34). Goalkeeper runs up t=30..50.

    Goal kick struck at t=51 (22 m/s). Teammate H1 controls at t=80 and carries t=80..120.
    GT: D (0..50) -> H (51..120).
    """
    n = 121
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..50: Dead ball stationary at (10.0, 34.0)
    for t in range(51):
        ball_xy[t] = [10.0, 34.0]
        # GK runs up from (2.0, 34.0) to (9.7, 34.0) during t=30..50
        if t < 30:
            players_xy[t, 0] = [2.0, 34.0]
        else:
            players_xy[t, 0] = [2.0 + (t - 30) * (7.7 / 20.0), 34.0]

    # 51..79: Ball flies into midfield at 22 m/s
    for t in range(51, 80):
        pos = [10.0 + (t - 50) * (22.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [40.0 + 0.3, 34.0]

    # 80..120: H1 controls and carries forward
    for t in range(80, 121):
        pos = [ball_xy[79, 0] + (t - 79) * (2.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["D"] * 51 + ["H"] * 70)
    return "Scenario 09: Restart After Stoppage (Goal Kick Strike)", ball_xy, players_xy, team_of_track, gt


def build_bench_scenario_10_difficult_negative_early_interception(fps: float = 25.0):
    """Scenario 10: Difficult Negative Case: H0 attempts 70-frame pass to H1.

    At t=55 (early in pass), defender A23 steps into the passing lane, cleanly
    intercepts and controls ball, carrying t=55..110.
    GT: H (0..54) -> A (55..110).
    CRUCIAL: Keeping possession with H would be an erroneous pass-continuation!
    """
    n = 111
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..39: H0 carries
    for t in range(40):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 40..54: Pass launched toward H1 far away at x=75.0 (18 m/s)
    base_x = ball_xy[39, 0]
    for t in range(40, 55):
        pos = [base_x + (t - 39) * (18.0 / fps), 30.0]
        ball_xy[t] = pos
        # A23 is moving into the passing lane at x=41.0
        players_xy[t, 23] = [pos[0] + (55 - t) * (1.0 / fps), 30.0]
        players_xy[t, 1] = [75.0 + 0.3, 30.0]

    # 55..110: A23 traps ball cleanly at x=41.0 and carries forward
    base_ix = ball_xy[54, 0]
    for t in range(55, 111):
        pos = [base_ix + (t - 54) * (2.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 1] = [75.0 + 0.3, 30.0]

    gt = np.array(["H"] * 55 + ["A"] * 56)
    return "Scenario 10: Difficult Negative Case (Early Opponent Interception)", ball_xy, players_xy, team_of_track, gt


def get_all_frozen_scenarios(fps: float = 25.0):
    """Return all 10 frozen synthetic scenarios."""
    return [
        build_bench_scenario_01_ground_pass_opponent_near(fps),
        build_bench_scenario_02_aerial_lofted_pass(fps),
        build_bench_scenario_03_in_flight_interception(fps),
        build_bench_scenario_04_first_time_clearance(fps),
        build_bench_scenario_05_headed_clearance_to_gk(fps),
        build_bench_scenario_06_brief_opponent_deflection(fps),
        build_bench_scenario_07_rapid_turnovers(fps),
        build_bench_scenario_08_ball_flight_occlusion(fps),
        build_bench_scenario_09_goal_kick_restart(fps),
        build_bench_scenario_10_difficult_negative_early_interception(fps),
    ]


def compute_boundary_timing_error(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute mean absolute error (in frames) across true boundary transitions."""
    gt_trans = np.where(gt_labels[:-1] != gt_labels[1:])[0]
    pred_trans = np.where(pred_labels[:-1] != pred_labels[1:])[0]

    if len(gt_trans) == 0:
        return 0.0 if len(pred_trans) == 0 else float(len(pred_trans) * 10.0)

    if len(pred_trans) == 0:
        return 50.0  # Heavy penalty for completely missing transitions

    errors = []
    for gt_t in gt_trans:
        nearest_err = np.min(np.abs(pred_trans - gt_t))
        errors.append(nearest_err)
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_evaluation_suite(
    pipeline_fn,
    pipeline_name: str,
    seeds: Tuple[int, ...] = (42, 101, 777),
    fps: float = 25.0,
) -> Dict[str, Any]:
    """Run full evaluation on frozen benchmark suite for a candidate pipeline."""
    scenarios = get_all_frozen_scenarios(fps)
    scenario_results = []
    boundary_errors = []

    for name, b_xy, p_xy, teams, gt in scenarios:
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)

        # Baseline unperturbed prediction
        pred = pipeline_fn(
            ball_xy=b_xy,
            players_xy=p_xy,
            team_of_track=teams,
            non_play_intervals=[],
            fps=fps,
        )
        pred_codes = pm.label_codes(pred)
        scores = pm.spell_scores(gt_codes, pred_codes, runs)
        f1 = scores["mean_f1"]
        b_err = compute_boundary_timing_error(gt, pred)

        # Perturbation testing
        noise_f1s = []
        for sigma in (0.20, 0.50):
            for s in seeds:
                np.random.seed(s)
                b_noisy = b_xy.copy() + np.random.normal(0, sigma, b_xy.shape)
                p_noisy = p_xy.copy() + np.random.normal(0, sigma, p_xy.shape)
                p_pred = pipeline_fn(b_noisy, p_noisy, teams, [], fps=fps)
                p_f1 = pm.spell_scores(gt_codes, pm.label_codes(p_pred), runs)["mean_f1"]
                noise_f1s.append(p_f1)

        drop_f1s = []
        for drop_rate in (0.05, 0.15):
            for s in seeds:
                np.random.seed(s)
                b_drop = b_xy.copy()
                mask = np.random.rand(len(b_drop)) < drop_rate
                b_drop[mask] = np.nan
                d_pred = pipeline_fn(b_drop, p_xy, teams, [], fps=fps)
                d_f1 = pm.spell_scores(gt_codes, pm.label_codes(d_pred), runs)["mean_f1"]
                drop_f1s.append(d_f1)

        scenario_results.append({
            "name": name,
            "spell_f1": f1,
            "boundary_error": b_err,
            "noise_f1_mean": float(np.mean(noise_f1s)),
            "noise_f1_std": float(np.std(noise_f1s)),
            "drop_f1_mean": float(np.mean(drop_f1s)),
            "drop_f1_std": float(np.std(drop_f1s)),
        })
        boundary_errors.append(b_err)

    mean_spell_f1 = float(np.mean([r["spell_f1"] for r in scenario_results]))
    mean_boundary_err = float(np.mean(boundary_errors))
    mean_noise_f1 = float(np.mean([r["noise_f1_mean"] for r in scenario_results]))
    mean_drop_f1 = float(np.mean([r["drop_f1_mean"] for r in scenario_results]))

    return {
        "pipeline": pipeline_name,
        "scenarios": scenario_results,
        "mean_spell_f1": mean_spell_f1,
        "mean_boundary_err": mean_boundary_err,
        "mean_noise_f1": mean_noise_f1,
        "mean_drop_f1": mean_drop_f1,
    }


class TestFlightHandoverBenchmark(unittest.TestCase):
    """Unit test wrapper for automated test runners."""

    def test_benchmark_scenarios_have_valid_shapes(self):
        scenarios = get_all_frozen_scenarios()
        self.assertEqual(len(scenarios), 10)
        for name, b_xy, p_xy, teams, gt in scenarios:
            n = len(gt)
            self.assertEqual(b_xy.shape, (n, 2))
            self.assertEqual(p_xy.shape, (n, 46, 2))
            self.assertEqual(len(teams), 46)
            self.assertEqual(len(gt), n)

    def test_v3_and_v4_baseline_execution(self):
        # Quick execution test on Scenario 1
        name, b_xy, p_xy, teams, gt = build_bench_scenario_01_ground_pass_opponent_near()
        v3_p = run_pipeline_v3(b_xy, p_xy, teams, [])
        v4_p = run_pipeline_v4(b_xy, p_xy, teams, [])
        self.assertEqual(len(v3_p), len(gt))
        self.assertEqual(len(v4_p), len(gt))


if __name__ == "__main__":
    print("=" * 86)
    print("PHANTOM-11: FROZEN INDEPENDENT BENCHMARK EVALUATION (v3 vs v4 BASELINE)")
    print("=" * 86)

    res_v3 = run_evaluation_suite(run_pipeline_v3, "PHANTOM-11 v3 (Published)")
    res_v4 = run_evaluation_suite(run_pipeline_v4, "PHANTOM-11 v4 (Experimental)")

    header = f"{'Scenario Name':<50} | {'v3 spellF1':<11} | {'v4 spellF1':<11} | {'Outcome'}"
    print(header)
    print("-" * 86)

    for s3, s4 in zip(res_v3["scenarios"], res_v4["scenarios"]):
        name = s3["name"]
        f1_3 = s3["spell_f1"]
        f1_4 = s4["spell_f1"]
        outcome = "TIED"
        if f1_4 > f1_3 + 0.001:
            outcome = f"v4 +{f1_4 - f1_3:.3f}"
        elif f1_3 > f1_4 + 0.001:
            outcome = f"v3 +{f1_3 - f1_4:.3f}"
        print(f"{name:<50} | {f1_3:>9.4f}   | {f1_4:>9.4f}   | {outcome}")

    print("-" * 86)
    m3 = res_v3["mean_spell_f1"]
    m4 = res_v4["mean_spell_f1"]
    diff_str = f"v4 +{m4 - m3:.4f}" if m4 >= m3 else f"v3 +{m3 - m4:.4f}"
    print(f"{'OVERALL MEAN spellF1':<50} | {m3:>9.4f}   | {m4:>9.4f}   | {diff_str}")
    print(f"{'MEAN BOUNDARY ERROR (frames)':<50} | {res_v3['mean_boundary_err']:>9.2f}   | {res_v4['mean_boundary_err']:>9.2f}   |")
    print(f"{'MEAN NOISE ROBUSTNESS spellF1':<50} | {res_v3['mean_noise_f1']:>9.4f}   | {res_v4['mean_noise_f1']:>9.4f}   |")
    print(f"{'MEAN DROPOUT ROBUSTNESS spellF1':<50} | {res_v3['mean_drop_f1']:>9.4f}   | {res_v4['mean_drop_f1']:>9.4f}   |")
    print("=" * 86)
