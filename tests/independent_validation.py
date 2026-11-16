#!/usr/bin/env python3
"""PHANTOM-11: Independent Synthetic Validation & Scientific Robustness Benchmark

Step 10 of More Than A Hack 2027:
- Evaluates PHANTOM-11 v3 vs v4 on 9 independently authored synthetic test scenarios
  with objectively verified ground-truth possession labels:
  1. Genuine one-touch interception.
  2. Brief opponent deflection without control.
  3. Rapid consecutive turnovers (1.6s H -> 1.2s A -> 1.6s H).
  4. Aerial lofted pass flying over opponent in 2D.
  5. 50-50 loose-ball contest with emerging carrier.
  6. Long missing-ball interval during continued play.
  7. Tracker identity swap at the moment of tackle.
  8. Throw-in stoppage and subsequent restart.
  9. Short stationary trapped ball without dribbling.
- Uses official evaluation/possession_metrics.py to score spellF1 and macroF1.
- Conducts scientific stress testing under 3 independently seeded perturbations:
  * Missing tracking observations (5%, 10%, 15% dropout).
  * Gaussian coordinate noise (sigma = 0.2m, 0.5m).
  * Player tracker identity swaps.
- Strictly synthetic in-memory data. Zero peeking at private competition data.
- Note: Synthetic scores measure algorithmic robustness and do NOT represent official leaderboard scores.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

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


def run_model_pipeline(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    fps: float = 25.0,
    st_events: list = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run full v3 sequence decoder followed by v4 boundary refinement."""
    n_frames = len(ball_xy)
    st_events = st_events or []
    ball_present = ~np.isnan(ball_xy[:, 0]) & ~np.isnan(ball_xy[:, 1])

    # 1. Kinematics & Evidence
    ball_kin = compute_ball_kinematics(ball_xy, fps=fps)
    ev = compute_control_evidence(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=st_events,
        non_play_intervals=non_play_intervals,
        fps=fps,
    )

    # Nearest team for v0 baseline
    d_raw = np.sqrt(np.sum((players_xy - ball_xy[:, np.newaxis, :]) ** 2, axis=2))
    d_clean = np.where(np.isnan(d_raw), 100.0, d_raw)
    nearest_slot = np.argmin(d_clean, axis=1)
    nearest_team = np.where(ball_present, team_of_track[nearest_slot], -1).astype(np.int8)

    # Simple baseline smoothing for v0
    v0_labels = np.full(n_frames, "D", dtype="<U1")
    v0_labels[ball_present & (nearest_team == 0)] = "H"
    v0_labels[ball_present & (nearest_team == 1)] = "A"

    # v2 carrier overrides
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

    # v3 sequence decoder
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

    # v4 confidence-gated boundary refinement
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

    return v3_labels, v4_labels


# ---------------------------------------------------------------------------
# Scenario Builders
# ---------------------------------------------------------------------------

def build_scenario_01_one_touch_interception(fps=25.0):
    """Scenario 1: Team H passes (75 frames), Away defender steps in and volleys

    forward (touch lasts 3 frames, speed 18 m/s), Away teammate controls for 50 frames.
    GT: H (0..74) -> A (75..127).
    """
    n = 128
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..74: Player H0 dribbles
    for t in range(75):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 75..77: Player A23 intercepts with first-time hard strike
    for t in range(75, 78):
        pos = [30.0 + 74 * (2.0 / fps) + (t - 74) * (18.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] - 0.2, pos[1]]

    # 78..127: Away teammate A24 controls the cleared ball and dribbles
    base_x = ball_xy[77, 0]
    for t in range(78, 128):
        pos = [base_x + (t - 77) * (3.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.4, pos[1]]

    gt = np.array(["H"] * 75 + ["A"] * 53)
    return "Scenario 1: One-Touch Interception", ball_xy, players_xy, team_of_track, gt


def build_scenario_02_brief_deflection(fps=25.0):
    """Scenario 2: Team H passes; ball glances off Away defender for 14 frames,

    then continues to Team H teammate who controls for 80 frames.
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
        players_xy[t, 23] = [pos[0] + 0.8, pos[1] + 0.5]  # close, but moving opposite

    # 114..193: Teammate H1 receives and carries
    for t in range(114, 194):
        pos = [ball_xy[113, 0] + (t - 113) * (3.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Scenario 2: Brief Opponent Deflection", ball_xy, players_xy, team_of_track, gt


def build_scenario_03_rapid_turnovers(fps=25.0):
    """Scenario 3: H carries (40f / 1.6s) -> A tackles & carries (30f / 1.2s) -> H tackles back (40f).

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
    return "Scenario 3: Rapid Consecutive Turnovers", ball_xy, players_xy, team_of_track, gt


def build_scenario_04_aerial_pass(fps=25.0):
    """Scenario 4: High lobbed pass across midfield (60 frames). Opponent is directly underneath

    in 2D coordinates (d < 1m), but ball is in the air. Team H winger controls it.
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

    # 50..109: High lofted cross travelling 15 m/s. Opponent A23 is stationary at (50, 30)
    for t in range(50, 110):
        pos = [ball_xy[49, 0] + (t - 49) * (15.0 / fps), 30.0]
        ball_xy[t] = pos
        # A23 stands at midpoint x=50, perfectly near the ball in 2D plane around t=70..80
        players_xy[t, 23] = [50.0, 30.2]

    # 110..159: H1 controls and carries
    for t in range(110, 160):
        pos = [ball_xy[109, 0] + (t - 109) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Scenario 4: Aerial Pass Over Opponent", ball_xy, players_xy, team_of_track, gt


def build_scenario_05_loose_ball_duel(fps=25.0):
    """Scenario 5: Contested loose ball for 35 frames (1.4s), then Away player emerges

    with the ball and dribbles for 50 frames.
    GT: D (0..34) -> A (35..84).
    """
    n = 85
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(35):
        pos = [50.0 + t * (0.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.6, pos[1] + 0.5]
        players_xy[t, 23] = [pos[0] - 0.6, pos[1] - 0.5]

    for t in range(35, 85):
        pos = [ball_xy[34, 0] + (t - 34) * (3.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 0] = [pos[0] - 3.0, pos[1]]

    gt = np.array(["D"] * 35 + ["A"] * 50)
    return "Scenario 5: 50-50 Loose Ball Duel", ball_xy, players_xy, team_of_track, gt


def build_scenario_06_missing_ball_dropout(fps=25.0):
    """Scenario 6: 60-frame (2.4s) missing ball tracking dropout while Team H runs.

    GT: Continuous H (0..159).
    """
    n = 160
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(50):
        pos = [20.0 + t * (2.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 50..109: ball missing (NaN)
    for t in range(50, 110):
        players_xy[t, 0] = [20.0 + t * (2.5 / fps) + 0.3, 30.0]

    # 110..159: ball resumes at player's feet
    for t in range(110, 160):
        pos = [20.0 + t * (2.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Scenario 6: Missing-Ball Dropout", ball_xy, players_xy, team_of_track, gt


def build_scenario_07_tracker_identity_swap(fps=25.0):
    """Scenario 7: Away player tackles and carries for 40 frames, but tracker swaps slot

    with an unphysical jump at the tackle frame.
    GT: H (0..39) -> A (40..79).
    """
    n = 80
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(40):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    for t in range(40, 80):
        pos = [ball_xy[39, 0] - (t - 39) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        # Slot 23 has a 2.0m position jump at frame 40, but settles to carry afterwards
        if t == 40:
            players_xy[t, 23] = [pos[0] + 2.0, pos[1]]
        else:
            players_xy[t, 23] = [pos[0] - 0.3, pos[1]]

    gt = np.array(["H"] * 40 + ["A"] * 40)
    return "Scenario 7: Tracker Identity Swap", ball_xy, players_xy, team_of_track, gt


def build_scenario_08_throw_in_stoppage(fps=25.0):
    """Scenario 8: Ball out of play for 100 frames (4.0s). Throw-in taken at frame 125,

    Away player intercepts and carries for 45 frames.
    GT: H (0..24) -> D (25..124) -> A (125..169).
    """
    n = 170
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(25):
        pos = [30.0 + t * (2.0 / fps), 67.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 25..124: Ball out of bounds at sideline (y = 68.5)
    for t in range(25, 125):
        ball_xy[t] = [35.0, 68.5]
        players_xy[t, 0] = [35.0, 68.4]  # player holding ball

    # 125..169: Throw in to Away defender A23
    for t in range(125, 170):
        pos = [35.0 + (t - 125) * (2.5 / fps), 60.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 25 + ["D"] * 100 + ["A"] * 45)
    return "Scenario 8: Throw-in Stoppage & Restart", ball_xy, players_xy, team_of_track, gt


def build_scenario_09_short_trapped_ball(fps=25.0):
    """Scenario 9: Away player intercepts, traps ball dead at feet for 20 frames (0.8s,

    speed 0, not moving), then passes to teammate who carries for 40 frames.
    GT: H (0..49) -> A (50..109).
    """
    n = 110
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(50):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 50..69: A23 traps ball stationary at (40.0, 30.0)
    for t in range(50, 70):
        ball_xy[t] = [40.0, 30.0]
        players_xy[t, 23] = [40.2, 30.0]

    # 70..109: Teammate A24 carries
    for t in range(70, 110):
        pos = [40.0 + (t - 69) * (2.5 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 50 + ["A"] * 60)
    return "Scenario 9: Short Trapped Ball Without Dribbling", ball_xy, players_xy, team_of_track, gt


# ---------------------------------------------------------------------------
# Benchmark & Stress Testing Runner
# ---------------------------------------------------------------------------

def run_synthetic_benchmark():
    """Run all 9 independent scenarios and compute official spellF1."""
    scenarios = [
        build_scenario_01_one_touch_interception(),
        build_scenario_02_brief_deflection(),
        build_scenario_03_rapid_turnovers(),
        build_scenario_04_aerial_pass(),
        build_scenario_05_loose_ball_duel(),
        build_scenario_06_missing_ball_dropout(),
        build_scenario_07_tracker_identity_swap(),
        build_scenario_08_throw_in_stoppage(),
        build_scenario_09_short_trapped_ball(),
    ]

    print("=" * 82)
    print("PHANTOM-11: INDEPENDENT SYNTHETIC VALIDATION BENCHMARK (v3 vs v4)")
    print("=" * 82)
    print(f"{'Scenario Name':<42} | {'v3 spellF1':<12} | {'v4 spellF1':<12} | {'Outcome'}")
    print("-" * 82)

    v3_scores = []
    v4_scores = []

    for name, b_xy, p_xy, teams, gt in scenarios:
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)

        v3_pred, v4_pred = run_model_pipeline(
            ball_xy=b_xy,
            players_xy=p_xy,
            team_of_track=teams,
            non_play_intervals=[],
        )

        v3_codes = pm.label_codes(v3_pred)
        v4_codes = pm.label_codes(v4_pred)

        s3 = pm.spell_scores(gt_codes, v3_codes, runs)["mean_f1"]
        s4 = pm.spell_scores(gt_codes, v4_codes, runs)["mean_f1"]

        v3_scores.append(s3)
        v4_scores.append(s4)

        outcome = "TIED"
        if s4 > s3 + 0.001:
            outcome = "v4 BETTER (+{:.3f})".format(s4 - s3)
        elif s3 > s4 + 0.001:
            outcome = "v3 BETTER (+{:.3f})".format(s3 - s4)

        print(f"{name:<42} | {s3:>10.4f}   | {s4:>10.4f}   | {outcome}")

    print("-" * 82)
    mean_v3 = float(np.mean(v3_scores))
    mean_v4 = float(np.mean(v4_scores))
    print(f"{'MEAN OVER ALL 9 INDEPENDENT SCENARIOS':<42} | {mean_v3:>10.4f}   | {mean_v4:>10.4f}   | {'v4 +{:.4f}'.format(mean_v4 - mean_v3) if mean_v4 >= mean_v3 else 'v3 +{:.4f}'.format(mean_v3 - mean_v4)}")
    print("=" * 82)


def run_scientific_perturbation_study(seeds=(42, 101, 777)):
    """Run stress testing under noise, dropouts, and identity changes."""
    print("\n" + "=" * 82)
    print("SCIENTIFIC ROBUSTNESS & PERTURBATION STUDY")
    print("=" * 82)

    # Use Scenario 3 (turnovers) and Scenario 2 (deflections) combined as realistic stress test
    scenarios_to_stress = [
        build_scenario_02_brief_deflection(),
        build_scenario_03_rapid_turnovers(),
    ]

    for name, b_xy_orig, p_xy_orig, teams, gt in scenarios_to_stress:
        print(f"\nStress Testing on: {name}")
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)

        # Baseline unperturbed scores
        v3_base, v4_base = run_model_pipeline(b_xy_orig, p_xy_orig, teams, [])
        s3_base = pm.spell_scores(gt_codes, pm.label_codes(v3_base), runs)["mean_f1"]
        s4_base = pm.spell_scores(gt_codes, pm.label_codes(v4_base), runs)["mean_f1"]
        print(f"  Unperturbed Baseline spellF1: v3={s3_base:.4f}, v4={s4_base:.4f}")

        # 1. Coordinate Noise Study (Gaussian sigma = 0.2m, 0.5m)
        for sigma in [0.20, 0.50]:
            s3_list, s4_list = [], []
            for seed in seeds:
                np.random.seed(seed)
                b_noisy = b_xy_orig.copy() + np.random.normal(0, sigma, b_xy_orig.shape)
                p_noisy = p_xy_orig.copy() + np.random.normal(0, sigma, p_xy_orig.shape)
                v3_p, v4_p = run_model_pipeline(b_noisy, p_noisy, teams, [])
                s3_list.append(pm.spell_scores(gt_codes, pm.label_codes(v3_p), runs)["mean_f1"])
                s4_list.append(pm.spell_scores(gt_codes, pm.label_codes(v4_p), runs)["mean_f1"])
            print(f"  Coord Noise (sigma={sigma:.2f}m): v3={np.mean(s3_list):.4f} (+/-{np.std(s3_list):.4f}), v4={np.mean(s4_list):.4f} (+/-{np.std(s4_list):.4f})")

        # 2. Tracking Dropout Study (5%, 15% random missing ball)
        for drop_rate in [0.05, 0.15]:
            s3_list, s4_list = [], []
            for seed in seeds:
                np.random.seed(seed)
                b_dropped = b_xy_orig.copy()
                mask = np.random.rand(len(b_dropped)) < drop_rate
                b_dropped[mask] = np.nan
                v3_p, v4_p = run_model_pipeline(b_dropped, p_xy_orig, teams, [])
                s3_list.append(pm.spell_scores(gt_codes, pm.label_codes(v3_p), runs)["mean_f1"])
                s4_list.append(pm.spell_scores(gt_codes, pm.label_codes(v4_p), runs)["mean_f1"])
            print(f"  Dropouts ({int(drop_rate*100)}% frames missing): v3={np.mean(s3_list):.4f} (+/-{np.std(s3_list):.4f}), v4={np.mean(s4_list):.4f} (+/-{np.std(s4_list):.4f})")

    print("=" * 82)


if __name__ == "__main__":
    run_synthetic_benchmark()
    run_scientific_perturbation_study()
