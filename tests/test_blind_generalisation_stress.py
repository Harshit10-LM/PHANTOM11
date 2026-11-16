#!/usr/bin/env python3
"""PHANTOM-11: Blind Generalisation Stress Test Suite (Step 14)

Frozen blind evaluation suite designed to probe complex physical mechanisms:
1. Blind-01: Curved / Swerving Ball Trajectory (cross-field bending pass).
2. Blind-02: Goalkeeper Long Drop-Kick Punt to Distant Winger (65m aerial flight).
3. Blind-03: Multi-Player Penalty Box Scramble (rapid pinball touches in dense traffic).
4. Blind-04: Slide Tackle into Touchline (turnover followed by touchline stoppage).
5. Blind-05: Overhit Ground Pass Rolling Past Defender to Goalkeeper (loose ball collection).
6. Blind-06: Back-Pass Under High Press with First-Time Clearance (consecutive first-time touches).

Deterministic, SI units (meters, seconds, m/s), frozen ground truth.
"""

import unittest
import numpy as np
import sys
import os
from typing import List, Tuple

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


def build_blind_01_curved_swerving_pass(fps: float = 25.0):
    """Blind-01: Swerving / curved cross-field pass curving around defenders.
    
    H0 strikes a curling pass from (20, 20) to H1 at (65, 45).
    Ball follows a curved arc y(t) = y_linear + 6.0 * sin(pi * t / T).
    Defender A23 stands at the linear midpoint (42.5, 32.5), but ball curves around him.
    GT: Continuous H (0..119).
    """
    n = 120
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..34: H0 carries
    for t in range(35):
        pos = [20.0 + t * (1.5 / fps), 20.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 35..84: Curling pass (50f = 2.0s)
    start_pos = np.array(ball_xy[34])
    dest_pos = np.array([65.0, 45.0])
    pass_dur = 50
    for t in range(35, 85):
        frac = (t - 35) / float(pass_dur)
        lin_pos = start_pos + frac * (dest_pos - start_pos)
        # Add lateral curve
        curve_offset = 6.0 * np.sin(np.pi * frac)
        pos = [lin_pos[0] - curve_offset * 0.5, lin_pos[1] + curve_offset]
        ball_xy[t] = pos
        # Defender stands near the linear chord
        players_xy[t, 23] = [42.5, 32.5]
        # Receiver waits at destination
        players_xy[t, 1] = [dest_pos[0] + 0.3, dest_pos[1]]

    # 85..119: H1 controls and carries
    for t in range(85, 120):
        pos = [dest_pos[0] + (t - 84) * (2.0 / fps), dest_pos[1]]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "Blind-01: Curved / Swerving Ball Trajectory", ball_xy, players_xy, team_of_track, gt


def build_blind_02_gk_long_dropkick_punt(fps: float = 25.0):
    """Blind-02: Goalkeeper long drop-kick punt (65m flight, 22 m/s).
    
    A0 (GK) holds ball at penalty area (15, 34), punts downfield to A1 winger at (80, 55).
    Ball takes 70 frames (2.8s) flying over congested midfield (3 Home players at x=45, 50, 55).
    GT: Continuous A (0..139).
    """
    n = 140
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..39: A0 GK holds and takes run-up
    for t in range(40):
        pos = [15.0 + t * (1.2 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.2, pos[1]] # A0 is slot 23

    # 40..109: Massive 65m punt to (80, 55)
    start_pos = np.array(ball_xy[39])
    dest_pos = np.array([80.0, 55.0])
    punt_dur = 70
    for t in range(40, 110):
        frac = (t - 40) / float(punt_dur)
        pos = start_pos + frac * (dest_pos - start_pos)
        ball_xy[t] = pos
        # Midfield defenders in flight path in 2D
        players_xy[t, 0] = [45.0, 42.0]
        players_xy[t, 1] = [52.0, 45.0]
        players_xy[t, 2] = [60.0, 48.0]
        # Winger A24 at destination
        players_xy[t, 24] = [dest_pos[0] + 0.3, dest_pos[1]]

    # 110..139: A24 controls and drives forward
    for t in range(110, 140):
        pos = [dest_pos[0] + (t - 109) * (2.5 / fps), dest_pos[1]]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["A"] * n)
    return "Blind-02: Goalkeeper Long Drop-Kick Punt to Distant Winger", ball_xy, players_xy, team_of_track, gt


def build_blind_03_penalty_box_scramble(fps: float = 25.0):
    """Blind-03: Multi-player scramble in penalty box.
    
    H0 shoots at goal (0..24). A0 (defender) blocks shot at t=25, ball rebounds to H1.
    H1 controls and strikes at t=45..69. A23 (GK) parries wide at t=70..100.
    GT: H (0..24) -> A (25..44) -> H (45..69) -> A (70..100).
    """
    n = 101
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: H0 carries and strikes
    for t in range(25):
        pos = [90.0 + t * (2.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 23] = [95.0, 34.0] # Defender

    # 25..44: Blocked rebound controlled by A0 defender
    for t in range(25, 45):
        pos = [95.0 - (t - 24) * (1.2 / fps), 36.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.2, pos[1]]
        players_xy[t, 1] = [92.0, 38.0] # Attacker H1

    # 45..69: Attacker H1 tackles/collects and shoots
    for t in range(45, 70):
        pos = [92.0 + (t - 44) * (2.5 / fps), 35.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 24] = [102.0, 34.0] # GK A24

    # 70..100: GK parries and secures
    for t in range(70, 101):
        pos = [102.0 + (t - 69) * (0.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.2, pos[1]]

    gt = np.array(["H"] * 25 + ["A"] * 20 + ["H"] * 25 + ["A"] * 31)
    return "Blind-03: Multi-Player Penalty Box Scramble", ball_xy, players_xy, team_of_track, gt


def build_blind_04_slide_tackle_to_touchline(fps: float = 25.0):
    """Blind-04: Slide tackle into touchline with clean out of bounds.
    
    H0 dribbles down touchline (0..44). Away defender A23 slides in and pokes ball at t=45.
    Ball slides toward touchline and crosses out at t=65. Stoppage D (65..100).
    GT: H (0..44) -> A (45..64) -> D (65..100).
    """
    n = 101
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..44: H0 dribbles near touchline y=66.0
    for t in range(45):
        pos = [60.0 + t * (2.5 / fps), 65.5]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 23] = [pos[0] + 1.5, pos[1] + 1.0]

    # 45..64: A23 slide-tackles ball toward touchline (10 m/s)
    for t in range(45, 65):
        pos = [ball_xy[44, 0] + (t - 44) * (2.0 / fps), 65.5 + (t - 44) * (3.0 / fps)]
        ball_xy[t] = pos
        players_xy[t, 23] = [ball_xy[44, 0] + 0.5, 66.0]

    # 65..100: Ball goes out of bounds (y > 68.0), dead ball D
    for t in range(65, 101):
        # Ball out of bounds or missing
        ball_xy[t] = [ball_xy[64, 0] + 0.5, 69.5] # Out of bounds
        players_xy[t, 23] = [ball_xy[64, 0], 67.0]

    gt = np.array(["H"] * 45 + ["A"] * 20 + ["D"] * 36)
    return "Blind-04: Slide Tackle into Touchline", ball_xy, players_xy, team_of_track, gt


def build_blind_05_overhit_pass_to_goalkeeper(fps: float = 25.0):
    """Blind-05: Overhit pass rolling past opponent into goalkeeper arms.
    
    H0 overhits through-ball at t=25 (14 m/s). Ball rolls past defender A23 at t=45 (d=1.8m),
    decelerates to 2.2 m/s, and Home GK H2 gathers cleanly at t=70..110.
    GT: Continuous H (0..110).
    """
    n = 111
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: H0 carries
    for t in range(25):
        pos = [50.0 - t * (1.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] - 0.3, pos[1]]

    # 25..69: Ball rolls back toward goalkeeper at (15, 34)
    # Passes defender A23 standing at (35, 36) (d~2.0m)
    start_pos = np.array(ball_xy[24])
    dest_pos = np.array([16.0, 34.0])
    roll_dur = 45
    for t in range(25, 70):
        frac = (t - 25) / float(roll_dur)
        pos = start_pos + frac * (dest_pos - start_pos)
        ball_xy[t] = pos
        players_xy[t, 23] = [35.0, 36.0]
        players_xy[t, 2] = [15.0, 34.0] # GK H2

    # 70..110: GK H2 secures and controls
    for t in range(70, 111):
        pos = [16.0 + (t - 70) * (0.8 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 2] = [pos[0] + 0.2, pos[1]]

    gt = np.array(["H"] * n)
    return "Blind-05: Overhit Pass Rolling Past Opponent to Goalkeeper", ball_xy, players_xy, team_of_track, gt


def build_blind_06_backpass_under_press_clearance(fps: float = 25.0):
    """Blind-06: Back-pass under press followed by first-time clearance.
    
    H0 (defender) under pressure from A23 plays back-pass to H1 (GK) at t=30.
    GK clears first-time downfield at t=55..100.
    GT: Continuous H (0..100).
    """
    n = 101
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..29: H0 shields ball under press from A23
    for t in range(30):
        pos = [35.0 - t * (1.0 / fps), 25.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]
        players_xy[t, 23] = [pos[0] + 1.2, pos[1]] # A23 pressing tight

    # 30..54: Firm back-pass to GK H1 at (15, 30) (15 m/s)
    start_pos = np.array(ball_xy[29])
    dest_pos = np.array([15.0, 30.0])
    pass_dur = 25
    for t in range(30, 55):
        frac = (t - 30) / float(pass_dur)
        pos = start_pos + frac * (dest_pos - start_pos)
        ball_xy[t] = pos
        players_xy[t, 23] = [32.0, 25.0] # pressing forward chases
        players_xy[t, 1] = [14.8, 30.0] # GK waiting

    # 55..100: GK first-time clearance downfield at 24 m/s
    for t in range(55, 101):
        pos = [15.0 + (t - 54) * (24.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [15.0, 30.0]

    gt = np.array(["H"] * n)
    return "Blind-06: Back-Pass Under Press with First-Time Clearance", ball_xy, players_xy, team_of_track, gt


def get_all_blind_scenarios():
    """Return all 6 blind stress scenarios."""
    builders = [
        build_blind_01_curved_swerving_pass,
        build_blind_02_gk_long_dropkick_punt,
        build_blind_03_penalty_box_scramble,
        build_blind_04_slide_tackle_to_touchline,
        build_blind_05_overhit_pass_to_goalkeeper,
        build_blind_06_backpass_under_press_clearance,
    ]
    return [b() for b in builders]


class TestBlindGeneralisationStress(unittest.TestCase):
    """Test suite comparing v5 and v6 across the blind stress scenarios."""

    def test_blind_stress_suite(self):
        scenarios = get_all_blind_scenarios()
        print("\n" + "=" * 105)
        print("BLIND GENERALISATION STRESS TEST SUITE (Step 14)")
        print("=" * 105)
        print(f"{'Scenario':<55} | {'v5 F1':<8} | {'v6 F1':<8} | {'Outcome'}")
        print("-" * 105)

        v5_scores = []
        v6_scores = []
        for name, b, p, t, gt in scenarios:
            runs = [(0, len(gt))]
            gt_codes = pm.label_codes(gt)
            p5 = run_pipeline_v5(b, p, t, [])
            p6 = run_pipeline_v6(b, p, t, [])

            s5 = pm.spell_scores(gt_codes, pm.label_codes(p5), runs)["mean_f1"]
            s6 = pm.spell_scores(gt_codes, pm.label_codes(p6), runs)["mean_f1"]
            v5_scores.append(s5)
            v6_scores.append(s6)

            diff = s6 - s5
            if diff > 0.001:
                outcome = f"v6 WINS (+{diff:.4f})"
            elif diff < -0.001:
                outcome = f"v5 WINS ({diff:.4f})"
            else:
                outcome = "TIED"

            print(f"{name:<55} | {s5:<8.4f} | {s6:<8.4f} | {outcome}")
            self.assertGreaterEqual(s6, 0.60, f"v6 scored unexpectedly low on {name}")

        mean_v5 = np.mean(v5_scores)
        mean_v6 = np.mean(v6_scores)
        print("-" * 105)
        print(f"{'MEAN SPELL F1 ACROSS BLIND STRESS SUITE':<55} | {mean_v5:<8.4f} | {mean_v6:<8.4f} |")
        print("=" * 105)
        self.assertGreaterEqual(mean_v6, 0.90)


if __name__ == "__main__":
    unittest.main()
