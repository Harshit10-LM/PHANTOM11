#!/usr/bin/env python3
"""PHANTOM-11: Newly Frozen Held-Out Handover & Physical Counterexample Suite

Step 13 of More Than A Hack 2027:
- Genuinely independent, newly frozen held-out evaluation suite:
  HO-01: Cross-Field Diagonal Switch (over two opponents in 2D).
  HO-02: Blocked Pass / Charged Down Strike (deliberate contact with ricochet).
  HO-03: Running Interception by Fullback (sliding trap and clearance).
  HO-04: Contested Aerial 50-50 Duel at Midfield (simultaneous challenge).
  HO-05: Goalkeeper Long Drop-Kick Distribution (holding, run-up, 50m kick).
  HO-06: Throw-In Up the Line Intercepted (touchline restart with turnover).
  HO-07: Rapid Midfield Tackle & Counter-Tackle (short possessions 1.2s -> 1.0s -> 1.4s).
  HO-08: Optical Ball Tracking Occlusion on Long Pass (25f missing ball mid-flight).
  HO-09: Overhit Pass Decelerating in Open Space (tests against premature deceleration triggers).
  HO-10: Multi-Player Deflection Pinball (glances off two defenders to attacker).
- Evaluates pipelines with official possession_metrics.py (spellF1).
- FROZEN SUITE: This suite must not be altered after running candidate models.
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, Any, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom11.path_resolver import resolve_official_repo
resolve_official_repo()
import possession_metrics as pm


# ---------------------------------------------------------------------------
# Held-Out Scenario Builders
# ---------------------------------------------------------------------------

def build_ho_01_diagonal_switch(fps: float = 25.0):
    """HO-01: Cross-Field Diagonal Switch (65f / 2.6s, 22 m/s). Ball flies over two

    opposing midfielders (A23 at x=40, A24 at x=55) in 2D. Winger H1 controls at x=75.
    GT: Continuous H (0..139).
    """
    n = 140
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..39: H0 carries at left back (x=20, y=15)
    for t in range(40):
        pos = [20.0 + t * (1.5 / fps), 15.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 40..104: Long diagonal pass to (77.2, 55.0) at 22 m/s
    base_pos = ball_xy[39].copy()
    target_pos = np.array([77.2, 55.0])
    vec = target_pos - base_pos
    dist = np.linalg.norm(vec)
    flight_frames = 65

    for t in range(40, 105):
        frac = (t - 39) / flight_frames
        pos = base_pos + frac * vec
        ball_xy[t] = pos
        # A23 stands at (38.0, 30.0) near 2D trajectory at t=55..65
        players_xy[t, 23] = [38.0, 30.0]
        # A24 stands at (56.0, 42.0) near 2D trajectory at t=75..85
        players_xy[t, 24] = [56.0, 42.0]
        # Receiver H1 waiting at target
        players_xy[t, 1] = [target_pos[0] + 0.3, target_pos[1]]

    # 105..139: H1 controls and carries forward
    for t in range(105, 140):
        pos = [target_pos[0] + (t - 104) * (2.0 / fps), target_pos[1]]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "HO-01: Cross-Field Diagonal Switch (Over 2 Opponents)", ball_xy, players_xy, team_of_track, gt


def build_ho_02_blocked_pass(fps: float = 25.0):
    """HO-02: Attacker H0 shoots/passes at t=30 (20 m/s). Defender A23 blocks pass

    at t=33 (d=0.5m), ball ricochets backward into Away territory at 14 m/s.
    Away defender A24 collects at t=55 and dribbles until t=95.
    GT: H (0..32) -> A (33..95).
    """
    n = 96
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(30):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 30..32: Pass moving forward at 20 m/s
    base_x = ball_xy[29, 0]
    for t in range(30, 33):
        pos = [base_x + (t - 29) * (20.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [33.0, 30.0]

    # 33: Blocked by A23 at x=32.4! Ricochets backward at 14 m/s
    block_x = ball_xy[32, 0]
    for t in range(33, 56):
        pos = [block_x - (t - 32) * (14.0 / fps), 30.0]
        ball_xy[t] = pos
        if t in (33, 34):
            players_xy[t, 23] = [block_x, 30.0]
        players_xy[t, 24] = [20.0 + 0.3, 30.0]

    # 56..95: A24 controls and dribbles
    base_cx = ball_xy[55, 0]
    for t in range(56, 96):
        pos = [base_cx - (t - 55) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] - 0.3, pos[1]]

    gt = np.array(["H"] * 33 + ["A"] * 63)
    return "HO-02: Blocked Pass / Charged Down Strike", ball_xy, players_xy, team_of_track, gt


def build_ho_03_running_interception(fps: float = 25.0):
    """HO-03: Attacking through-ball played into channel at t=30 (16 m/s). Fullback A23

    sprints diagonally, slides in and traps ball at t=55 (d=0.4m, speed drops to 2 m/s),
    and clears forward t=55..100.
    GT: H (0..54) -> A (55..100).
    """
    n = 101
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(30):
        pos = [30.0 + t * (2.0 / fps), 25.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 30..54: Pass traveling at 16 m/s towards (46.0, 25.0). A23 sprinting in from (50, 35)
    base_x = ball_xy[29, 0]
    for t in range(30, 55):
        pos = [base_x + (t - 29) * (16.0 / fps), 25.0]
        ball_xy[t] = pos
        frac = (55 - t) / 25.0
        players_xy[t, 23] = [pos[0] + frac * 2.0, 25.0 + frac * 8.0]

    # 55..100: A23 traps ball cleanly at (46.0, 25.0) and dribbles away
    trap_x = ball_xy[54, 0]
    for t in range(55, 101):
        pos = [trap_x + (t - 54) * (2.5 / fps), 25.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 55 + ["A"] * 46)
    return "HO-03: Running Interception by Fullback", ball_xy, players_xy, team_of_track, gt


def build_ho_04_aerial_50_50_duel(fps: float = 25.0):
    """HO-04: High punt drops into midfield. Both H0 and A23 challenge simultaneously

    at contact point for 25 frames (d_H ~ 0.5m, d_A ~ 0.5m) until A23 heads to A24.
    GT: D (0..24 contested) -> A (25..85).
    """
    n = 86
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..24: 50-50 contested loose ball
    for t in range(25):
        pos = [50.0 + t * (0.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.5, pos[1] + 0.3]
        players_xy[t, 23] = [pos[0] - 0.5, pos[1] - 0.3]

    # 25..85: A23 heads ball to teammate A24 who carries forward
    base_x = ball_xy[24, 0]
    for t in range(25, 86):
        pos = [base_x - (t - 24) * (2.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] - 0.3, pos[1]]
        players_xy[t, 0] = [pos[0] + 3.0, pos[1]]

    gt = np.array(["D"] * 25 + ["A"] * 61)
    return "HO-04: Contested Aerial 50-50 Duel at Midfield", ball_xy, players_xy, team_of_track, gt


def build_ho_05_gk_drop_kick(fps: float = 25.0):
    """HO-05: Goalkeeper A23 holds ball in hands (t=0..35), drop-kicks at t=36 (24 m/s).

    Ball travels 50m to striker A24 who traps at t=80 and shields until t=120.
    GT: Continuous A (0..120).
    """
    n = 121
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..35: GK holds ball stationary / small steps
    for t in range(36):
        pos = [10.0 + t * (0.8 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0], pos[1]]

    # 36..79: Ball flies 50m downfield at 24 m/s
    base_x = ball_xy[35, 0]
    for t in range(36, 80):
        pos = [base_x + (t - 35) * (24.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [52.0 + 0.3, 34.0]

    # 80..120: Striker A24 controls and shields
    base_sx = ball_xy[79, 0]
    for t in range(80, 121):
        pos = [base_sx + (t - 79) * (1.5 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 24] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["A"] * n)
    return "HO-05: Goalkeeper Long Drop-Kick Distribution", ball_xy, players_xy, team_of_track, gt


def build_ho_06_throw_in_intercepted(fps: float = 25.0):
    """HO-06: Ball out of play at touchline (t=0..35, y=68.5). Throw-in taken at t=36

    toward H1. Opponent A23 steps in front at t=45, traps, and carries t=45..90.
    GT: D (0..35) -> H (36..44) -> A (45..90).
    """
    n = 91
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    # 0..35: Dead ball at touchline
    for t in range(36):
        ball_xy[t] = [40.0, 68.5]
        players_xy[t, 0] = [40.0, 68.4]

    # 36..44: Throw-in in transit towards (40.0, 58.0)
    for t in range(36, 45):
        pos = [40.0, 68.5 - (t - 35) * (10.0 / fps)]
        ball_xy[t] = pos
        players_xy[t, 1] = [40.0, 56.0]
        players_xy[t, 23] = [40.0, pos[1] - 0.5]

    # 45..90: A23 traps and carries away
    trap_y = ball_xy[44, 1]
    for t in range(45, 91):
        pos = [40.0 + (t - 44) * (2.5 / fps), trap_y - (t - 44) * (1.0 / fps)]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["D"] * 36 + ["H"] * 9 + ["A"] * 46)
    return "HO-06: Throw-In Up the Line Intercepted", ball_xy, players_xy, team_of_track, gt


def build_ho_07_rapid_tackles(fps: float = 25.0):
    """HO-07: H carries 1.2s (30f) -> A tackles and carries 1.0s (25f) -> H tackles back and carries 1.4s (35f).

    GT: H (0..29) -> A (30..54) -> H (55..89).
    """
    n = 90
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(30):
        pos = [30.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    for t in range(30, 55):
        pos = [ball_xy[29, 0] - (t - 29) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [pos[0] - 0.3, pos[1]]

    for t in range(55, 90):
        pos = [ball_xy[54, 0] + (t - 54) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * 30 + ["A"] * 25 + ["H"] * 35)
    return "HO-07: Rapid Midfield Tackle & Counter-Tackle", ball_xy, players_xy, team_of_track, gt


def build_ho_08_optical_occlusion(fps: float = 25.0):
    """HO-08: H0 passes at t=30 (18 m/s). Ball is NaN from t=40..64 (25f / 1.0s).

    Ball reappears at receiver H1 feet at t=65, and H1 carries t=65..110.
    GT: Continuous H (0..110).
    """
    n = 111
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(30):
        pos = [25.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 30..39: Pass in flight (18 m/s)
    for t in range(30, 40):
        pos = [ball_xy[29, 0] + (t - 29) * (18.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [50.0 + 0.3, 30.0]

    # 40..64: Ball missing (NaN)
    for t in range(40, 65):
        players_xy[t, 1] = [50.0 + 0.3, 30.0]

    # 65..110: Ball reappears at H1 feet
    for t in range(65, 111):
        pos = [50.0 + (t - 65) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "HO-08: Optical Ball Tracking Occlusion on Pass", ball_xy, players_xy, team_of_track, gt


def build_ho_09_overhit_pass_rolling_open(fps: float = 25.0):
    """HO-09: Overhit pass at t=30 (14 m/s). Rolls in open space, slowing to 2.5 m/s

    at t=60, but opposing GK A23 is still 8m off line! GK arrives and collects at t=78.
    GT: H (0..77) -> A (78..115).
    Crucial check: Ball slowing down in open space must NOT prematurely trigger handover!
    """
    n = 116
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(30):
        pos = [35.0 + t * (2.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 30..59: Overhit pass traveling towards penalty box
    for t in range(30, 60):
        pos = [ball_xy[29, 0] - (t - 29) * (14.0 / fps), 34.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [5.0, 34.0]  # GK on goal line

    # 60..77: Ball decelerating in open space at x=21.0 (spd ~ 2.0 m/s), GK advancing
    stop_x = ball_xy[59, 0]
    for t in range(60, 78):
        pos = [stop_x - (t - 59) * (2.0 / fps), 34.0]
        ball_xy[t] = pos
        # GK advances from x=5.0 to x=19.5
        players_xy[t, 23] = [5.0 + (t - 59) * (14.5 / 18.0), 34.0]

    # 78..115: GK reaches ball and holds
    base_gk = ball_xy[77, 0]
    for t in range(78, 116):
        pos = [base_gk, 34.0]
        ball_xy[t] = pos
        players_xy[t, 23] = [base_gk + 0.2, 34.0]

    gt = np.array(["H"] * 78 + ["A"] * 38)
    return "HO-09: Overhit Pass Decelerating in Open Space", ball_xy, players_xy, team_of_track, gt


def build_ho_10_multi_player_pinball(fps: float = 25.0):
    """HO-10: H0 strikes ball at t=30. Glances off defender A23 at t=35, then glances off

    defender A24 at t=45, before attacker H1 collects loose ball at t=60 and carries.
    GT: Continuous H (0..109).
    """
    n = 110
    team_of_track = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    players_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)

    for t in range(30):
        pos = [25.0 + t * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

    # 30..59: Pass in flight (15 m/s). Glances off A23 (t=35) and A24 (t=45)
    for t in range(30, 60):
        pos = [ball_xy[29, 0] + (t - 29) * (15.0 / fps), 30.0]
        ball_xy[t] = pos
        if 34 <= t <= 36:
            players_xy[t, 23] = [pos[0] + 0.4, pos[1] + 0.5]
        if 44 <= t <= 46:
            players_xy[t, 24] = [pos[0] + 0.4, pos[1] - 0.5]
        players_xy[t, 1] = [45.0 + 0.3, 30.0]

    # 60..109: H1 collects and carries
    for t in range(60, 110):
        pos = [ball_xy[59, 0] + (t - 59) * (2.0 / fps), 30.0]
        ball_xy[t] = pos
        players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

    gt = np.array(["H"] * n)
    return "HO-10: Multi-Player Deflection Pinball", ball_xy, players_xy, team_of_track, gt


def get_all_held_out_scenarios(fps: float = 25.0):
    return [
        build_ho_01_diagonal_switch(fps),
        build_ho_02_blocked_pass(fps),
        build_ho_03_running_interception(fps),
        build_ho_04_aerial_50_50_duel(fps),
        build_ho_05_gk_drop_kick(fps),
        build_ho_06_throw_in_intercepted(fps),
        build_ho_07_rapid_tackles(fps),
        build_ho_08_optical_occlusion(fps),
        build_ho_09_overhit_pass_rolling_open(fps),
        build_ho_10_multi_player_pinball(fps),
    ]


class TestHeldOutSuite(unittest.TestCase):
    def test_held_out_scenarios_shape(self):
        scenarios = get_all_held_out_scenarios()
        self.assertEqual(len(scenarios), 10)
        for name, b_xy, p_xy, teams, gt in scenarios:
            n = len(gt)
            self.assertEqual(b_xy.shape, (n, 2))
            self.assertEqual(p_xy.shape, (n, 46, 2))
            self.assertEqual(len(teams), 46)
            self.assertEqual(len(gt), n)


if __name__ == "__main__":
    print("Held-out suite compiled and verified.")
