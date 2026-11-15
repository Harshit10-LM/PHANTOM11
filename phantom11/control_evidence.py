#!/usr/bin/env python3
"""PHANTOM-11: Team-Control Evidence Integration Module

Step 6 of More Than A Hack 2027:
- Combines kinematics, player co-movement, team geometry, and tactical tags into
  transparent, interpretable team-control evidence scores:
  * score_h: Home evidence in [0.0, 1.0]
  * score_a: Away evidence in [0.0, 1.0]
  * reliability: Confidence / certainty metric in [0.0, 1.0]
  * reason_code: Primary evidence justification string
- Strictly follows physical principles. No uncalibrated pseudo-probabilities.
- Explicitly flags low-confidence, contested duels, loose balls, and tracking dropouts.
"""

import os
import sys
import numpy as np

# Ensure repository root and starter directory are on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
STARTER_DIR = os.path.join(REPO_ROOT, "starter")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)

import games
try:
    from phantom11.movement_features import (
        compute_ball_kinematics,
        compute_player_kinematics,
        compute_relative_motion,
        detect_ball_carrier,
    )
    from phantom11.team_geometry import compute_team_geometry
    from phantom11.tactical_tags import parse_smart_tagging_timeline
except ImportError:
    from movement_features import (
        compute_ball_kinematics,
        compute_player_kinematics,
        compute_relative_motion,
        detect_ball_carrier,
    )
    from team_geometry import compute_team_geometry
    from tactical_tags import parse_smart_tagging_timeline


# ---------------------------------------------------------------------------
# Transparent Scoring Parameters (Documented Physical Meanings)
# ---------------------------------------------------------------------------
# Weight for confirmed active ball carrier (close distance, aligned velocity)
W_CARRIER = 0.55

# Weight for clear spatial proximity dominance (d_opp - d_own >= 1.5m)
W_PROXIMITY = 0.25

# Weight for local numerical overload / support (more teammates within 3m)
W_OVERLOAD = 0.10

# Weight for soft tactical prior from Smart Tagging attacking/defending phases
W_TACTICAL = 0.10


def compute_control_evidence(
    ball_xy,
    players_xy,
    team_of_track,
    smart_tagging_events,
    non_play_intervals=None,
    fps=25.0,
):
    """Compute frame-by-frame team-control evidence scores and diagnostic reason codes.

    Parameters
    ----------
    ball_xy : np.ndarray of shape (N, 2), float32
        Ball coordinates in metres (NaN when missing).
    players_xy : np.ndarray of shape (N, S, 2), float32
        Player coordinates in metres. S is 46 track slots.
    team_of_track : np.ndarray of shape (S,), int8
        0 for Home, 1 for Away.
    smart_tagging_events : list of dict
        Events from starter/load_smart_tagging.py.
    non_play_intervals : list of (start, end)
        Match boundary ranges (pre-match, halftime, post-match).
    fps : float
        Frames per second (25.0).

    Returns
    -------
    dict with:
        'score_h': (N,) float32 in [0.0, 1.0]
        'score_a': (N,) float32 in [0.0, 1.0]
        'reliability': (N,) float32 in [0.0, 1.0]
        'reason_code': (N,) object array of str
        'is_carrier_h': (N,) bool
        'is_carrier_a': (N,) bool
        'control_state': (N,) int8
    """
    n_frames = len(ball_xy)
    non_play_intervals = non_play_intervals or []

    # 1. Kinematics & Relative Motion
    ball_kin = compute_ball_kinematics(ball_xy, fps=fps)
    player_kin = compute_player_kinematics(players_xy, fps=fps)

    rel_motion = compute_relative_motion(
        ball_xy=ball_xy,
        ball_vel=ball_kin["velocity"],
        ball_speed=ball_kin["speed"],
        players_xy=players_xy,
        player_vel=player_kin["velocities"],
        player_speed=player_kin["speeds"],
    )

    # 2. Ball Carrier Detection
    carrier_slot, is_carrier, carrier_conf = detect_ball_carrier(
        distances=rel_motion["distances"],
        rel_speeds=rel_motion["rel_speeds"],
        alignments=rel_motion["alignments"],
        ball_speed=ball_kin["speed"],
        player_speed=player_kin["speeds"],
        player_jump_flag=player_kin["player_jump_flag"],
        ball_jump_flag=ball_kin["ball_jump_flag"],
    )

    # Map carrier slot to team
    carrier_team = np.full(n_frames, -1, dtype=np.int8)
    has_carrier = (carrier_slot >= 0)
    carrier_team[has_carrier] = team_of_track[carrier_slot[has_carrier]]

    is_carrier_h = (carrier_team == 0)
    is_carrier_a = (carrier_team == 1)

    # 3. Team Geometry & Proximity
    geom = compute_team_geometry(
        distances=rel_motion["distances"],
        team_of_track=team_of_track,
        ball_xy=ball_xy,
        ball_present=ball_kin["ball_present"],
        players_xy=players_xy,
        player_vel=player_kin["velocities"],
    )

    # 4. Smart Tagging Soft Prior
    tact = parse_smart_tagging_timeline(smart_tagging_events, n_frames)

    # 5. Composite Score Calculation
    score_h = np.zeros(n_frames, dtype=np.float32)
    score_a = np.zeros(n_frames, dtype=np.float32)
    reliability = np.zeros(n_frames, dtype=np.float32)
    reason_code = np.full(n_frames, "UNRESOLVED", dtype=object)

    # Signal 1: Carrier evidence
    score_h[is_carrier_h] += W_CARRIER * carrier_conf[is_carrier_h]
    score_a[is_carrier_a] += W_CARRIER * carrier_conf[is_carrier_a]

    # Signal 2: Spatial Proximity Dominance (clear closer player)
    d_h1 = geom["d_h1"]
    d_a1 = geom["d_a1"]
    margin_d = geom["margin_d"]  # d_a1 - d_h1
    ball_present = ball_kin["ball_present"]

    # Home proximity advantage
    h_prox = ball_present & (margin_d >= 1.5) & (d_h1 <= 2.5)
    score_h[h_prox] += W_PROXIMITY

    # Away proximity advantage
    a_prox = ball_present & (margin_d <= -1.5) & (d_a1 <= 2.5)
    score_a[a_prox] += W_PROXIMITY

    # Signal 3: Local Overload
    h_overload = ball_present & (geom["n_home_3m"] > geom["n_away_3m"]) & (d_h1 <= 3.0)
    score_h[h_overload] += W_OVERLOAD

    a_overload = ball_present & (geom["n_away_3m"] > geom["n_home_3m"]) & (d_a1 <= 3.0)
    score_a[a_overload] += W_OVERLOAD

    # Signal 4: Tactical Prior
    st_score = tact["soft_team_score"]
    st_h = (st_score > 0.15)
    score_h[st_h] += W_TACTICAL * st_score[st_h]

    st_a = (st_score < -0.15)
    score_a[st_a] += W_TACTICAL * np.abs(st_score[st_a])

    # 6. Reliability Metric and Reason Codes
    # Base reliability components
    rel_ball = np.where(ball_present, 0.40, 0.05).astype(np.float32)
    rel_prox = np.where(ball_present & ((d_h1 <= 2.5) | (d_a1 <= 2.5)), 0.30, 0.05).astype(np.float32)
    rel_tracking = np.where(~ball_kin["ball_jump_flag"], 0.20, 0.0).astype(np.float32)
    rel_unambig = np.where(
        (geom["control_state"] == 0) | (geom["control_state"] == 1), 0.10, 0.0
    ).astype(np.float32)

    reliability = rel_ball + rel_prox + rel_tracking + rel_unambig

    # Reason code assignment (hierarchy of explanatory power)
    # 1. Non-play match periods
    in_non_play = np.zeros(n_frames, dtype=bool)
    for s, e in non_play_intervals:
        s_c = max(0, s)
        e_c = min(n_frames - 1, e)
        if s_c <= e_c:
            in_non_play[s_c:e_c + 1] = True
            reason_code[s_c:e_c + 1] = "MATCH_NON_PLAY"
            score_h[s_c:e_c + 1] = 0.0
            score_a[s_c:e_c + 1] = 0.0
            reliability[s_c:e_c + 1] = 1.0  # Confirmed non-play

    # 2. Ball missing
    missing_mask = ~ball_present & ~in_non_play
    reason_code[missing_mask] = "BALL_MISSING"
    reliability[missing_mask] = 0.10

    # 3. Carrier detected
    live_mask = ball_present & ~in_non_play
    reason_code[live_mask & is_carrier_h] = "CARRIER_H"
    reason_code[live_mask & is_carrier_a] = "CARRIER_A"

    # 4. Proximity dominance (when not carrier)
    h_prox_only = live_mask & ~is_carrier_h & h_prox
    reason_code[h_prox_only] = "PROXIMITY_H"

    a_prox_only = live_mask & ~is_carrier_a & a_prox
    reason_code[a_prox_only] = "PROXIMITY_A"

    # 5. Contested duel
    contested_mask = live_mask & (geom["control_state"] == 2) & ~is_carrier_h & ~is_carrier_a
    reason_code[contested_mask] = "CONTESTED_DUEL"
    reliability[contested_mask] = np.minimum(reliability[contested_mask], 0.35)

    # 6. Loose ball
    loose_mask = live_mask & (geom["control_state"] == 3) & ~is_carrier_h & ~is_carrier_a
    reason_code[loose_mask] = "LOOSE_BALL"
    reliability[loose_mask] = np.minimum(reliability[loose_mask], 0.30)

    # 7. Tracking jump anomaly
    jump_mask = live_mask & ball_kin["ball_jump_flag"]
    reason_code[jump_mask] = "TRACKING_JUMP_GLITCH"
    reliability[jump_mask] = np.minimum(reliability[jump_mask], 0.20)

    return {
        "score_h": score_h,
        "score_a": score_a,
        "reliability": reliability,
        "reason_code": reason_code,
        "is_carrier_h": is_carrier_h,
        "is_carrier_a": is_carrier_a,
        "control_state": geom["control_state"],
        "d_h1": geom["d_h1"],
        "d_a1": geom["d_a1"],
        "margin_d": geom["margin_d"],
        "ball_present": ball_present,
    }
