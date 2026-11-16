#!/usr/bin/env python3
"""PHANTOM-11: Boundary Evidence and Change-Point Extraction Module

Step 7 of More Than A Hack 2027:
- Identifies candidate possession change points and physical transition readiness:
  * Ball trajectory deflections and kick bursts.
  * Carrier transfers (when a player from a different team takes the ball).
  * Ball visibility transitions (entering or exiting tracking dropouts).
  * Smart Tagging phase transitions and restart windows.
- Dynamically derives non-play match boundaries from official games.py constants.
- Purely deterministic, fully reproducible, SI units throughout.
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


def derive_non_play_intervals(game=games.GAME):
    """Derive non-play intervals (pre-match, halftime, post-match) dynamically from official constants."""
    blocks = games.half_video_blocks(game)
    n_frames = games.source_frames("tactical", game)
    h1_start, h1_end = blocks[0][1], blocks[0][2]
    h2_start, h2_end = blocks[1][1], blocks[1][2]

    return [
        (0, h1_start - 1),                # Pre-match
        (h1_end + 1, h2_start - 1),        # Halftime
        (h2_end + 1, n_frames - 1),       # Post-match
    ]


def extract_boundary_evidence(
    ball_present,
    is_kick_or_deflection,
    is_carrier_h,
    is_carrier_a,
    nearest_team,
    smart_tagging_stoppage,
    non_play_intervals=None,
):
    """Extract physical change points and transition readiness signals across all frames.

    Parameters
    ----------
    ball_present : (N,) bool
    is_kick_or_deflection : (N,) bool
    is_carrier_h : (N,) bool
    is_carrier_a : (N,) bool
    nearest_team : (N,) int8 (0=Home, 1=Away, -1=Nobody)
    smart_tagging_stoppage : (N,) float32
    non_play_intervals : list of (start, end)

    Returns
    -------
    dict with:
        'turnover_readiness_ha': (N,) float32 in [0.0, 1.0] (readiness for H -> A)
        'turnover_readiness_ah': (N,) float32 in [0.0, 1.0] (readiness for A -> H)
        'stoppage_readiness': (N,) float32 in [0.0, 1.0] (readiness for Live -> D)
        'restart_readiness': (N,) float32 in [0.0, 1.0] (readiness for D -> Live)
        'is_candidate_boundary': (N,) bool
        'in_non_play': (N,) bool
    """
    n_frames = len(ball_present)
    non_play_intervals = derive_non_play_intervals() if non_play_intervals is None else non_play_intervals

    in_non_play = np.zeros(n_frames, dtype=bool)
    for s, e in non_play_intervals:
        s_c = max(0, s)
        e_c = min(n_frames - 1, e)
        if s_c <= e_c:
            in_non_play[s_c:e_c + 1] = True

    # 1. Candidate boundary flags
    # Nearest team change
    team_changed = np.zeros(n_frames, dtype=bool)
    team_changed[1:] = (nearest_team[1:] != nearest_team[:-1]) & (nearest_team[1:] >= 0) & (nearest_team[:-1] >= 0)

    # Ball presence edge
    ball_edge = np.zeros(n_frames, dtype=bool)
    ball_edge[1:] = (ball_present[1:] != ball_present[:-1])

    # Carrier emergence / change
    carrier_h_emerged = np.zeros(n_frames, dtype=bool)
    carrier_h_emerged[1:] = is_carrier_h[1:] & ~is_carrier_h[:-1]

    carrier_a_emerged = np.zeros(n_frames, dtype=bool)
    carrier_a_emerged[1:] = is_carrier_a[1:] & ~is_carrier_a[:-1]

    is_candidate_boundary = (
        is_kick_or_deflection
        | team_changed
        | ball_edge
        | carrier_h_emerged
        | carrier_a_emerged
    )

    # 2. Turnover readiness (physical readiness to switch team)
    # Baseline readiness is low (0.10) to prevent spurious switches.
    turnover_readiness_ha = np.full(n_frames, 0.10, dtype=np.float32)
    turnover_readiness_ah = np.full(n_frames, 0.10, dtype=np.float32)

    # When Away carrier emerges, readiness for H -> A is high
    turnover_readiness_ha[carrier_a_emerged] = 0.90
    turnover_readiness_ha[is_carrier_a] = np.maximum(turnover_readiness_ha[is_carrier_a], 0.75)

    # When Home carrier emerges, readiness for A -> H is high
    turnover_readiness_ah[carrier_h_emerged] = 0.90
    turnover_readiness_ah[is_carrier_h] = np.maximum(turnover_readiness_ah[is_carrier_h], 0.75)

    # Deflection or kick during contested or changing play increases turnover readiness
    kick_and_changed = is_kick_or_deflection & team_changed
    turnover_readiness_ha[kick_and_changed & (nearest_team == 1)] = 0.70
    turnover_readiness_ah[kick_and_changed & (nearest_team == 0)] = 0.70

    # 3. Stoppage readiness (Live -> Dead Ball D)
    stoppage_readiness = np.full(n_frames, 0.10, dtype=np.float32)
    stoppage_readiness[~ball_present] = 0.60
    stoppage_readiness[smart_tagging_stoppage > 0.3] = np.maximum(
        stoppage_readiness[smart_tagging_stoppage > 0.3], 0.70
    )
    stoppage_readiness[in_non_play] = 1.0

    # 4. Restart readiness (Dead Ball D -> Live)
    restart_readiness = np.full(n_frames, 0.10, dtype=np.float32)
    restart_readiness[ball_present & (is_carrier_h | is_carrier_a)] = 0.85
    restart_readiness[ball_edge & ball_present] = 0.70
    restart_readiness[in_non_play] = 0.0

    return {
        "turnover_readiness_ha": turnover_readiness_ha,
        "turnover_readiness_ah": turnover_readiness_ah,
        "stoppage_readiness": stoppage_readiness,
        "restart_readiness": restart_readiness,
        "is_candidate_boundary": is_candidate_boundary,
        "in_non_play": in_non_play,
    }
