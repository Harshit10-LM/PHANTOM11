#!/usr/bin/env python3
"""PHANTOM-11: Conservative Missing-Ball Possession Continuity Module

Step 5 of More Than A Hack 2027:
- Distinguishes tracking dropouts from genuine dead-ball stoppages.
- Uses physical evidence: team control before/after, displacement, and velocity plausibility.
- Uses Smart Tagging set pieces as supporting evidence to preserve true stoppages.
- Correctly handles match boundaries (pre-match, halftime, post-match).
- Preserves baseline v0 predictions when evidence is ambiguous or insufficient.
- Purely algorithmic and deterministic: NO ground-truth labels used.
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

# ---------------------------------------------------------------------------
# Physical Constants & Justifications (Not tuned against game labels)
# ---------------------------------------------------------------------------
# In football, typical pass durations or visual occlusions last 1 to 5 seconds.
# An official referee stoppage (whistle, ball out, restart preparation) takes
# at least 5-10 seconds. We conservatively cap continuity bridging at 150 frames (6.0s).
MAX_CONTINUITY_GAP_FRAMES = 150  # 6.0 seconds at 25 fps

# Maximum plausible ball displacement across a 5-6s tracking dropout during
# continuous possession (e.g. cross or pass): 30.0 meters.
MAX_BALL_DISPLACEMENT_METERS = 30.0

# Maximum plausible average ball speed across the gap: 25.0 m/s (~90 km/h).
MAX_BALL_SPEED_MPS = 25.0

# Pre/post window to verify consistent team control before and after gap
EVIDENCE_WINDOW_FRAMES = 5

# Match boundary definitions (from games.py constants for BAR-ATH)
# Half 1: 11185 to 90651 | Halftime: 90652 to 110807 | Half 2: 110808 to 190197
NON_PLAY_INTERVALS = [
    (0, 11184),         # Pre-match before 1st half kickoff
    (90652, 110807),     # Halftime break
    (190198, 199318),    # Post-match after final whistle
]


def find_missing_ball_gaps(ball_present):
    """Find contiguous intervals where ball_present is False."""
    gaps = []
    in_gap = False
    gap_start = 0
    n = len(ball_present)

    for i in range(n):
        if not ball_present[i]:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                in_gap = False
                gaps.append((gap_start, i - 1))

    if in_gap:
        gaps.append((gap_start, n - 1))

    return gaps


def get_dominant_team(nearest_team, start, end):
    """Return dominant team in window if one team has >= 60% presence."""
    window = nearest_team[start:end + 1]
    valid = window[window >= 0]
    if len(valid) == 0:
        return -1
    counts = np.bincount(valid, minlength=2)
    if counts[0] > counts[1] and counts[0] / len(valid) >= 0.6:
        return 0  # Home (Barcelona)
    elif counts[1] > counts[0] and counts[1] / len(valid) >= 0.6:
        return 1  # Away (Athletic Club)
    return -1


def apply_continuity(
    v0_labels,
    ball_present,
    ball_coords,
    nearest_team,
    set_pieces_intervals=None,
    pitch=(105.3, 68.0),
    max_gap_frames=MAX_CONTINUITY_GAP_FRAMES,
    max_disp_m=MAX_BALL_DISPLACEMENT_METERS,
    max_speed_mps=MAX_BALL_SPEED_MPS,
    non_play_intervals=NON_PLAY_INTERVALS,
):
    """Apply conservative possession continuity to baseline v0 labels.

    Parameters:
    -----------
    v0_labels: np.ndarray of shape (N,) with str dtype ('H', 'A', 'D', 'X')
    ball_present: np.ndarray of shape (N,) bool
    ball_coords: np.ndarray of shape (N, 2) float (metric x, y)
    nearest_team: np.ndarray of shape (N,) int8 (0=Home, 1=Away, -1=Nobody)
    set_pieces_intervals: list of (start_frame, end_frame) for set piece stoppages
    pitch: (length, width) in meters
    """
    n_frames = len(v0_labels)
    v1_labels = v0_labels.copy()
    set_pieces_intervals = set_pieces_intervals or []

    # 1. Enforce Stoppage Protection on Non-Match Periods
    # (Pre-match, Halftime, Post-match cannot be active gameplay)
    clamped_non_play_frames = 0
    for s, e in non_play_intervals:
        s_clamped = max(0, s)
        e_clamped = min(n_frames - 1, e)
        if s_clamped <= e_clamped:
            mask = (v1_labels[s_clamped:e_clamped + 1] != "D")
            clamped_non_play_frames += int(np.sum(mask))
            v1_labels[s_clamped:e_clamped + 1] = "D"

    # 2. Find missing-ball gaps
    gaps = find_missing_ball_gaps(ball_present)

    # Tracking diagnostic counters
    decisions = {
        "RECOVER_CONTINUITY": 0,
        "PRESERVE_NON_PLAY": 0,
        "PRESERVE_SET_PIECE": 0,
        "AMBIGUOUS_TURNOVER": 0,
        "EXCEEDS_MAX_DURATION": 0,
        "IMPLAUSIBLE_DISPLACEMENT": 0,
        "INSUFFICIENT_EVIDENCE": 0,
    }

    recovered_gap_details = []
    frames_recovered_to_h = 0
    frames_recovered_to_a = 0

    pitch_len, pitch_wid = pitch

    for s, e in gaps:
        dur = e - s + 1

        # Check if inside non-play intervals
        in_non_play = any(not (e < nps or s > npe) for nps, npe in non_play_intervals)
        if in_non_play:
            decisions["PRESERVE_NON_PLAY"] += 1
            continue

        # Check if exceeds conservative duration threshold
        if dur > max_gap_frames:
            decisions["EXCEEDS_MAX_DURATION"] += 1
            continue

        # Check if overlaps with confirmed Smart Tagging set piece stoppage
        sp_overlap = any(not (e < sp_s or s > sp_e) for sp_s, sp_e in set_pieces_intervals)
        if sp_overlap:
            decisions["PRESERVE_SET_PIECE"] += 1
            continue

        # Check evidence before and after
        idx_before = s - 1
        idx_after = e + 1

        if idx_before < 0 or idx_after >= n_frames:
            decisions["INSUFFICIENT_EVIDENCE"] += 1
            continue

        # Ball must be tracked on the boundary frames
        if not (ball_present[idx_before] and ball_present[idx_after]):
            decisions["INSUFFICIENT_EVIDENCE"] += 1
            continue

        # Team evidence before and after
        w_before_start = max(0, s - EVIDENCE_WINDOW_FRAMES)
        w_after_end = min(n_frames - 1, e + EVIDENCE_WINDOW_FRAMES)

        team_before = get_dominant_team(nearest_team, w_before_start, idx_before)
        team_after = get_dominant_team(nearest_team, idx_after, w_after_end)

        # Fallback to single immediate boundary frame if window was too small
        if team_before < 0:
            team_before = nearest_team[idx_before]
        if team_after < 0:
            team_after = nearest_team[idx_after]

        if team_before not in (0, 1) or team_after not in (0, 1):
            decisions["INSUFFICIENT_EVIDENCE"] += 1
            continue

        # Opposing teams before vs after -> contested turnover / duel
        if team_before != team_after:
            decisions["AMBIGUOUS_TURNOVER"] += 1
            continue

        # Check ball coordinates & displacement
        p_b = ball_coords[idx_before]
        p_a = ball_coords[idx_after]

        # Coordinates must be valid pitch positions
        if not (0 <= p_b[0] <= pitch_len and 0 <= p_b[1] <= pitch_wid and
                0 <= p_a[0] <= pitch_len and 0 <= p_a[1] <= pitch_wid):
            decisions["INSUFFICIENT_EVIDENCE"] += 1
            continue

        dx = p_a[0] - p_b[0]
        dy = p_a[1] - p_b[1]
        displacement = float(np.sqrt(dx * dx + dy * dy))
        time_elapsed_s = dur / 25.0
        speed_mps = displacement / time_elapsed_s if time_elapsed_s > 0 else 0.0

        if displacement > max_disp_m or speed_mps > max_speed_mps:
            decisions["IMPLAUSIBLE_DISPLACEMENT"] += 1
            continue

        # All conservative criteria satisfied: recover continuity
        decisions["RECOVER_CONTINUITY"] += 1
        cont_label = "H" if team_before == 0 else "A"
        prev_labels = set(v1_labels[s:e + 1])

        # Overwrite gap with the verified controlling team
        v1_labels[s:e + 1] = cont_label

        if cont_label == "H":
            frames_recovered_to_h += dur
        else:
            frames_recovered_to_a += dur

        recovered_gap_details.append({
            "start": s,
            "end": e,
            "duration": dur,
            "team": cont_label,
            "displacement": displacement,
            "speed": speed_mps,
            "v0_labels": prev_labels,
        })

    # Summary diagnostics
    total_frames_changed = int(np.sum(v0_labels != v1_labels))

    diag = {
        "total_gaps": len(gaps),
        "decisions": decisions,
        "clamped_non_play_frames": clamped_non_play_frames,
        "recovered_gaps_count": len(recovered_gap_details),
        "recovered_gaps": recovered_gap_details,
        "frames_recovered_to_h": frames_recovered_to_h,
        "frames_recovered_to_a": frames_recovered_to_a,
        "total_frames_changed": total_frames_changed,
    }

    return v1_labels, diag
