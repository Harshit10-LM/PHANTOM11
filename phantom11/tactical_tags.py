#!/usr/bin/env python3
"""PHANTOM-11: Tactical Tags Soft Evidence Module

Step 6 of More Than A Hack 2027:
- Maps Smart Tagging XML events to the tactical video timeline (video_frame = round(s * 25)).
- Translates tactical phases and team tags into soft possession prior signals:
  * Attacking phases (BUILD UP, PROGRESSION, etc.) support the tagged team.
  * Defensive phases (DEFENDING, DEFENSIVE TRANSITION) support the opponent team.
  * SET PIECES and GOALS provide soft stoppage / restart evidence (not hard dead-ball labels).
  * Explicitly handles 'N/A' and missing team tags.
  * Detects and records overlapping or conflicting tactical annotations.
- Adheres to competition rules: soft evidence only, never copying annotations as hard truth.
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
import load_smart_tagging


# Tactical Code Categories
ATTACKING_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "LONG BALL",
    "PLAYERS IN THE BOX",
}

DEFENDING_CODES = {
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
}

STOPPAGE_RESTART_CODES = {
    "SET PIECES",
    "GOALS",
}

# Team string mapping
TEAM_MAP = {
    "barcelona": 0,       # Home
    "atletic club": 1,    # Away (official spelling in tag)
    "athletic club": 1,   # Alternate spelling defense
}


def parse_smart_tagging_timeline(events, n_frames, team_map=None):
    """Convert Smart Tagging events into per-frame soft evidence arrays.

    Parameters
    ----------
    events : list of dict
        Events returned by starter/load_smart_tagging.py:
        {'code', 'start_frame', 'end_frame', 'tags'}
    n_frames : int
        Total frames in match.
    team_map : dict, optional
        Mapping of lowercase team names to 0 (Home) and 1 (Away).
        Defaults to TEAM_MAP (Barcelona=0, Athletic Club=1).

    Returns
    -------
    dict with keys:
        'soft_team_score': (N,) float32 in [-1.0, 1.0]
            Positive values support Home, negative values support Away.
        'stoppage_support': (N,) float32 in [0.0, 1.0]
            Strength of evidence that ball is dead or in restart window.
        'has_conflict': (N,) bool
            True if multiple active events provide opposing team support.
        'active_event_count': (N,) int16
            Number of simultaneous tactical tags on this frame.
    """
    effective_team_map = team_map if team_map is not None else TEAM_MAP
    home_votes = np.zeros(n_frames, dtype=np.float32)
    away_votes = np.zeros(n_frames, dtype=np.float32)
    stoppage_support = np.zeros(n_frames, dtype=np.float32)
    active_event_count = np.zeros(n_frames, dtype=np.int16)

    if not events:
        return {
            "soft_team_score": np.zeros(n_frames, dtype=np.float32),
            "stoppage_support": np.zeros(n_frames, dtype=np.float32),
            "has_conflict": np.zeros(n_frames, dtype=bool),
            "active_event_count": np.zeros(n_frames, dtype=np.int16),
        }

    for ev in events:
        code = (ev["code"] or "").strip().upper()
        s = max(0, ev["start_frame"])
        e = min(n_frames - 1, ev["end_frame"])
        if s > e:
            continue

        active_event_count[s:e + 1] += 1
        team_str = (ev["tags"].get("Team") or "").strip().lower()
        team_id = effective_team_map.get(team_str, -1)

        # 1. Stoppage / Restart evidence
        if code in STOPPAGE_RESTART_CODES:
            # Soft weight for stoppage (0.60; not 1.0, because ~5s window includes live restart)
            stoppage_support[s:e + 1] = np.maximum(stoppage_support[s:e + 1], 0.60)

        # 2. Attacking Phase evidence
        if code in ATTACKING_CODES:
            if team_id == 0:
                home_votes[s:e + 1] += 0.40
            elif team_id == 1:
                away_votes[s:e + 1] += 0.40

        # 3. Defending Phase evidence (defending team implies opposing team possesses)
        elif code in DEFENDING_CODES:
            if team_id == 0:
                # Home defending -> Away possesses
                away_votes[s:e + 1] += 0.35
            elif team_id == 1:
                # Away defending -> Home possesses
                home_votes[s:e + 1] += 0.35

    # Conflict detection: both teams have positive support
    has_conflict = (home_votes > 0.1) & (away_votes > 0.1)

    # Net soft score in [-1.0, 1.0]
    total_votes = home_votes + away_votes + 1e-6
    soft_team_score = np.where(
        has_conflict,
        0.0,  # zero out conflicted frames
        np.clip((home_votes - away_votes) / np.maximum(1.0, total_votes), -1.0, 1.0)
    ).astype(np.float32)

    return {
        "soft_team_score": soft_team_score,
        "stoppage_support": stoppage_support,
        "has_conflict": has_conflict,
        "active_event_count": active_event_count,
    }
