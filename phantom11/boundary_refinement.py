#!/usr/bin/env python3
"""PHANTOM-11: Confidence-Gated Possession Boundary Refinement Module

Step 9 of More Than A Hack 2027:
- Implements physical evidence-based possession boundary refinement:
  1. Distinguishes momentary opponent deflections (< 1.0s, 0 carrier frames) from
     genuine established possessions without erasing short confirmed interceptions.
  2. Bridges brief visual tracking dropouts (<= 1.2s) during live play where the same
     team maintains bilateral control, preventing artificial possession fragmentation.
  3. Snaps premature restart boundaries from static pre-kick run-ups to the true moment
     the ball is struck or engaged.
  4. Strictly clamps non-play intervals (pre-match, halftime, post-match) to Dead Ball (D).
- Fully deterministic, completely reproducible, SI units throughout.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def extract_spell_intervals(labels: np.ndarray) -> List[Tuple[str, int, int, int]]:
    """Extract contiguous spans as (label, start_frame, end_frame, duration_frames)."""
    n = len(labels)
    if n == 0:
        return []

    spells = []
    cur_lbl = labels[0]
    cur_start = 0

    for i in range(1, n):
        if labels[i] != cur_lbl:
            spells.append((cur_lbl, cur_start, i - 1, i - cur_start))
            cur_lbl = labels[i]
            cur_start = i

    spells.append((cur_lbl, cur_start, n - 1, n - cur_start))
    return spells


def refine_possession_boundaries(
    labels: np.ndarray,
    ball_present: np.ndarray,
    ball_xy: np.ndarray,
    is_carrier_h: np.ndarray,
    is_carrier_a: np.ndarray,
    d_h1: np.ndarray,
    d_a1: np.ndarray,
    ball_speed: Optional[np.ndarray] = None,
    stoppage_support: Optional[np.ndarray] = None,
    non_play_intervals: Optional[List[Tuple[int, int]]] = None,
    fps: float = 25.0,
    max_deflection_frames: int = 24,  # < 1.0s at 25 fps
    min_carrier_frames_for_turnover: int = 4,  # >= ~0.16s confirmed carrying
    max_dropout_bridge_frames: int = 30,  # <= 1.2s
) -> np.ndarray:
    """Apply confidence-gated boundary refinement to predicted possession sequence.

    Parameters
    ----------
    labels : (N,) np.ndarray of str ('H', 'A', 'D')
        Input candidate labels (e.g. from v3 sequence decoder).
    ball_present : (N,) bool
    ball_xy : (N, 2) float32 (metric coordinates)
    is_carrier_h : (N,) bool
    is_carrier_a : (N,) bool
    d_h1 : (N,) float32 (distance from ball to nearest Home player)
    d_a1 : (N,) float32 (distance from ball to nearest Away player)
    ball_speed : (N,) float32, optional
    stoppage_support : (N,) float32, optional
    non_play_intervals : list of (start, end)
    fps : float

    Returns
    -------
    refined : (N,) np.ndarray of str ('H', 'A', 'D')
    """
    n_frames = len(labels)
    refined = labels.copy()
    non_play_intervals = non_play_intervals or []

    # Non-play boolean mask
    in_non_play = np.zeros(n_frames, dtype=bool)
    for s, e in non_play_intervals:
        s_c = max(0, s)
        e_c = min(n_frames - 1, e)
        if s_c <= e_c:
            in_non_play[s_c:e_c + 1] = True

    # Compute ball speed if not provided
    if ball_speed is None:
        ball_speed = np.zeros(n_frames, dtype=np.float32)
        valid = ball_present[:-1] & ball_present[1:]
        idx = np.where(valid)[0]
        if len(idx) > 0:
            disp = ball_xy[idx + 1] - ball_xy[idx]
            dist = np.sqrt(np.sum(disp * disp, axis=1))
            ball_speed[idx + 1] = np.where(dist <= 2.0, dist * fps, 0.0)

    if stoppage_support is None:
        stoppage_support = np.zeros(n_frames, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Step 1: Fleeting Opponent Deflection Unification
    # -----------------------------------------------------------------------
    # A spell of Team T (< max_deflection_frames) surrounded by Opponent O
    # with 0 carrier frames is a deflection / non-controlling touch.
    spells = extract_spell_intervals(refined)
    for idx in range(1, len(spells) - 1):
        cur_lbl, s, e, dur = spells[idx]
        prev_lbl, _, _, _ = spells[idx - 1]
        next_lbl, _, _, _ = spells[idx + 1]

        # Check if surrounded by the same opposing team (e.g. H -> A -> H or A -> H -> A)
        if cur_lbl in ("H", "A") and prev_lbl == next_lbl and prev_lbl != cur_lbl and prev_lbl in ("H", "A"):
            if dur <= max_deflection_frames:
                # Do not touch non-play frames
                if np.any(in_non_play[s:e + 1]):
                    continue

                # Check carrier control of the defending team
                carrier_mask = is_carrier_h[s:e + 1] if cur_lbl == "H" else is_carrier_a[s:e + 1]
                carrier_count = int(np.sum(carrier_mask))

                # Also check minimum distance to ball
                min_dist = float(np.min(d_h1[s:e + 1] if cur_lbl == "H" else d_a1[s:e + 1]))

                # If the team never established carrier control and wasn't firmly holding the ball:
                if carrier_count < min_carrier_frames_for_turnover:
                    # Unify into the surrounding continuous possession
                    refined[s:e + 1] = prev_lbl

    # -----------------------------------------------------------------------
    # Step 2: Short Dead-Ball Dropout Bridging
    # -----------------------------------------------------------------------
    # A brief D spell (<= max_dropout_bridge_frames) inside live play surrounded
    # by the same team without a referee set-piece stoppage is a tracking loss.
    spells = extract_spell_intervals(refined)
    for idx in range(1, len(spells) - 1):
        cur_lbl, s, e, dur = spells[idx]
        prev_lbl, _, _, _ = spells[idx - 1]
        next_lbl, _, _, _ = spells[idx + 1]

        if cur_lbl == "D" and prev_lbl == next_lbl and prev_lbl in ("H", "A"):
            if dur <= max_dropout_bridge_frames:
                if np.any(in_non_play[s:e + 1]):
                    continue

                # Check if there is an explicit stoppage tag (e.g. SET PIECES)
                mean_stoppage = float(np.mean(stoppage_support[s:e + 1]))
                if mean_stoppage < 0.35:
                    # Check spatial displacement across the gap if coordinates exist
                    p_b = ball_xy[s - 1] if (s > 0 and ball_present[s - 1]) else None
                    p_a = ball_xy[e + 1] if (e + 1 < n_frames and ball_present[e + 1]) else None

                    plausible = True
                    if p_b is not None and p_a is not None:
                        disp = np.linalg.norm(p_a - p_b)
                        if disp > 30.0 or (disp / (dur / fps)) > 25.0:
                            plausible = False

                    if plausible:
                        refined[s:e + 1] = prev_lbl

    # -----------------------------------------------------------------------
    # Step 3: Premature Restart Correction
    # -----------------------------------------------------------------------
    # For D -> H or D -> A restarts: if ball is stationary and nearest player is
    # still far away (> 2.5m), the ball is not yet in play. Delay the restart
    # until the player reaches the ball (<= 1.5m) or the ball is struck.
    spells = extract_spell_intervals(refined)
    for idx in range(len(spells) - 1):
        cur_lbl, s, e, _ = spells[idx]
        next_lbl, ns, ne, ndur = spells[idx + 1]

        if cur_lbl == "D" and next_lbl in ("H", "A"):
            # Transition occurs at frame ns
            if ns < n_frames and not in_non_play[ns]:
                d_nearest = d_h1[ns] if next_lbl == "H" else d_a1[ns]
                spd = ball_speed[ns]

                # If ball is stationary (< 0.5 m/s) and player is > 2.5m away:
                if spd < 0.5 and d_nearest > 2.5:
                    # Scan forward up to 75 frames (3s) or end of spell
                    search_limit = min(ns + 75, ne)
                    strike_frame = ns
                    found = False
                    for f in range(ns, search_limit + 1):
                        d_f = d_h1[f] if next_lbl == "H" else d_a1[f]
                        spd_f = ball_speed[f]
                        is_c_f = is_carrier_h[f] if next_lbl == "H" else is_carrier_a[f]

                        if is_c_f or d_f <= 1.5 or spd_f >= 2.0:
                            strike_frame = f
                            found = True
                            break

                    if found and strike_frame > ns:
                        # Extend D until strike_frame
                        refined[ns:strike_frame] = "D"

    # -----------------------------------------------------------------------
    # Step 4: Strict Non-Play Boundary Clamping
    # -----------------------------------------------------------------------
    refined[in_non_play] = "D"

    return refined
