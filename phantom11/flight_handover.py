#!/usr/bin/env python3
"""PHANTOM-11: Flight-Aware Possession Handover Module

Step 11 of More Than A Hack 2027:
- Implements physically grounded flight-aware possession handover:
  1. Distinguishes high-speed passes in transit (same-team passes, aerial crosses, lobbed balls)
     from opposing-team possessions, resolving 2D proximity ambiguities where an opponent
     happens to be closest in Euclidean distance while the ball is airborne overhead.
  2. Protects genuine one-touch turnovers: allows immediate handover at the contact frame
     when an opponent executes a deliberate interception, volleyed clearance, or headed pass.
  3. Enforces the official competition possession definition:
     "while live, loose or contested possession generally remains with the last controlling
      team until an opponent gains controlled possession."
  4. Preserves genuine rapid turnovers (>= 1.0s of confirmed carrying) intact.
  5. Deterministic, fully reproducible, SI units throughout (meters, seconds, m/s).
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from phantom11.boundary_refinement import extract_spell_intervals


def detect_ball_flight_frames(
    ball_present: np.ndarray,
    ball_speed: np.ndarray,
    min_flight_speed: float = 7.0,
) -> np.ndarray:
    """Identify frames where the ball is traveling at sustained flight speed.

    Parameters
    ----------
    ball_present : (N,) bool
    ball_speed : (N,) float32 (m/s)
    min_flight_speed : float (default 7.0 m/s = 25.2 km/h)

    Returns
    -------
    is_flight : (N,) bool
    """
    valid_speed = np.nan_to_num(ball_speed, nan=0.0)
    return ball_present & (valid_speed >= min_flight_speed)


def refine_flight_handovers(
    labels: np.ndarray,
    ball_present: np.ndarray,
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    d_h1: np.ndarray,
    d_a1: np.ndarray,
    is_carrier_h: np.ndarray,
    is_carrier_a: np.ndarray,
    ball_speed: Optional[np.ndarray] = None,
    non_play_intervals: Optional[List[Tuple[int, int]]] = None,
    fps: float = 25.0,
    min_flight_speed: float = 7.0,          # >= 7 m/s indicates ball in flight / kicked
    max_transit_duration_s: float = 3.5,     # Aerial passes rarely exceed 3.5s
    receiver_proximity_threshold: float = 2.5, # Receiver must be within 2.5m at arrival
    gate_deceleration_by_proximity: bool = False, # If True, requires d <= 2.2m for deceleration trap
) -> np.ndarray:
    """Apply flight-aware possession handover refinement.

    Parameters
    ----------
    labels : (N,) np.ndarray of str ('H', 'A', 'D')
        Input labels (e.g. from v4 boundary refinement).
    ball_present : (N,) bool
    ball_xy : (N, 2) float32 (meters)
    players_xy : (N, 46, 2) float32 (meters)
    team_of_track : (46,) int8 (0=Home, 1=Away)
    d_h1 : (N,) float32 (meters to nearest Home player)
    d_a1 : (N,) float32 (meters to nearest Away player)
    is_carrier_h : (N,) bool
    is_carrier_a : (N,) bool
    ball_speed : (N,) float32, optional (m/s)
    non_play_intervals : list of (start, end)
    fps : float (frames per second)
    min_flight_speed : float (m/s)
    max_transit_duration_s : float (seconds)
    receiver_proximity_threshold : float (meters)

    Returns
    -------
    refined : (N,) np.ndarray of str ('H', 'A', 'D')
    """
    n_frames = len(labels)
    refined = labels.copy()
    non_play_intervals = non_play_intervals or []

    # Non-play mask
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
            ball_speed[idx + 1] = np.where(dist <= 2.5, dist * fps, 0.0)

    clean_speed = np.nan_to_num(ball_speed, nan=0.0)
    max_transit_frames = int(max_transit_duration_s * fps)

    # -----------------------------------------------------------------------
    # Step 1: Aerial Pass Continuity (Same-Team Long Pass / Lofted Cross)
    # -----------------------------------------------------------------------
    # Pattern: Team T1 -> Team T2 -> Team T1, where Team T2 was momentarily
    # closest in 2D while the ball was in high-speed flight across the pitch.
    # If Team T2 never established carrier control, ball was in sustained transit,
    # and Team T1 receiver successfully collects the ball at the end:
    # -> Unify into continuous possession of Team T1.
    spells = extract_spell_intervals(refined)
    for idx in range(1, len(spells) - 1):
        cur_lbl, s, e, dur = spells[idx]
        prev_lbl, ps, pe, pdur = spells[idx - 1]
        next_lbl, ns, ne, ndur = spells[idx + 1]

        # Only evaluate opposing-team interruptions surrounded by the same team
        if cur_lbl in ("H", "A") and prev_lbl == next_lbl and prev_lbl in ("H", "A") and cur_lbl != prev_lbl:
            if np.any(in_non_play[s:e + 1]):
                continue

            # Check if duration is within realistic airborne flight time (<= 3.5s)
            if dur > max_transit_frames:
                continue

            # Check carrier control of the defending team T2 during the spell
            is_c_opp = is_carrier_h[s:e + 1] if cur_lbl == "H" else is_carrier_a[s:e + 1]
            carrier_count = int(np.sum(is_c_opp))

            # SAFEGUARD: If defending team had confirmed carrier control (>= 4 frames = 0.16s),
            # this was a genuine tackle/turnover. NEVER merge!
            if carrier_count >= 4:
                continue

            # Calculate mean ball speed during the spell
            seg_speeds = clean_speed[s:e + 1]
            seg_valid_speeds = seg_speeds[ball_present[s:e + 1]]
            mean_speed = float(np.mean(seg_valid_speeds)) if len(seg_valid_speeds) > 0 else 0.0

            # Check if ball was ever trapped or brought to a controlled stop at defending feet
            d_opp = d_h1[s:e + 1] if cur_lbl == "H" else d_a1[s:e + 1]
            stopped_at_feet = np.any((d_opp <= 1.2) & (seg_speeds <= 3.5))
            if stopped_at_feet:
                continue

            # Check receiver proximity: did Team T1 receive/collect the ball in the next spell?
            d_team_next = d_a1[ns:ne + 1] if cur_lbl == "H" else d_h1[ns:ne + 1]
            min_d_next = float(np.min(d_team_next)) if len(d_team_next) > 0 else 100.0

            # If the ball was traveling at flight speed and was received by a teammate:
            if mean_speed >= min_flight_speed and min_d_next <= receiver_proximity_threshold:
                # Unify spurious in-flight spell into continuous possession
                refined[s:e + 1] = prev_lbl

    # -----------------------------------------------------------------------
    # Step 2: Premature Interception Handover Alignment
    # -----------------------------------------------------------------------
    # When Team T1 passes and Team T2 intercepts downfield, naive 2D proximity
    # switches to Team T2 mid-flight before Team T2's player reaches the ball.
    # Possession should remain with kicking team T1 until the intercepting
    # player reaches within contact distance (<= 1.5m) or controls the ball.
    spells = extract_spell_intervals(refined)
    for idx in range(len(spells) - 1):
        cur_lbl, s, e, dur = spells[idx]
        next_lbl, ns, ne, ndur = spells[idx + 1]

        if cur_lbl in ("H", "A") and next_lbl in ("H", "A") and cur_lbl != next_lbl:
            if in_non_play[ns]:
                continue

            # Transition occurs at frame ns
            spd_ns = clean_speed[ns]
            d_next_ns = d_h1[ns] if next_lbl == "H" else d_a1[ns]
            is_c_next_ns = is_carrier_h[ns] if next_lbl == "H" else is_carrier_a[ns]

            # If ball is traveling at flight speed and the interceptor is still far away (> 2.0m)
            # without carrier control:
            if spd_ns >= min_flight_speed and d_next_ns > 2.0 and not is_c_next_ns:
                # Scan forward up to 75 frames (3.0s) for the moment of interception
                limit = min(ns + 75, ne)
                engage_frame = ns
                for f in range(ns, limit + 1):
                    d_f = d_h1[f] if next_lbl == "H" else d_a1[f]
                    is_c_f = is_carrier_h[f] if next_lbl == "H" else is_carrier_a[f]
                    spd_f = clean_speed[f]

                    # Interception confirmed when interceptor reaches ball, traps at feet, or carries
                    if gate_deceleration_by_proximity:
                        if is_c_f or d_f <= 1.5 or (d_f <= 2.2 and spd_f <= 3.5):
                            engage_frame = f
                            break
                    else:
                        if is_c_f or d_f <= 1.5 or spd_f <= 3.5:
                            engage_frame = f
                            break

                if engage_frame > ns:
                    # Delay handover until the intercepting player actually engages
                    refined[ns:engage_frame] = cur_lbl

    # -----------------------------------------------------------------------
    # Step 3: Strict Non-Play Clamping
    # -----------------------------------------------------------------------
    refined[in_non_play] = "D"

    return refined
