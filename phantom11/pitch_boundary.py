#!/usr/bin/env python3
"""PHANTOM-11: Pitch Boundary Awareness and Out-of-Bounds Detection Module

Step 16 of More Than A Hack 2027:
- Implements IFAB Law 9 (Ball in and out of play):
  "The ball is out of play when it has wholly passed over the goal line
   or touchline on the ground or in the air."
- Uses measured pitch dimensions (length, width) from MatchConfig / games.py.
- Accounts for ball radius (~0.11m), line width (~0.12m), and optical tracking uncertainty.
- Protects boundary touches: margin threshold of 0.35m ensures the entire ball
  has unequivocally crossed the boundary line before declaring Dead Ball ('D').
- Filters single-frame tracking noise near lines using consecutive-frame persistence.
- Re-entry dynamics: Once the ball crosses back into the field of play, live play resumes.
"""

import numpy as np
from typing import Tuple, Optional, List


DEFAULT_PITCH_MARGIN_M = 0.35  # Ball radius (0.11m) + line width (0.12m) + optical noise buffer (0.12m)
DEEP_OOB_THRESHOLD_M = 0.80    # Beyond 0.8m is unequivocally outside play, even for single frames


def compute_out_of_bounds_distance(
    ball_xy: np.ndarray,
    ball_present: np.ndarray,
    pitch: Tuple[float, float] = (105.3, 68.0),
) -> np.ndarray:
    """Compute the perpendicular/Euclidean distance of the ball outside the pitch rectangle.

    Parameters
    ----------
    ball_xy : (N, 2) float32
        Ball coordinates in metres.
    ball_present : (N,) bool
        Boolean mask indicating valid (non-NaN) ball tracking.
    pitch : (length, width) in metres (default: 105.3, 68.0)

    Returns
    -------
    dist_oob : (N,) float32
        Distance in metres outside the pitch rectangle (0.0 if inside or on line).
    """
    length, width = pitch
    n_frames = len(ball_xy)
    dist_oob = np.zeros(n_frames, dtype=np.float32)

    valid = ball_present & ~np.isnan(ball_xy[:, 0]) & ~np.isnan(ball_xy[:, 1])
    if not np.any(valid):
        return dist_oob

    bx = ball_xy[valid, 0]
    by = ball_xy[valid, 1]

    dx = np.maximum(0.0, np.maximum(-bx, bx - length))
    dy = np.maximum(0.0, np.maximum(-by, by - width))

    dist_oob[valid] = np.sqrt(dx * dx + dy * dy)
    return dist_oob


def detect_out_of_bounds_mask(
    ball_xy: np.ndarray,
    ball_present: np.ndarray,
    pitch: Tuple[float, float] = (105.3, 68.0),
    margin: float = DEFAULT_PITCH_MARGIN_M,
    min_consecutive_frames: int = 2,
) -> np.ndarray:
    """Identify frames where the ball is physically out of bounds (out of play).

    Parameters
    ----------
    ball_xy : (N, 2) float32
    ball_present : (N,) bool
    pitch : (length, width) in metres
    margin : float (default: 0.35m)
        Buffer accounting for ball radius and line width under IFAB Law 9.
    min_consecutive_frames : int (default: 2 frames = 0.08s)
        Persistence filter to avoid classifying 1-frame boundary calibration jitter.

    Returns
    -------
    is_oob : (N,) bool
        True on frames where the ball is confirmed out of play.
    """
    dist_oob = compute_out_of_bounds_distance(ball_xy, ball_present, pitch)
    n_frames = len(dist_oob)

    raw_oob = dist_oob >= margin
    deep_oob = dist_oob >= DEEP_OOB_THRESHOLD_M

    is_oob = np.zeros(n_frames, dtype=bool)

    # 1. Any deep out-of-bounds (> 0.8m) is immediately confirmed
    is_oob[deep_oob] = True

    # 2. Moderate out-of-bounds (0.35m to 0.8m) requires at least min_consecutive frames
    # to filter out single-frame tracking noise
    if min_consecutive_frames > 1:
        # Run-length filter for raw_oob
        breaks = np.flatnonzero(raw_oob[1:] != raw_oob[:-1]) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n_frames]))

        for s, e in zip(starts, ends):
            if raw_oob[s]:
                duration = e - s
                # If persistent or containing deep observations, confirm
                if duration >= min_consecutive_frames or np.any(deep_oob[s:e]):
                    is_oob[s:e] = True
    else:
        is_oob[raw_oob] = True

    return is_oob


def apply_pitch_boundary_clamping(
    labels: np.ndarray,
    ball_xy: np.ndarray,
    ball_present: np.ndarray,
    pitch: Tuple[float, float] = (105.3, 68.0),
    non_play_intervals: Optional[List[Tuple[int, int]]] = None,
    margin: float = DEFAULT_PITCH_MARGIN_M,
) -> np.ndarray:
    """Enforce IFAB Law 9 by clamping confirmed out-of-bounds frames to Dead Ball ('D').

    Parameters
    ----------
    labels : (N,) str or object array
        Predicted possession labels ('H', 'A', 'D', 'X').
    ball_xy : (N, 2) float32
    ball_present : (N,) bool
    pitch : (length, width) in metres
    non_play_intervals : list of (start, stop), optional
    margin : float (default: 0.35m)

    Returns
    -------
    corrected_labels : (N,) str
    """
    corrected = labels.copy()
    is_oob = detect_out_of_bounds_mask(ball_xy, ball_present, pitch=pitch, margin=margin)

    # Set confirmed out-of-bounds frames to 'D'
    corrected[is_oob] = "D"

    # Strictly preserve official match non-play intervals (pre-match, halftime, post-match)
    if non_play_intervals:
        for start, end in non_play_intervals:
            s = max(0, start)
            e = min(len(corrected), end + 1)
            corrected[s:e] = "D"

    return corrected
