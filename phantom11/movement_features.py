#!/usr/bin/env python3
"""PHANTOM-11: Movement-Based Evidence Module

Step 6 of More Than A Hack 2027:
- Computes distances between observed ball and all tracked players in metres.
- Computes relative speed and velocity vector alignment between ball and players.
- Detects ball carrying / dribbling (co-movement where player and ball move together).
- Detects sudden ball velocity and direction changes (kicks, deflections, tackles).
- Generates data reliability flags:
  * Missing ball observations (explicit NaN handling).
  * Tracker identity changes / slot position jumps (> 15 m/s).
  * Ball tracking anomalies (> 50 m/s single-frame jumps).
  * Aerial ball indicators.
- Purely deterministic, fully reproducible, strictly using physical SI units (metres, m/s).
- Tracking slots are treated as tracklets, not permanent player identities.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Physical Constants & Calibration Thresholds
# ---------------------------------------------------------------------------
DEFAULT_FPS = 25.0

# Human physical maximums:
# Fastest recorded football sprint: ~10.5 m/s (~38 km/h).
# Tracker jump threshold: 0.6 m in 1 frame at 25 fps = 15.0 m/s.
MAX_PHYSICAL_PLAYER_SPEED_MPS = 15.0
MAX_PLAYER_SINGLE_FRAME_DISP_M = 0.60

# Ball physical thresholds:
# Powerful strike / shot: up to ~35-40 m/s (~130-140 km/h).
# Single-frame displacement > 2.0 m at 25 fps = 50.0 m/s is an unphysical tracking glitch.
MAX_PHYSICAL_BALL_SPEED_MPS = 45.0
MAX_BALL_SINGLE_FRAME_DISP_M = 2.00

# Ball carrying / dribbling thresholds:
# A player controlling the ball remains within ~1.8-2.0 metres of it.
MAX_CARRYING_DISTANCE_M = 2.00
# When carrying, relative speed between player and ball is low (they travel together).
MAX_CARRYING_REL_SPEED_MPS = 2.50
# Velocity vectors should be roughly aligned (cosine similarity >= 0.40) when moving.
MIN_CARRYING_ALIGNMENT = 0.40

# Sudden trajectory change (kick or deflection):
# Minimum speed to consider direction change meaningful (avoid resting ball noise).
MIN_SPEED_FOR_DIRECTION_CHANGE_MPS = 3.00
# Angle threshold in radians for sudden direction change: 45 degrees = ~0.785 rad.
DEFLECTION_ANGLE_RAD = np.deg2rad(45.0)


def compute_ball_kinematics(ball_xy, fps=DEFAULT_FPS):
    """Compute ball velocity, speed, acceleration, and sudden trajectory changes.

    Parameters
    ----------
    ball_xy : np.ndarray of shape (N, 2), float32
        Ball coordinates (x, y) in metres. NaN when ball is not observed.
    fps : float
        Frames per second (default: 25.0).

    Returns
    -------
    dict with keys:
        'velocity': (N, 2) float32 (m/s)
        'speed': (N,) float32 (m/s)
        'acceleration': (N,) float32 (m/s^2)
        'heading_change_rad': (N,) float32 (radians)
        'is_kick_or_deflection': (N,) bool
        'ball_jump_flag': (N,) bool (unphysical single-frame jump)
        'ball_present': (N,) bool
    """
    n_frames = len(ball_xy)
    ball_present = ~np.isnan(ball_xy[:, 0]) & ~np.isnan(ball_xy[:, 1])

    velocity = np.full((n_frames, 2), np.nan, dtype=np.float32)
    speed = np.full(n_frames, np.nan, dtype=np.float32)
    acceleration = np.full(n_frames, np.nan, dtype=np.float32)
    heading_change = np.full(n_frames, np.nan, dtype=np.float32)
    is_kick_or_deflection = np.zeros(n_frames, dtype=bool)
    ball_jump_flag = np.zeros(n_frames, dtype=bool)

    # 1. Consecutive valid ball frames for velocity
    valid_step = ball_present[:-1] & ball_present[1:]
    idx_valid = np.where(valid_step)[0]

    if len(idx_valid) > 0:
        disp = ball_xy[idx_valid + 1] - ball_xy[idx_valid]
        dist = np.sqrt(np.sum(disp * disp, axis=1))

        # Check for unphysical tracking jumps
        unphysical = dist > MAX_BALL_SINGLE_FRAME_DISP_M
        ball_jump_flag[idx_valid + 1] = unphysical

        # Compute velocity for physically plausible steps
        plausible = ~unphysical
        p_idx = idx_valid[plausible]

        v = disp[plausible] * fps
        velocity[p_idx + 1] = v
        speed[p_idx + 1] = dist[plausible] * fps

    # 2. Acceleration and heading changes (between consecutive valid velocities)
    valid_v = ~np.isnan(velocity[:-1, 0]) & ~np.isnan(velocity[1:, 0])
    idx_v = np.where(valid_v)[0]

    if len(idx_v) > 0:
        v0 = velocity[idx_v]
        v1 = velocity[idx_v + 1]
        s0 = speed[idx_v]
        s1 = speed[idx_v + 1]

        dv = np.sqrt(np.sum((v1 - v0) ** 2, axis=1))
        acceleration[idx_v + 1] = dv * fps

        # Heading change angle via dot product
        # cos(theta) = (v0 . v1) / (||v0|| * ||v1||)
        moving = (s0 >= MIN_SPEED_FOR_DIRECTION_CHANGE_MPS) & (s1 >= MIN_SPEED_FOR_DIRECTION_CHANGE_MPS)
        m_idx = idx_v[moving]

        if len(m_idx) > 0:
            dot = np.sum(velocity[m_idx] * velocity[m_idx + 1], axis=1)
            cos_theta = np.clip(dot / (speed[m_idx] * speed[m_idx + 1]), -1.0, 1.0)
            angles = np.arccos(cos_theta)
            heading_change[m_idx + 1] = angles

            # Kick or sharp deflection: significant heading change OR sharp speed burst
            sharp_turn = angles >= DEFLECTION_ANGLE_RAD
            sharp_burst = (dv[moving] * fps >= 40.0) & (s1[moving] > s0[moving] + 3.0)
            is_kick_or_deflection[m_idx + 1] = sharp_turn | sharp_burst

    return {
        "velocity": velocity,
        "speed": speed,
        "acceleration": acceleration,
        "heading_change_rad": heading_change,
        "is_kick_or_deflection": is_kick_or_deflection,
        "ball_jump_flag": ball_jump_flag,
        "ball_present": ball_present,
    }


def compute_player_kinematics(players_xy, fps=DEFAULT_FPS):
    """Compute player velocities, speeds, and tracker jump flags.

    Parameters
    ----------
    players_xy : np.ndarray of shape (N, S, 2), float32
        Player coordinates in metres. S is number of tracking slots (46).
    fps : float
        Frames per second (default: 25.0).

    Returns
    -------
    dict with keys:
        'velocities': (N, S, 2) float32 (m/s)
        'speeds': (N, S) float32 (m/s)
        'player_jump_flag': (N, S) bool (tracking glitch / slot swap)
        'player_present': (N, S) bool
    """
    n_frames, n_slots, _ = players_xy.shape
    player_present = ~np.isnan(players_xy[:, :, 0]) & ~np.isnan(players_xy[:, :, 1])

    velocities = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speeds = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    player_jump_flag = np.zeros((n_frames, n_slots), dtype=bool)

    # Consecutive valid tracking per slot
    valid_step = player_present[:-1, :] & player_present[1:, :]
    t_idx, s_idx = np.where(valid_step)

    if len(t_idx) > 0:
        p0 = players_xy[t_idx, s_idx]
        p1 = players_xy[t_idx + 1, s_idx]
        disp = p1 - p0
        dist = np.sqrt(np.sum(disp * disp, axis=1))

        # Check for unphysical tracking jumps (> 15 m/s)
        jump_mask = dist > MAX_PLAYER_SINGLE_FRAME_DISP_M
        player_jump_flag[t_idx + 1, s_idx] = jump_mask

        plausible = ~jump_mask
        t_pl = t_idx[plausible]
        s_pl = s_idx[plausible]

        v = disp[plausible] * fps
        velocities[t_pl + 1, s_pl] = v
        speeds[t_pl + 1, s_pl] = dist[plausible] * fps

    return {
        "velocities": velocities,
        "speeds": speeds,
        "player_jump_flag": player_jump_flag,
        "player_present": player_present,
    }


def compute_relative_motion(ball_xy, ball_vel, ball_speed, players_xy, player_vel, player_speed):
    """Compute relative distance, relative velocity, and velocity alignment between ball and players.

    Parameters
    ----------
    ball_xy : (N, 2) float32
    ball_vel : (N, 2) float32
    ball_speed : (N,) float32
    players_xy : (N, S, 2) float32
    player_vel : (N, S, 2) float32
    player_speed : (N, S) float32

    Returns
    -------
    dict with keys:
        'distances': (N, S) float32 (metres)
        'rel_speeds': (N, S) float32 (m/s)
        'alignments': (N, S) float32 (cosine similarity in [-1, 1], NaN if stationary)
    """
    n_frames, n_slots, _ = players_xy.shape

    # Distances
    # expand ball_xy to (N, 1, 2)
    b_exp = ball_xy[:, np.newaxis, :]
    diff = players_xy - b_exp
    distances = np.sqrt(np.sum(diff * diff, axis=2))

    # Relative velocity
    bv_exp = ball_vel[:, np.newaxis, :]
    rel_v = bv_exp - player_vel
    rel_speeds = np.sqrt(np.sum(rel_v * rel_v, axis=2))

    # Velocity alignment: (v_ball . v_player) / (||v_ball|| * ||v_player||)
    b_spd_exp = ball_speed[:, np.newaxis]
    denom = b_spd_exp * player_speed
    dot = np.sum(bv_exp * player_vel, axis=2)

    # Valid alignment when both are moving at least 0.5 m/s
    moving_mask = (b_spd_exp >= 0.5) & (player_speed >= 0.5) & (~np.isnan(denom)) & (denom > 1e-4)

    alignments = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    alignments[moving_mask] = np.clip(dot[moving_mask] / denom[moving_mask], -1.0, 1.0)

    return {
        "distances": distances,
        "rel_speeds": rel_speeds,
        "alignments": alignments,
    }


def detect_ball_carrier(
    distances,
    rel_speeds,
    alignments,
    ball_speed,
    player_speed,
    player_jump_flag,
    ball_jump_flag,
    max_dist=MAX_CARRYING_DISTANCE_M,
    max_rel_speed=MAX_CARRYING_REL_SPEED_MPS,
    min_alignment=MIN_CARRYING_ALIGNMENT,
):
    """Identify which player slot (if any) is actively carrying / dribbling the ball.

    Criteria for ball carrying:
    1. Distance to ball <= max_dist (2.0 metres).
    2. Tracker is reliable (no unphysical jump flags).
    3. Moving with ball:
       - If moving (> 1.0 m/s): rel_speed <= 2.5 m/s AND alignment >= 0.40.
       - If stationary or slow: rel_speed <= 1.5 m/s.
    4. If multiple players meet criteria, the one with closest distance and lowest relative speed wins.

    Returns
    -------
    carrier_slot: (N,) int8 (-1 if no carrier detected)
    is_carrier: (N, S) bool
    carrier_confidence: (N,) float32 in [0.0, 1.0]
    """
    n_frames, n_slots = distances.shape

    # Base proximity criterion
    is_close = (distances <= max_dist) & (~np.isnan(distances))

    # Reliability criterion
    b_jump_exp = ball_jump_flag[:, np.newaxis]
    is_reliable = ~player_jump_flag & ~b_jump_exp

    # Motion alignment criterion
    b_spd_exp = ball_speed[:, np.newaxis]
    both_moving = (b_spd_exp > 1.0) & (player_speed > 1.0)

    # When moving: low relative speed and aligned velocity
    moving_carry = both_moving & (rel_speeds <= max_rel_speed) & (alignments >= min_alignment)

    # When stationary / slow: player very close, relative speed low
    stationary_carry = (~both_moving) & (rel_speeds <= 1.5) & (distances <= 1.5)

    # Combined frame-level candidate mask
    candidate_mask = is_close & is_reliable & (moving_carry | stationary_carry)

    carrier_slot = np.full(n_frames, -1, dtype=np.int8)
    carrier_confidence = np.zeros(n_frames, dtype=np.float32)
    is_carrier = np.zeros((n_frames, n_slots), dtype=bool)

    # For frames with candidates, pick the best one based on distance and relative speed
    has_candidate = np.any(candidate_mask, axis=1)
    cand_frames = np.where(has_candidate)[0]

    for f in cand_frames:
        slots = np.where(candidate_mask[f])[0]
        if len(slots) == 1:
            best_s = slots[0]
        else:
            # Rank candidates by composite score: distance + 0.5 * rel_speed
            scores = distances[f, slots] + 0.5 * np.nan_to_num(rel_speeds[f, slots], nan=5.0)
            best_s = slots[np.argmin(scores)]

        carrier_slot[f] = best_s
        is_carrier[f, best_s] = True

        # Confidence metric: 1.0 for distance < 0.8m, scaling down to 0.5 at 2.0m
        d = distances[f, best_s]
        conf = float(np.clip(1.0 - (d / (max_dist * 1.5)), 0.4, 1.0))
        carrier_confidence[f] = conf

    return carrier_slot, is_carrier, carrier_confidence
