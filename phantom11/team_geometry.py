#!/usr/bin/env python3
"""PHANTOM-11: Team Geometry and Spatial Support Module

Step 6 of More Than A Hack 2027:
- Computes permutation-invariant team-level spatial metrics around the ball:
  1. Counts of Home and Away players within 1.5m, 3.0m, and 6.0m radii.
  2. Nearest distances to the closest few players of each team (dH1, dH2, dA1, dA2).
  3. Spatial control margin (dA1 - dH1) in physical metres.
  4. Team-level radial convergence towards the ball (closing down / rushing onto pass).
  5. Geometric control state (Unambiguous Home, Unambiguous Away, Contested Duel, Loose Ball).
- Permutation-invariant: depends only on team membership, not on player tracking slot indices.
- Purely deterministic and reproducible.
"""

import numpy as np


# Physical radius constants (in metres)
RADIUS_IMMEDIATE_M = 1.50   # Immediate tackle / control sphere
RADIUS_PRESSURE_M = 3.00    # Direct pressure / challenge sphere
RADIUS_SUPPORT_M = 6.00     # Tactical support / local overload sphere

# Large default distance for unobserved players
DEFAULT_MAX_DIST_M = 100.0


def compute_team_geometry(
    distances,
    team_of_track,
    ball_xy,
    ball_present,
    players_xy,
    player_vel,
):
    """Compute permutation-invariant team-level spatial and movement features around the ball.

    Parameters
    ----------
    distances : np.ndarray of shape (N, S), float32
        Distance in metres from ball to each player slot.
    team_of_track : np.ndarray of shape (S,), int8
        0 for Home, 1 for Away.
    ball_xy : np.ndarray of shape (N, 2), float32
    ball_present : np.ndarray of shape (N,), bool
    players_xy : np.ndarray of shape (N, S, 2), float32
    player_vel : np.ndarray of shape (N, S, 2), float32

    Returns
    -------
    dict with team-level geometry features:
        'd_h1': (N,) float32 - Closest Home player distance (metres)
        'd_h2': (N,) float32 - Second closest Home player distance
        'd_a1': (N,) float32 - Closest Away player distance
        'd_a2': (N,) float32 - Second closest Away player distance
        'margin_d': (N,) float32 - (d_a1 - d_h1) in metres (>0 favours Home, <0 favours Away)
        'n_home_1_5m': (N,) int16 - Home players within 1.5m
        'n_away_1_5m': (N,) int16 - Away players within 1.5m
        'n_home_3m': (N,) int16 - Home players within 3.0m
        'n_away_3m': (N,) int16 - Away players within 3.0m
        'n_home_6m': (N,) int16 - Home players within 6.0m
        'n_away_6m': (N,) int16 - Away players within 6.0m
        'home_convergence': (N,) float32 - Max radial velocity of Home player towards ball (m/s)
        'away_convergence': (N,) float32 - Max radial velocity of Away player towards ball (m/s)
        'control_state': (N,) int8 - 0: Unambiguous Home, 1: Unambiguous Away,
                                     2: Contested Duel, 3: Loose Ball, -1: Ball Missing
    """
    n_frames, n_slots = distances.shape

    home_mask = (team_of_track == 0)
    away_mask = (team_of_track == 1)

    # Distances split by team with NaNs replaced by DEFAULT_MAX_DIST_M
    d_clean = np.where(np.isnan(distances), DEFAULT_MAX_DIST_M, distances)

    d_home = d_clean[:, home_mask]
    d_away = d_clean[:, away_mask]

    # 1. Sorted distances to nearest players
    sorted_h = np.sort(d_home, axis=1)
    sorted_a = np.sort(d_away, axis=1)

    d_h1 = sorted_h[:, 0]
    d_h2 = sorted_h[:, 1] if sorted_h.shape[1] > 1 else np.full(n_frames, DEFAULT_MAX_DIST_M, dtype=np.float32)

    d_a1 = sorted_a[:, 0]
    d_a2 = sorted_a[:, 1] if sorted_a.shape[1] > 1 else np.full(n_frames, DEFAULT_MAX_DIST_M, dtype=np.float32)

    margin_d = d_a1 - d_h1

    # When ball is missing, distances are invalid
    d_h1 = np.where(ball_present, d_h1, np.nan)
    d_h2 = np.where(ball_present, d_h2, np.nan)
    d_a1 = np.where(ball_present, d_a1, np.nan)
    d_a2 = np.where(ball_present, d_a2, np.nan)
    margin_d = np.where(ball_present, margin_d, np.nan)

    # 2. Player count density within radii
    n_home_1_5m = np.sum((d_home <= RADIUS_IMMEDIATE_M), axis=1).astype(np.int16)
    n_away_1_5m = np.sum((d_away <= RADIUS_IMMEDIATE_M), axis=1).astype(np.int16)

    n_home_3m = np.sum((d_home <= RADIUS_PRESSURE_M), axis=1).astype(np.int16)
    n_away_3m = np.sum((d_away <= RADIUS_PRESSURE_M), axis=1).astype(np.int16)

    n_home_6m = np.sum((d_home <= RADIUS_SUPPORT_M), axis=1).astype(np.int16)
    n_away_6m = np.sum((d_away <= RADIUS_SUPPORT_M), axis=1).astype(np.int16)

    # Zero out counts on missing ball frames
    n_home_1_5m[~ball_present] = 0
    n_away_1_5m[~ball_present] = 0
    n_home_3m[~ball_present] = 0
    n_away_3m[~ball_present] = 0
    n_home_6m[~ball_present] = 0
    n_away_6m[~ball_present] = 0

    # 3. Team-level radial convergence towards ball
    # Vector from player to ball: r = ball_xy - players_xy
    b_exp = ball_xy[:, np.newaxis, :]  # (N, 1, 2)
    vec_to_ball = b_exp - players_xy    # (N, S, 2)
    dist_safe = np.where(distances < 0.1, 0.1, distances)[:, :, np.newaxis]
    unit_vec = vec_to_ball / dist_safe

    # Radial velocity = dot product of player_vel and unit_vec
    # Positive = moving towards ball, Negative = moving away
    rad_vel = np.sum(player_vel * unit_vec, axis=2)  # (N, S)

    # Only consider players within support radius (<= 6m) for active convergence
    within_support = (distances <= RADIUS_SUPPORT_M) & (~np.isnan(rad_vel))
    rad_vel_support = np.where(within_support, rad_vel, -100.0)

    # Max convergence speed for each team
    rad_vel_h = rad_vel_support[:, home_mask]
    rad_vel_a = rad_vel_support[:, away_mask]

    home_convergence = np.max(rad_vel_h, axis=1)
    away_convergence = np.max(rad_vel_a, axis=1)

    # Clamp floor at 0.0 m/s
    home_convergence = np.where(ball_present & (home_convergence > 0.0), home_convergence, 0.0)
    away_convergence = np.where(ball_present & (away_convergence > 0.0), away_convergence, 0.0)

    # 4. Geometric control state
    # 0: Unambiguous Home (dH1 < 1.5m and dA1 > 2.5m)
    # 1: Unambiguous Away (dA1 < 1.5m and dH1 > 2.5m)
    # 2: Contested Duel (both within 2.5m and |margin_d| <= 1.0m)
    # 3: Loose Ball (both > 3.0m)
    # 4: Asymmetric Proximity (one is closer, but not in immediate 1.5m range)
    # -1: Ball Missing
    control_state = np.full(n_frames, -1, dtype=np.int8)

    valid_frames = ball_present & ~np.isnan(d_h1) & ~np.isnan(d_a1)

    # Loose ball
    is_loose = valid_frames & (d_h1 > RADIUS_PRESSURE_M) & (d_a1 > RADIUS_PRESSURE_M)
    control_state[is_loose] = 3

    # Contested duel
    is_contested = (
        valid_frames
        & ~is_loose
        & (d_h1 <= RADIUS_PRESSURE_M)
        & (d_a1 <= RADIUS_PRESSURE_M)
        & (np.abs(margin_d) <= 1.0)
    )
    control_state[is_contested] = 2

    # Unambiguous Home
    is_unambig_h = (
        valid_frames
        & ~is_loose
        & ~is_contested
        & (d_h1 <= RADIUS_IMMEDIATE_M)
        & (d_a1 >= 2.5)
    )
    control_state[is_unambig_h] = 0

    # Unambiguous Away
    is_unambig_a = (
        valid_frames
        & ~is_loose
        & ~is_contested
        & (d_a1 <= RADIUS_IMMEDIATE_M)
        & (d_h1 >= 2.5)
    )
    control_state[is_unambig_a] = 1

    # Asymmetric proximity (clear closer player between 1.5m and 3m)
    is_closer_h = valid_frames & (control_state == -1) & (margin_d >= 1.0)
    control_state[is_closer_h] = 0

    is_closer_a = valid_frames & (control_state == -1) & (margin_d <= -1.0)
    control_state[is_closer_a] = 1

    # Remaining ambiguous frames
    is_remaining = valid_frames & (control_state == -1)
    control_state[is_remaining] = 2  # treat unresolved as contested

    return {
        "d_h1": d_h1,
        "d_h2": d_h2,
        "d_a1": d_a1,
        "d_a2": d_a2,
        "margin_d": margin_d,
        "n_home_1_5m": n_home_1_5m,
        "n_away_1_5m": n_away_1_5m,
        "n_home_3m": n_home_3m,
        "n_away_3m": n_away_3m,
        "n_home_6m": n_home_6m,
        "n_away_6m": n_away_6m,
        "home_convergence": home_convergence.astype(np.float32),
        "away_convergence": away_convergence.astype(np.float32),
        "control_state": control_state,
    }
