#!/usr/bin/env python3
"""PHANTOM-11: Fast Structured Tracking Data Loader

Step 6 of More Than A Hack 2027:
- Reads the 199,319 frames of tactical tracking data into structured numpy arrays.
- Memory efficient: float32 coordinates, boolean presence masks (< 90 MB total).
- Reuses official games.py asset paths and baseline.py team mapping.
- Validates frame count and alignment: video_frame = frame_id - 1.
"""

import os
import sys
import numpy as np

# Ensure repository root and starter directory are on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

try:
    from phantom11.path_resolver import resolve_official_repo, resolve_data_dir
except ImportError:
    try:
        from path_resolver import resolve_official_repo, resolve_data_dir
    except ImportError:
        resolve_official_repo, resolve_data_dir = None, None

try:
    import games
    from baseline import team_of_each_track
except ImportError:
    if resolve_official_repo:
        resolve_official_repo(auto_fetch=False)
        import games
        from baseline import team_of_each_track
    else:
        raise


def load_full_tracking(game=None, data_dir=None, match_config=None):
    """Load complete tactical tracking data into memory.

    Parameters
    ----------
    game : str, optional (defaults to games.GAME, 'BAR-ATH')
    data_dir : str, optional (path to directory containing tactical files)
    match_config : MatchConfig, optional (reusable match configuration)

    Returns
    -------
    ball_xy : np.ndarray of shape (N, 2), float32 (in metres, NaN when unobserved)
    ball_present : np.ndarray of shape (N,), bool
    players_xy : np.ndarray of shape (N, S, 2), float32 (in metres, NaN when unobserved)
    players_present : np.ndarray of shape (N, S), bool
    team_of_track : np.ndarray of shape (S,), int8 (0=Home, 1=Away)
    pitch : tuple of (length, width) in metres
    n_frames : int
    """
    if match_config is not None:
        game = match_config.match_id
        data_dir = data_dir or match_config.data_dir
        pitch = match_config.pitch
        n_frames = match_config.n_frames
    else:
        game = game or games.GAME
        pitch = games.GAMES[game]["pitch"]
        n_frames = games.source_frames("tactical", game)

    if data_dir is None and resolve_data_dir:
        try:
            data_dir = resolve_data_dir()
        except FileNotFoundError:
            pass

    paths = games.asset_paths(game=game, source="tactical", data_dir=data_dir)
    raw_path = paths["atd_raw"]
    meta_path = paths["atd_meta"]
    length, width = pitch

    if match_config is not None and match_config.team_of_track is not None:
        team_of_track = match_config.team_of_track
    else:
        team_of_track = team_of_each_track(meta_path)

    n_slots = len(team_of_track)

    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_present = np.zeros(n_frames, dtype=bool)
    players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    players_present = np.zeros((n_frames, n_slots), dtype=bool)

    print(f"Loading tracking data from {os.path.basename(raw_path)} ({n_frames:,} frames, {n_slots} slots)...")
    with open(raw_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            video_frame = frame_id - 1
            if video_frame < 0 or video_frame >= n_frames:
                continue

            # Ball
            ball_str = parts[2]
            if not ball_str.startswith("N"):
                c1 = ball_str.index(",")
                c2 = ball_str.index(",", c1 + 1) if "," in ball_str[c1 + 1:] else len(ball_str)
                bx = float(ball_str[:c1]) * length
                by = float(ball_str[c1 + 1:c2]) * width
                ball_xy[video_frame, 0] = bx
                ball_xy[video_frame, 1] = by
                ball_present[video_frame] = True

            # Players (dynamic slots)
            player_tokens = parts[1].split(";")
            for slot, p_str in enumerate(player_tokens):
                if slot >= n_slots:
                    break
                if not p_str.startswith("N"):
                    c1 = p_str.index(",")
                    c2 = p_str.index(",", c1 + 1) if "," in p_str[c1 + 1:] else len(p_str)
                    px = float(p_str[:c1]) * length
                    py = float(p_str[c1 + 1:c2]) * width
                    players_xy[video_frame, slot, 0] = px
                    players_xy[video_frame, slot, 1] = py
                    players_present[video_frame, slot] = True

    print(f"Tracking loaded successfully: {int(np.sum(ball_present)):,} frames with ball visible.")
    return ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, n_frames
