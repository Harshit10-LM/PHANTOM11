#!/usr/bin/env python3
"""PHANTOM-11: Missing-Ball Gap Analysis Tool

Step 5 of More Than A Hack 2027:
- Analyzes consecutive intervals where ball coordinates are unavailable (NaN).
- Examines gap durations, percentiles, and categories (short, medium, long).
- Analyzes team-control evidence immediately before and after gaps.
- Correlates gaps with Smart Tagging set pieces and restarts.
- Compares missing-ball intervals with baseline v0 predictions.
- Adheres strictly to the competition rules: no ground truth used.
"""

import os
import sys
import numpy as np
from collections import Counter

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
from baseline import (
    HOME,
    AWAY,
    NOBODY,
    team_of_each_track,
    team_names,
)


def extract_tracking_and_gaps(raw_path, meta_path, pitch, n_frames):
    """Single streaming pass to extract ball presence, coordinates, nearest team, and gaps."""
    length, width = pitch
    team_of_track = team_of_each_track(meta_path)

    # Arrays to store per-frame data
    ball_present = np.zeros(n_frames, dtype=bool)
    ball_coords = np.full((n_frames, 2), np.nan, dtype=np.float32)
    nearest_team = np.full(n_frames, NOBODY, dtype=np.int8)

    print(f"Streaming {os.path.basename(raw_path)} ({n_frames:,} frames)...")
    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            video_frame = frame_id - 1
            if video_frame < 0 or video_frame >= n_frames:
                continue

            ball_str = parts[2]
            if ball_str.startswith("N"):  # NaN
                continue

            # Ball is present
            comma1 = ball_str.index(",")
            comma2 = ball_str.index(",", comma1 + 1) if "," in ball_str[comma1 + 1:] else len(ball_str)
            bx_norm = float(ball_str[:comma1])
            by_norm = float(ball_str[comma1 + 1:comma2])

            ball_present[video_frame] = True
            ball_coords[video_frame, 0] = bx_norm * length
            ball_coords[video_frame, 1] = by_norm * width

            # Find nearest player
            best_dist = None
            best_team = NOBODY
            for slot, group in enumerate(parts[1].split(";")):
                if group.startswith("N"):
                    continue
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1) if "," in group[c1 + 1:] else len(group)
                px = float(group[:c1]) * length - ball_coords[video_frame, 0]
                py = float(group[c1 + 1:c2]) * width - ball_coords[video_frame, 1]
                dist_sq = px * px + py * py
                if best_dist is None or dist_sq < best_dist:
                    best_dist = dist_sq
                    best_team = team_of_track[slot]

            nearest_team[video_frame] = best_team

    # Identify contiguous missing-ball intervals
    # A gap is a contiguous run where ball_present == False
    gaps = []
    in_gap = False
    gap_start = 0

    for f in range(n_frames):
        if not ball_present[f]:
            if not in_gap:
                in_gap = True
                gap_start = f
        else:
            if in_gap:
                in_gap = False
                gaps.append((gap_start, f - 1))

    if in_gap:
        gaps.append((gap_start, n_frames - 1))

    return ball_present, ball_coords, nearest_team, gaps


def analyze_missing_ball(game=games.GAME):
    paths = games.asset_paths(game=game, source="tactical")
    raw_path = paths["atd_raw"]
    meta_path = paths["atd_meta"]
    pitch = games.GAMES[game]["pitch"]
    n_frames = games.source_frames("tactical", game)
    fps = games.GAMES[game]["fps"]

    home_name, away_name = team_names(meta_path)
    print("=" * 78)
    print(f"PHANTOM-11: MISSING-BALL GAP ANALYSIS ({game})")
    print(f"Match: {home_name} (H) vs {away_name} (A) | Total Frames: {n_frames:,}")
    print("=" * 78)

    # 1. Extract tracking data and gaps
    ball_present, ball_coords, nearest_team, gaps = extract_tracking_and_gaps(
        raw_path, meta_path, pitch, n_frames
    )

    n_gaps = len(gaps)
    total_missing_frames = int(np.sum(~ball_present))
    durations_frames = np.array([end - start + 1 for start, end in gaps], dtype=int)
    durations_s = durations_frames / float(fps)

    print("\n--- 1. Overall Missing-Ball Statistics ---")
    print(f"  Total Missing-Ball Frames:   {total_missing_frames:,} ({total_missing_frames / n_frames * 100:.2f}%)")
    print(f"  Total Missing-Ball Gaps:     {n_gaps:,}")
    print(f"  Mean Gap Duration:           {np.mean(durations_frames):.1f} frames ({np.mean(durations_s):.2f} s)")
    print(f"  Median Gap Duration:         {np.median(durations_frames):.1f} frames ({np.median(durations_s):.2f} s)")
    print(f"  Min / Max Gap Duration:      {np.min(durations_frames)} frames ({np.min(durations_s):.2f}s) / "
          f"{np.max(durations_frames):,} frames ({np.max(durations_s):.1f}s)")

    # 2. Duration Percentiles
    pcts = [10, 25, 50, 75, 90, 95, 99]
    perc_vals_frames = np.percentile(durations_frames, pcts)
    perc_vals_s = perc_vals_frames / float(fps)
    print("\n--- 2. Duration Percentiles ---")
    for p, vf, vs in zip(pcts, perc_vals_frames, perc_vals_s):
        print(f"  {p:>2}th percentile:              {vf:>6.1f} frames ({vs:>5.2f} s)")

    # 3. Categorization by Duration
    short_mask = durations_frames <= 25  # <= 1.0 s
    med_mask = (durations_frames > 25) & (durations_frames <= 125)  # 1.0 s - 5.0 s
    long_mask = durations_frames > 125  # > 5.0 s

    print("\n--- 3. Gap Categories ---")
    print(f"  Short gaps (<= 25 frames / <= 1.0s):   {np.sum(short_mask):>5} gaps ({np.sum(short_mask)/n_gaps*100:5.1f}%) "
          f"covering {np.sum(durations_frames[short_mask]):>6,} frames")
    print(f"  Medium gaps (26-125 frames / 1-5s):    {np.sum(med_mask):>5} gaps ({np.sum(med_mask)/n_gaps*100:5.1f}%) "
          f"covering {np.sum(durations_frames[med_mask]):>6,} frames")
    print(f"  Long gaps (> 125 frames / > 5.0s):     {np.sum(long_mask):>5} gaps ({np.sum(long_mask)/n_gaps*100:5.1f}%) "
          f"covering {np.sum(durations_frames[long_mask]):>6,} frames")

    # 4. Context Evidence Before & After Gaps
    print("\n--- 4. Team Evidence Before and After Gaps ---")
    gap_types = Counter()
    displacement_by_type = {
        "Same Team (H...H)": [],
        "Same Team (A...A)": [],
        "Turnover (H...A)": [],
        "Turnover (A...H)": [],
        "Boundary / Unknown": [],
    }

    for start, end in gaps:
        team_before = nearest_team[start - 1] if start > 0 else NOBODY
        team_after = nearest_team[end + 1] if end + 1 < n_frames else NOBODY

        # Displacement if coordinates available
        disp = None
        if start > 0 and end + 1 < n_frames and ball_present[start - 1] and ball_present[end + 1]:
            dx = ball_coords[end + 1, 0] - ball_coords[start - 1, 0]
            dy = ball_coords[end + 1, 1] - ball_coords[start - 1, 1]
            disp = float(np.sqrt(dx * dx + dy * dy))

        if team_before == HOME and team_after == HOME:
            cat = "Same Team (H...H)"
            gap_types["H...H"] += 1
        elif team_before == AWAY and team_after == AWAY:
            cat = "Same Team (A...A)"
            gap_types["A...A"] += 1
        elif team_before == HOME and team_after == AWAY:
            cat = "Turnover (H...A)"
            gap_types["H...A"] += 1
        elif team_before == AWAY and team_after == HOME:
            cat = "Turnover (A...H)"
            gap_types["A...H"] += 1
        else:
            cat = "Boundary / Unknown"
            gap_types["Boundary / Unknown"] += 1

        if disp is not None:
            displacement_by_type[cat].append((end - start + 1, disp))

    for gtype, count in gap_types.most_common():
        print(f"  {gtype:<20}: {count:>5} gaps ({count/n_gaps*100:5.1f}%)")

    # 5. Smart Tagging Overlap Analysis
    print("\n--- 5. Overlap with Smart Tagging (Set Pieces & Restarts) ---")
    st_events = load_smart_tagging.load(game=game)
    set_pieces = [e for e in st_events if e["code"] == "SET PIECES"]
    print(f"  Total Smart Tagging events:  {len(st_events)}")
    print(f"  Total SET PIECES events:     {len(set_pieces)}")

    # Check how many gaps overlap with set pieces
    gaps_overlapping_sp = 0
    sp_types_in_gaps = Counter()

    for start, end in gaps:
        overlap = False
        for sp in set_pieces:
            sp_start, sp_end = sp["start_frame"], sp["end_frame"]
            if not (end < sp_start or start > sp_end):
                overlap = True
                sp_type = sp["tags"].get("Type", "Unknown")
                sp_types_in_gaps[sp_type] += 1
        if overlap:
            gaps_overlapping_sp += 1

    print(f"  Missing-ball gaps overlapping SET PIECES: {gaps_overlapping_sp} of {n_gaps} ({gaps_overlapping_sp/n_gaps*100:.1f}%)")
    print("  Set Piece types observed during gaps:")
    for spt, cnt in sp_types_in_gaps.most_common(8):
        print(f"    - {spt:<24}: {cnt}")

    # 6. Comparison with Baseline v0 Predictions
    print("\n--- 6. Missing-Ball Gaps vs. Baseline v0 Predictions ---")
    v0_path = os.path.join(SCRIPT_DIR, "outputs", "phantom11_v0.csv")
    if os.path.isfile(v0_path):
        import pandas as pd
        v0_df = pd.read_csv(v0_path)
        v0_labels = v0_df["label"].values

        missing_mask = ~ball_present
        v0_on_missing = v0_labels[missing_mask]
        v0_on_present = v0_labels[ball_present]

        print("  On missing-ball frames (total 85,831 frames):")
        for lbl in ["H", "A", "D", "X"]:
            cnt = int(np.sum(v0_on_missing == lbl))
            print(f"    v0 predicted {lbl}: {cnt:>6,} frames ({cnt/total_missing_frames*100:5.1f}%)")

        print("\n  On ball-visible frames (total 113,488 frames):")
        for lbl in ["H", "A", "D", "X"]:
            cnt = int(np.sum(v0_on_present == lbl))
            print(f"    v0 predicted {lbl}: {cnt:>6,} frames ({cnt/len(v0_on_present)*100:5.1f}%)")
    else:
        print("  phantom11_v0.csv not found; skipping baseline comparison.")

    print("\n" + "=" * 78)
    print("MISSING-BALL GAP ANALYSIS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    analyze_missing_ball()
