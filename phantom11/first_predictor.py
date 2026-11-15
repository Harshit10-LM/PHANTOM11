#!/usr/bin/env python3
"""PHANTOM-11: First Baseline Possession Predictor (v0)

Step 4 of More Than A Hack 2027:
- Reuses official baseline logic: nearest player to the ball scaled to pitch dimensions.
- Applies the official 50-frame majority-vote smoothing window (+/- 2 seconds at 25 fps).
- Preserves strict video frame alignment: video_frame = frame_id - 1 (0 to 199,318).
- Saves formatted submission to phantom11/outputs/phantom11_v0.csv.
- Computes comprehensive descriptive diagnostics.
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
from baseline import (
    HOME,
    AWAY,
    NOBODY,
    LABELS,
    build,
    smooth,
    write_submission,
)


def compute_diagnostics(raw_codes, smoothed_labels, fps=25):
    """Compute detailed descriptive statistics on predicted possession sequence."""
    n_frames = len(smoothed_labels)
    counts = {lbl: int(np.sum(smoothed_labels == lbl)) for lbl in ["H", "A", "D", "X"]}

    # Identify possession segments (maximal continuous runs of H or A)
    # Stoppages (D or X) separate possessions.
    segments = []
    current_label = None
    start_idx = 0

    for i, label in enumerate(smoothed_labels):
        if label in ("H", "A"):
            if current_label is None:
                current_label = label
                start_idx = i
            elif label != current_label:
                # Direct switch between H and A
                segments.append((current_label, start_idx, i, i - start_idx))
                current_label = label
                start_idx = i
        else:
            if current_label is not None:
                segments.append((current_label, start_idx, i, i - start_idx))
                current_label = None

    if current_label is not None:
        segments.append((current_label, start_idx, n_frames, n_frames - start_idx))

    h_segments = [seg for seg in segments if seg[0] == "H"]
    a_segments = [seg for seg in segments if seg[0] == "A"]
    all_durations_s = [seg[3] / float(fps) for seg in segments]

    mean_duration = float(np.mean(all_durations_s)) if all_durations_s else 0.0
    median_duration = float(np.median(all_durations_s)) if all_durations_s else 0.0

    # Number of direct H/A switches (turnovers without an intervening D/X frame)
    ha_switches = 0
    for i in range(len(smoothed_labels) - 1):
        c1, c2 = smoothed_labels[i], smoothed_labels[i + 1]
        if (c1 == "H" and c2 == "A") or (c1 == "A" and c2 == "H"):
            ha_switches += 1

    # Analysis of missing-ball frames receiving H or A after smoothing
    missing_ball_mask = (raw_codes == NOBODY)
    n_missing = int(np.sum(missing_ball_mask))
    smoothed_on_missing = smoothed_labels[missing_ball_mask]
    n_missing_given_ha = int(np.sum((smoothed_on_missing == "H") | (smoothed_on_missing == "A")))
    pct_missing_given_ha = (n_missing_given_ha / n_missing * 100.0) if n_missing > 0 else 0.0

    return {
        "n_frames": n_frames,
        "counts": counts,
        "n_segments": len(segments),
        "n_h_segments": len(h_segments),
        "n_a_segments": len(a_segments),
        "mean_duration_s": mean_duration,
        "median_duration_s": median_duration,
        "ha_switches": ha_switches,
        "n_missing_ball": n_missing,
        "n_missing_given_ha": n_missing_given_ha,
        "pct_missing_given_ha": pct_missing_given_ha,
    }


def main():
    print("=" * 78)
    print("PHANTOM-11: First Baseline Possession Predictor (v0)")
    print("=" * 78)

    # 1. Generate raw nearest-player codes using official baseline build
    print("\n[1] Extracting raw nearest player to ball across all frames...")
    raw_codes = build()

    # 2. Apply official 50-frame smoothing half-window
    window = 50
    print(f"\n[2] Applying centered majority-vote smoothing (half-window = +/-{window} frames, +/-2.0s)...")
    smoothed_labels = smooth(raw_codes, window)

    # 3. Save to phantom11/outputs/phantom11_v0.csv
    output_dir = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phantom11_v0.csv")

    print(f"\n[3] Writing submission CSV to {output_path}...")
    write_submission(output_path, smoothed_labels)
    print("    Done writing file.")

    # 4. Verify basic file integrity
    print("\n[4] Basic file format checks:")
    print(f"    Total data rows: {len(smoothed_labels):,}")
    print(f"    Frame index:     0 .. {len(smoothed_labels) - 1}")
    allowed_labels = {"H", "A", "D", "X"}
    unique_labels = set(smoothed_labels)
    print(f"    Unique labels:   {unique_labels} (all valid: {unique_labels.issubset(allowed_labels)})")

    # 5. Compute and print descriptive diagnostics
    print("\n[5] Descriptive Diagnostics:")
    diag = compute_diagnostics(raw_codes, smoothed_labels)
    print(f"    Total Frames:                {diag['n_frames']:,}")
    print(f"    Predicted 'H' (Barcelona):   {diag['counts']['H']:,} ({diag['counts']['H']/diag['n_frames']*100:.2f}%)")
    print(f"    Predicted 'A' (Athletic):    {diag['counts']['A']:,} ({diag['counts']['A']/diag['n_frames']*100:.2f}%)")
    print(f"    Predicted 'D' (Dead Ball):   {diag['counts']['D']:,} ({diag['counts']['D']/diag['n_frames']*100:.2f}%)")
    print(f"    Predicted 'X' (Abstain):     {diag['counts']['X']:,} ({diag['counts']['X']/diag['n_frames']*100:.2f}%)")
    print(f"    Total Possession Spells:     {diag['n_segments']:,} (H: {diag['n_h_segments']:,}, A: {diag['n_a_segments']:,})")
    print(f"    Mean Possession Duration:    {diag['mean_duration_s']:.2f} seconds")
    print(f"    Median Possession Duration:  {diag['median_duration_s']:.2f} seconds")
    print(f"    Direct H/A Switches:         {diag['ha_switches']:,}")
    print(f"    Missing-Ball Frames (raw):   {diag['n_missing_ball']:,}")
    print(f"    Missing-Ball Frames Given H/A after smoothing: {diag['n_missing_given_ha']:,} ({diag['pct_missing_given_ha']:.2f}%)")

    print("\n" + "=" * 78)
    print("PHANTOM-11 v0 GENERATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
