#!/usr/bin/env python3
"""PHANTOM-11: Version 1 Predictor (Conservative Missing-Ball Continuity)

Step 5 of More Than A Hack 2027:
- Generates/loads baseline v0.
- Applies conservative continuity module from phantom11/continuity.py.
- Saves result to phantom11/outputs/phantom11_v1.csv (preserving v0 unchanged).
- Computes comprehensive comparative diagnostics against v0.
"""

import os
import sys
import numpy as np
import pandas as pd
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
    LABELS,
    build,
    smooth,
    write_submission,
)
from continuity import apply_continuity, NON_PLAY_INTERVALS
from missing_ball_analysis import extract_tracking_and_gaps


def get_possession_spells(labels, fps=25):
    """Compute runs of H or A (stoppages D or X split possessions)."""
    spells = []
    current_label = None
    start_idx = 0

    for i, lbl in enumerate(labels):
        if lbl in ("H", "A"):
            if current_label is None:
                current_label = lbl
                start_idx = i
            elif lbl != current_label:
                spells.append((current_label, start_idx, i - 1, (i - start_idx) / float(fps)))
                current_label = lbl
                start_idx = i
        else:
            if current_label is not None:
                spells.append((current_label, start_idx, i - 1, (i - start_idx) / float(fps)))
                current_label = None

    if current_label is not None:
        spells.append((current_label, start_idx, len(labels) - 1, (len(labels) - start_idx) / float(fps)))

    return spells


def get_dead_ball_intervals(labels, fps=25):
    """Compute contiguous intervals of dead ball (D)."""
    intervals = []
    in_d = False
    start_idx = 0

    for i, lbl in enumerate(labels):
        if lbl == "D":
            if not in_d:
                in_d = True
                start_idx = i
        else:
            if in_d:
                in_d = False
                intervals.append((start_idx, i - 1, (i - start_idx) / float(fps)))

    if in_d:
        intervals.append((start_idx, len(labels) - 1, (len(labels) - start_idx) / float(fps)))

    return intervals


def count_ha_switches(labels):
    """Count immediate H -> A or A -> H turnovers without intervening D."""
    switches = 0
    for i in range(len(labels) - 1):
        c1, c2 = labels[i], labels[i + 1]
        if (c1 == "H" and c2 == "A") or (c1 == "A" and c2 == "H"):
            switches += 1
    return switches


def run_predictor_v1():
    print("=" * 78)
    print("PHANTOM-11: PREDICTOR VERSION 1 (CONTINUITY EXPERIMENT)")
    print("=" * 78)

    # 1. Load or generate baseline v0
    v0_path = os.path.join(SCRIPT_DIR, "outputs", "phantom11_v0.csv")
    if os.path.isfile(v0_path):
        print(f"\n[1] Loading existing baseline v0 from {v0_path}...")
        v0_df = pd.read_csv(v0_path)
        v0_labels = v0_df["label"].values.astype("<U1")
    else:
        print("\n[1] Building baseline v0...")
        raw_codes = build()
        v0_labels = smooth(raw_codes, 50)
        write_submission(v0_path, v0_labels)

    # 2. Extract tracking data for continuity module
    print("\n[2] Extracting tracking coordinates and missing-ball intervals...")
    paths = games.asset_paths(source="tactical")
    pitch = games.GAMES["BAR-ATH"]["pitch"]
    n_frames = games.source_frames("tactical")

    ball_present, ball_coords, nearest_team, gaps = extract_tracking_and_gaps(
        paths["atd_raw"], paths["atd_meta"], pitch, n_frames
    )

    # 3. Load Smart Tagging set pieces
    print("\n[3] Loading Smart Tagging set pieces as supporting evidence...")
    st_events = load_smart_tagging.load(game="BAR-ATH")
    set_pieces_intervals = [
        (e["start_frame"], e["end_frame"])
        for e in st_events
        if e["code"] in ("SET PIECES", "GOALS")
    ]
    print(f"    Loaded {len(set_pieces_intervals)} set-piece / restart stoppage windows.")

    # 4. Apply Conservative Continuity Module
    print("\n[4] Applying conservative continuity algorithm...")
    v1_labels, diag = apply_continuity(
        v0_labels=v0_labels,
        ball_present=ball_present,
        ball_coords=ball_coords,
        nearest_team=nearest_team,
        set_pieces_intervals=set_pieces_intervals,
        pitch=pitch,
        non_play_intervals=NON_PLAY_INTERVALS,
    )

    # 5. Save PHANTOM-11 v1 output
    v1_path = os.path.join(SCRIPT_DIR, "outputs", "phantom11_v1.csv")
    print(f"\n[5] Writing PHANTOM-11 v1 output to {v1_path}...")
    write_submission(v1_path, v1_labels)
    print("    Saved successfully (v0 preserved intact).")

    # 6. Detailed Comparative Diagnostics (v0 vs v1)
    print("\n" + "=" * 78)
    print("COMPARATIVE DIAGNOSTICS: v0 vs v1 (NO GROUND TRUTH ACCURACY USED)")
    print("=" * 78)

    # Label counts
    v0_counts = Counter(v0_labels)
    v1_counts = Counter(v1_labels)

    print("\n--- A. Overall Label Distribution ---")
    print(f"{'Label':<8} {'v0 Frames':<14} {'v0 Pct':<10} {'v1 Frames':<14} {'v1 Pct':<10} {'Difference'}")
    print("-" * 68)
    for lbl in ["H", "A", "D", "X"]:
        c0 = v0_counts[lbl]
        c1 = v1_counts[lbl]
        diff = c1 - c0
        sign = "+" if diff > 0 else ""
        print(f"{lbl:<8} {c0:<14,} {c0/n_frames*100:6.2f}%   {c1:<14,} {c1/n_frames*100:6.2f}%   {sign}{diff:,}")

    # Changes breakdown
    diff_mask = (v0_labels != v1_labels)
    total_changed = int(np.sum(diff_mask))
    print(f"\n--- B. Label Transitions (v0 -> v1) ---")
    print(f"  Total Frames Changed: {total_changed:,} ({total_changed / n_frames * 100:.2f}%)")

    transition_counts = Counter()
    for o, n in zip(v0_labels[diff_mask], v1_labels[diff_mask]):
        transition_counts[f"{o} -> {n}"] += 1

    for trans, count in transition_counts.most_common():
        print(f"    {trans:<12}: {count:>6,} frames")

    # Possession Spells
    v0_spells = get_possession_spells(v0_labels)
    v1_spells = get_possession_spells(v1_labels)

    v0_durations = [s[3] for s in v0_spells]
    v1_durations = [s[3] for s in v1_spells]

    print("\n--- C. Possession Structure (Spells) ---")
    print(f"  v0 Total Possessions:        {len(v0_spells):,} (H: {sum(1 for s in v0_spells if s[0]=='H')}, A: {sum(1 for s in v0_spells if s[0]=='A')})")
    print(f"  v1 Total Possessions:        {len(v1_spells):,} (H: {sum(1 for s in v1_spells if s[0]=='H')}, A: {sum(1 for s in v1_spells if s[0]=='A')})")
    print(f"  v0 Mean / Median Duration:   {np.mean(v0_durations):.2f}s / {np.median(v0_durations):.2f}s")
    print(f"  v1 Mean / Median Duration:   {np.mean(v1_durations):.2f}s / {np.median(v1_durations):.2f}s")

    # Turnovers (Direct H <-> A switches)
    v0_switches = count_ha_switches(v0_labels)
    v1_switches = count_ha_switches(v1_labels)
    print(f"  Direct H <-> A Switches:     v0: {v0_switches:,} -> v1: {v1_switches:,} ({v1_switches - v0_switches:+d})")

    # Dead Ball Intervals
    v0_d_intervals = get_dead_ball_intervals(v0_labels)
    v1_d_intervals = get_dead_ball_intervals(v1_labels)
    v0_d_durs = [d[2] for d in v0_d_intervals]
    v1_d_durs = [d[2] for d in v1_d_intervals]

    print("\n--- D. Dead Ball (D) Stoppages ---")
    print(f"  v0 Stoppage Intervals:       {len(v0_d_intervals):,} (mean {np.mean(v0_d_durs):.2f}s, median {np.median(v0_d_durs):.2f}s)")
    print(f"  v1 Stoppage Intervals:       {len(v1_d_intervals):,} (mean {np.mean(v1_d_durs):.2f}s, median {np.median(v1_d_durs):.2f}s)")

    # Continuity Module Decisions
    print("\n--- E. Continuity Decisions on Missing-Ball Gaps ---")
    print(f"  Total Missing-Ball Gaps:     {diag['total_gaps']:,}")
    print(f"  Gaps Recovered (Continuity): {diag['decisions']['RECOVER_CONTINUITY']:,}")
    print(f"    - Recovered to Home (H):   {diag['frames_recovered_to_h']:,} frames")
    print(f"    - Recovered to Away (A):   {diag['frames_recovered_to_a']:,} frames")
    print(f"  Ambiguous Turnovers Left Unchanged: {diag['decisions']['AMBIGUOUS_TURNOVER']:,}")
    print(f"  Set Pieces Preserved as Stoppage:   {diag['decisions']['PRESERVE_SET_PIECE']:,}")
    print(f"  Non-Play Gaps Preserved as Stoppage:{diag['decisions']['PRESERVE_NON_PLAY']:,}")
    print(f"  Gaps Exceeding Max Duration:        {diag['decisions']['EXCEEDS_MAX_DURATION']:,}")

    # Halftime Boundary Verification
    print("\n--- F. Match Boundary & Halftime Integrity Verification ---")
    ht_s, ht_e = 90652, 110807
    ht_v1_d = int(np.sum(v1_labels[ht_s:ht_e + 1] == "D"))
    ht_total = ht_e - ht_s + 1
    print(f"  Halftime Frames (90,652..110,807): {ht_total:,} frames")
    print(f"  Halftime Preserved as Dead Ball:   {ht_v1_d:,} of {ht_total:,} ({ht_v1_d/ht_total*100:.1f}%)")
    print(f"  Halftime Boundaries Intact:        {'YES (Cleanly Clamped)' if ht_v1_d == ht_total else 'NO'}")

    print("\n" + "=" * 78)
    print("PHANTOM-11 v1 EXECUTION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    run_predictor_v1()
