#!/usr/bin/env python3
"""Audit possession fragmentation between v5 and v5-A across BAR-ATH full match (199,319 frames)."""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from phantom11.match_config import MatchConfig
from phantom11.load_tracking import load_full_tracking
from phantom11.pitch_boundary import compute_out_of_bounds_distance
from phantom11.boundary_refinement import extract_spell_intervals

cfg = MatchConfig.from_official_game()
ball_xy, ball_present, players_xy, _, team_of_track, pitch, _ = load_full_tracking(match_config=cfg)
d_oob = compute_out_of_bounds_distance(ball_xy, ball_present, pitch=pitch)

v5 = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "phantom11_v5.csv"))["label"].values
v5a = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "phantom11_v5a.csv"))["label"].values

diff_idx = np.where(v5 != v5a)[0]
print(f"Total modified frames: {len(diff_idx)}")

# Group into continuous intervals
intervals = []
cur_start = diff_idx[0]
cur_end = diff_idx[0]
for idx in diff_idx[1:]:
    if idx == cur_end + 1:
        cur_end = idx
    else:
        intervals.append((cur_start, cur_end, cur_end - cur_start + 1))
        cur_start = idx
        cur_end = idx
intervals.append((cur_start, cur_end, cur_end - cur_start + 1))

print(f"Total continuous modified intervals: {len(intervals)}")

# Analyze context of each interval in v5 vs v5a
categories = {
    "H -> D -> H (Same-team split)": [],
    "A -> D -> A (Same-team split)": [],
    "H -> D -> A (Turnover transition)": [],
    "A -> D -> H (Turnover transition)": [],
    "Other/Adjacent to D": [],
}

detailed_records = []

for s, e, l in intervals:
    # Context before and after
    lbl_before = v5[s - 1] if s > 0 else "D"
    lbl_after = v5[e + 1] if e < len(v5) - 1 else "D"
    lbl_mid_v5 = v5[s] # all frames in interval have same v5 label
    
    # Boundary type
    bx = np.nanmean(ball_xy[s:e+1, 0])
    by = np.nanmean(ball_xy[s:e+1, 1])
    mean_d = np.mean(d_oob[s:e+1])
    max_d = np.max(d_oob[s:e+1])
    dur_s = l / 25.0

    if by > 68.0:
        b_type = "Top Touchline (Y>68)"
    elif by < 0.0:
        b_type = "Bottom Touchline (Y<0)"
    elif bx < 0.0:
        b_type = "Left Goal Line (X<0)"
    else:
        b_type = "Right Goal Line (X>105.3)"

    # Classify transition
    cat_key = None
    if lbl_before == "H" and lbl_after == "H":
        cat_key = "H -> D -> H (Same-team split)"
    elif lbl_before == "A" and lbl_after == "A":
        cat_key = "A -> D -> A (Same-team split)"
    elif lbl_before == "H" and lbl_after == "A":
        cat_key = "H -> D -> A (Turnover transition)"
    elif lbl_before == "A" and lbl_after == "H":
        cat_key = "A -> D -> H (Turnover transition)"
    else:
        cat_key = "Other/Adjacent to D"

    categories[cat_key].append((s, e, l, dur_s, mean_d, max_d, b_type))
    detailed_records.append({
        "start": s,
        "end": e,
        "len": l,
        "dur_s": dur_s,
        "cat": cat_key,
        "before": lbl_before,
        "v5": lbl_mid_v5,
        "after": lbl_after,
        "b_type": b_type,
        "mean_d": mean_d,
        "max_d": max_d,
    })

print("\n" + "=" * 80)
print("FRAGMENTATION BREAKDOWN BY TRANSITION TYPE")
print("=" * 80)
for k, v in categories.items():
    durations = [item[3] for item in v]
    short_count = sum(1 for d in durations if d <= 1.0)
    ultra_short_count = sum(1 for d in durations if d < 0.5)
    mean_dur = np.mean(durations) if durations else 0.0
    tot_frames = sum(item[2] for item in v)
    print(f"\n{k}: {len(v)} intervals ({tot_frames:,} frames)")
    if durations:
        print(f"  Duration: mean={mean_dur:.2f}s, min={min(durations):.2f}s, max={max(durations):.2f}s")
        print(f"  Short intervals (<= 1.0s): {short_count} ({short_count/len(v)*100:.1f}%)")
        print(f"  Ultra-short (< 0.5s):       {ultra_short_count} ({ultra_short_count/len(v)*100:.1f}%)")

print("\n" + "=" * 80)
print("AUDIT OF ULTRA-SHORT INTERVALS (< 0.5s / 12 FRAMES)")
print("=" * 80)
ultra_short = [r for r in detailed_records if r["dur_s"] < 0.5]
print(f"Total ultra-short out-of-bounds intervals: {len(ultra_short)}")
for r in ultra_short:
    print(f"  Frames [{r['start']:,} .. {r['end']:,}] ({r['len']:2d}f, {r['dur_s']:.2f}s): {r['before']} -> D -> {r['after']} | {r['b_type']} | d_oob: mean={r['mean_d']:.2f}m, max={r['max_d']:.2f}m")
