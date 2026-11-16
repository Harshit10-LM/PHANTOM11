#!/usr/bin/env python3
"""Audit sample of 3,577 pitch-boundary corrections in BAR-ATH full match."""

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
import load_smart_tagging

cfg = MatchConfig.from_official_game()
ball_xy, ball_present, players_xy, _, team_of_track, pitch, _ = load_full_tracking(match_config=cfg)
st_events = load_smart_tagging.load(game=cfg.match_id, data_dir=cfg.data_dir)

v6 = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "phantom11_v6.csv"))["label"].values
v7a = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "phantom11_v7a.csv"))["label"].values

diff_idx = np.where(v6 != v7a)[0]
d_oob = compute_out_of_bounds_distance(ball_xy, ball_present, pitch=pitch)

# Find continuous intervals of diff_idx
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

print(f"Total modified frames: {len(diff_idx)}")
print(f"Total continuous modified intervals: {len(intervals)}")

durations_s = [l / 25.0 for _, _, l in intervals]
print(f"Interval duration (s): min={np.min(durations_s):.2f}s, median={np.median(durations_s):.2f}s, mean={np.mean(durations_s):.2f}s, max={np.max(durations_s):.2f}s")

# Categorize intervals by boundary location
touchline_top = 0     # y > 68.0
touchline_bottom = 0  # y < 0.0
goalline_left = 0     # x < 0.0
goalline_right = 0    # x > 105.3

for s, e, l in intervals:
    sub_x = ball_xy[s:e+1, 0]
    sub_y = ball_xy[s:e+1, 1]
    mean_x = np.nanmean(sub_x)
    mean_y = np.nanmean(sub_y)
    if mean_y > 68.0:
        touchline_top += 1
    elif mean_y < 0.0:
        touchline_bottom += 1
    elif mean_x < 0.0:
        goalline_left += 1
    elif mean_x > 105.3:
        goalline_right += 1

print("\n--- Interval Boundary Breakdown ---")
print(f"Top Touchline (y > 68.0m):    {touchline_top} intervals")
print(f"Bottom Touchline (y < 0.0m): {touchline_bottom} intervals")
print(f"Left Goal Line (x < 0.0m):   {goalline_left} intervals")
print(f"Right Goal Line (x > 105.3m): {goalline_right} intervals")

# Sample 10 intervals across different durations and locations
print("\n--- Detailed Inspection of Representative Sample (10 Intervals) ---")
step = max(1, len(intervals) // 10)
sample_intervals = [intervals[i] for i in range(0, len(intervals), step)][:10]

for i, (s, e, l) in enumerate(sample_intervals, 1):
    sub_v6 = v6[s:e+1]
    lbl_v6 = sub_v6[0]
    bx = np.nanmean(ball_xy[s:e+1, 0])
    by = np.nanmean(ball_xy[s:e+1, 1])
    mean_dist = np.mean(d_oob[s:e+1])
    max_dist = np.max(d_oob[s:e+1])

    # Find nearby Smart Tagging events within 10 seconds (250 frames)
    nearby_events = [ev for ev in st_events if abs(ev.get("frame", -999) - s) <= 250]
    event_descs = [f"{ev.get('type') or ev.get('label', 'event')} @ f{ev.get('frame')}" for ev in nearby_events[:2]]
    ev_str = "; ".join(event_descs) if event_descs else "None nearby"

    # Identify boundary type
    if by > 68.0:
        b_type = f"Top Touchline (y={by:.1f}m)"
    elif by < 0.0:
        b_type = f"Bottom Touchline (y={by:.1f}m)"
    elif bx < 0.0:
        b_type = f"Left Goal Line (x={bx:.1f}m)"
    else:
        b_type = f"Right Goal Line (x={bx:.1f}m)"

    print(f"Sample {i:2d}: [{s:,} .. {e:,}] ({l:3d}f = {l/25.0:5.2f}s) | v6={lbl_v6} -> v7a=D | {b_type:<28} | d_oob: mean={mean_dist:4.2f}m, max={max_dist:4.2f}m | Events: {ev_str}")
