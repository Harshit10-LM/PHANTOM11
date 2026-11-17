#!/usr/bin/env python3
"""Investigate the disagreement between confirmed out-of-bounds frames and GT window."""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from phantom11.match_config import MatchConfig
from phantom11.load_tracking import load_full_tracking
from phantom11.pitch_boundary import compute_out_of_bounds_distance, detect_out_of_bounds_mask
from phantom11.path_resolver import resolve_official_repo

cfg = MatchConfig.from_official_game()
ball_xy, ball_present, players_xy, _, team_of_track, pitch, _ = load_full_tracking(match_config=cfg)

official_dir = resolve_official_repo()
gt_path = os.path.join(official_dir, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
gt_df = pd.read_csv(gt_path)
gt_labels = gt_df["label"].values.astype("<U1")
gt_frames = gt_df["frame"].values

print(f"GT frames count: {len(gt_frames)} [first={gt_frames[0]}, last={gt_frames[-1]}]")

s, e = gt_frames[0], gt_frames[-1]
sub_ball_xy = ball_xy[s:e+1]
sub_ball_present = ball_present[s:e+1]

d_oob = compute_out_of_bounds_distance(sub_ball_xy, sub_ball_present, pitch=pitch)
mask = detect_out_of_bounds_mask(sub_ball_xy, sub_ball_present, pitch=pitch, margin=0.35)

print(f"Confirmed OOB frames in GT window (d_oob >= 0.35m): {np.sum(mask)}")
disagree_idx = np.where(mask & (gt_labels != "D"))[0]
print(f"Disagreements count: {len(disagree_idx)}")

for idx in disagree_idx:
    abs_frame = s + idx
    print("=" * 65)
    print(f"DISAGREEMENT AT FRAME {abs_frame} (relative index {idx})")
    print("=" * 65)
    print(f"GT Label: {gt_labels[idx]}")
    print(f"Ball coords: ({sub_ball_xy[idx, 0]:.3f}, {sub_ball_xy[idx, 1]:.3f})")
    print(f"Pitch (L, W): {pitch}")
    print(f"d_oob: {d_oob[idx]:.3f} m")

    c_s = max(0, idx - 10)
    c_e = min(len(gt_labels), idx + 11)
    print("\nSurrounding Context (20 frames around):")
    print(f"{'Frame':<8} | {'GT':<4} | {'Ball (X, Y)':<18} | {'d_oob (m)':<10} | {'Mask':<5}")
    print("-" * 55)
    for c in range(c_s, c_e):
        f = s + c
        bx, by = sub_ball_xy[c, 0], sub_ball_xy[c, 1]
        ball_str = f"({bx:.2f}, {by:.2f})" if not np.isnan(bx) else "(NaN, NaN)"
        marker = " <--- DISAGREEMENT" if c == idx else ""
        print(f"{f:<8} | {gt_labels[c]:<4} | {ball_str:<18} | {d_oob[c]:<10.3f} | {str(mask[c]):<5}{marker}")
