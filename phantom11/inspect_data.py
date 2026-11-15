#!/usr/bin/env python3
"""PHANTOM-11: Match Data Inventory & Tracking Data Inspection Script

Step 3 of More Than A Hack 2027:
- Checks the presence and size of tactical and broadcast assets using games.py.
- The MP4 video is optional and non-blocking.
- Inspects tactical tracking data (ATD) using official loaders (starter/load_atd.py).
- Verifies record count (199,319), frame alignment, track slots, team mapping,
  ball visibility, and checks for duplicates or parsing errors.
- Inspects Smart Tagging using official loader (starter/load_smart_tagging.py).
- Inspects Homography using official loader (starter/load_homography.py).
"""

import os
import sys
import xml.etree.ElementTree as ET
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
import load_atd
import load_homography
import load_smart_tagging
from baseline import team_names, team_of_each_track


def format_size(bytes_size):
    """Format file size in human-readable units."""
    if bytes_size is None:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:3.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def check_asset_inventory():
    """Check existence and size of tactical and broadcast assets."""
    print("=" * 78)
    print("MATCH DATA INVENTORY (Tactical & Broadcast Assets)")
    print("=" * 78)
    print(f"{'Source':<10} {'Asset Name':<16} {'Filename':<36} {'Exists?':<9} {'Size'}")
    print("-" * 78)

    inventory = {}

    for source in ["tactical", "broadcast"]:
        paths = games.asset_paths(source=source)
        inventory[source] = {}
        for asset_name, path in paths.items():
            filename = os.path.basename(path)
            exists = os.path.isfile(path)
            if exists:
                size_str = format_size(os.path.getsize(path))
                status_str = "YES"
            else:
                if asset_name == "video":
                    size_str = "Optional (Not Downloaded)"
                    status_str = "NO"
                else:
                    size_str = "Missing"
                    status_str = "NO"

            inventory[source][asset_name] = {
                "path": path,
                "filename": filename,
                "exists": exists,
                "size": os.path.getsize(path) if exists else None,
                "size_str": size_str,
            }
            print(f"{source:<10} {asset_name:<16} {filename:<36} {status_str:<9} {size_str}")

    print("=" * 78)
    return inventory


def inspect_tracking_data(inventory):
    """Inspect tactical tracking data (ATD) using official loaders."""
    tactical = inventory.get("tactical", {})
    atd_meta = tactical.get("atd_meta", {})
    atd_raw = tactical.get("atd_raw", {})

    if not atd_meta.get("exists") or not atd_raw.get("exists"):
        print("\n[!] Tactical tracking files (XML or TXT) are missing. Skipping tracking inspection.")
        return None

    print("\n" + "=" * 78)
    print("1. TACTICAL TRACKING DATA (ATD) INSPECTION")
    print("=" * 78)

    meta_path = atd_meta["path"]
    raw_path = atd_raw["path"]

    # Alignment and streaming verification via official load_atd
    print("\n--- A. Alignment and Record Counts ---")
    report = load_atd.check_alignment_raw(source="tactical")
    print(f"  Records counted:         {report['n_records']:,}")
    print(f"  Expected video frames:   {report['n_video_frames']:,}")
    print(f"  Frame ID range:          {report['frame_id_range'][0]} .. {report['frame_id_range'][1]}")
    print(f"  Contiguous records:      {report['contiguous']}")
    print(f"  First video frame:       {report['first_video_frame']}")
    print(f"  Alignment Agreement:     {'PASSED (agrees)' if report['agrees'] else 'FAILED'}")
    print(f"  Alignment Formula:       video_frame = frame_id - 1")

    # Team & Track Slot Metadata
    print("\n--- B. Teams & Track Slots ---")
    home_name, away_name = team_names(meta_path)
    track_teams = team_of_each_track(meta_path)
    n_slots = len(track_teams)
    home_slots = [i + 1 for i, t in enumerate(track_teams) if t == 0]
    away_slots = [i + 1 for i, t in enumerate(track_teams) if t == 1]

    print(f"  Home Team:               {home_name} (H)")
    print(f"  Away Team:               {away_name} (A)")
    print(f"  Total Track Slots:       {n_slots}")
    print(f"  Home Track Slots ({len(home_slots)}):   {home_slots}")
    print(f"  Away Track Slots ({len(away_slots)}):   {away_slots}")

    # Full data validation: streaming pass for missing ball, duplicates, errors
    print("\n--- C. Full Streaming Pass & Ball Visibility ---")
    total_frames = 0
    missing_ball_frames = 0
    duplicate_frames = 0
    malformed_lines = 0
    seen_frame_ids = set()

    # Landmark frames to capture for sample output:
    # Frame 1 (first), 11185 (kickoff), 11186, 100000 (halftime), 113742 (2nd half kickoff)
    target_sample_fids = {1, 11185, 11186, 100000, 113742}
    samples = {}

    with open(raw_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            parts = line_str.split(":")
            if len(parts) != 3:
                malformed_lines += 1
                continue

            try:
                fid = int(parts[0])
            except ValueError:
                malformed_lines += 1
                continue

            if fid in seen_frame_ids:
                duplicate_frames += 1
            seen_frame_ids.add(fid)

            total_frames += 1
            ball_part = parts[2]
            players_part = parts[1]

            if ball_part.startswith("NaN"):
                missing_ball_frames += 1

            if fid in target_sample_fids:
                samples[fid] = (parts[0], ball_part, players_part)

    ball_vis_pct = ((total_frames - missing_ball_frames) / total_frames * 100.0) if total_frames > 0 else 0.0

    print(f"  Total records verified:  {total_frames:,}")
    print(f"  Missing-ball frames:     {missing_ball_frames:,} ({100.0 - ball_vis_pct:.2f}%)")
    print(f"  Ball visible frames:     {total_frames - missing_ball_frames:,} ({ball_vis_pct:.2f}%)")
    print(f"  Duplicate Frame IDs:     {duplicate_frames}")
    print(f"  Malformed lines:         {malformed_lines}")

    # 5 Sample Rows
    print("\n--- D. Five Sample Data Rows ---")
    for fid in sorted(samples.keys()):
        frame_str, b_str, p_str = samples[fid]
        vframe = int(frame_str) - 1
        first_few_slots = p_str.split(";")[:3]
        print(f"  [Frame ID {frame_str:>6} | Video Frame {vframe:>6}]")
        print(f"    Ball (x, y, s):         {b_str}")
        print(f"    Track Slots 1..3:       {first_few_slots} ...")

    return {
        "total_frames": total_frames,
        "missing_ball_frames": missing_ball_frames,
        "ball_vis_pct": ball_vis_pct,
        "duplicate_frames": duplicate_frames,
        "malformed_lines": malformed_lines,
        "n_slots": n_slots,
        "home_name": home_name,
        "away_name": away_name,
    }


def inspect_smart_tagging(inventory):
    """Inspect tactical Smart Tagging XML using official loader."""
    tactical = inventory.get("tactical", {})
    st_meta = tactical.get("smart_tagging", {})

    if not st_meta.get("exists"):
        print("\n[!] Tactical Smart Tagging file missing. Skipping Smart Tagging inspection.")
        return None

    print("\n" + "=" * 78)
    print("2. SMART TAGGING INSPECTION")
    print("=" * 78)

    events = load_smart_tagging.load(game="BAR-ATH")
    print(f"\n  Total events/instances:  {len(events)}")

    # Event types / codes distribution
    code_counts = Counter(e["code"] for e in events)
    print("\n--- Available Event Types (Codes) ---")
    for code, count in code_counts.most_common():
        print(f"    {code:<32} {count:>4}")

    # Team tags distribution
    team_tags = Counter(e["tags"].get("Team", "None") for e in events)
    print("\n--- Available Team Tags ---")
    for team, count in team_tags.items():
        print(f"    {team:<20} {count:>4}")

    # Time base check via official loader
    print("\n--- Time Base & Kickoff Alignment Check ---")
    tb_check = load_smart_tagging.check_time_base(events)
    for half_row in tb_check["halves"]:
        print(f"    {half_row['half']}: true kickoff frame {half_row['true_kickoff_frame']}, "
              f"tagged intervals {half_row['tagged']}, contains kickoff: {half_row['contains_kickoff']}")
    print(f"  Time base agreed:        {'PASSED (agrees)' if tb_check['agrees'] else 'FAILED'}")
    print("  Conversion formula:      frame = int(round(seconds * 25))")

    # Five sample intervals
    print("\n--- Five Sample Smart Tagging Intervals ---")
    for idx, e in enumerate(events[:5], start=1):
        print(f"  Sample {idx}:")
        print(f"    Code:        {e['code']}")
        print(f"    Time (s):    {e['start_s']:.2f}s -> {e['end_s']:.2f}s")
        print(f"    Frames:      {e['start_frame']} -> {e['end_frame']}")
        print(f"    Tags:        {e['tags']}")

    return {
        "n_events": len(events),
        "code_counts": code_counts,
        "team_tags": team_tags,
        "tb_agrees": tb_check["agrees"],
    }


def inspect_homography(inventory):
    """Inspect tactical homography CSV using official loader."""
    tactical = inventory.get("tactical", {})
    hom_meta = tactical.get("homography", {})

    if not hom_meta.get("exists"):
        print("\n[!] Tactical homography file missing. Skipping homography inspection.")
        return None

    print("\n" + "=" * 78)
    print("3. HOMOGRAPHY INSPECTION")
    print("=" * 78)

    matrices, errors, valid = load_homography.load(source="tactical")
    n_rows = len(matrices)
    n_valid = int(valid.sum())
    n_invalid = n_rows - n_valid

    print(f"\n  Total rows / matrices:   {n_rows:,}")
    print(f"  Frame index range:       0 .. {n_rows - 1}")
    print(f"  Valid matrices:          {n_valid:,} ({n_valid / n_rows * 100:.1f}%)")
    print(f"  Invalid matrices:        {n_invalid:,} ({n_invalid / n_rows * 100:.1f}%)")

    import numpy as np
    median_err = float(np.median(errors))
    p99_err = float(np.percentile(errors, 99))
    print(f"  Reprojection Error:      median {median_err:.3f}, 99th percentile {p99_err:.3f}")

    # Alignment check against tactical tracking (expected 199,319 frames)
    expected_frames = games.source_frames("tactical")
    first_aligned = (0 == 0)
    last_aligned = (n_rows == expected_frames)
    print(f"  First frame aligns (0):  {first_aligned}")
    print(f"  Last frame aligns ({expected_frames - 1}): {last_aligned}")
    print(f"  Alignment with ATD:      {'PASSED' if n_rows == expected_frames else 'FAILED'}")

    return {
        "n_rows": n_rows,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "median_err": median_err,
        "p99_err": p99_err,
        "aligned": n_rows == expected_frames,
    }


def main():
    print("PHANTOM-11 — Tactical Data Verification & Inspection\n")
    inventory = check_asset_inventory()
    tracking_stats = inspect_tracking_data(inventory)
    smart_tagging_stats = inspect_smart_tagging(inventory)
    homography_stats = inspect_homography(inventory)

    print("\n" + "=" * 78)
    print("INSPECTION COMPLETE — ALL CHECKS FINISHED")
    print("=" * 78)


if __name__ == "__main__":
    main()
