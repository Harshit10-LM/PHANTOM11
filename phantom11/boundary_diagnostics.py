#!/usr/bin/env python3
"""PHANTOM-11: Possession Boundary Diagnostics and Consistency Tool

Step 9 of More Than A Hack 2027:
- Analyzes the physical consistency of predicted possession transitions against permitted tracking data.
- Groups suspected boundary errors by physical mechanism:
  1. Fleeting opponent deflections mistaken for interceptions.
  2. In-flight turnovers (switching while a pass is airborne without control).
  3. False possession splits caused by brief tracking dropouts.
  4. Premature restarts before ball is put into play.
  5. Premature stoppages during active dribbling / carrying.
- Reports aggregate statistical distributions and physical justifications.
- Strictly adheres to competition rules: zero private frame-level leaks, zero hand-tuning.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from phantom11.match_config import MatchConfig
    from phantom11.load_tracking import load_full_tracking
    from phantom11.movement_features import compute_ball_kinematics
    from phantom11.control_evidence import compute_control_evidence
    from phantom11.path_resolver import resolve_data_dir
except ImportError:
    from match_config import MatchConfig
    from load_tracking import load_full_tracking
    from movement_features import compute_ball_kinematics
    from control_evidence import compute_control_evidence
    from path_resolver import resolve_data_dir

import games
import load_smart_tagging


def extract_spells(labels: np.ndarray, fps: float = 25.0) -> List[Dict[str, Any]]:
    """Extract contiguous spells of H, A, and D."""
    spells = []
    n = len(labels)
    if n == 0:
        return spells

    cur_lbl = labels[0]
    cur_start = 0

    for i in range(1, n):
        if labels[i] != cur_lbl:
            spells.append({
                "label": cur_lbl,
                "start": cur_start,
                "end": i - 1,
                "duration_frames": i - cur_start,
                "duration_s": (i - cur_start) / fps,
            })
            cur_lbl = labels[i]
            cur_start = i

    spells.append({
        "label": cur_lbl,
        "start": cur_start,
        "end": n - 1,
        "duration_frames": n - cur_start,
        "duration_s": (n - cur_start) / fps,
    })
    return spells


def diagnose_possession_boundaries(
    labels: np.ndarray,
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    smart_tagging_events: list,
    non_play_intervals: list,
    fps: float = 25.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run comprehensive physical consistency diagnosis on predicted labels."""
    n_frames = len(labels)
    spells = extract_spells(labels, fps=fps)
    possession_spells = [s for s in spells if s["label"] in ("H", "A")]
    stoppage_spells = [s for s in spells if s["label"] == "D"]

    # Compute physical signals
    ball_kin = compute_ball_kinematics(ball_xy, fps=fps)
    ev = compute_control_evidence(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=smart_tagging_events,
        non_play_intervals=non_play_intervals,
        fps=fps,
    )

    ball_present = ball_kin["ball_present"]
    ball_speed = np.nan_to_num(ball_kin["speed"], nan=0.0)

    # -----------------------------------------------------------------------
    # 1. Transitions analysis
    # -----------------------------------------------------------------------
    transitions = []
    for i in range(len(spells) - 1):
        s1, s2 = spells[i], spells[i + 1]
        t_frame = s2["start"]
        transitions.append({
            "frame": t_frame,
            "from": s1["label"],
            "to": s2["label"],
            "prev_duration_s": s1["duration_s"],
            "next_duration_s": s2["duration_s"],
        })

    trans_counts = {
        "H->A": sum(1 for t in transitions if t["from"] == "H" and t["to"] == "A"),
        "A->H": sum(1 for t in transitions if t["from"] == "A" and t["to"] == "H"),
        "H->D": sum(1 for t in transitions if t["from"] == "H" and t["to"] == "D"),
        "A->D": sum(1 for t in transitions if t["from"] == "A" and t["to"] == "D"),
        "D->H": sum(1 for t in transitions if t["from"] == "D" and t["to"] == "H"),
        "D->A": sum(1 for t in transitions if t["from"] == "D" and t["to"] == "A"),
    }

    # -----------------------------------------------------------------------
    # 2. Suspected Error Modes
    # -----------------------------------------------------------------------
    # Mode 1: Fleeting opponent deflection (< 1.0s without active carrier control)
    deflections = []
    for s in possession_spells:
        if s["duration_s"] < 1.0:
            s_idx = slice(s["start"], s["end"] + 1)
            is_c = ev["is_carrier_h"][s_idx] if s["label"] == "H" else ev["is_carrier_a"][s_idx]
            carrier_frames = int(np.sum(is_c))
            if carrier_frames == 0:
                deflections.append(s)

    # Mode 2: In-flight turnover transition (H <-> A while ball speed > 10 m/s and neither team has carrier)
    in_flight_turnovers = []
    for t in transitions:
        if (t["from"], t["to"]) in [("H", "A"), ("A", "H")]:
            f = t["frame"]
            w_start = max(0, f - 3)
            w_end = min(n_frames - 1, f + 3)
            mean_spd = np.mean(ball_speed[w_start:w_end + 1])
            has_carrier = np.any(ev["is_carrier_h"][w_start:w_end + 1] | ev["is_carrier_a"][w_start:w_end + 1])
            if mean_spd > 10.0 and not has_carrier:
                in_flight_turnovers.append((t, mean_spd))

    # Mode 3: False possession split by short dead-ball gap (< 1.5s D surrounded by same team)
    split_possessions = []
    for i in range(1, len(spells) - 1):
        prev_s = spells[i - 1]
        cur_s = spells[i]
        next_s = spells[i + 1]
        if cur_s["label"] == "D" and cur_s["duration_s"] <= 1.5:
            if prev_s["label"] in ("H", "A") and prev_s["label"] == next_s["label"]:
                # Check if ball was actually missing vs just a drop
                f_idx = slice(cur_s["start"], cur_s["end"] + 1)
                missing_ratio = float(np.mean(~ball_present[f_idx]))
                split_possessions.append((cur_s, prev_s["label"], missing_ratio))

    # Mode 4: Premature restart (D -> H or D -> A while ball is stationary and unapproached)
    premature_restarts = []
    for t in transitions:
        if t["from"] == "D" and t["to"] in ("H", "A"):
            f = t["frame"]
            # Check player distance to ball
            d_ball = ev["d_h1"][f] if t["to"] == "H" else ev["d_a1"][f]
            spd = ball_speed[f]
            # If ball is stationary and player is still > 2.5m away
            if spd < 0.5 and d_ball > 2.5:
                premature_restarts.append((t, d_ball, spd))

    # Mode 5: Premature stoppage (Live -> D while player is actively carrying ball)
    premature_stoppages = []
    for t in transitions:
        if t["from"] in ("H", "A") and t["to"] == "D":
            f = t["frame"]
            is_c = ev["is_carrier_h"][f] if t["from"] == "H" else ev["is_carrier_a"][f]
            if is_c and ball_present[f]:
                premature_stoppages.append(t)

    # Compile diagnostic summary
    durations = [s["duration_s"] for s in possession_spells]
    results = {
        "n_total_spells": len(spells),
        "n_possessions": len(possession_spells),
        "n_stoppages": len(stoppage_spells),
        "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "median_duration_s": float(np.median(durations)) if durations else 0.0,
        "durations_under_1s": sum(1 for d in durations if d < 1.0),
        "durations_1s_to_2s": sum(1 for d in durations if 1.0 <= d < 2.0),
        "durations_2s_to_5s": sum(1 for d in durations if 2.0 <= d < 5.0),
        "durations_over_5s": sum(1 for d in durations if d >= 5.0),
        "transitions": trans_counts,
        "suspected_fleeting_deflections": len(deflections),
        "suspected_in_flight_turnovers": len(in_flight_turnovers),
        "suspected_split_possessions": len(split_possessions),
        "suspected_premature_restarts": len(premature_restarts),
        "suspected_premature_stoppages": len(premature_stoppages),
    }

    if verbose:
        print("=" * 78)
        print("PHANTOM-11: POSSESSION BOUNDARY DIAGNOSTIC AUDIT")
        print("=" * 78)
        print(f"Total Predicted Possession Spells: {len(possession_spells)}")
        print(f"  Mean Duration:   {results['mean_duration_s']:.2f} s")
        print(f"  Median Duration: {results['median_duration_s']:.2f} s")
        print(f"  Spells < 1.0s:   {results['durations_under_1s']} ({results['durations_under_1s']/len(possession_spells)*100:.1f}%)")
        print(f"  Spells 1.0-2.0s: {results['durations_1s_to_2s']} ({results['durations_1s_to_2s']/len(possession_spells)*100:.1f}%)")
        print(f"  Spells 2.0-5.0s: {results['durations_2s_to_5s']} ({results['durations_2s_to_5s']/len(possession_spells)*100:.1f}%)")
        print(f"  Spells >= 5.0s:  {results['durations_over_5s']} ({results['durations_over_5s']/len(possession_spells)*100:.1f}%)")
        print("-" * 78)
        print("Transition Distribution:")
        for k, v in trans_counts.items():
            print(f"  {k:>6}: {v:>4} transitions")
        print("-" * 78)
        print("Physical Consistency & Boundary Anomaly Detection:")
        print(f"  1. Fleeting Deflections without Control (< 1.0s, 0 carrier frames): {len(deflections):>4}")
        print(f"  2. In-Flight Turnovers (ball speed > 10 m/s without carrier):        {len(in_flight_turnovers):>4}")
        print(f"  3. False Possession Splits (short D <= 1.5s inside same team):      {len(split_possessions):>4}")
        print(f"  4. Premature Restarts (ball stationary & player > 2.5m away):        {len(premature_restarts):>4}")
        print(f"  5. Premature Stoppages (Live -> D while player actively carrying):   {len(premature_stoppages):>4}")
        print("=" * 78)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHANTOM-11 Boundary Diagnostics")
    parser.add_argument("--submission", default="outputs/phantom11_v3.csv", help="Path to submission CSV")
    parser.add_argument("--game", default="BAR-ATH", help="Game identifier")
    parser.add_argument("--data-dir", default=None, help="Tactical data directory")
    args = parser.parse_args()

    sub_path = os.path.join(PROJECT_ROOT, args.submission) if not os.path.isabs(args.submission) else args.submission
    if not os.path.isfile(sub_path):
        raise FileNotFoundError(f"Submission CSV not found at {sub_path}")

    print(f"Loading submission from {sub_path}...")
    df = pd.read_csv(sub_path)
    labels = df["label"].values.astype("<U1")

    cfg = MatchConfig.from_official_game(game=args.game, data_dir=args.data_dir)
    ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, n_frames = load_full_tracking(
        match_config=cfg
    )
    st_events = load_smart_tagging.load(game=cfg.match_id, data_dir=cfg.data_dir)
    non_play = cfg.get_non_play_intervals()

    diagnose_possession_boundaries(
        labels=labels,
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=st_events,
        non_play_intervals=non_play,
        fps=cfg.fps,
        verbose=True,
    )
