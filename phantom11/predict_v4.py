#!/usr/bin/env python3
"""PHANTOM-11: Version 4 Experimental Predictor (Confidence-Gated Boundary Refinement)

Step 9 of More Than A Hack 2027:
- Takes PHANTOM-11 v3 predictions as input baseline.
- Preserves strongly supported v3 possession intervals and confirmed turnovers.
- Applies confidence-gated physical boundary refinement:
  * Eliminates fleeting opponent deflections (< 1.0s, 0 carrier frames) without control.
  * Bridges brief live tracking dropouts (<= 1.2s) with bilateral same-team control.
  * Snaps premature restart boundaries from static pre-kick setups to true strikes/touches.
  * Strictly preserves non-play intervals (pre-match, halftime, post-match).
- Generates outputs/phantom11_v4.csv (preserving v0, v1, v2, and v3 intact).
- Evaluates diagnostic differences against physical evidence.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from phantom11.match_config import MatchConfig
    from phantom11.load_tracking import load_full_tracking
    from phantom11.control_evidence import compute_control_evidence
    from phantom11.movement_features import compute_ball_kinematics
    from phantom11.boundary_refinement import refine_possession_boundaries
    from phantom11.tactical_tags import parse_smart_tagging_timeline
    from phantom11.path_resolver import resolve_data_dir
except ImportError:
    from match_config import MatchConfig
    from load_tracking import load_full_tracking
    from control_evidence import compute_control_evidence
    from movement_features import compute_ball_kinematics
    from boundary_refinement import refine_possession_boundaries
    from tactical_tags import parse_smart_tagging_timeline
    from path_resolver import resolve_data_dir

from baseline import write_submission
import load_smart_tagging


def run_predictor_v4(
    v3_path=None,
    output_path=None,
    game="BAR-ATH",
    data_dir=None,
    verbose=True,
):
    """Generate PHANTOM-11 v4 candidate from v3 and physical boundary refinement."""
    t_start = time.time()
    v3_path = v3_path or os.path.join(PROJECT_ROOT, "outputs", "phantom11_v3.csv")
    output_path = output_path or os.path.join(PROJECT_ROOT, "outputs", "phantom11_v4.csv")

    if not os.path.isfile(v3_path):
        raise FileNotFoundError(f"v3 output not found at {v3_path}. Run v3 first.")

    if verbose:
        print("=" * 78)
        print("PHANTOM-11: PREDICTOR VERSION 4 (CONFIDENCE-GATED BOUNDARY REFINEMENT)")
        print("=" * 78)
        print(f"Loading input v3 from: {v3_path}")

    v3_df = pd.read_csv(v3_path)
    v3_labels = v3_df["label"].values.astype("<U1")
    n_frames = len(v3_labels)

    # 1. Load Match Configuration & Tracking
    cfg = MatchConfig.from_official_game(game=game, data_dir=data_dir)
    ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, _ = load_full_tracking(
        match_config=cfg
    )
    st_events = load_smart_tagging.load(game=cfg.match_id, data_dir=cfg.data_dir)
    non_play = cfg.get_non_play_intervals()
    fps = cfg.fps

    if verbose:
        print("\n[1] Computing physical kinematics & control evidence...")
    ball_kin = compute_ball_kinematics(ball_xy, fps=fps)
    ev = compute_control_evidence(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=st_events,
        non_play_intervals=non_play,
        fps=fps,
    )
    tact = parse_smart_tagging_timeline(st_events, n_frames, team_map=cfg.get_team_map())

    if verbose:
        print("\n[2] Applying confidence-gated possession boundary refinement...")
    v4_labels = refine_possession_boundaries(
        labels=v3_labels,
        ball_present=ball_present,
        ball_xy=ball_xy,
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        d_h1=ev["d_h1"],
        d_a1=ev["d_a1"],
        ball_speed=ball_kin["speed"],
        stoppage_support=tact["stoppage_support"],
        non_play_intervals=non_play,
        fps=fps,
    )

    # 2. Write v4 submission
    write_submission(output_path, v4_labels)
    t_end = time.time()

    # 3. Diagnostic comparisons
    changed_frames = int(np.sum(v3_labels != v4_labels))
    flips_ha = int(np.sum((v3_labels == "H") & (v4_labels == "A")))
    flips_ah = int(np.sum((v3_labels == "A") & (v4_labels == "H")))
    flips_live_d = int(np.sum(np.isin(v3_labels, ["H", "A"]) & (v4_labels == "D")))
    flips_d_live = int(np.sum((v3_labels == "D") & np.isin(v4_labels, ["H", "A"])))

    if verbose:
        print(f"\n[3] Saved final v4 to: {output_path}")
        print(f"    Runtime: {t_end - t_start:.2f} seconds.")
        print("-" * 78)
        print("Comparative Diagnostics (v3 vs v4):")
        print(f"  Total Changed Frames: {changed_frames:,} / {n_frames:,} ({changed_frames/n_frames*100:.3f}%)")
        print(f"  H -> A changes:       {flips_ha:>5} frames")
        print(f"  A -> H changes:       {flips_ah:>5} frames")
        print(f"  Live -> Dead changes: {flips_live_d:>5} frames")
        print(f"  Dead -> Live changes: {flips_d_live:>5} frames")
        print("=" * 78)

    return v4_labels, output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PHANTOM-11 v4 Predictor")
    parser.add_argument("--v3-path", default=None, help="Path to v3 CSV")
    parser.add_argument("--output", default=None, help="Path to output v4 CSV")
    parser.add_argument("--game", default="BAR-ATH", help="Game identifier")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    args = parser.parse_args()

    run_predictor_v4(
        v3_path=args.v3_path,
        output_path=args.output,
        game=args.game,
        data_dir=args.data_dir,
        verbose=True,
    )
