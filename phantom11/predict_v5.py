#!/usr/bin/env python3
"""PHANTOM-11: Version 5 Experimental Predictor (Flight-Aware Possession Handover)

Step 11 of More Than A Hack 2027:
- Takes PHANTOM-11 v4 predictions as input candidate.
- Applies physically grounded flight-aware possession handover:
  1. Resolves in-flight aerial passes and lofted crosses where 2D proximity
     momentarily favors an opposing player while the ball is airborne overhead.
  2. Aligns turnover boundaries to the exact moment the interceptor reaches
     or engages the ball (<= 1.5m, deliberate strike, or carrier status).
  3. Preserves genuine rapid turnovers (>= 1.0s of confirmed carrying) intact.
  4. Strictly clamps non-play intervals (pre-match, halftime, post-match).
- Generates outputs/phantom11_v5.csv (leaving v0, v1, v2, v3, and v4 completely untouched).
- Evaluates diagnostic differences between v4 and v5 against physical evidence.
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
    from phantom11.flight_handover import refine_flight_handovers
    from phantom11.boundary_refinement import extract_spell_intervals
    from phantom11.tactical_tags import parse_smart_tagging_timeline
    from phantom11.path_resolver import resolve_data_dir
except ImportError:
    from match_config import MatchConfig
    from load_tracking import load_full_tracking
    from control_evidence import compute_control_evidence
    from movement_features import compute_ball_kinematics
    from flight_handover import refine_flight_handovers
    from boundary_refinement import extract_spell_intervals
    from tactical_tags import parse_smart_tagging_timeline
    from path_resolver import resolve_data_dir

from baseline import write_submission
import load_smart_tagging


def run_predictor_v5(
    v4_path: str = None,
    output_path: str = None,
    game: str = "BAR-ATH",
    data_dir: str = None,
    verbose: bool = True,
):
    """Generate PHANTOM-11 v5 candidate from v4 and flight-aware handover refinement."""
    t_start = time.time()
    v4_path = v4_path or os.path.join(PROJECT_ROOT, "outputs", "phantom11_v4.csv")
    output_path = output_path or os.path.join(PROJECT_ROOT, "outputs", "phantom11_v5.csv")

    if not os.path.isfile(v4_path):
        raise FileNotFoundError(f"v4 output not found at {v4_path}. Run v4 first.")

    if verbose:
        print("=" * 82)
        print("PHANTOM-11: PREDICTOR VERSION 5 (FLIGHT-AWARE POSSESSION HANDOVER)")
        print("=" * 82)
        print(f"Loading input v4 from: {v4_path}")

    v4_df = pd.read_csv(v4_path)
    v4_labels = v4_df["label"].values.astype("<U1")
    n_frames = len(v4_labels)

    # 1. Load Match Configuration & Tracking
    cfg = MatchConfig.from_official_game(game=game, data_dir=data_dir)
    ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, _ = load_full_tracking(
        match_config=cfg
    )
    st_events = load_smart_tagging.load(game=cfg.match_id, data_dir=cfg.data_dir)
    non_play = cfg.get_non_play_intervals()
    fps = cfg.fps

    if verbose:
        print("\n[1] Computing kinematics & control evidence...")
    ball_kin = compute_ball_kinematics(ball_xy, fps=fps)
    ev = compute_control_evidence(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=st_events,
        non_play_intervals=non_play,
        fps=fps,
    )

    if verbose:
        print("\n[2] Applying flight-aware possession handover refinement...")
    v5_labels = refine_flight_handovers(
        labels=v4_labels,
        ball_present=ball_present,
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        d_h1=ev["d_h1"],
        d_a1=ev["d_a1"],
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        ball_speed=ball_kin["speed"],
        non_play_intervals=non_play,
        fps=fps,
    )

    # 2. Write v5 submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_submission(output_path, v5_labels)
    t_end = time.time()

    # 3. Diagnostic comparisons between v4 and v5
    changed_frames = int(np.sum(v4_labels != v5_labels))
    flips_ha = int(np.sum((v4_labels == "H") & (v5_labels == "A")))
    flips_ah = int(np.sum((v4_labels == "A") & (v5_labels == "H")))
    flips_live_d = int(np.sum(np.isin(v4_labels, ["H", "A"]) & (v5_labels == "D")))
    flips_d_live = int(np.sum((v4_labels == "D") & np.isin(v5_labels, ["H", "A"])))

    spells_v4 = extract_spell_intervals(v4_labels)
    spells_v5 = extract_spell_intervals(v5_labels)
    poss_v4 = [s for s in spells_v4 if s[0] in ("H", "A")]
    poss_v5 = [s for s in spells_v5 if s[0] in ("H", "A")]

    durs_v4 = [s[3] / fps for s in poss_v4]
    durs_v5 = [s[3] / fps for s in poss_v5]

    if verbose:
        print(f"\n[3] v5 Candidate Generation Complete ({t_end - t_start:.2f}s)")
        print(f"    Output written to: {output_path}")
        print(f"    Total match frames: {n_frames:,}")
        print(f"    Modified frames (v4 -> v5): {changed_frames:,} ({changed_frames / n_frames * 100:.3f}%)")
        print(f"      * Away -> Home (in-flight pass unifications): {flips_ah:,}")
        print(f"      * Home -> Away (in-flight pass unifications): {flips_ha:,}")
        print(f"      * Live -> Dead Ball: {flips_live_d:,}")
        print(f"      * Dead Ball -> Live: {flips_d_live:,}")
        print(f"\n    Possession Spells Summary:")
        print(f"      * v4 Possessions: {len(poss_v4):<4} (mean: {np.mean(durs_v4):.2f}s, median: {np.median(durs_v4):.2f}s)")
        print(f"      * v5 Possessions: {len(poss_v5):<4} (mean: {np.mean(durs_v5):.2f}s, median: {np.median(durs_v5):.2f}s)")
        print(f"      * Net spell reduction: {len(poss_v4) - len(poss_v5):+d} spells")
        print("=" * 82)

    return {
        "v4_path": v4_path,
        "output_path": output_path,
        "changed_frames": changed_frames,
        "flips_ah": flips_ah,
        "flips_ha": flips_ha,
        "flips_live_d": flips_live_d,
        "flips_d_live": flips_d_live,
        "n_spells_v4": len(poss_v4),
        "n_spells_v5": len(poss_v5),
        "elapsed_s": t_end - t_start,
    }


if __name__ == "__main__":
    run_predictor_v5()
