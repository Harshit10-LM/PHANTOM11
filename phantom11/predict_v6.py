#!/usr/bin/env python3
"""PHANTOM-11: Version 6 Experimental Predictor (Evidence-Gated Handover Alignment)

Step 13 of More Than A Hack 2027:
- Takes PHANTOM-11 v4 predictions as input candidate.
- Applies evidence-gated flight-aware possession handover:
  1. Resolves in-flight aerial passes and lofted crosses (Step 1).
  2. Gated handover alignment (Step 2): prevents premature turnovers when a ball
     slows down in open space while the interceptor is still far away (> 2.2m).
     Requires confirmed contact proximity (<= 1.5m), carrier status, or trap at feet.
  3. Preserves genuine rapid turnovers (>= 1.0s of confirmed carrying) intact.
  4. Strictly clamps non-play intervals (pre-match, halftime, post-match).
- Generates outputs/phantom11_v6.csv (leaving v0, v1, v2, v3, v4, and v5 untouched).
- Evaluates diagnostic differences between v5 and v6 against physical evidence.
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
    from phantom11.path_resolver import resolve_data_dir
except ImportError:
    from match_config import MatchConfig
    from load_tracking import load_full_tracking
    from control_evidence import compute_control_evidence
    from movement_features import compute_ball_kinematics
    from flight_handover import refine_flight_handovers
    from boundary_refinement import extract_spell_intervals
    from path_resolver import resolve_data_dir

from baseline import write_submission
import load_smart_tagging


def run_predictor_v6(
    v4_path: str = None,
    output_path: str = None,
    game: str = "BAR-ATH",
    data_dir: str = None,
    verbose: bool = True,
):
    """Generate PHANTOM-11 v6 candidate with evidence-gated handover refinement."""
    t_start = time.time()
    v4_path = v4_path or os.path.join(PROJECT_ROOT, "outputs", "phantom11_v4.csv")
    output_path = output_path or os.path.join(PROJECT_ROOT, "outputs", "phantom11_v6.csv")

    if not os.path.isfile(v4_path):
        raise FileNotFoundError(f"v4 output not found at {v4_path}. Run v4 first.")

    if verbose:
        print("=" * 82)
        print("PHANTOM-11: PREDICTOR VERSION 6 (EVIDENCE-GATED HANDOVER ALIGNMENT)")
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
        print("\n[2] Applying evidence-gated possession handover refinement...")
    v6_labels = refine_flight_handovers(
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
        gate_deceleration_by_proximity=True,
    )

    # 2. Write v6 submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_submission(output_path, v6_labels)
    t_end = time.time()

    # 3. Diagnostic comparisons between v5 and v6
    v5_path = os.path.join(PROJECT_ROOT, "outputs", "phantom11_v5.csv")
    v5_labels = pd.read_csv(v5_path)["label"].values.astype("<U1") if os.path.isfile(v5_path) else v4_labels

    changed_frames = int(np.sum(v5_labels != v6_labels))
    flips_ha = int(np.sum((v5_labels == "H") & (v6_labels == "A")))
    flips_ah = int(np.sum((v5_labels == "A") & (v6_labels == "H")))
    flips_live_d = int(np.sum(np.isin(v5_labels, ["H", "A"]) & (v6_labels == "D")))
    flips_d_live = int(np.sum((v5_labels == "D") & np.isin(v6_labels, ["H", "A"])))

    spells_v5 = extract_spell_intervals(v5_labels)
    spells_v6 = extract_spell_intervals(v6_labels)
    poss_v5 = [s for s in spells_v5 if s[0] in ("H", "A")]
    poss_v6 = [s for s in spells_v6 if s[0] in ("H", "A")]

    durs_v5 = [s[3] / fps for s in poss_v5]
    durs_v6 = [s[3] / fps for s in poss_v6]

    if verbose:
        print(f"\n[3] v6 Candidate Generation Complete ({t_end - t_start:.2f}s)")
        print(f"    Output written to: {output_path}")
        print(f"    Total match frames: {n_frames:,}")
        print(f"    Modified frames (v5 -> v6): {changed_frames:,} ({changed_frames / n_frames * 100:.3f}%)")
        print(f"      * Away -> Home (un-triggered premature deceleration): {flips_ah:,}")
        print(f"      * Home -> Away (un-triggered premature deceleration): {flips_ha:,}")
        print(f"      * Live -> Dead Ball: {flips_live_d:,}")
        print(f"      * Dead Ball -> Live: {flips_d_live:,}")
        print(f"\n    Possession Spells Summary:")
        print(f"      * v5 Possessions: {len(poss_v5):<4} (mean: {np.mean(durs_v5):.2f}s, median: {np.median(durs_v5):.2f}s)")
        print(f"      * v6 Possessions: {len(poss_v6):<4} (mean: {np.mean(durs_v6):.2f}s, median: {np.median(durs_v6):.2f}s)")
        print(f"      * Net spell count change: {len(poss_v6) - len(poss_v5):+d} spells")
        print("=" * 82)

    return {
        "v4_path": v4_path,
        "output_path": output_path,
        "changed_frames": changed_frames,
        "flips_ah": flips_ah,
        "flips_ha": flips_ha,
        "flips_live_d": flips_live_d,
        "flips_d_live": flips_d_live,
        "n_spells_v5": len(poss_v5),
        "n_spells_v6": len(poss_v6),
        "elapsed_s": t_end - t_start,
    }


if __name__ == "__main__":
    run_predictor_v6()
