#!/usr/bin/env python3
"""PHANTOM-11: Full End-to-End Prediction Pipeline

Executes the complete, deterministic pipeline from raw data to final v3 submission:
  Raw Tactical ATD -> v0 Baseline -> v1 Continuity -> v2 Evidence -> v3 Sequence Decoder

Can run against existing intermediate files or generate everything from scratch.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


try:
    from phantom11.path_resolver import resolve_official_repo, resolve_data_dir
except ImportError:
    from path_resolver import resolve_official_repo, resolve_data_dir

# Resolve official repo
official_root = resolve_official_repo()
import games
import load_smart_tagging
from baseline import build, smooth, write_submission, team_of_each_track
from phantom11.continuity import apply_continuity, NON_PLAY_INTERVALS
from phantom11.load_tracking import load_full_tracking
from phantom11.control_evidence import compute_control_evidence
from phantom11.boundary_evidence import extract_boundary_evidence, derive_non_play_intervals
from phantom11.movement_features import compute_ball_kinematics
from phantom11.sequence_decoder import decode_possession_sequence, DEFAULT_CONFIG
from phantom11.tactical_tags import parse_smart_tagging_timeline
from phantom11.match_config import MatchConfig


def run_pipeline(
    data_dir=None,
    output_dir=None,
    game="BAR-ATH",
    match_config=None,
    save_intermediates=True,
    verbose=True,
    candidate="v5a",
):
    """Run the complete PHANTOM-11 pipeline and return the candidate labels.

    Parameters
    ----------
    data_dir : str, optional
        Path to directory containing tactical data files.
    output_dir : str, optional
        Directory where outputs are saved (defaults to PROJECT_ROOT/outputs).
    game : str
        Game identifier ('BAR-ATH').
    match_config : MatchConfig, optional
        Configured match structure (if None, derived from games.py and game).
    save_intermediates : bool
        Whether to save intermediate candidates along with final.
    verbose : bool
        Whether to print diagnostic progress.
    candidate : str
        Candidate version to generate ('v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v5a').
        Default: 'v5a' (Primary Competition Candidate).

    Returns
    -------
    final_labels : np.ndarray
        Array of prediction labels ('H', 'A', 'D').
    output_paths : dict
        Mapping of version to output CSV path.
    """
    t_start = time.time()
    data_dir = data_dir or resolve_data_dir()
    output_dir = output_dir or os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Resolve match configuration
    if match_config is None:
        match_config = MatchConfig.from_official_game(game=game, data_dir=data_dir)

    is_valid, issues = match_config.validate()
    if not is_valid:
        raise ValueError(
            f"Invalid MatchConfig for match '{match_config.match_id}':\n  " + "\n  ".join(issues)
        )

    game = match_config.match_id
    fps = match_config.fps
    pitch = match_config.pitch
    n_frames = match_config.n_frames
    non_play = match_config.get_non_play_intervals()
    team_map = match_config.get_team_map()

    if verbose:
        print("=" * 78)
        print("PHANTOM-11: REPRODUCIBLE END-TO-END PIPELINE (v0 -> v1 -> v2 -> v3)")
        print("=" * 78)
        print(f"Match:            {match_config.home_team} vs {match_config.away_team} [{game}]")
        print(f"Frames:           {n_frames:,} ({n_frames/fps/60:.1f} mins @ {fps} fps)")
        print(f"Pitch Dimensions: {pitch[0]:.1f}m x {pitch[1]:.1f}m")
        print(f"Data directory:   {data_dir}")
        print(f"Output directory: {output_dir}")

    # -----------------------------------------------------------------------
    # Step 4: Baseline v0 (Nearest player with 50-frame majority smoothing)
    # -----------------------------------------------------------------------
    v0_path = os.path.join(output_dir, "phantom11_v0.csv")
    if os.path.isfile(v0_path):
        if verbose:
            print(f"\n[Step 4] Loading existing v0 from {v0_path}...")
        v0_labels = pd.read_csv(v0_path)["label"].values.astype("<U1")
    else:
        if verbose:
            print("\n[Step 4] Building baseline v0 from raw tactical ATD...")
        raw_codes = build(game=game, data_dir=data_dir)
        v0_labels = smooth(raw_codes, 50)
        if save_intermediates:
            write_submission(v0_path, v0_labels)
            if verbose:
                print(f"         Saved v0 to {v0_path}")

    # -----------------------------------------------------------------------
    # Step 5: Continuity v1 (Conservative missing-ball continuity)
    # -----------------------------------------------------------------------
    v1_path = os.path.join(output_dir, "phantom11_v1.csv")
    # Load tracking and Smart Tagging
    ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, n_frames = load_full_tracking(
        game=game, data_dir=data_dir, match_config=match_config
    )
    st_events = load_smart_tagging.load(game=game, data_dir=data_dir)

    if os.path.isfile(v1_path):
        if verbose:
            print(f"\n[Step 5] Loading existing v1 from {v1_path}...")
        v1_labels = pd.read_csv(v1_path)["label"].values.astype("<U1")
    else:
        if verbose:
            print("\n[Step 5] Applying conservative missing-ball continuity (v1)...")
        # Nearest team array
        d_raw = np.sqrt(np.sum((players_xy - ball_xy[:, np.newaxis, :]) ** 2, axis=2))
        d_clean = np.where(np.isnan(d_raw), 100.0, d_raw)
        nearest_slot = np.argmin(d_clean, axis=1)
        nearest_team = np.where(ball_present, team_of_track[nearest_slot], -1).astype(np.int8)

        set_pieces_intervals = [
            (e["start_frame"], e["end_frame"])
            for e in st_events
            if e["code"] in ("SET PIECES", "GOALS")
        ]

        v1_labels, _ = apply_continuity(
            v0_labels=v0_labels,
            ball_present=ball_present,
            ball_coords=ball_xy,
            nearest_team=nearest_team,
            set_pieces_intervals=set_pieces_intervals,
            pitch=pitch,
            non_play_intervals=non_play,
        )
        if save_intermediates:
            write_submission(v1_path, v1_labels)
            if verbose:
                print(f"         Saved v1 to {v1_path}")

    # -----------------------------------------------------------------------
    # Step 6: Evidence v2 (Active carrier overrides)
    # -----------------------------------------------------------------------
    v2_path = os.path.join(output_dir, "phantom11_v2.csv")
    if os.path.isfile(v2_path):
        if verbose:
            print(f"\n[Step 6] Loading existing v2 from {v2_path}...")
        v2_labels = pd.read_csv(v2_path)["label"].values.astype("<U1")
        # Compute evidence needed for v3
        ev = compute_control_evidence(
            ball_xy=ball_xy,
            players_xy=players_xy,
            team_of_track=team_of_track,
            smart_tagging_events=st_events,
            non_play_intervals=non_play,
            fps=fps,
        )
    else:
        if verbose:
            print("\n[Step 6] Computing control evidence and applying carrier overrides (v2)...")
        ev = compute_control_evidence(
            ball_xy=ball_xy,
            players_xy=players_xy,
            team_of_track=team_of_track,
            smart_tagging_events=st_events,
            non_play_intervals=non_play,
            fps=fps,
        )
        v2_labels = v1_labels.copy()

        in_non_play = np.zeros(n_frames, dtype=bool)
        for s, e in non_play:
            s_c = max(0, s)
            e_c = min(n_frames - 1, e)
            if s_c <= e_c:
                in_non_play[s_c:e_c + 1] = True

        c_h_mask = (
            ev["is_carrier_h"]
            & ~in_non_play
            & (ev["score_h"] >= 0.55)
            & (ev["d_h1"] <= 1.80)
            & (ev["margin_d"] >= 1.00)
        )
        c_a_mask = (
            ev["is_carrier_a"]
            & ~in_non_play
            & (ev["score_a"] >= 0.55)
            & (ev["d_a1"] <= 1.80)
            & (ev["margin_d"] <= -1.00)
        )
        v2_labels[c_h_mask & (v1_labels != "H")] = "H"
        v2_labels[c_a_mask & (v1_labels != "A")] = "A"

        for s, e in non_play:
            s_c = max(0, s)
            e_c = min(n_frames - 1, e)
            if s_c <= e_c:
                v2_labels[s_c:e_c + 1] = "D"

        if save_intermediates:
            write_submission(v2_path, v2_labels)
            if verbose:
                print(f"         Saved v2 to {v2_path}")

    # -----------------------------------------------------------------------
    # Step 7: Sequence Decoder v3 (Duration-Aware Dynamic Programming)
    # -----------------------------------------------------------------------
    v3_path = os.path.join(output_dir, "phantom11_v3.csv")
    if verbose:
        print("\n[Step 7] Running duration-aware sequence decoder (v3)...")

    ball_kin = compute_ball_kinematics(ball_xy, fps=fps)
    tact = parse_smart_tagging_timeline(st_events, n_frames, team_map=team_map)
    geom_nearest = np.where(ev["d_h1"] <= ev["d_a1"], 0, 1).astype(np.int8)

    b_ev = extract_boundary_evidence(
        ball_present=ball_present,
        is_kick_or_deflection=ball_kin["is_kick_or_deflection"],
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        nearest_team=geom_nearest,
        smart_tagging_stoppage=tact["stoppage_support"],
        non_play_intervals=non_play,
    )

    v3_labels = decode_possession_sequence(
        v2_labels=v2_labels,
        score_h=ev["score_h"],
        score_a=ev["score_a"],
        reliability=ev["reliability"],
        ball_present=ball_present,
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        stoppage_support=tact["stoppage_support"],
        turnover_readiness_ha=b_ev["turnover_readiness_ha"],
        turnover_readiness_ah=b_ev["turnover_readiness_ah"],
        stoppage_readiness=b_ev["stoppage_readiness"],
        restart_readiness=b_ev["restart_readiness"],
        in_non_play=b_ev["in_non_play"],
        config=DEFAULT_CONFIG,
    )

    write_submission(v3_path, v3_labels)
    if verbose:
        print(f"         Saved v3 to {v3_path}")

    output_paths = {
        "v0": v0_path,
        "v1": v1_path,
        "v2": v2_path,
        "v3": v3_path,
    }
    final_labels = v3_labels

    if candidate in ("v4", "v5", "v5a"):
        from phantom11.boundary_refinement import refine_possession_boundaries
        v4_path = os.path.join(output_dir, "phantom11_v4.csv")
        if verbose:
            print("\n[Step 8] Applying physical boundary refinement (v4)...")
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
        if save_intermediates or candidate == "v4":
            write_submission(v4_path, v4_labels)
            if verbose:
                print(f"         Saved v4 to {v4_path}")
        output_paths["v4"] = v4_path
        final_labels = v4_labels

    if candidate in ("v5", "v5a"):
        from phantom11.flight_handover import refine_flight_handovers
        v5_path = os.path.join(output_dir, "phantom11_v5.csv")
        if verbose:
            print("\n[Step 9] Applying flight-aware possession handover (v5)...")
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
            gate_deceleration_by_proximity=False,
        )
        if save_intermediates or candidate == "v5":
            write_submission(v5_path, v5_labels)
            if verbose:
                print(f"         Saved v5 to {v5_path}")
        output_paths["v5"] = v5_path
        final_labels = v5_labels

    if candidate == "v5a":
        from phantom11.pitch_boundary import apply_pitch_boundary_clamping
        v5a_path = os.path.join(output_dir, "phantom11_v5a.csv")
        team_named_path = os.path.join(output_dir, "No Data No Goal.csv")
        algo_named_path = os.path.join(output_dir, "PHANTOM-11.csv")
        if verbose:
            print("\n[Step 10] Applying IFAB Law 9 pitch-boundary clamping (v5-A)...")
        v5a_labels = apply_pitch_boundary_clamping(
            labels=v5_labels,
            ball_xy=ball_xy,
            ball_present=ball_present,
            pitch=pitch,
            non_play_intervals=non_play,
        )
        write_submission(v5a_path, v5a_labels)
        write_submission(team_named_path, v5a_labels)
        write_submission(algo_named_path, v5a_labels)
        if verbose:
            print(f"         Saved primary candidate v5-A to {v5a_path}")
            print(f"         Saved official team submission to {team_named_path}")
            print(f"         Saved algorithm-named submission to {algo_named_path}")
        output_paths["v5a"] = v5a_path
        output_paths["submission"] = team_named_path
        output_paths["algo_submission"] = algo_named_path
        final_labels = v5a_labels

    t_end = time.time()
    if verbose:
        print(f"\nPipeline finished in {t_end - t_start:.2f} seconds.")
        print(f"Final Candidate Generated: {candidate.upper()} ({output_paths[candidate]})")
        print(f"Total rows: {len(final_labels):,} | Format valid: 0..{len(final_labels)-1}")
        print("=" * 78)

    return final_labels, output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHANTOM-11 End-to-End Prediction Pipeline")
    parser.add_argument("--candidate", "--version", default="v5a", choices=["v0", "v1", "v2", "v3", "v4", "v5", "v5a"], help="Model candidate version to generate (default: v5a)")
    parser.add_argument("--data-dir", default=None, help="Directory containing tactical data files")
    parser.add_argument("--output-dir", default=None, help="Directory to save generated CSV files")
    parser.add_argument("--official-dir", default=None, help="Path to official more-than-a-hack-2027 repository")
    args = parser.parse_args()

    if args.official_dir:
        resolve_official_repo(args.official_dir)

    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, candidate=args.candidate)
