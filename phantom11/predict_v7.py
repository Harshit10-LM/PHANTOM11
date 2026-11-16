#!/usr/bin/env python3
"""PHANTOM-11: Version 7 Experimental Predictor (Pitch-Boundary Awareness & Refined Carrier Detection)

Step 16 of More Than A Hack 2027:
- Implements ablation variants:
  * v7-A: Pitch-boundary correction only (IFAB Law 9 boundary clamping to 'D').
  * v7-B: Carrier detection correction only (tightened moving carrier thresholds: max_dist=1.2m, max_rel_speed=1.5m/s).
  * v7-C: Both pitch-boundary and carrier-detection corrections combined.
- Preserves all previous outputs (v0..v6) untouched.
- Generates outputs/phantom11_v7a.csv, outputs/phantom11_v7b.csv, outputs/phantom11_v7c.csv.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom11.match_config import MatchConfig
from phantom11.load_tracking import load_full_tracking
from phantom11.control_evidence import compute_control_evidence
from phantom11.movement_features import compute_ball_kinematics
from phantom11.flight_handover import refine_flight_handovers
from phantom11.pitch_boundary import apply_pitch_boundary_clamping, detect_out_of_bounds_mask
from phantom11.path_resolver import resolve_data_dir, resolve_official_repo
from tests.test_flight_handover_benchmark import run_pipeline_v4
from baseline import write_submission
import load_smart_tagging


def run_pipeline_v5(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    fps: float = 25.0,
    st_events: list = None,
) -> np.ndarray:
    """Run full PHANTOM-11 v5 pipeline."""
    v4 = run_pipeline_v4(ball_xy, players_xy, team_of_track, non_play_intervals, fps=fps, st_events=st_events)
    ev = compute_control_evidence(ball_xy, players_xy, team_of_track, st_events or [], non_play_intervals, fps=fps)
    kin = compute_ball_kinematics(ball_xy, fps=fps)
    v5 = refine_flight_handovers(
        labels=v4,
        ball_present=kin["ball_present"],
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        d_h1=ev["d_h1"],
        d_a1=ev["d_a1"],
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        ball_speed=kin["speed"],
        non_play_intervals=non_play_intervals,
        fps=fps,
        gate_deceleration_by_proximity=False,
    )
    return v5


def run_pipeline_v5a(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    pitch: Tuple[float, float] = (105.3, 68.0),
    fps: float = 25.0,
    st_events: list = None,
) -> np.ndarray:
    """v5-A: Pitch-boundary awareness applied to v5."""
    v5 = run_pipeline_v5(ball_xy, players_xy, team_of_track, non_play_intervals, fps=fps, st_events=st_events)
    kin = compute_ball_kinematics(ball_xy, fps=fps)
    return apply_pitch_boundary_clamping(
        labels=v5,
        ball_xy=ball_xy,
        ball_present=kin["ball_present"],
        pitch=pitch,
        non_play_intervals=non_play_intervals,
    )


def run_pipeline_v6(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    fps: float = 25.0,
    st_events: list = None,
) -> np.ndarray:
    """Run full PHANTOM-11 v6 pipeline."""
    v4 = run_pipeline_v4(ball_xy, players_xy, team_of_track, non_play_intervals, fps=fps, st_events=st_events)
    ev = compute_control_evidence(ball_xy, players_xy, team_of_track, st_events or [], non_play_intervals, fps=fps)
    kin = compute_ball_kinematics(ball_xy, fps=fps)
    v6 = refine_flight_handovers(
        labels=v4,
        ball_present=kin["ball_present"],
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        d_h1=ev["d_h1"],
        d_a1=ev["d_a1"],
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        ball_speed=kin["speed"],
        non_play_intervals=non_play_intervals,
        fps=fps,
        gate_deceleration_by_proximity=True,
    )
    return v6


def run_pipeline_v7a(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    pitch: Tuple[float, float] = (105.3, 68.0),
    fps: float = 25.0,
    st_events: list = None,
) -> np.ndarray:
    """v7-A: Pitch-boundary awareness only applied to v6."""
    v6 = run_pipeline_v6(ball_xy, players_xy, team_of_track, non_play_intervals, fps=fps, st_events=st_events)
    kin = compute_ball_kinematics(ball_xy, fps=fps)
    return apply_pitch_boundary_clamping(
        labels=v6,
        ball_xy=ball_xy,
        ball_present=kin["ball_present"],
        pitch=pitch,
        non_play_intervals=non_play_intervals,
    )


def run_pipeline_v7b(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    pitch: Tuple[float, float] = (105.3, 68.0),
    fps: float = 25.0,
    st_events: list = None,
    v3_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """v7-B: Refined carrier detection only (max_dist=1.2m, max_rel_speed=1.5m/s)."""
    # 1. Run v4 with refined carrier detection
    # In v4, boundary refinement uses control evidence
    ev = compute_control_evidence(
        ball_xy, players_xy, team_of_track, st_events or [], non_play_intervals,
        fps=fps, max_carrier_dist=1.2, max_carrier_rel_speed=1.5, min_carrier_alignment=0.50
    )
    kin = compute_ball_kinematics(ball_xy, fps=fps)
    
    # We run the standard v4 decoder base with refined evidence
    from tests.test_flight_handover_benchmark import run_pipeline_v3
    from phantom11.boundary_refinement import refine_possession_boundaries
    n_frames = len(ball_xy)
    if v3_labels is None:
        v3 = run_pipeline_v3(ball_xy, players_xy, team_of_track, non_play_intervals, fps=fps, st_events=st_events)
    else:
        v3 = v3_labels
    v4_b = refine_possession_boundaries(
        labels=v3,
        ball_present=kin["ball_present"],
        ball_xy=ball_xy,
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        d_h1=ev["d_h1"],
        d_a1=ev["d_a1"],
        ball_speed=kin["speed"],
        stoppage_support=np.zeros(n_frames, dtype=np.float32),
        non_play_intervals=non_play_intervals,
        fps=fps,
    )
    
    # 2. Run flight handover with refined carrier flags
    v7b = refine_flight_handovers(
        labels=v4_b,
        ball_present=kin["ball_present"],
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        d_h1=ev["d_h1"],
        d_a1=ev["d_a1"],
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        ball_speed=kin["speed"],
        non_play_intervals=non_play_intervals,
        fps=fps,
        gate_deceleration_by_proximity=True,
    )
    return v7b


def run_pipeline_v7c(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    team_of_track: np.ndarray,
    non_play_intervals: List[Tuple[int, int]],
    pitch: Tuple[float, float] = (105.3, 68.0),
    fps: float = 25.0,
    st_events: list = None,
    v3_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """v7-C: Combined pitch-boundary awareness and refined carrier detection."""
    v7b = run_pipeline_v7b(
        ball_xy, players_xy, team_of_track, non_play_intervals,
        pitch=pitch, fps=fps, st_events=st_events, v3_labels=v3_labels
    )
    kin = compute_ball_kinematics(ball_xy, fps=fps)
    return apply_pitch_boundary_clamping(
        labels=v7b,
        ball_xy=ball_xy,
        ball_present=kin["ball_present"],
        pitch=pitch,
        non_play_intervals=non_play_intervals,
    )


def generate_full_match_v7(variant: str = "v7c", output_path: str = None, game: str = "BAR-ATH", data_dir: str = None, verbose: bool = True):
    """Generate full 199,319-frame prediction for a v7 variant or v5a."""
    t_start = time.time()
    variant = variant.lower()
    if variant not in ["v5a", "v7a", "v7b", "v7c"]:
        raise ValueError(f"Invalid variant {variant}. Choose v5a, v7a, v7b, or v7c.")

    output_path = output_path or os.path.join(PROJECT_ROOT, "outputs", f"phantom11_{variant}.csv")
    if verbose:
        print("=" * 85)
        print(f"PHANTOM-11: GENERATING FULL-MATCH EXPERIMENTAL CANDIDATE {variant.upper()}")
        print("=" * 85)

    cfg = MatchConfig.from_official_game(game=game, data_dir=data_dir)
    ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, _ = load_full_tracking(match_config=cfg)
    st_events = load_smart_tagging.load(game=cfg.match_id, data_dir=cfg.data_dir)
    non_play = cfg.get_non_play_intervals()
    fps = cfg.fps

    if variant == "v5a":
        # Load v5 and clamp
        v5_path = os.path.join(PROJECT_ROOT, "outputs", "phantom11_v5.csv")
        if verbose:
            print(f"Loading base v5 from: {v5_path}")
        v5_df = pd.read_csv(v5_path)
        v5_labels = v5_df["label"].values.astype("<U1")
        labels = apply_pitch_boundary_clamping(
            labels=v5_labels,
            ball_xy=ball_xy,
            ball_present=ball_present,
            pitch=pitch,
            non_play_intervals=non_play,
        )
    elif variant == "v7a":
        # Load v6 and clamp
        v6_path = os.path.join(PROJECT_ROOT, "outputs", "phantom11_v6.csv")
        if verbose:
            print(f"Loading base v6 from: {v6_path}")
        v6_df = pd.read_csv(v6_path)
        v6_labels = v6_df["label"].values.astype("<U1")
        kin = compute_ball_kinematics(ball_xy, fps=fps)
        labels = apply_pitch_boundary_clamping(
            labels=v6_labels,
            ball_xy=ball_xy,
            ball_present=ball_present,
            pitch=pitch,
            non_play_intervals=non_play,
        )
    elif variant == "v7b":
        v3_path = os.path.join(PROJECT_ROOT, "outputs", "phantom11_v3.csv")
        if verbose:
            print(f"Loading base v3 from: {v3_path}")
        v3_df = pd.read_csv(v3_path)
        v3_labels = v3_df["label"].values.astype("<U1")
        if verbose:
            print("Running full pipeline with tightened carrier detection...")
        labels = run_pipeline_v7b(
            ball_xy=ball_xy,
            players_xy=players_xy,
            team_of_track=team_of_track,
            non_play_intervals=non_play,
            pitch=pitch,
            fps=fps,
            st_events=st_events,
            v3_labels=v3_labels,
        )
    elif variant == "v7c":
        v3_path = os.path.join(PROJECT_ROOT, "outputs", "phantom11_v3.csv")
        if verbose:
            print(f"Loading base v3 from: {v3_path}")
        v3_df = pd.read_csv(v3_path)
        v3_labels = v3_df["label"].values.astype("<U1")
        if verbose:
            print("Running combined pipeline (tightened carrier + pitch boundary clamping)...")
        labels = run_pipeline_v7c(
            ball_xy=ball_xy,
            players_xy=players_xy,
            team_of_track=team_of_track,
            non_play_intervals=non_play,
            pitch=pitch,
            fps=fps,
            st_events=st_events,
            v3_labels=v3_labels,
        )

    # Write output CSV
    write_submission(output_path, labels)
    elapsed = time.time() - t_start
    if verbose:
        print(f"Successfully wrote {len(labels):,} frames to {output_path} in {elapsed:.2f}s")
    return labels, output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PHANTOM-11 v7 Experimental Predictor")
    parser.add_argument("--variant", choices=["v5a", "v7a", "v7b", "v7c", "all"], default="all")
    args = parser.parse_args()

    if args.variant == "all":
        for v in ["v7a", "v7b", "v7c"]:
            generate_full_match_v7(variant=v)
    else:
        generate_full_match_v7(variant=args.variant)
