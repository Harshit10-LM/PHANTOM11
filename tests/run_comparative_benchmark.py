#!/usr/bin/env python3
"""Run comprehensive comparative evaluation on v3, v4, and v5 across all 10 frozen scenarios."""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_flight_handover_benchmark import (
    get_all_frozen_scenarios,
    run_pipeline_v3,
    run_pipeline_v4,
    compute_boundary_timing_error,
)
from phantom11.flight_handover import refine_flight_handovers
from phantom11.control_evidence import compute_control_evidence
from phantom11.movement_features import compute_ball_kinematics
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm


def run_pipeline_v5(ball_xy, players_xy, team_of_track, non_play_intervals, fps=25.0, st_events=None):
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
    )
    return v5


def main():
    scenarios = get_all_frozen_scenarios()
    seeds = (42, 101, 777)

    print("=" * 102)
    print("{:<52} | {:<8} | {:<8} | {:<8} | {:<10} | {:<12}".format(
        "Scenario Name", "v3 F1", "v4 F1", "v5 F1", "v5 vs v4", "Outcome"
    ))
    print("-" * 102)

    v3_scores, v4_scores, v5_scores = [], [], []
    v3_b_errs, v4_b_errs, v5_b_errs = [], [], []
    v3_noise, v4_noise, v5_noise = [], [], []
    v3_drop, v4_drop, v5_drop = [], [], []

    for name, b_xy, p_xy, teams, gt in scenarios:
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)

        p3 = run_pipeline_v3(b_xy, p_xy, teams, [])
        p4 = run_pipeline_v4(b_xy, p_xy, teams, [])
        p5 = run_pipeline_v5(b_xy, p_xy, teams, [])

        s3 = pm.spell_scores(gt_codes, pm.label_codes(p3), runs)["mean_f1"]
        s4 = pm.spell_scores(gt_codes, pm.label_codes(p4), runs)["mean_f1"]
        s5 = pm.spell_scores(gt_codes, pm.label_codes(p5), runs)["mean_f1"]

        e3 = compute_boundary_timing_error(gt, p3)
        e4 = compute_boundary_timing_error(gt, p4)
        e5 = compute_boundary_timing_error(gt, p5)

        v3_scores.append(s3)
        v4_scores.append(s4)
        v5_scores.append(s5)
        v3_b_errs.append(e3)
        v4_b_errs.append(e4)
        v5_b_errs.append(e5)

        # Perturbations: Coordinate Noise (sigma = 0.2m, 0.5m)
        n3_list, n4_list, n5_list = [], [], []
        for sigma in (0.20, 0.50):
            for s in seeds:
                np.random.seed(s)
                b_n = b_xy.copy() + np.random.normal(0, sigma, b_xy.shape)
                p_n = p_xy.copy() + np.random.normal(0, sigma, p_xy.shape)
                n3_list.append(pm.spell_scores(gt_codes, pm.label_codes(run_pipeline_v3(b_n, p_n, teams, [])), runs)["mean_f1"])
                n4_list.append(pm.spell_scores(gt_codes, pm.label_codes(run_pipeline_v4(b_n, p_n, teams, [])), runs)["mean_f1"])
                n5_list.append(pm.spell_scores(gt_codes, pm.label_codes(run_pipeline_v5(b_n, p_n, teams, [])), runs)["mean_f1"])
        v3_noise.append(np.mean(n3_list))
        v4_noise.append(np.mean(n4_list))
        v5_noise.append(np.mean(n5_list))

        # Perturbations: Tracking Dropouts (5%, 15%)
        d3_list, d4_list, d5_list = [], [], []
        for drop in (0.05, 0.15):
            for s in seeds:
                np.random.seed(s)
                b_d = b_xy.copy()
                b_d[np.random.rand(len(b_d)) < drop] = np.nan
                d3_list.append(pm.spell_scores(gt_codes, pm.label_codes(run_pipeline_v3(b_d, p_xy, teams, [])), runs)["mean_f1"])
                d4_list.append(pm.spell_scores(gt_codes, pm.label_codes(run_pipeline_v4(b_d, p_xy, teams, [])), runs)["mean_f1"])
                d5_list.append(pm.spell_scores(gt_codes, pm.label_codes(run_pipeline_v5(b_d, p_xy, teams, [])), runs)["mean_f1"])
        v3_drop.append(np.mean(d3_list))
        v4_drop.append(np.mean(d4_list))
        v5_drop.append(np.mean(d5_list))

        diff = s5 - s4
        outcome = "TIED"
        if diff > 0.001:
            outcome = "v5 IMPROVED"
        elif diff < -0.001:
            outcome = "v5 REGRESSED"
        print("{:<52} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>+8.4f}   | {:<12}".format(
            name, s3, s4, s5, diff, outcome
        ))

    m3, m4, m5 = np.mean(v3_scores), np.mean(v4_scores), np.mean(v5_scores)
    be3, be4, be5 = np.mean(v3_b_errs), np.mean(v4_b_errs), np.mean(v5_b_errs)
    n3, n4, n5 = np.mean(v3_noise), np.mean(v4_noise), np.mean(v5_noise)
    d3, d4, d5 = np.mean(v3_drop), np.mean(v4_drop), np.mean(v5_drop)

    print("-" * 102)
    print("{:<52} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>+8.4f}   | v5 {:+.4f}".format(
        "OVERALL MEAN spellF1", m3, m4, m5, m5 - m4, m5 - m4
    ))
    print("{:<52} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>+8.2f}   | {:<12}".format(
        "MEAN BOUNDARY ERROR (frames)", be3, be4, be5, be5 - be4, "REDUCED" if be5 <= be4 else "INCREASED"
    ))
    print("{:<52} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>+8.4f}   | {:<12}".format(
        "MEAN NOISE ROBUSTNESS spellF1", n3, n4, n5, n5 - n4, "ROBUST" if n5 >= n4 else "DEGRADED"
    ))
    print("{:<52} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>+8.4f}   | {:<12}".format(
        "MEAN DROPOUT ROBUSTNESS spellF1", d3, d4, d5, d5 - d4, "ROBUST" if d5 >= d4 else "DEGRADED"
    ))
    print("=" * 102)


if __name__ == "__main__":
    main()
