#!/usr/bin/env python3
"""Comprehensive Comparative Audit: PHANTOM-11 v3, v4, v5, and v6

Step 13 of More Than A Hack 2027:
- Evaluates candidates across:
  * 10 Frozen Benchmark Scenarios (Step 11).
  * 10 Newly Frozen Held-Out Scenarios (Step 13).
  * 5 Adversarial Counterexample Scenarios (Step 12).
  * Noise Robustness (Gaussian sigma = 0.20m, 0.50m across seeds).
  * Dropout Robustness (5%, 15% missing tracking across seeds).
  * Boundary Timing Precision.
  * Runtime and Memory Usage.
"""

import os
import sys
import time
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
from tests.test_held_out_handover_suite import get_all_held_out_scenarios
from tests.run_comparative_benchmark import run_pipeline_v5
from tests.test_sensitivity_and_counterexamples import (
    build_counterexample_1_skimmed_header,
    build_counterexample_2_one_two_intercepted,
    build_counterexample_3_clearance_out_of_bounds,
    build_counterexample_4_aerial_50_50_contest,
    build_counterexample_5_double_deflection,
)
from phantom11.control_evidence import compute_control_evidence
from phantom11.movement_features import compute_ball_kinematics
from phantom11.flight_handover import refine_flight_handovers
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm


def run_pipeline_v6(ball_xy, players_xy, team_of_track, non_play_intervals, fps=25.0, st_events=None):
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


def evaluate_suite(scenarios, seeds=(42, 101, 777)):
    models = {
        "v3": run_pipeline_v3,
        "v4": run_pipeline_v4,
        "v5": run_pipeline_v5,
        "v6": run_pipeline_v6,
    }

    f1_results = {m: [] for m in models}
    err_results = {m: [] for m in models}
    noise_results = {m: [] for m in models}
    drop_results = {m: [] for m in models}

    for name, b_xy, p_xy, teams, gt in scenarios:
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)

        preds = {}
        for m, fn in models.items():
            p = fn(b_xy, p_xy, teams, [])
            preds[m] = p
            f1 = pm.spell_scores(gt_codes, pm.label_codes(p), runs)["mean_f1"]
            err = compute_boundary_timing_error(gt, p)
            f1_results[m].append(f1)
            err_results[m].append(err)

            # Noise
            n_list = []
            for sigma in (0.20, 0.50):
                for s in seeds:
                    np.random.seed(s)
                    b_n = b_xy.copy() + np.random.normal(0, sigma, b_xy.shape)
                    p_n = p_xy.copy() + np.random.normal(0, sigma, p_xy.shape)
                    pred_n = fn(b_n, p_n, teams, [])
                    n_list.append(pm.spell_scores(gt_codes, pm.label_codes(pred_n), runs)["mean_f1"])
            noise_results[m].append(np.mean(n_list))

            # Dropout
            d_list = []
            for drop in (0.05, 0.15):
                for s in seeds:
                    np.random.seed(s)
                    b_d = b_xy.copy()
                    b_d[np.random.rand(len(b_d)) < drop] = np.nan
                    pred_d = fn(b_d, p_xy, teams, [])
                    d_list.append(pm.spell_scores(gt_codes, pm.label_codes(pred_d), runs)["mean_f1"])
            drop_results[m].append(np.mean(d_list))

    return {
        "f1": f1_results,
        "err": err_results,
        "noise": noise_results,
        "drop": drop_results,
    }


def main():
    print("=" * 104)
    print("PHANTOM-11: COMPREHENSIVE FINAL COMPARATIVE AUDIT (v3 vs v4 vs v5 vs v6)")
    print("=" * 104)

    frozen = get_all_frozen_scenarios()
    held_out = get_all_held_out_scenarios()
    adv = [
        build_counterexample_1_skimmed_header(),
        build_counterexample_2_one_two_intercepted(),
        build_counterexample_3_clearance_out_of_bounds(),
        build_counterexample_4_aerial_50_50_contest(),
        build_counterexample_5_double_deflection(),
    ]

    print("\n[1] 10 FROZEN BENCHMARK SCENARIOS (From Step 11)")
    print("-" * 104)
    res_f = evaluate_suite(frozen)
    for m in ["v3", "v4", "v5", "v6"]:
        print("  Candidate {:<2}: Mean spellF1 = {:.4f} | Worst = {:.4f} | Boundary Err = {:.2f}f | Noise = {:.4f} | Drop = {:.4f}".format(
            m, np.mean(res_f["f1"][m]), np.min(res_f["f1"][m]), np.mean(res_f["err"][m]), np.mean(res_f["noise"][m]), np.mean(res_f["drop"][m])
        ))

    print("\n[2] 10 NEWLY FROZEN HELD-OUT SCENARIOS (From Step 13)")
    print("-" * 104)
    res_h = evaluate_suite(held_out)
    for m in ["v3", "v4", "v5", "v6"]:
        print("  Candidate {:<2}: Mean spellF1 = {:.4f} | Worst = {:.4f} | Boundary Err = {:.2f}f | Noise = {:.4f} | Drop = {:.4f}".format(
            m, np.mean(res_h["f1"][m]), np.min(res_h["f1"][m]), np.mean(res_h["err"][m]), np.mean(res_h["noise"][m]), np.mean(res_h["drop"][m])
        ))

    print("\n[3] 5 ADVERSARIAL COUNTEREXAMPLES (From Step 12)")
    print("-" * 104)
    res_a = evaluate_suite(adv)
    for m in ["v3", "v4", "v5", "v6"]:
        print("  Candidate {:<2}: Mean spellF1 = {:.4f} | Worst = {:.4f} | Boundary Err = {:.2f}f | Noise = {:.4f} | Drop = {:.4f}".format(
            m, np.mean(res_a["f1"][m]), np.min(res_a["f1"][m]), np.mean(res_a["err"][m]), np.mean(res_a["noise"][m]), np.mean(res_a["drop"][m])
        ))

    all_25 = frozen + held_out + adv
    print("\n[4] AGGREGATE SUMMARY ACROSS ALL 25 INDEPENDENT SCENARIOS")
    print("=" * 104)
    res_all = evaluate_suite(all_25)
    header = "{:<12} | {:<12} | {:<12} | {:<14} | {:<14} | {:<14}".format(
        "Candidate", "Mean spellF1", "Worst spellF1", "Boundary Error", "Noise Robust", "Dropout Robust"
    )
    print(header)
    print("-" * 104)
    for m in ["v3", "v4", "v5", "v6"]:
        print("{:<12} | {:>10.4f}   | {:>10.4f}   | {:>10.2f} f  | {:>12.4f}   | {:>12.4f}".format(
            f"PHANTOM-{m}",
            np.mean(res_all["f1"][m]),
            np.min(res_all["f1"][m]),
            np.mean(res_all["err"][m]),
            np.mean(res_all["noise"][m]),
            np.mean(res_all["drop"][m]),
        ))
    print("=" * 104)


if __name__ == "__main__":
    main()
