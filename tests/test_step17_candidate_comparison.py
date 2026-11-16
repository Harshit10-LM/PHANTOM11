#!/usr/bin/env python3
"""Step 17: Comprehensive Candidate Comparison & Evaluation Suite

Compares:
- v5: Flight-aware handover baseline
- v6: Evidence-gated handover alignment
- v7-A: Pitch-boundary awareness only (IFAB Law 9 applied to v6)
- v7-C: Combined pitch-boundary awareness and refined carrier detection
- v5-A: Pitch-boundary awareness applied to v5

Evaluates across:
1. 10 Frozen Benchmark Scenarios (Step 11)
2. 10 Frozen Held-Out Scenarios (Step 13)
3. 5 Adversarial Counterexamples (Step 12)
4. 6 Step 16 Untouched Evaluation Scenarios
5. Total Aggregate Across All 31 Scenarios
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Any

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
from tests.test_sensitivity_and_counterexamples import (
    build_counterexample_1_skimmed_header,
    build_counterexample_2_one_two_intercepted,
    build_counterexample_3_clearance_out_of_bounds,
    build_counterexample_4_aerial_50_50_contest,
    build_counterexample_5_double_deflection,
)
from phantom11.predict_v7 import (
    run_pipeline_v5,
    run_pipeline_v5a,
    run_pipeline_v6,
    run_pipeline_v7a,
    run_pipeline_v7c,
)
from tests.test_step16_independent_evaluation import get_all_step16_eval_scenarios
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm


def evaluate_candidates_on_scenarios(scenarios: list, suite_name: str) -> dict:
    candidates = ["v5", "v6", "v7a", "v7c", "v5a"]
    scores = {c: [] for c in candidates}
    b_errs = {c: [] for c in candidates}

    print("\n" + "=" * 105)
    print(f"EVALUATING SUITE: {suite_name} ({len(scenarios)} scenarios)")
    print("=" * 105)
    header = "{:<50} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7}".format(
        "Scenario Name", "v5", "v6", "v7-A", "v7-C", "v5-A"
    )
    print(header)
    print("-" * 105)

    pipeline_map = {
        "v5": run_pipeline_v5,
        "v6": run_pipeline_v6,
        "v7a": run_pipeline_v7a,
        "v7c": run_pipeline_v7c,
        "v5a": run_pipeline_v5a,
    }

    for item in scenarios:
        if len(item) == 5:
            name, b, p, t, gt = item
        else:
            b, p, t, gt = item
            name = "Scenario"
        runs = [(0, len(gt))]
        gt_codes = pm.label_codes(gt)

        row_scores = {}
        for c in candidates:
            fn = pipeline_map[c]
            pred = fn(b, p, t, [])
            res = pm.spell_scores(gt_codes, pm.label_codes(pred), runs)
            s = res["mean_f1"]
            err = compute_boundary_timing_error(gt, pred)
            scores[c].append(s)
            b_errs[c].append(err)
            row_scores[c] = s

        print("{:<50} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f}".format(
            name[:50], row_scores["v5"], row_scores["v6"], row_scores["v7a"], row_scores["v7c"], row_scores["v5a"]
        ))

    print("-" * 105)
    mean_str = "{:<50} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f}".format(
        "MEAN SPELL F1", np.mean(scores["v5"]), np.mean(scores["v6"]), np.mean(scores["v7a"]), np.mean(scores["v7c"]), np.mean(scores["v5a"])
    )
    print(mean_str)
    err_str = "{:<50} | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f | {:>6.2f}f".format(
        "MEAN BOUNDARY ERROR (frames)", np.mean(b_errs["v5"]), np.mean(b_errs["v6"]), np.mean(b_errs["v7a"]), np.mean(b_errs["v7c"]), np.mean(b_errs["v5a"])
    )
    print(err_str)
    print("=" * 105)
    return {
        "scores": scores,
        "b_errs": b_errs,
    }


def main():
    frozen = get_all_frozen_scenarios()
    held_out = get_all_held_out_scenarios()
    adv = [
        build_counterexample_1_skimmed_header(),
        build_counterexample_2_one_two_intercepted(),
        build_counterexample_3_clearance_out_of_bounds(),
        build_counterexample_4_aerial_50_50_contest(),
        build_counterexample_5_double_deflection(),
    ]
    step16_eval = get_all_step16_eval_scenarios()

    r1 = evaluate_candidates_on_scenarios(frozen, "1. 10 FROZEN BENCHMARK SCENARIOS (Step 11)")
    r2 = evaluate_candidates_on_scenarios(held_out, "2. 10 HELD-OUT SCENARIOS (Step 13)")
    r3 = evaluate_candidates_on_scenarios(adv, "3. 5 ADVERSARIAL COUNTEREXAMPLES (Step 12)")
    r4 = evaluate_candidates_on_scenarios(step16_eval, "4. 6 STEP 16 UNTOUCHED EVALUATION SCENARIOS")

    candidates = ["v5", "v6", "v7a", "v7c", "v5a"]
    all_scores = {c: r1["scores"][c] + r2["scores"][c] + r3["scores"][c] + r4["scores"][c] for c in candidates}
    all_errs = {c: r1["b_errs"][c] + r2["b_errs"][c] + r3["b_errs"][c] + r4["b_errs"][c] for c in candidates}

    print("\n" + "=" * 105)
    print(f"OVERALL AGGREGATE SUMMARY ACROSS ALL 31 INDEPENDENT SCENARIOS")
    print("=" * 105)
    print("{:<20} | {:<15} | {:<20}".format("Candidate Model", "Mean Spell F1", "Mean Boundary Error"))
    print("-" * 65)
    for model in candidates:
        mean_f1 = np.mean(all_scores[model])
        mean_err = np.mean(all_errs[model])
        print("{:<20} | {:<15.4f} | {:<15.2f} frames".format(model.upper(), mean_f1, mean_err))
    print("=" * 105)


if __name__ == "__main__":
    main()
