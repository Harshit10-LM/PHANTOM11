#!/usr/bin/env python3
"""Step 15 - Task 1: Deep Diagnostic of the Five Weak Scenarios (< 0.8000)

Examines the 5 scenarios:
1. Frozen 06: Brief Opponent Deflection (0.3000)
2. Frozen 09: Restart (Goal Kick Strike) (0.6000)
3. Held-Out 04: Contested 50-50 Aerial Duel (0.6667)
4. Held-Out 06: Throw-In Up the Line Intercepted (0.5000)
5. Adversarial CE3: Clearance Directly Out of Bounds (0.5333)
"""

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_flight_handover_benchmark import (
    build_bench_scenario_06_brief_opponent_deflection,
    build_bench_scenario_09_goal_kick_restart,
    run_pipeline_v3,
    run_pipeline_v4,
)
from tests.test_held_out_handover_suite import (
    build_ho_04_aerial_50_50_duel,
    build_ho_06_throw_in_intercepted,
)
from tests.test_sensitivity_and_counterexamples import (
    build_counterexample_3_clearance_out_of_bounds,
)
from tests.run_v5_v6_comparative_audit import run_pipeline_v6
from phantom11.control_evidence import compute_control_evidence
from phantom11.movement_features import compute_ball_kinematics, compute_player_kinematics, compute_relative_motion, detect_ball_carrier
from phantom11.sequence_decoder import decode_possession_sequence
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm

def investigate_scenario(name, b_xy, p_xy, teams, gt, focus_frames=None):
    print("=" * 100)
    print(f"INVESTIGATING: {name}")
    print("=" * 100)
    runs = [(0, len(gt))]
    gt_codes = pm.label_codes(gt)
    
    p3 = run_pipeline_v3(b_xy, p_xy, teams, [])
    p4 = run_pipeline_v4(b_xy, p_xy, teams, [])
    p6 = run_pipeline_v6(b_xy, p_xy, teams, [])

    s3 = pm.spell_scores(gt_codes, pm.label_codes(p3), runs)["mean_f1"]
    s4 = pm.spell_scores(gt_codes, pm.label_codes(p4), runs)["mean_f1"]
    s6 = pm.spell_scores(gt_codes, pm.label_codes(p6), runs)["mean_f1"]

    print(f"spellF1 Scores: v3={s3:.4f} | v4={s4:.4f} | v6={s6:.4f}")
    gt_sp = pm.possession_spells(gt_codes, runs)
    p6_sp = pm.possession_spells(pm.label_codes(p6), runs)
    print(f"GT Spells:   {gt_sp}")
    print(f"Pred Spells: {p6_sp}")

    # Overlaps
    overlaps = pm.spell_overlaps(gt_sp, p6_sp)
    print("Spell Overlaps:")
    for iou, gi, pi in overlaps:
        print(f"  GT[{gi}] {gt_sp[gi]} vs Pred[{pi}] {p6_sp[pi]} -> IoU = {iou:.4f}")

    # Kinematics & Evidence
    fps = 25.0
    b_kin = compute_ball_kinematics(b_xy, fps=fps)
    p_kin = compute_player_kinematics(p_xy, fps=fps)
    rel = compute_relative_motion(b_xy, b_kin["velocity"], b_kin["speed"], p_xy, p_kin["velocities"], p_kin["speeds"])
    carrier_slot, is_carrier, carrier_conf = detect_ball_carrier(
        rel["distances"], rel["rel_speeds"], rel["alignments"], b_kin["speed"], p_kin["speeds"], p_kin["player_jump_flag"], b_kin["ball_jump_flag"]
    )
    ev = compute_control_evidence(b_xy, p_xy, teams, [], [], fps=fps)

    if focus_frames:
        print("\nFrame-Level Diagnostics across Focus Window:")
        for f in focus_frames:
            if 0 <= f < len(gt):
                dh = ev["d_h1"][f]
                da = ev["d_a1"][f]
                bspd = b_kin["speed"][f]
                cslot = carrier_slot[f]
                cteam = "None" if cslot == -1 else ("H" if teams[cslot] == 0 else "A")
                print(f"  f={f:3d} | GT={gt[f]} v3={p3[f]} v4={p4[f]} v6={p6[f]} | b_spd={bspd:.2f} m/s | d_h={dh:.2f}m d_a={da:.2f}m | carrier={cteam} (slot {cslot})")

def main():
    # 1. Frozen 06: Deflection Without Control
    name, b, p, t, gt = build_bench_scenario_06_brief_opponent_deflection()
    investigate_scenario(name, b, p, t, gt, focus_frames=range(95, 120))

    # 2. Frozen 09: Restart (Goal Kick Strike)
    name, b, p, t, gt = build_bench_scenario_09_goal_kick_restart()
    investigate_scenario(name, b, p, t, gt, focus_frames=range(45, 60))

    # 3. Held-Out 04: Contested Aerial Duel
    name, b, p, t, gt = build_ho_04_aerial_50_50_duel()
    investigate_scenario(name, b, p, t, gt, focus_frames=range(20, 35))

    # 4. Held-Out 06: Throw-In Intercepted
    name, b, p, t, gt = build_ho_06_throw_in_intercepted()
    investigate_scenario(name, b, p, t, gt, focus_frames=range(30, 50))

    # 5. Adversarial CE3: Clearance Out of Bounds
    cename, b, p, t, gt = build_counterexample_3_clearance_out_of_bounds()
    investigate_scenario(cename, b, p, t, gt, focus_frames=range(35, 65))

if __name__ == "__main__":
    main()
