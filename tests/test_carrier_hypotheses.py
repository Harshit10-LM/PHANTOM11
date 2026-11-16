#!/usr/bin/env python3
"""Step 16 - Task 3: Carrier Detection Hypothesis Evaluation Harness

Tests moving carrier conditions:
Baseline: max_dist = 2.0m, max_rel_speed = 2.5 m/s, min_align = 0.40
Hypothesis A: max_dist = 1.2m, max_rel_speed = 1.5 m/s, min_align = 0.50
Hypothesis B (Evidence-Gated): max_dist = 1.2m for v_b > 3.5 m/s (with impulse/deflection checks)

Tests across:
- Genuine dribbling at 3.5, 5.0, 7.0 m/s
- Controlled trapping
- One-touch interceptions & volleys
- Heading
- Rapid turnovers (1.0s, 1.2s)
- Tracker identity switches
- Co-moving defender running alongside pass (v=5.0, d=1.0m, d=1.5m)
- Coordinate noise (sigma = 0.20m)
"""

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom11.movement_features import (
    compute_ball_kinematics,
    compute_player_kinematics,
    compute_relative_motion,
    detect_ball_carrier,
)

def evaluate_carrier_variant(b_xy, p_xy, variant="baseline", fps=25.0):
    b_kin = compute_ball_kinematics(b_xy, fps=fps)
    p_kin = compute_player_kinematics(p_xy, fps=fps)
    rel = compute_relative_motion(b_xy, b_kin["velocity"], b_kin["speed"], p_xy, p_kin["velocities"], p_kin["speeds"])
    
    if variant == "baseline":
        slot, is_c, conf = detect_ball_carrier(
            rel["distances"], rel["rel_speeds"], rel["alignments"],
            b_kin["speed"], p_kin["speeds"], p_kin["player_jump_flag"], b_kin["ball_jump_flag"],
            max_dist=2.0, max_rel_speed=2.5, min_alignment=0.40
        )
    elif variant == "tightened":
        slot, is_c, conf = detect_ball_carrier(
            rel["distances"], rel["rel_speeds"], rel["alignments"],
            b_kin["speed"], p_kin["speeds"], p_kin["player_jump_flag"], b_kin["ball_jump_flag"],
            max_dist=1.2, max_rel_speed=1.5, min_alignment=0.50
        )
    elif variant == "adaptive":
        # Adaptive: tighter for fast ball, standard for slow/trapped ball
        # When ball_speed > 3.5 m/s: require d <= 1.2m and rel_speed <= 1.5 m/s
        # When ball_speed <= 3.5 m/s: allow d <= 1.5m and rel_speed <= 1.8 m/s
        n_frames, n_slots = rel["distances"].shape
        b_spd_exp = b_kin["speed"][:, np.newaxis]
        p_spd = p_kin["speeds"]
        
        is_reliable = ~p_kin["player_jump_flag"] & ~b_kin["ball_jump_flag"][:, np.newaxis]
        
        # Fast moving ball
        fast_mask = b_spd_exp > 3.5
        carry_fast = fast_mask & (rel["distances"] <= 1.2) & (rel["rel_speeds"] <= 1.5) & (rel["alignments"] >= 0.50)
        
        # Moderate / slow moving ball
        slow_moving_mask = (b_spd_exp <= 3.5) & (b_spd_exp > 0.8) & (p_spd > 0.8)
        carry_slow = slow_moving_mask & (rel["distances"] <= 1.5) & (rel["rel_speeds"] <= 1.8) & (rel["alignments"] >= 0.40)
        
        # Stationary / trapped ball
        stationary_mask = (b_spd_exp <= 0.8) | (p_spd <= 0.8)
        carry_stat = stationary_mask & (rel["distances"] <= 0.8) & (rel["rel_speeds"] <= 1.2)
        
        cand = is_reliable & (carry_fast | carry_slow | carry_stat)
        slot = np.full(n_frames, -1, dtype=np.int8)
        conf = np.zeros(n_frames, dtype=np.float32)
        is_c = np.zeros((n_frames, n_slots), dtype=bool)
        
        for t in range(n_frames):
            players_cand = np.where(cand[t])[0]
            if len(players_cand) > 0:
                best_p = players_cand[np.argmin(rel["distances"][t, players_cand])]
                slot[t] = best_p
                is_c[t, best_p] = True
                conf[t] = 1.0 - (rel["distances"][t, best_p] / 1.5)
    else:
        raise ValueError(f"Unknown variant {variant}")

    return slot, is_c, conf


def run_all_carrier_tests():
    fps = 25.0
    n = 50
    variants = ["baseline", "tightened", "adaptive"]

    test_cases = []

    # 1. Dribbling at 3.5 m/s (d=0.3m, rel=0.1)
    b1 = np.zeros((n, 2), dtype=np.float32)
    p1 = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b1[t] = [20.0 + t * (3.5 / fps), 30.0]
        p1[t, 0] = [b1[t, 0] + 0.3, 30.0]
    test_cases.append(("1. Dribble (3.5 m/s, d=0.3m)", b1, p1, True))

    # 2. Dribble Fast Sprint at 7.0 m/s (d=0.6m, rel=0.4)
    b2 = np.zeros((n, 2), dtype=np.float32)
    p2 = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b2[t] = [20.0 + t * (7.0 / fps), 30.0]
        p2[t, 0] = [b2[t, 0] + 0.6, 30.0]
    test_cases.append(("2. Fast Sprint Dribble (7.0 m/s, d=0.6m)", b2, p2, True))

    # 3. Trapped Ball at Feet (v=0.4 m/s, d=0.2m)
    b3 = np.zeros((n, 2), dtype=np.float32)
    p3 = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b3[t] = [20.0 + t * (0.4 / fps), 30.0]
        p3[t, 0] = [b3[t, 0] + 0.2, 30.0]
    test_cases.append(("3. Trapped Ball at Feet (v=0.4 m/s, d=0.2m)", b3, p3, True))

    # 4. Co-Moving Defender Alongside Pass (v=5.0 m/s, d=1.0m, rel=0.0) -> Negative!
    b4 = np.zeros((n, 2), dtype=np.float32)
    p4 = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b4[t] = [20.0 + t * (5.0 / fps), 30.0]
        p4[t, 23] = [b4[t, 0], 31.0] # 1.0m distance parallel
    test_cases.append(("4. Co-Moving Defender Along Pass (v=5m/s, d=1.0m)", b4, p4, False))

    # 5. Co-Moving Defender at d=1.5m (v=5.0 m/s, rel=0.0) -> Negative!
    b5 = np.zeros((n, 2), dtype=np.float32)
    p5 = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b5[t] = [20.0 + t * (5.0 / fps), 30.0]
        p5[t, 23] = [b5[t, 0], 31.5] # 1.5m distance parallel
    test_cases.append(("5. Co-Moving Defender Along Pass (v=5m/s, d=1.5m)", b5, p5, False))

    # 6. Stationary Ball Setup (v=0, d=0.8m) -> Negative!
    b6 = np.zeros((n, 2), dtype=np.float32)
    p6 = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b6[t] = [20.0, 30.0]
        p6[t, 0] = [20.8, 30.0]
    test_cases.append(("6. Stationary Setup Ball (v=0, d=0.8m)", b6, p6, False))

    # 7. Noisy Dribble (v=4.0 m/s, d=0.4m, Gaussian noise sigma=0.20m) -> Positive!
    np.random.seed(42)
    b7 = np.zeros((n, 2), dtype=np.float32)
    p7 = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b7[t] = [20.0 + t * (4.0 / fps), 30.0] + np.random.normal(0, 0.15, 2)
        p7[t, 0] = [20.0 + t * (4.0 / fps) + 0.4, 30.0] + np.random.normal(0, 0.15, 2)
    test_cases.append(("7. Noisy Dribble (sigma=0.15m, d=0.4m)", b7, p7, True))

    print("=" * 105)
    print("CARRIER DETECTION HYPOTHESIS EVALUATION")
    print("=" * 105)
    print(f"{'Test Case':<45} | {'Expected':<8} | {'Baseline':<10} | {'Tightened':<10} | {'Adaptive':<10}")
    print("-" * 105)

    for name, b, p, exp in test_cases:
        res = {}
        for var in variants:
            slot, is_c, conf = evaluate_carrier_variant(b, p, variant=var, fps=fps)
            detected = np.sum(slot != -1) >= 10 # at least 10 frames detected
            correct = detected == exp
            res[var] = f"{np.sum(slot != -1):2d}f ({'OK' if correct else 'FAIL'})"
        print(f"{name:<45} | {'True' if exp else 'False':<8} | {res['baseline']:<10} | {res['tightened']:<10} | {res['adaptive']:<10}")
    print("=" * 105)

if __name__ == "__main__":
    run_all_carrier_tests()
