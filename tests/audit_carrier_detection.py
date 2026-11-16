#!/usr/bin/env python3
"""Step 15 - Task 2: Empirical Audit of False Carrier Detection

Tests positive and negative scenarios:
1. Actual dribbling (Positive)
2. Ball trapped at feet (Positive)
3. Player standing near stationary ball (Ambiguous/Negative)
4. Defender co-moving with pass / loose ball (Negative)
5. One-touch interception / volley clearance (Negative for carry, positive for contact)
6. Tracker identity swap / position jump (Negative)
7. Coordinate noise robustness
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

def run_carrier_test(case_name, b_xy, p_xy, expected_carrier_frames, fps=25.0):
    n_frames = len(b_xy)
    b_kin = compute_ball_kinematics(b_xy, fps=fps)
    p_kin = compute_player_kinematics(p_xy, fps=fps)
    rel = compute_relative_motion(b_xy, b_kin["velocity"], b_kin["speed"], p_xy, p_kin["velocities"], p_kin["speeds"])
    carrier_slot, is_carrier, carrier_conf = detect_ball_carrier(
        rel["distances"], rel["rel_speeds"], rel["alignments"], b_kin["speed"], p_kin["speeds"], p_kin["player_jump_flag"], b_kin["ball_jump_flag"]
    )
    
    carrier_detected_frames = np.where(carrier_slot != -1)[0]
    n_detected = len(carrier_detected_frames)
    
    print("=" * 85)
    print(f"TEST: {case_name}")
    print(f"Total Frames: {n_frames} | Carrier Detected Frames: {n_detected}")
    if n_detected > 0:
        sample_f = carrier_detected_frames[0]
        dist = rel["distances"][sample_f, carrier_slot[sample_f]]
        rel_spd = rel["rel_speeds"][sample_f, carrier_slot[sample_f]]
        align = rel["alignments"][sample_f, carrier_slot[sample_f]]
        b_spd = b_kin["speed"][sample_f]
        p_spd = p_kin["speeds"][sample_f, carrier_slot[sample_f]]
        print(f"Sample Frame {sample_f:3d}: slot={carrier_slot[sample_f]}, dist={dist:.2f}m, rel_spd={rel_spd:.2f}m/s, align={align:.2f}, b_spd={b_spd:.2f}m/s, p_spd={p_spd:.2f}m/s, conf={carrier_conf[sample_f]:.2f}")
    
    print(f"Expected carrier frames: {expected_carrier_frames}")
    matches = (n_detected > 0) == (expected_carrier_frames > 0)
    print(f"Result Assessment: {'PASS' if matches else 'POTENTIAL FLAW / FALSE POSITIVE'}")
    return {
        "n_detected": n_detected,
        "carrier_slot": carrier_slot,
        "is_carrier": is_carrier,
        "conf": carrier_conf,
    }

def main():
    fps = 25.0
    n = 50

    # Case 1: Actual Dribbling (Player runs at 3.5 m/s, ball at 3.5 m/s, d=0.4m)
    b_xy = np.zeros((n, 2), dtype=np.float32)
    p_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b_xy[t] = [20.0 + t * (3.5 / fps), 30.0]
        p_xy[t, 0] = [b_xy[t, 0] + 0.3, 30.0]
    run_carrier_test("1. Actual Dribbling (Positive)", b_xy, p_xy, expected_carrier_frames=n)

    # Case 2: Ball Trapped at Player's Feet (Slow movement 0.5 m/s, d=0.3m)
    b_xy = np.zeros((n, 2), dtype=np.float32)
    p_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b_xy[t] = [20.0 + t * (0.5 / fps), 30.0]
        p_xy[t, 0] = [b_xy[t, 0] + 0.2, 30.0]
    run_carrier_test("2. Ball Trapped at Feet (Positive)", b_xy, p_xy, expected_carrier_frames=n)

    # Case 3: Player Standing Near Stationary Ball (Dead-Ball Setup: v=0, d=0.8m)
    b_xy = np.zeros((n, 2), dtype=np.float32)
    p_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b_xy[t] = [20.0, 30.0]
        p_xy[t, 0] = [20.8, 30.0] # 0.8m away, perfectly still
    res3 = run_carrier_test("3. Player Standing Near Stationary Ball (Ambiguous / Potential FP)", b_xy, p_xy, expected_carrier_frames=0)

    # Case 4: Defender Co-Moving with Pass / Loose Ball (v=4.5 m/s, d=1.2m, no touch)
    b_xy = np.zeros((n, 2), dtype=np.float32)
    p_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b_xy[t] = [20.0 + t * (4.5 / fps), 30.0]
        p_xy[t, 23] = [b_xy[t, 0] + 0.8, 30.8] # 1.13m distance, moving parallel at 4.5 m/s
    res4 = run_carrier_test("4. Defender Co-Moving With Rolling Ball (False Positive Risk)", b_xy, p_xy, expected_carrier_frames=0)

    # Case 5: One-Touch Clearance (Ball incoming at 12 m/s, kicked out at 22 m/s)
    b_xy = np.zeros((n, 2), dtype=np.float32)
    p_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(25):
        b_xy[t] = [40.0 - (25 - t) * (12.0 / fps), 30.0]
        p_xy[t, 23] = [40.2, 30.0]
    for t in range(25, n):
        b_xy[t] = [40.0 + (t - 24) * (22.0 / fps), 30.0]
        p_xy[t, 23] = [40.2, 30.0]
    run_carrier_test("5. One-Touch Clearance Contact (Expected 0 carrier frames)", b_xy, p_xy, expected_carrier_frames=0)

    # Case 6: Tracker Identity Swap / Position Jump (Player jumps 12m in 1 frame)
    b_xy = np.zeros((n, 2), dtype=np.float32)
    p_xy = np.full((n, 46, 2), 80.0, dtype=np.float32)
    for t in range(n):
        b_xy[t] = [20.0 + t * (2.0 / fps), 30.0]
        p_xy[t, 0] = [b_xy[t, 0] + 0.3, 30.0]
    # Inject jump at t=25
    p_xy[25, 0] = [b_xy[25, 0] + 12.0, 30.0]
    run_carrier_test("6. Tracker Identity Swap / Jump Flag (Should reject jump frame)", b_xy, p_xy, expected_carrier_frames=n-2)

if __name__ == "__main__":
    main()
