#!/usr/bin/env python3
"""PHANTOM-11: Version 3 Predictor (Duration-Aware Sequence Decoder)

Step 7 of More Than A Hack 2027:
- Takes PHANTOM-11 v2 as input.
- Computes physical team-control evidence and boundary change points.
- Applies the constrained, duration-aware Viterbi dynamic programming sequence decoder.
- Eliminates unsupported micro-islands and fragmentation while preserving genuine short possessions.
- Generates phantom11/outputs/phantom11_v3.csv (preserving v0, v1, and v2 intact).
- Computes comprehensive comparative diagnostics across v0, v1, v2, and v3.
"""

import os
import sys
import time
import json
import resource
import numpy as np
import pandas as pd
from collections import Counter

# Ensure repository root and starter directory are on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
STARTER_DIR = os.path.join(REPO_ROOT, "starter")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import games
import load_smart_tagging
from baseline import write_submission
from load_tracking import load_full_tracking
from control_evidence import compute_control_evidence
from boundary_evidence import extract_boundary_evidence, derive_non_play_intervals
from movement_features import compute_ball_kinematics
from sequence_decoder import decode_possession_sequence, DEFAULT_CONFIG
from tactical_tags import parse_smart_tagging_timeline


def get_possession_spells(labels, fps=25):
    """Compute contiguous runs of H or A (stoppages D or X split possessions)."""
    spells = []
    current_label = None
    start_idx = 0

    for i, lbl in enumerate(labels):
        if lbl in ("H", "A"):
            if current_label is None:
                current_label = lbl
                start_idx = i
            elif lbl != current_label:
                spells.append((current_label, start_idx, i - 1, (i - start_idx) / float(fps)))
                current_label = lbl
                start_idx = i
        else:
            if current_label is not None:
                spells.append((current_label, start_idx, i - 1, (i - start_idx) / float(fps)))
                current_label = None

    if current_label is not None:
        spells.append((current_label, start_idx, len(labels) - 1, (len(labels) - start_idx) / float(fps)))

    return spells


def count_ha_switches(labels):
    """Count immediate H -> A or A -> H turnovers without intervening D."""
    switches = 0
    for i in range(len(labels) - 1):
        c1, c2 = labels[i], labels[i + 1]
        if (c1 == "H" and c2 == "A") or (c1 == "A" and c2 == "H"):
            switches += 1
    return switches


def run_predictor_v3():
    t_start = time.time()
    print("=" * 78)
    print("PHANTOM-11: PREDICTOR VERSION 3 (DURATION-AWARE SEQUENCE DECODER)")
    print("=" * 78)

    # 1. Load baseline v2
    v2_path = os.path.join(SCRIPT_DIR, "outputs", "phantom11_v2.csv")
    if not os.path.isfile(v2_path):
        raise FileNotFoundError(f"v2 output not found at {v2_path}. Run predict_v2.py first.")

    print(f"\n[1] Loading PHANTOM-11 v2 from {v2_path}...")
    v2_df = pd.read_csv(v2_path)
    v2_labels = v2_df["label"].values.astype("<U1")
    n_frames = len(v2_labels)

    # Load v0 and v1 for comparative tracking
    v0_path = os.path.join(SCRIPT_DIR, "outputs", "phantom11_v0.csv")
    v1_path = os.path.join(SCRIPT_DIR, "outputs", "phantom11_v1.csv")
    v0_labels = pd.read_csv(v0_path)["label"].values.astype("<U1") if os.path.isfile(v0_path) else None
    v1_labels = pd.read_csv(v1_path)["label"].values.astype("<U1") if os.path.isfile(v1_path) else None

    # 2. Load Tracking and Smart Tagging Data
    print("\n[2] Loading full tactical tracking and Smart Tagging data...")
    ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, _ = load_full_tracking()
    st_events = load_smart_tagging.load(game=games.GAME)
    non_play = derive_non_play_intervals()

    # 3. Compute Control Evidence and Kinematics
    print("\n[3] Computing physical control evidence and boundary signals...")
    t_ev_0 = time.time()
    ev = compute_control_evidence(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=st_events,
        non_play_intervals=non_play,
        fps=25.0,
    )
    ball_kin = compute_ball_kinematics(ball_xy, fps=25.0)
    tact = parse_smart_tagging_timeline(st_events, n_frames)

    nearest_team = np.where(ev["d_h1"] <= ev["d_a1"], 0, 1).astype(np.int8)

    b_ev = extract_boundary_evidence(
        ball_present=ball_present,
        is_kick_or_deflection=ball_kin["is_kick_or_deflection"],
        is_carrier_h=ev["is_carrier_h"],
        is_carrier_a=ev["is_carrier_a"],
        nearest_team=nearest_team,
        smart_tagging_stoppage=tact["stoppage_support"],
        non_play_intervals=non_play,
    )
    t_ev_1 = time.time()
    print(f"    Evidence computed in {t_ev_1 - t_ev_0:.2f}s.")

    # 4. Load Decoder Configuration
    cfg_path = os.path.join(SCRIPT_DIR, "decoder_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            cfg_data = json.load(f)
            config = cfg_data.get("parameters", DEFAULT_CONFIG)
            print(f"\n[4] Loaded decoder configuration from {cfg_path}")
    else:
        config = DEFAULT_CONFIG
        print("\n[4] Using default decoder configuration")

    # 5. Run Duration-Aware Sequence Decoder
    print("\n[5] Running constrained Viterbi sequence decoder...")
    t_dec_0 = time.time()
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
        config=config,
    )
    t_dec_1 = time.time()
    print(f"    Decoding completed in {t_dec_1 - t_dec_0:.3f}s.")

    # 6. Save PHANTOM-11 v3 Output
    output_dir = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    v3_path = os.path.join(output_dir, "phantom11_v3.csv")
    print(f"\n[6] Writing submission CSV to {v3_path}...")
    write_submission(v3_path, v3_labels)
    print("    Done writing v3 (v0, v1, and v2 preserved intact).")

    t_end = time.time()
    # Memory usage in MB
    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)

    # 7. Comprehensive Comparative Diagnostics
    print("\n" + "=" * 78)
    print("COMPARATIVE DIAGNOSTICS: v0 -> v1 -> v2 -> v3")
    print("=" * 78)

    print("\n--- A. Overall Execution & Resource Metrics ---")
    print(f"  Total Runtime:               {t_end - t_start:.2f} seconds")
    print(f"  Decoder Dynamic Programming: {t_dec_1 - t_dec_0:.3f} seconds")
    print(f"  Peak Resident Memory:        {peak_mem_mb:.1f} MB")

    v2_counts = Counter(v2_labels)
    v3_counts = Counter(v3_labels)

    print("\n--- B. Label Distribution (v2 vs v3) ---")
    print(f"{'Label':<8} {'v2 Frames':<14} {'v2 Pct':<10} {'v3 Frames':<14} {'v3 Pct':<10} {'Difference'}")
    print("-" * 68)
    for lbl in ["H", "A", "D", "X"]:
        c2 = v2_counts[lbl]
        c3 = v3_counts[lbl]
        diff = c3 - c2
        sign = "+" if diff > 0 else ""
        print(f"{lbl:<8} {c2:<14,} {c2/n_frames*100:6.2f}%   {c3:<14,} {c3/n_frames*100:6.2f}%   {sign}{diff:,}")

    # Label transitions v2 -> v3
    diff_mask = (v2_labels != v3_labels)
    total_changed = int(np.sum(diff_mask))
    print(f"\n--- C. Label Transitions (v2 -> v3) ---")
    print(f"  Total Frames Changed: {total_changed:,} ({total_changed / n_frames * 100:.3f}%)")

    transition_counts = Counter()
    for o, n in zip(v2_labels[diff_mask], v3_labels[diff_mask]):
        transition_counts[f"{o} -> {n}"] += 1

    for trans, count in transition_counts.most_common():
        print(f"    {trans:<12}: {count:>6,} frames")

    # Full 4-way comparison of possession structure
    all_versions = [("v0", v0_labels), ("v1", v1_labels), ("v2", v2_labels), ("v3", v3_labels)]

    print("\n--- D. Multi-Version Evolution (v0, v1, v2, v3) ---")
    print(f"{'Metric':<32} {'v0':<12} {'v1':<12} {'v2':<12} {'v3':<12}")
    print("-" * 78)

    spells_dict = {}
    for name, lbls in all_versions:
        if lbls is not None:
            spells = get_possession_spells(lbls)
            spells_dict[name] = spells
            durations = [s[3] for s in spells]
            short_durations = sum(1 for d in durations if d < 0.50)
            switches = count_ha_switches(lbls)
        else:
            spells_dict[name] = []

    print(f"{'Total Possessions':<32} {len(spells_dict['v0']):<12,} {len(spells_dict['v1']):<12,} {len(spells_dict['v2']):<12,} {len(spells_dict['v3']):<12,}")

    mean_durs = [f"{np.mean([s[3] for s in spells_dict[m]]):.2f}s" for m in ["v0", "v1", "v2", "v3"]]
    print(f"{'Mean Duration':<32} {mean_durs[0]:<12} {mean_durs[1]:<12} {mean_durs[2]:<12} {mean_durs[3]:<12}")

    med_durs = [f"{np.median([s[3] for s in spells_dict[m]]):.2f}s" for m in ["v0", "v1", "v2", "v3"]]
    print(f"{'Median Duration':<32} {med_durs[0]:<12} {med_durs[1]:<12} {med_durs[2]:<12} {med_durs[3]:<12}")

    micro_counts = [f"{sum(1 for s in spells_dict[m] if s[3] < 0.50):,}" for m in ["v0", "v1", "v2", "v3"]]
    print(f"{'Micro-Possessions (< 0.5s)':<32} {micro_counts[0]:<12} {micro_counts[1]:<12} {micro_counts[2]:<12} {micro_counts[3]:<12}")

    sw_counts = [f"{count_ha_switches(lbls):,}" for _, lbls in all_versions]
    print(f"{'Direct Turnovers (H <-> A)':<32} {sw_counts[0]:<12} {sw_counts[1]:<12} {sw_counts[2]:<12} {sw_counts[3]:<12}")

    # Non-play boundary integrity check
    ht_s, ht_e = 90652, 110807
    ht_v3_d = int(np.sum(v3_labels[ht_s:ht_e + 1] == "D"))
    ht_total = ht_e - ht_s + 1
    print("\n--- E. Halftime & Match Boundary Integrity ---")
    print(f"  Halftime Frames (90,652..110,807): {ht_total:,} frames")
    print(f"  Halftime Preserved as Dead Ball:   {ht_v3_d:,} of {ht_total:,} ({ht_v3_d/ht_total*100:.1f}%)")

    print("\n" + "=" * 78)
    print("PHANTOM-11 v3 EXECUTION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    run_predictor_v3()
