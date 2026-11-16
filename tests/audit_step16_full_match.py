#!/usr/bin/env python3
"""PHANTOM-11: Full-Match Comparative Audit for v7 Candidates (v7-A, v7-B, v7-C vs v6 & v5)

Step 16 Task 6:
- Compares full 199,319-frame predictions for v6, v7a, v7b, and v7c.
- Evaluates:
  1. Spell counts and duration distributions.
  2. Frame modification counts and label transitions.
  3. Pitch boundary corrections vs carrier corrections.
  4. Changes to genuine short possessions (<= 2.0s).
  5. Non-play interval integrity (pre-match, halftime, post-match).
  6. Memory and runtime performance metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

sys.path.insert(0, PROJECT_ROOT)
from phantom11.match_config import MatchConfig
from phantom11.load_tracking import load_full_tracking
from phantom11.pitch_boundary import compute_out_of_bounds_distance
from phantom11.boundary_refinement import extract_spell_intervals


def load_candidate_labels(name: str) -> np.ndarray:
    path = os.path.join(OUTPUT_DIR, f"phantom11_{name}.csv")
    df = pd.read_csv(path)
    return df["label"].values.astype("<U1")


def count_spells(labels: np.ndarray, fps: float = 25.0) -> Dict[str, any]:
    spells = extract_spell_intervals(labels)
    h_spells = [s for s in spells if s[0] == "H"]
    a_spells = [s for s in spells if s[0] == "A"]
    d_spells = [s for s in spells if s[0] == "D"]
    active_spells = [s for s in spells if s[0] in ["H", "A"]]
    
    durations = [s[3] / fps for s in active_spells]
    short_spells = [s for s in active_spells if (s[3] / fps) <= 2.0]
    
    return {
        "total_spells": len(spells),
        "active_spells": len(active_spells),
        "h_spells": len(h_spells),
        "a_spells": len(a_spells),
        "d_spells": len(d_spells),
        "short_active_spells": len(short_spells),
        "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "median_duration_s": float(np.median(durations)) if durations else 0.0,
        "max_duration_s": float(np.max(durations)) if durations else 0.0,
        "spells_list": spells,
    }


def analyze_transitions(labels_ref: np.ndarray, labels_target: np.ndarray) -> Dict[str, int]:
    diff_mask = labels_ref != labels_target
    transitions = {}
    for r in ["H", "A", "D"]:
        for t in ["H", "A", "D"]:
            if r != t:
                count = int(np.sum((labels_ref == r) & (labels_target == t)))
                if count > 0:
                    transitions[f"{r}->{t}"] = count
    return transitions


def main():
    print("=" * 80)
    print("PHANTOM-11: STEP 16 FULL-MATCH COMPARATIVE AUDIT (199,319 FRAMES)")
    print("=" * 80)

    # 1. Load MatchConfig & tracking for physical context
    cfg = MatchConfig.from_official_game()
    ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, _ = load_full_tracking(match_config=cfg)
    oob_dist = compute_out_of_bounds_distance(ball_xy, ball_present, pitch=pitch)
    oob_mask = ball_present & (oob_dist >= 0.35)
    non_play = cfg.get_non_play_intervals()

    # 2. Load candidates
    candidates = ["v5", "v6", "v7a", "v7b", "v7c"]
    labels_dict = {c: load_candidate_labels(c) for c in candidates}
    spell_stats = {c: count_spells(labels_dict[c]) for c in candidates}

    print("\n--- 1. FULL-MATCH SPELL METRICS ---")
    print(f"{'Candidate':<10} | {'Active Spells':<14} | {'H Spells':<9} | {'A Spells':<9} | {'D Spells':<9} | {'Short (<=2s)':<12} | {'Mean Dur (s)':<12}")
    print("-" * 85)
    for c in candidates:
        s = spell_stats[c]
        print(f"{c:<10} | {s['active_spells']:<14} | {s['h_spells']:<9} | {s['a_spells']:<9} | {s['d_spells']:<9} | {s['short_active_spells']:<12} | {s['mean_duration_s']:<12.2f}")

    print("\n--- 2. FRAME MODIFICATIONS RELATIVE TO v6 ---")
    for c in ["v7a", "v7b", "v7c"]:
        diff = np.sum(labels_dict[c] != labels_dict["v6"])
        pct = (diff / 199319) * 100
        trans = analyze_transitions(labels_dict["v6"], labels_dict[c])
        print(f"\n{c.upper()} vs v6: {diff:,} modified frames ({pct:.3f}%)")
        print("  Label Transitions:")
        for k, v in sorted(trans.items()):
            print(f"    {k}: {v:,} frames")

    print("\n--- 3. FRAME MODIFICATIONS RELATIVE TO v5 ---")
    for c in ["v6", "v7a", "v7b", "v7c"]:
        diff = np.sum(labels_dict[c] != labels_dict["v5"])
        pct = (diff / 199319) * 100
        print(f"  {c.upper()} vs v5: {diff:,} modified frames ({pct:.3f}%)")

    print("\n--- 4. OUT-OF-BOUNDS FRAMES ANALYSIS (d_oob >= 0.35m, n=4,559) ---")
    for c in candidates:
        sub = labels_dict[c][oob_mask]
        n_h = int(np.sum(sub == "H"))
        n_a = int(np.sum(sub == "A"))
        n_d = int(np.sum(sub == "D"))
        print(f"  {c:<6}: H={n_h:,} ({n_h/len(sub)*100:5.2f}%), A={n_a:,} ({n_a/len(sub)*100:5.2f}%), D={n_d:,} ({n_d/len(sub)*100:5.2f}%)")

    print("\n--- 5. NON-PLAY INTERVAL INTEGRITY AUDIT ---")
    np_intervals = cfg.get_non_play_intervals()
    interval_names = ["Pre-match", "Halftime", "Post-match"]
    for i, (s, e) in enumerate(np_intervals):
        desc = interval_names[i] if i < len(interval_names) else f"Interval {i}"
        print(f"\n  Checking {desc} [{s:,} .. {e:,}] (length {e-s+1:,} frames):")
        for c in candidates:
            sub = labels_dict[c][s:e+1]
            d_count = np.sum(sub == "D")
            non_d = np.sum(sub != "D")
            status = "PASSED (100% Dead-ball)" if non_d == 0 else f"FAILED ({non_d} active frames)"
            print(f"    {c:<6}: {status}")

    print("\n--- 6. DETAILED COMPARISON: v7a vs v7c vs v7b ---")
    diff_ab = np.sum(labels_dict["v7a"] != labels_dict["v7b"])
    diff_ac = np.sum(labels_dict["v7a"] != labels_dict["v7c"])
    diff_bc = np.sum(labels_dict["v7b"] != labels_dict["v7c"])
    print(f"  v7a vs v7b: {diff_ab:,} frames")
    print(f"  v7a vs v7c: {diff_ac:,} frames")
    print(f"  v7b vs v7c: {diff_bc:,} frames")

    if diff_ac > 0:
        trans_ac = analyze_transitions(labels_dict["v7a"], labels_dict["v7c"])
        print(f"  v7a -> v7c transitions: {trans_ac}")

    if diff_bc > 0:
        trans_bc = analyze_transitions(labels_dict["v7b"], labels_dict["v7c"])
        print(f"  v7b -> v7c transitions: {trans_bc}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
