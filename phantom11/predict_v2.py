#!/usr/bin/env python3
"""PHANTOM-11: Version 2 Predictor (Evidence-Aware Tactical Intelligence)

Step 6 of More Than A Hack 2027:
- Takes PHANTOM-11 v1 as the solid baseline (retaining missing-ball continuity & halftime clamping).
- Incorporates physical team-control evidence from:
  * Active ball carrying (co-movement, close control, low relative speed).
  * Team geometry and spatial margins (d_A1 - d_H1).
  * Tactical Smart Tagging soft priors.
- Conservative override: modifies v1 ONLY when multiple independent physical signals
  strongly agree that an active carrier was mislabeled due to smoothing window lag.
- Ironclad match-boundary protection: strictly preserves pre-match, halftime, and post-match as 'D'.
- Preserves v0 and v1 intact.
- Generates a transparent decision log documenting every modified frame.
"""

import os
import sys
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

import games
import load_smart_tagging
from baseline import write_submission
try:
    from phantom11.continuity import NON_PLAY_INTERVALS
    from phantom11.load_tracking import load_full_tracking
    from phantom11.control_evidence import compute_control_evidence
except ImportError:
    from continuity import NON_PLAY_INTERVALS
    from load_tracking import load_full_tracking
    from control_evidence import compute_control_evidence


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


def run_predictor_v2():
    print("=" * 78)
    print("PHANTOM-11: PREDICTOR VERSION 2 (TACTICAL EVIDENCE MODULE)")
    print("=" * 78)

    # 1. Load baseline v1
    v1_path = os.path.join(SCRIPT_DIR, "outputs", "phantom11_v1.csv")
    if not os.path.isfile(v1_path):
        raise FileNotFoundError(f"v1 output not found at {v1_path}. Run predict_v1.py first.")

    print(f"\n[1] Loading PHANTOM-11 v1 from {v1_path}...")
    v1_df = pd.read_csv(v1_path)
    v1_labels = v1_df["label"].values.astype("<U1")
    n_frames = len(v1_labels)

    # 2. Load Tracking and Smart Tagging Data
    print("\n[2] Loading full tactical tracking and Smart Tagging data...")
    ball_xy, ball_present, players_xy, players_present, team_of_track, pitch, _ = load_full_tracking()
    st_events = load_smart_tagging.load(game=games.GAME)

    # 3. Compute Control Evidence
    print("\n[3] Computing physical control evidence (kinematics, geometry, tags)...")
    ev = compute_control_evidence(
        ball_xy=ball_xy,
        players_xy=players_xy,
        team_of_track=team_of_track,
        smart_tagging_events=st_events,
        non_play_intervals=NON_PLAY_INTERVALS,
        fps=25.0,
    )

    # 4. Apply Conservative Evidence-Based Overrides
    print("\n[4] Applying conservative evidence rules...")
    v2_labels = v1_labels.copy()
    decision_records = []

    # Non-play boundary mask
    in_non_play = np.zeros(n_frames, dtype=bool)
    for s, e in NON_PLAY_INTERVALS:
        s_c = max(0, s)
        e_c = min(n_frames - 1, e)
        if s_c <= e_c:
            in_non_play[s_c:e_c + 1] = True

    # Rule 1: High-Confidence Carrier Override (Home)
    # A Barcelona player is carrying the ball (d <= 1.8m, margin >= 1.0m, score >= 0.55)
    c_h_mask = (
        ev["is_carrier_h"]
        & ~in_non_play
        & (ev["score_h"] >= 0.55)
        & (ev["d_h1"] <= 1.80)
        & (ev["margin_d"] >= 1.00)
    )

    override_h_indices = np.where(c_h_mask & (v1_labels != "H"))[0]
    for idx in override_h_indices:
        prev = v1_labels[idx]
        v2_labels[idx] = "H"
        decision_records.append({
            "frame": int(idx),
            "v1_label": prev,
            "v2_label": "H",
            "reason": "CARRIER_OVERRIDE_H",
            "d_h1": float(round(ev["d_h1"][idx], 2)),
            "d_a1": float(round(ev["d_a1"][idx], 2)),
            "score_h": float(round(ev["score_h"][idx], 3)),
            "score_a": float(round(ev["score_a"][idx], 3)),
            "reliability": float(round(ev["reliability"][idx], 2)),
        })

    # Rule 2: High-Confidence Carrier Override (Away)
    # An Athletic Club player is carrying the ball (d <= 1.8m, margin <= -1.0m, score >= 0.55)
    c_a_mask = (
        ev["is_carrier_a"]
        & ~in_non_play
        & (ev["score_a"] >= 0.55)
        & (ev["d_a1"] <= 1.80)
        & (ev["margin_d"] <= -1.00)
    )

    override_a_indices = np.where(c_a_mask & (v1_labels != "A"))[0]
    for idx in override_a_indices:
        prev = v1_labels[idx]
        v2_labels[idx] = "A"
        decision_records.append({
            "frame": int(idx),
            "v1_label": prev,
            "v2_label": "A",
            "reason": "CARRIER_OVERRIDE_A",
            "d_h1": float(round(ev["d_h1"][idx], 2)),
            "d_a1": float(round(ev["d_a1"][idx], 2)),
            "score_h": float(round(ev["score_h"][idx], 3)),
            "score_a": float(round(ev["score_a"][idx], 3)),
            "reliability": float(round(ev["reliability"][idx], 2)),
        })

    # Strict enforcement of non-play boundaries (Pre-match, Halftime, Post-match)
    for s, e in NON_PLAY_INTERVALS:
        s_c = max(0, s)
        e_c = min(n_frames - 1, e)
        if s_c <= e_c:
            v2_labels[s_c:e_c + 1] = "D"

    # 5. Save Decision Log and v2 Submission
    output_dir = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    log_df = pd.DataFrame(decision_records)
    log_path = os.path.join(output_dir, "v2_decision_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"\n[5] Decision log saved to {log_path} ({len(log_df)} modifications recorded).")

    v2_path = os.path.join(output_dir, "phantom11_v2.csv")
    write_submission(v2_path, v2_labels)
    print(f"    PHANTOM-11 v2 saved to {v2_path} (v0 and v1 preserved intact).")

    # 6. Detailed Comparative Diagnostics (v1 vs v2)
    print("\n" + "=" * 78)
    print("COMPARATIVE DIAGNOSTICS: v1 vs v2 (EVIDENCE-BASED EVOLUTION)")
    print("=" * 78)

    v1_counts = Counter(v1_labels)
    v2_counts = Counter(v2_labels)

    print("\n--- A. Overall Label Distribution ---")
    print(f"{'Label':<8} {'v1 Frames':<14} {'v1 Pct':<10} {'v2 Frames':<14} {'v2 Pct':<10} {'Difference'}")
    print("-" * 68)
    for lbl in ["H", "A", "D", "X"]:
        c1 = v1_counts[lbl]
        c2 = v2_counts[lbl]
        diff = c2 - c1
        sign = "+" if diff > 0 else ""
        print(f"{lbl:<8} {c1:<14,} {c1/n_frames*100:6.2f}%   {c2:<14,} {c2/n_frames*100:6.2f}%   {sign}{diff:,}")

    diff_mask = (v1_labels != v2_labels)
    total_changed = int(np.sum(diff_mask))
    print(f"\n--- B. Label Transitions (v1 -> v2) ---")
    print(f"  Total Frames Changed: {total_changed:,} ({total_changed / n_frames * 100:.3f}%)")

    transition_counts = Counter()
    for o, n in zip(v1_labels[diff_mask], v2_labels[diff_mask]):
        transition_counts[f"{o} -> {n}"] += 1

    for trans, count in transition_counts.most_common():
        print(f"    {trans:<12}: {count:>6,} frames")

    # Possession Spells
    v1_spells = get_possession_spells(v1_labels)
    v2_spells = get_possession_spells(v2_labels)

    v1_durations = [s[3] for s in v1_spells]
    v2_durations = [s[3] for s in v2_spells]

    print("\n--- C. Possession Structure (Spells) ---")
    print(f"  v1 Total Possessions:        {len(v1_spells):,} (H: {sum(1 for s in v1_spells if s[0]=='H')}, A: {sum(1 for s in v1_spells if s[0]=='A')})")
    print(f"  v2 Total Possessions:        {len(v2_spells):,} (H: {sum(1 for s in v2_spells if s[0]=='H')}, A: {sum(1 for s in v2_spells if s[0]=='A')})")
    print(f"  v1 Mean / Median Duration:   {np.mean(v1_durations):.2f}s / {np.median(v1_durations):.2f}s")
    print(f"  v2 Mean / Median Duration:   {np.mean(v2_durations):.2f}s / {np.median(v2_durations):.2f}s")

    # Turnovers (Direct H <-> A switches)
    v1_switches = count_ha_switches(v1_labels)
    v2_switches = count_ha_switches(v2_labels)
    print(f"  Direct H <-> A Switches:     v1: {v1_switches:,} -> v2: {v2_switches:,} ({v2_switches - v1_switches:+d})")

    # Non-play boundary integrity check
    ht_s, ht_e = 90652, 110807
    ht_v2_d = int(np.sum(v2_labels[ht_s:ht_e + 1] == "D"))
    ht_total = ht_e - ht_s + 1
    print("\n--- D. Halftime & Match Boundary Integrity ---")
    print(f"  Halftime Frames (90,652..110,807): {ht_total:,} frames")
    print(f"  Halftime Preserved as Dead Ball:   {ht_v2_d:,} of {ht_total:,} ({ht_v2_d/ht_total*100:.1f}%)")

    print("\n" + "=" * 78)
    print("PHANTOM-11 v2 EXECUTION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    run_predictor_v2()
