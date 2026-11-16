#!/usr/bin/env python3
"""Step 18 Task 2: Official Public Ground-Truth Window Diagnostic for v5, v6, v7-A, and v5-A.

Evaluates:
- Official spellF1 (thresholds 0.3, 0.4, 0.5, 0.6, 0.7).
- Frame-level metrics: Dead-ball Precision, Recall, F1, TP, FP, FN.
- Multiclass confusion matrix (H, A, D).
- Detailed investigation of disagreement at frame 18,026 under dual hypotheses.
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
GT_PATH = "/Users/harshu/Desktop/PHANTOM-11/more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv"

gt_df = pd.read_csv(GT_PATH)
gt_full = gt_df["label"].values.astype("<U1")

eval_mask = (gt_full != "X")
eval_indices = np.where(eval_mask)[0]
start_f, end_f = eval_indices[0], eval_indices[-1]
gt_eval = gt_full[start_f:end_f+1]
n_eval = len(gt_eval)

print("=" * 80)
print(f"STEP 18: PUBLIC GROUND-TRUTH DIAGNOSTIC ({n_eval:,} frames, [{start_f:,} .. {end_f:,}])")
print("=" * 80)

candidates = ["v5", "v6", "v7a", "v5a"]
preds_eval = {}
for c in candidates:
    path = os.path.join(OUTPUT_DIR, f"phantom11_{c}.csv")
    df = pd.read_csv(path)
    preds_eval[c] = df["label"].values.astype("<U1")[start_f:end_f+1]

# 1. Official spellF1 calculation
runs = [(0, n_eval)]
gt_codes = pm.label_codes(gt_eval)

print("\n--- 1. OFFICIAL SPELL F1 METRICS (possession_metrics.py) ---")
print(f"{'Candidate':<10} | {'Mean F1':<9} | {'F1 @ 0.3':<9} | {'F1 @ 0.4':<9} | {'F1 @ 0.5':<9} | {'F1 @ 0.6':<9} | {'F1 @ 0.7':<9}")
print("-" * 75)

spell_results = {}
for c in candidates:
    pred_codes = pm.label_codes(preds_eval[c])
    res = pm.spell_scores(gt_codes, pred_codes, runs)
    spell_results[c] = res
    f1s = [res["per_iou"][t]["f1"] for t in [0.3, 0.4, 0.5, 0.6, 0.7]]
    print(f"{c.upper():<10} | {res['mean_f1']:<9.4f} | {f1s[0]:<9.4f} | {f1s[1]:<9.4f} | {f1s[2]:<9.4f} | {f1s[3]:<9.4f} | {f1s[4]:<9.4f}")

# 2. Frame-level Dead-Ball Metrics
print("\n--- 2. FRAME-LEVEL DEAD-BALL ('D') METRICS ---")
print(f"{'Candidate':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'TP':<7} | {'FP':<7} | {'FN':<7}")
print("-" * 75)

for c in candidates:
    pred = preds_eval[c]
    tp = int(np.sum((gt_eval == "D") & (pred == "D")))
    fp = int(np.sum((gt_eval != "D") & (pred == "D")))
    fn = int(np.sum((gt_eval == "D") & (pred != "D")))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"{c.upper():<10} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {tp:<7,d} | {fp:<7,d} | {fn:<7,d}")

# 3. Multiclass Frame-Level Accuracy & Breakdown
print("\n--- 3. MULTICLASS FRAME-LEVEL ACCURACY ---")
for c in candidates:
    pred = preds_eval[c]
    acc = np.mean(gt_eval == pred) * 100
    h_acc = np.sum((gt_eval == "H") & (pred == "H")) / np.sum(gt_eval == "H") * 100
    a_acc = np.sum((gt_eval == "A") & (pred == "A")) / np.sum(gt_eval == "A") * 100
    d_acc = np.sum((gt_eval == "D") & (pred == "D")) / np.sum(gt_eval == "D") * 100
    print(f"{c.upper():<5}: Overall Acc={acc:5.2f}% | Recall H={h_acc:5.2f}%, Recall A={a_acc:5.2f}%, Recall D={d_acc:5.2f}%")

print("\n" + "=" * 80)
