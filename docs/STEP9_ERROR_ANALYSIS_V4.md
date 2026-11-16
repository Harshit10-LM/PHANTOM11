# PHANTOM-11: Possession Boundary Error Analysis & v4 Candidate Report

**Project:** PHANTOM-11 (More Than A Hack 2027)  
**Date:** November 2026  
**Status:** Step 9 Possession Boundary Diagnostics & Experimental v4 Candidate  

---

## 1. Executive Summary

Following the successful publication and reproducibility verification of PHANTOM-11 v3, we conducted a systematic, physics-based possession boundary error analysis to investigate potential failure modes that directly penalize the official ranking metric, **spellF1**.

Key outcomes:
1. **Official Metric Analysis:** Analyzed how temporal IoU matching ($\theta \in [0.3, 0.7]$) penalizes false possession splits (fragmentation), suppressed genuine turnovers (merges), and boundary phase lag.
2. **Diagnostic Tool Developed:** Created `phantom11/boundary_diagnostics.py` to evaluate the physical consistency of predicted transitions without hand-tuning or leaking private frame data.
3. **Core Anomaly Discovered in v3:** Out of 34 micro-possessions under $1.0$s in v3, **29 spells (85%) had zero confirmed carrier frames**—they were fleeting opponent touches or deflections where the opponent never established control. Each false touch split an ongoing possession into two.
4. **Confidence-Gated Refinement Engine (v4):** Designed `phantom11/boundary_refinement.py` with 4 physical gates:
   - Deflection vs. Interception Gate (requires confirmed carrier control for turnovers $< 1.0$s).
   - Live Tracking Dropout Bridging (bridges short $\le 1.2$s occlusions with bilateral same-team control).
   - Premature Restart Snapping (delays restart from static run-up to true kick/touch).
   - Strict Non-Play Clamping (halftime, pre-match, post-match).
5. **Rigorous Synthetic Verification:** Created `tests/test_refinement.py` covering 11 synthetic edge cases and robustness tests. All **45 unit tests** in the project pass in $0.25$s.
6. **Experimental v4 Candidate Generated:** Generated `outputs/phantom11_v4.csv` (`f479648...`) modifying only 237 frames (0.119% of the match) and reducing possession spells from 493 to 471.
7. **v3 Baseline Preserved:** `outputs/phantom11_v3.csv` (`289be9ad...`) remains 100% byte-for-byte untouched as our safe competition fallback.

---

## 2. Understanding the Official Metric: spellF1

The hackathon ranking metric is `spell_f1_mean` defined in `evaluation/possession_metrics.py`.

### How spellF1 Works
- Possessions are maximal contiguous runs of $H$ or $A$. Stoppages ($D$) split possessions.
- Predicted spells are matched one-to-one against true spells using **temporal Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{\min(\text{stop}_{\text{gt}}, \text{stop}_{\text{pred}}) - \max(\text{start}_{\text{gt}}, \text{start}_{\text{pred}})}{\max(\text{stop}_{\text{gt}}, \text{stop}_{\text{pred}}) - \min(\text{start}_{\text{gt}}, \text{start}_{\text{pred}})}$$
- Matching is performed greedily at 5 IoU thresholds: $\theta \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$.
- The final score is the unweighted mean of $F_1(\theta)$ across all 5 thresholds.

### Mathematical Error Penalties
1. **False Possession Split (Double Penalty):**
   If a true 10-second possession is split into two predicted spells (e.g. 4s and 5s by a 1-frame false touch):
   - Only *one* predicted spell can match the true spell; the second becomes a False Positive ($FP \uparrow$).
   - The matched spell has $\text{IoU} \le 5 / 10 = 0.50$, completely failing thresholds $0.6$ and $0.7$ ($TP \downarrow$).
   - A single spurious touch destroys both precision and recall at high IoU thresholds.
2. **Merged Possessions:**
   Smoothing over a genuine turnover or stoppage merges two distinct possessions into one. The long prediction matches at most one ground-truth spell (at degraded IoU due to large union), leaving the other as a False Negative.
3. **Boundary Timing Jitter:**
   A 25-frame (1.0-second) shift on an 8-second possession reduces IoU from $1.0$ to $0.75$; on a 3-second possession, a 1.0-second shift reduces IoU to $0.50$, failing $\theta \in \{0.6, 0.7\}$.

---

## 3. Physical Boundary Diagnostics on v3

Running `phantom11/boundary_diagnostics.py` against `phantom11_v3.csv` revealed:

| Diagnostic Metric | Measured Value in v3 | Physical Implication |
|---|---|---|
| **Total Possession Spells** | 493 | Baseline was 669; v3 unified many flickers. |
| **Mean Duration** | 8.71 s | Realistic match average. |
| **Median Duration** | 5.64 s | Typical passing sequence length. |
| **Spells < 1.0s** | 34 (6.9%) | Shortest remaining spells. |
| **Fleeting Deflections (0 carrier frames)** | **29 spells** | **85% of spells < 1.0s have NO carrier control.** |
| **In-Flight Turnovers** | 134 transitions | Ball speed $> 10\text{ m/s}$ during $H \leftrightarrow A$ transition. |
| **False Dead-Ball Splits ($\le 1.5$s)** | 4 instances | Short tracking loss splitting same-team possession. |
| **Premature Restarts** | 4 instances | Restart declared while ball stationary and player $> 2.5\text{m}$. |

### Why Did v3 Allow 29 Fleeting Deflections?
In `sequence_decoder.py`, duration regularisation had threshold `min_sustained_frames = 12` ($0.48$s). Any deflection lasting 13 to 24 frames ($0.5$s to $0.96$s) survived as a full possession spell if the opponent player happened to be closest in 2D Euclidean distance, even though the player never controlled or carried the ball!

---

## 4. The v4 Confidence-Gated Refinement Architecture

Rather than retraining or guessing ground-truth counts, v4 implements 4 deterministic physical gates in `phantom11/boundary_refinement.py`:

```
                Input v3 Prediction (493 spells)
                              │
    ┌─────────────────────────┴─────────────────────────┐
    ▼                                                   ▼
Gate 1: Deflection vs Interception Gate    Gate 2: Live Tracking Dropout Bridging
- Check spells < 1.0s (24 frames)          - Check D gaps <= 1.2s (30 frames)
- If surrounded by opposing team AND       - If same team controls before & after
  ZERO carrier frames (< 4 frames):        - No referee set-piece tag (< 0.35)
  → Unify into continuous possession       - Plausible displacement (<= 25 m/s)
- SAFEGUARD: If >= 4 carrier frames        → Bridge to surrounding team
  (confirmed dribble) → NEVER MERGE!
    └─────────────────────────┬─────────────────────────┘
                              │
    ┌─────────────────────────┴─────────────────────────┐
    ▼                                                   ▼
Gate 3: Premature Restart Correction       Gate 4: Strict Non-Play Clamping
- Transitions D -> H / D -> A              - Pre-match (0..11184)
- If ball stationary (< 0.5 m/s) and       - Halftime (90652..110807)
  player > 2.5m away:                      - Post-match (190198..199318)
  → Snap restart to true kick / reach      → Strictly clamp to D
    └─────────────────────────┬─────────────────────────┘
                              │
                Output v4 Prediction (471 spells)
```

---

## 5. Comparative Evaluation: v3 vs. v4

### 5.1 Frame and Transition Differences

| Metric | PHANTOM-11 v3 | PHANTOM-11 v4 (Experimental) | Delta ($\Delta$) |
|---|---:|---:|---:|
| **Total Frames** | 199,319 | 199,319 | 0 |
| **Changed Frames** | — | **237 (0.119%)** | +237 |
| **Home (H) Frames** | 61,241 (30.7%) | 61,303 (30.8%) | +62 |
| **Away (A) Frames** | 46,121 (23.1%) | 46,092 (23.1%) | -29 |
| **Dead (D) Frames** | 91,957 (46.1%) | 91,924 (46.1%) | -33 |
| **Total Possession Spells** | **493** | **471** | **-22 spells** |
| **Mean Duration** | 8.71 s | **9.12 s** | +0.41 s |
| **Median Duration** | 5.64 s | **5.96 s** | +0.32 s |
| **Spells < 1.0s** | 34 (6.9%) | **22 (4.7%)** | -12 spells |
| **Fleeting Deflections (0 carrier)** | 29 | **17** | **-12 deflections** |
| **In-Flight Turnovers** | 134 | **126** | -8 transitions |
| **False Dead-Ball Splits** | 4 | **2** | -2 splits |
| **Direct Turnovers ($H \leftrightarrow A$)** | 356 | **336** | -20 transitions |
| **Official Validator Status** | **VALID** | **VALID** | — |
| **Pipeline Runtime** | 4.59 s | **3.28 s** | -1.31 s |
| **SHA-256 Checksum** | `289be9ad...` | `f479648...` | Verified |

### 5.2 Breakdown of the 237 Changed Frames
- **$A \to H$ (123 frames):** 7 fleeting Athletic Club deflections occurring in the middle of Barcelona passing sequences unified into Barcelona possession.
- **$H \to A$ (81 frames):** 5 fleeting Barcelona deflections occurring during Athletic Club sequences unified into Athletic Club possession.
- **$D \to \text{Live}$ (33 frames):** 2 brief optical tracking occlusion dropouts ($< 0.8$s) inside continuous possessions bridged.
- **$\text{Live} \to D$ (0 frames):** Zero false stoppages created.

---

## 6. Synthetic & Robustness Testing

The complete test suite in `PHANTOM11-public` was expanded to **45 unit tests**:
- `tests/test_continuity.py`: 6 tests (missing-ball continuity, boundary clamping).
- `tests/test_evidence.py`: 11 tests (kinematics, carrier detection, deflection angles).
- `tests/test_decoder.py`: 12 tests (Viterbi DP, turnover readiness, dwell penalties).
- `tests/test_match_config.py`: 5 tests (match metadata, team alias mapping, validation).
- `tests/test_refinement.py`: 11 tests (deflection unification, confirmed interception protection, 5% dropout robustness, $\pm 0.3$m Gaussian coordinate noise).

**Result:** `Ran 45 tests in 0.252s — ALL OK`.

---

## 7. Final Assessment & Recommendation

### Why v4 is a Convincing Technical Refinement
1. **Physically Grounded:** Every changed frame corresponds to a confirmed physical scenario: eliminating opponent touches where relative speed and velocity vector alignment confirmed zero dribbling/carrying.
2. **Preserves Genuine Short Interceptions:** The safeguard condition (`carrier_count >= min_carrier_frames_for_turnover`) ensures that true rapid turnovers are never smoothed away.
3. **Conservative Scope:** v4 modifies only **237 frames out of 199,319** (one-tenth of one percent of the match), eliminating 22 artificial possession splits.
4. **Protects High-IoU Spell Matching:** By eliminating false possession cuts, unbroken possessions can achieve $\text{IoU} \ge 0.6$ and $0.7$, which directly raises `spellF1`.

### Prudent Competition Strategy
- **Status of v3:** `phantom11_v3.csv` (`289be9ad...`) is fully validated, reproducible, and already published as our safe fallback candidate.
- **Status of v4:** `phantom11_v4.csv` (`f479648...`) is generated locally, verified by the official validator, and thoroughly documented.
- **Next Step:** Both candidates are ready. As required by competition rules, neither candidate was selected by peeking at private ground truth. We await user authorization before any potential commit or push.
