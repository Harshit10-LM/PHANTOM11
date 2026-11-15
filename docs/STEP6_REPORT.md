# PHANTOM-11: Step 6 Technical Report
## Tactical Intelligence & Team-Control Evidence Module

**Date:** 2026-11-15  
**Version:** PHANTOM-11 v2  
**Target Match:** FC Barcelona vs Athletic Club (`BAR-ATH`), 2024-25 Matchday 2  
**Status:** Completed and Validated  

---

## 1. Executive Summary

In Step 6 of PHANTOM-11, we advanced from a simple majority-vote baseline with gap continuity (v1) to an **evidence-aware tactical intelligence system (v2)**.

Instead of guessing team possession purely from who happens to be closest in 2D space, we developed physical evidence extractors that measure:
1. **Relative kinematics:** Whether a tracked player is actually moving together with the ball (active ball carrying / dribbling).
2. **Team spatial geometry:** Permutation-invariant player density spheres (1.5m, 3.0m, 6.0m) and spatial control margins.
3. **Tactical soft priors:** Integration of official Metrica Smart Tagging phases (attacking vs. defending vs. restarts) as probabilistic soft support rather than hard labels.
4. **Data reliability flags:** Automated detection of tracker swaps (> 15 m/s jumps), ball glitches (> 50 m/s jumps), missing observations, and 50-50 contested duels.

Using these signals, we created **PHANTOM-11 v2** (`phantom11/outputs/phantom11_v2.csv`). v2 modifies v1 **only when multiple independent physical signals strongly agree** that v1's 50-frame smoothing window lagged behind a genuine turnover.

All **11 synthetic and regression unit tests pass**, official validation reports **OK (199,319 rows)**, and all match boundaries (pre-match, halftime, post-match) are strictly preserved.

---

## 2. New Architectural Modules

```
phantom11/
├── load_tracking.py        # Fast structured ATD tracking loader (< 90 MB, ~2s)
├── movement_features.py    # Ball/player kinematics, relative motion & carrier detection
├── team_geometry.py        # Permutation-invariant team density & spatial margins
├── tactical_tags.py        # Smart Tagging alignment & soft evidence translation
├── control_evidence.py     # Composite evidence integration & reason codes
├── predict_v2.py           # Reversible v2 candidate generator & diagnostic reporter
├── test_evidence.py        # 11 unit & regression tests (all passing)
└── outputs/
    ├── phantom11_v0.csv    # Baseline reproduction (intact)
    ├── phantom11_v1.csv    # Continuity experiment (intact)
    ├── phantom11_v2.csv    # Evidence-aware candidate (validated)
    └── v2_decision_log.csv # Modification audit log (242 entries)
```

### Module 1: `movement_features.py`
Computes physical SI kinematics:
- **Ball Velocity & Heading:** Detects kicks and deflections (heading change $\ge 45^\circ$ or acceleration burst $> 40 \text{ m/s}^2$ at speeds $\ge 3 \text{ m/s}$).
- **Relative Motion:** Vector difference $\mathbf{v}_{\text{rel}} = \mathbf{v}_{\text{ball}} - \mathbf{v}_{\text{player}}$ and velocity alignment $\cos(\theta) = \frac{\mathbf{v}_b \cdot \mathbf{v}_p}{\|\mathbf{v}_b\| \|\mathbf{v}_p\|}$.
- **Ball Carrier Detection:** Identifies active ball carrying when:
  - Distance $d \le 2.0\text{ m}$.
  - Tracking is reliable (no position jumps).
  - When moving ($> 1\text{ m/s}$): $v_{\text{rel}} \le 2.5\text{ m/s}$ and $\cos(\theta) \ge 0.40$.
  - When slow / stationary: $v_{\text{rel}} \le 1.5\text{ m/s}$ and $d \le 1.5\text{ m}$.

### Module 2: `team_geometry.py`
Computes permutation-invariant team spatial metrics around the ball:
- **Player Density Counts:** Number of Home and Away players within immediate tackle radius ($1.5\text{ m}$), pressure radius ($3.0\text{ m}$), and tactical support radius ($6.0\text{ m}$).
- **Nearest Distances:** Sorted distances $d_{H, 1}, d_{H, 2}, d_{A, 1}, d_{A, 2}$.
- **Control Margin:** $\Delta d = d_{A, 1} - d_{H, 1}$ in metres.
- **Team Convergence:** Maximum radial speed of any player closing down the ball.
- **Geometric Control State:** Categorizes frames into Unambiguous Home, Unambiguous Away, Contested Duel ($|d_{A, 1} - d_{H, 1}| \le 1.0\text{ m}$ with both $\le 2.5\text{ m}$), or Loose Ball (both $> 3.0\text{ m}$).

### Module 3: `tactical_tags.py`
Integrates Smart Tagging as soft evidence:
- Translates attacking phases (`BUILD UP`, `PROGRESSION`, etc.) into soft support for the tagged team (+0.40).
- Translates defensive phases (`DEFENDING`, `DEFENSIVE TRANSITION`) into soft support for the opponent team (+0.35).
- Identifies conflicting tags and neutralizes them (0.0).
- Elevates stoppage support during `SET PIECES` and `GOALS` without enforcing hard labels.

### Module 4: `control_evidence.py`
Synthesizes all signals into transparent, interpretable outputs:
- `score_h`: float in $[0.0, 1.0]$
- `score_a`: float in $[0.0, 1.0]$
- `reliability`: confidence in $[0.0, 1.0]$ (low on missing ball, tracker jumps, duels)
- `reason_code`: Primary explanation (`CARRIER_H`, `CARRIER_A`, `PROXIMITY_H`, `PROXIMITY_A`, `CONTESTED_DUEL`, `LOOSE_BALL`, `BALL_MISSING`, `MATCH_NON_PLAY`, `TRACKING_JUMP_GLITCH`)

---

## 3. Full-Match Physical Reason Breakdown (199,319 frames)

| Reason Code | Frame Count | Percentage | Physical Interpretation |
|---|---:|---:|---|
| `BALL_MISSING` | 64,643 | 32.43% | Untracked ball during live play intervals |
| `MATCH_NON_PLAY` | 40,462 | 20.30% | Pre-match (11,185), Halftime (20,156), Post-match (9,121) |
| `LOOSE_BALL` | 32,553 | 16.33% | Pass in flight, clearance, cross, shot (both teams > 3m) |
| `CARRIER_H` | 15,461 | 7.76% | Active dribbling / close control by Barcelona |
| `PROXIMITY_H` | 14,244 | 7.15% | Barcelona player clearly nearest (margin $\ge 1.5$m) |
| `CARRIER_A` | 10,290 | 5.16% | Active dribbling / close control by Athletic Club |
| `PROXIMITY_A` | 8,918 | 4.47% | Athletic Club player clearly nearest (margin $\ge 1.5$m) |
| `UNRESOLVED` | 6,711 | 3.37% | Ambiguous distance / transitioning play |
| `CONTESTED_DUEL` | 5,918 | 2.97% | 50-50 physical duel (both teams within 1.5m) |
| `TRACKING_JUMP_GLITCH` | 119 | 0.06% | Single-frame tracking jump anomaly |

---

## 4. Why and Where v1 Made Errors

Comparing v1 labels against verified physical carriers revealed a major insight:
- **Smoothing Window Lag:** v1's 50-frame majority vote centered smoothing window spans $\pm 2.0\text{ s}$ (101 frames total).
- Around turnovers or tackles, this moving window averages past and future frames together.
- In **474 frames**, a Barcelona player was actively carrying the ball ($d \le 1.8\text{ m}$, opponent distant), but v1 was still predicting Athletic Club (`A`).
- In **355 frames**, an Athletic Club player was actively carrying the ball ($d \le 1.8\text{ m}$, Barcelona distant), but v1 was still predicting Barcelona (`H`).
- In **0 live frames** did v1 predict dead ball (`D`) while a player was carrying the ball. (All carrier/D coincidences occurred during halftime when non-players kicked balls around the pitch).

### Conservative Override Rule
v2 applies overrides **only** when multiple independent physical signals strongly agree:
1. `is_carrier` is True.
2. Distance to player $d \le 1.80\text{ m}$.
3. Spatial margin to opponent $\ge 1.00\text{ m}$ (opponent is at least 1m further away).
4. Evidence score $\ge 0.55$.
5. Strictly outside `NON_PLAY_INTERVALS`.

---

## 5. Comparative Diagnostics: v1 vs v2

### A. Label Counts
| Label | v1 Frames | v1 Pct | v2 Frames | v2 Pct | Net Difference |
|---|---:|---:|---:|---:|---:|
| **H** | 61,454 | 30.83% | 61,482 | 30.85% | **+28** |
| **A** | 46,409 | 23.28% | 46,381 | 23.27% | **-28** |
| **D** | 91,456 | 45.88% | 91,456 | 45.88% | **0** |
| **X** | 0 | 0.00% | 0 | 0.00% | **0** |
| **Total** | **199,319** | **100.00%** | **199,319** | **100.00%** | — |

### B. Transition Table (v1 $\rightarrow$ v2)
| Transition | Frames Changed | Reason |
|---|---:|---|
| **A $\rightarrow$ H** | 135 | `CARRIER_OVERRIDE_H` (Barcelona active carrier lagging in v1) |
| **H $\rightarrow$ A** | 107 | `CARRIER_OVERRIDE_A` (Athletic active carrier lagging in v1) |
| **Total Changed** | **242 (0.121%)** | All 242 documented in `v2_decision_log.csv` |

### C. Possession Structure
- **v1 Possessions:** 579 (mean 7.45s, median 4.76s)
- **v2 Possessions:** 629 (mean 6.86s, median 4.12s)
- **Direct Switches (H $\leftrightarrow$ A):** v1: 455 $\rightarrow$ v2: 505 (+50)
- **Halftime Integrity:** 20,156 / 20,156 frames (100.0% preserved as D).

---

## 6. Official Validation

```
OK    phantom11_v2.csv
      199319 rows, frame 0..199318

      label counts
        H    61482   30.8%
        A    46381   23.3%
        D    91456   45.9%
        X        0    0.0%

      reports 629 possessions of mean 6.9 s
```

---

## 7. Testing Summary

Unit tests in `phantom11/test_evidence.py` verified 11 specific physical scenarios:
1. `test_1_ball_carrier`: High-confidence co-movement detection (**PASS**).
2. `test_2_pass_between_teammates`: High-speed pass passing opponent without false carry (**PASS**).
3. `test_3_opponent_deflection`: Sudden 90-degree trajectory deflection (**PASS**).
4. `test_4_controlled_interception`: Opponent decelerates pass and establishes control (**PASS**).
5. `test_5_contested_duel`: 50-50 duel flagged as contested with low reliability (**PASS**).
6. `test_6_missing_ball`: Explicit NaN handling and low reliability (**PASS**).
7. `test_7_tracking_identity_swap`: >15 m/s tracker jump flagged as glitch (**PASS**).
8. `test_8_large_ball_jump`: >50 m/s unphysical ball jump flagged (**PASS**).
9. `test_9_ambiguous_smart_tagging`: Conflicting tactical tags neutralized (**PASS**).
10. `test_10_restart_stoppage`: Set-piece soft stoppage evidence (**PASS**).
11. `test_11_v1_audit_cross_flips_regression`: All 141 audit cross-flip frames verified as missing ball without false carrier evidence (**PASS**).

---

## 8. Recommendations for Step 7 (HSMM / Sequential Decoder)

With frame-level team evidence scores (`score_h`, `score_a`, `reliability`), we now have the ideal input representations for Step 7:
1. **Emission Probabilities:** Transform `score_h`, `score_a`, and dead-ball indicators into emission likelihoods for an Explicit-Duration Semi-Markov Model (HSMM) or Viterbi state decoder.
2. **Elimination of Window Smoothing:** A state-transition model with physical duration priors (e.g. gamma or Poisson distribution over possession spell lengths) will naturally prevent smoothing window lag at turnover boundaries, making ad-hoc majority-vote windows obsolete.
3. **Contested Duel Handling:** During 50-50 duels, the state decoder can maintain the previous team's possession until a clean takeover is established, preventing artificial flickering.
