# PHANTOM-11: Evidence-Aware Possession Sequence Decoding
## Technical Documentation — More Than A Hack 2027

**Team:** PHANTOM-11  
**Competition:** More Than A Hack 2027 (Metrica Sports)  
**Match:** FC Barcelona vs Athletic Club (`BAR-ATH`), LaLiga EA Sports 2024-25, Matchday 2  
**Primary Submission Candidate:** PHANTOM-11 v5-A (`outputs/phantom11_v5a.csv`)  
**Verified SHA-256 Checksum:** `6acdd06563420e135f154cff29194260f187410aacc5888fe071ccc4e9bfd6c4`  
**Official Submission Validator Status:** `PASSED (err=None, 199,319 frames, 0..199318, 487 active spells)`  
**Baseline / Fallback Candidate:** PHANTOM-11 v3 (`289be9ad33748690c6197a4659060a459858e344c78e84dbfdcc51a373952d62`)  

---

## 1. Executive Summary & Architectural Overview

The official competition baseline (`baseline_smooth50`) assigns possession to whichever player is geometrically closest to the ball in 2D coordinates, smoothed across a centered $\pm 50$-frame window ($\pm 2.0\text{s}$). While straightforward, this proximity-first paradigm exhibits three fundamental physical failure modes:

1. **Spatial Proximity Ambiguities:** A high-speed aerial pass or lofted cross flying overhead at $15\text{ m/s}$ frequently passes within $2.0\text{m}$ of an opposing player on the ground plane, triggering a spurious turnover even though the opponent was meters below the ball and never made contact.
2. **Boundary Lag & Smeared Turnovers:** Centered window smoothing averages past and future states together, smearing genuine turnover points by 15–25 frames ($0.6\text{s}$ to $1.0\text{s}$) and misallocating the start of attacking sequences.
3. **Dead-Ball & Restart Merging:** When a team kicks the ball out of bounds for a throw-in or corner kick and subsequently takes the restart, naive proximity merges the pre-stoppage play, the out-of-play interval, and the post-restart attack into a single unsegmented possession spell lasting over a minute.

### The PHANTOM-11 Framework
PHANTOM-11 formulates possession recognition as an **evidence-gated, constrained sequence decoding pipeline**:
* **Geometric & Kinematic Evidence:** Evaluates ball-player relative velocities, velocity vector alignment, and spatial density spheres to identify genuine ball-carrying and deliberate touches.
* **Duration-Aware Dynamic Programming:** Solves sequence continuity as a constrained Hidden Semi-Markov Model (HSMM) via exact Viterbi dynamic programming in $< 0.5\text{s}$, eliminating micro-flickers while strictly preserving genuine short possessions ($\ge 1.0\text{s}$).
* **Flight-Aware Possession Handover:** Distinguishes high-speed passes in transit from opponent control, delaying turnovers until the intercepting player reaches contact distance ($\le 1.5\text{m}$) or traps the ball, while immediately recognizing first-time volleys and sliding clearances.
* **IFAB Law 9 Pitch-Boundary Enforcement:** Implements strict geometric pitch boundaries ($105.3\text{m} \times 68.0\text{m}$) with a calibrated $0.35\text{m}$ buffer (accounting for ball radius, line width, and optical uncertainty), eliminating out-of-play false active possessions.

---

## 2. Match Assets & Architectural Evolution

| Asset Name | Usage | Role in Inference |
| :--- | :---: | :--- |
| `BAR-ATH_tactical_atd.txt` | **Primary** | Full-match tracking coordinates for ball and 46 player slots (199,319 frames). |
| `BAR-ATH_tactical_atd.xml` | **Primary** | FIFA/EPTS metadata mapping track slots to Home (0..23) vs Away (24..45). |
| `BAR-ATH_tactical_smart_tagging.xml` | **Supporting** | Soft priors for tactical phases (`BUILD UP`, `PROGRESSION`) and set-piece stoppages. |
| `BAR-ATH_tactical_homography.csv` | **Audited** | Inspected; tactical ATD is already projected into pitch-plane metric coordinates. |
| Video Footage (`.mp4`) | **Dropped** | Unnecessary; metric tracking provides complete physical kinematics without video decode overhead. |

### Incremental Evolution (v0 $\to$ v5-A)

```
[Raw Tactical ATD]
       │
       ▼
[v0 Baseline] ──────── 50-frame majority smoothing (669 spells)
       │
       ▼
[v1 Continuity] ────── Missing-ball gap bridging (<= 6.0s) & non-play clamping (579 spells)
       │
       ▼
[v2 Carrier Overrides] ─ Ball-carrier kinematics (v_rel <= 2.5m/s, cos(theta) >= 0.40) (629 spells)
       │
       ▼
[v3 Sequence Decoder] ── Duration-regularized Viterbi DP (493 spells, mean duration 8.71s)
       │
       ▼
[v4 Boundary Refine] ─── Confidence-gated boundary snapping & deflection elimination (471 spells)
       │
       ▼
[v5 Flight Handover] ─── Aerial pass continuity & responsive turnover timing (431 active spells)
       │
       ▼
[v5-A Final Candidate] ─ IFAB Law 9 boundary clamping (487 active spells, 204 dead spells)
```

---

## 3. Core Algorithmic Components

### 3.1 Ball-Carrier Kinematics & Geometric Control
A player slot $s$ is identified as an active ball carrier if and only if:
1. Distance to ball $d \le 2.0\text{m}$.
2. Tracker reliability: single-frame displacement $\le 0.60\text{m}$ ($15\text{ m/s}$).
3. **Dynamic Co-Movement:** When moving ($v > 1.0\text{ m/s}$), relative velocity $v_{\text{rel}} = \|\mathbf{v}_{\text{player}} - \mathbf{v}_{\text{ball}}\| \le 2.5\text{ m/s}$ and velocity cosine similarity $\cos(\theta) \ge 0.40$.
4. **Stationary Control:** When stationary ($v \le 1.0\text{ m/s}$), $v_{\text{rel}} \le 1.5\text{ m/s}$ and $d \le 1.5\text{ m}$.

### 3.2 Flight-Aware Possession Handovers
Under the official competition rules:
> *"While live, loose or contested possession generally remains with the last controlling team until an opponent gains controlled possession."*

1. **Aerial Pass Continuity:** When Team A kicks a pass that travels at high speed ($v_{\text{ball}} \ge 7.0\text{ m/s}$ / $25.2\text{ km/h}$) and is received downfield by a teammate within $2.5\text{m}$, any intervening 2D proximity blip favoring an opponent is unified into continuous Team A possession, provided the opponent never established carrier control ($\ge 4$ frames).
2. **Premature Handover Alignment:** When an opponent intercepts a downfield pass, naive 2D proximity flips possession while the ball is still in flight $20\text{m}$ away. `v5-A` delays handover until the interceptor engages the ball ($d \le 1.5\text{m}$ or traps $v_{\text{ball}} \le 3.5\text{ m/s}$).
3. **Responsive Turnover Preservation:** By avoiding restrictive deceleration gating (`gate_deceleration_by_proximity=False`), first-time volleys, sliding clearances, and direct interception passes are recognized immediately at contact ($\text{spellF1} = 1.0000$).

### 3.3 IFAB Law 9 Pitch-Boundary Enforcement
IFAB Law 9 mandates that the ball is out of play when it has wholly crossed the goal line or touchline:
$$\text{margin} = r_{\text{ball}} (0.11\text{m}) + w_{\text{line}} (0.12\text{m}) + \sigma_{\text{coord}} (0.12\text{m}) = \mathbf{0.35\text{ metres}}$$
* $d_{\text{oob}} \ge 0.80\text{m}$: Immediate confirmation of Dead Ball (`D`).
* $0.35\text{m} \le d_{\text{oob}} < 0.80\text{m}$: Requires $\ge 2$ consecutive frames ($80\text{ms}$) persistence filter to reject single-frame tracking calibration jitter.
* Frames with missing ball observations (`NaN`) strictly assign $d_{\text{oob}} = 0.0\text{m}$, preventing missing tracking from ever triggering false Dead Ball labels.
* Full-match non-play intervals (pre-match, halftime, post-match) are strictly enforced as $100\%$ Dead Ball (`D`).

---

## 4. Diagnostics & Comparative Evaluation

### 4.1 Synthetic Stress Benchmark (36 Scenarios Across 6 Suites)

| Candidate | Mean SpellF1 | F1 @ 0.3 | F1 @ 0.5 | F1 @ 0.7 | Mean Boundary Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v3** (Baseline Fallback) | 0.6333 | 0.8333 | 0.6667 | 0.3667 | 16.42 frames |
| **v5** (Flight Handover) | 0.8648 | 0.9576 | 0.9000 | 0.6727 | 11.87 frames |
| **v6** (Decel Gated) | 0.8426 | 0.9455 | 0.8848 | 0.6364 | 12.92 frames |
| **v7-A** (v6 + Boundary) | 0.8667 | 0.9636 | 0.9030 | 0.6788 | 10.21 frames |
| **PHANTOM-11 v5-A** | **0.8889** | **0.9758** | **0.9303** | **0.7182** | **9.15 frames (366ms)** |

* `v5-A` achieves the **highest empirical score** and the **lowest boundary timing error** across all evaluated architectures.

### 4.2 Official Public Ground-Truth Diagnostic (7,500 Frames)
Evaluated across `data/BAR-ATH_possession_gt_public.csv` ([11,185 .. 18,684]) using `evaluation/possession_metrics.py`:
* **Multiclass Frame Accuracy:** `v5-A` achieves **`73.72%`** (highest overall; `v5` is `72.39%`).
* **Dead-Ball ('D') Classification:**
  - Precision: **`0.7305`** (vs `0.6650` for `v5`)
  - Recall: **`0.2491`** (vs `0.1811` for `v5`)
  - F1-Score: **`0.3716`** (vs `0.2846` for `v5`)
  - True Positives: **366** (an increase of $+100\text{ TP}$ over `v5`).
* **Frame 18,026 Disagreement Dissection:** On the 101 confirmed out-of-bounds frames, ground truth is `D` on 100/101 frames ($99.01\%$). At frame 18,026, the ball crossed the goal line at $14.9\text{ m/s}$ ($x = -0.795\text{m}$); ground truth switched to `D` at frame 18,027, confirming automated boundary fidelity within **$40\text{ms}$ (1 video frame)** of human perception.
* *Policy Note:* Zero parameters or thresholds were tuned against this public window.

---

## 5. Cross-Match Generalisation & Match-Independent Design

All domain logic in PHANTOM-11 references the abstract `MatchConfig` object:
* **Frame Rates:** Temporal parameters are defined in seconds ($T_{\text{sec}}$) and dynamically scaled by `match_config.fps` (verified across 20, 25, and 30 FPS with $\text{spellF1} \ge 0.90$).
* **Pitch Geometry:** Out-of-bounds coordinates dynamically scale with `match_config.pitch_length` and `match_config.pitch_width` (verified across $100\times64\text{m}$, $105\times68\text{m}$, and $110\times72\text{m}$).
* **Tracking Slots:** Dynamically reads Home and Away slot mappings (verified across 28, 46, and 56 slots).
* **Zero Team Dependencies:** Verified with arbitrary match identifiers (e.g. `"LIV_MCI_2027"`).

> **Scientific Limitation:** While synthetic evaluation confirms mathematical invariance, **we do not claim verified real-match generalisation** without independently labelled external match data.

---

## 6. Exact Reproduction Instructions

### 1. Requirements
* Python 3.11 or 3.12 (standard CPU, zero GPU dependencies).
* Official competition repository checkout containing `starter/baseline.py`, `games.py`, and `evaluation/`.

### 2. Single Regeneration Command
From the root of `PHANTOM11-public`:

```bash
# Generate the primary competition candidate (v5-A) from scratch:
python run_phantom11.py --candidate v5a
```

### 3. Verification & Validation
```bash
# Verify official submission format:
python ../more-than-a-hack-2027/evaluation/validate_submission.py outputs/phantom11_v5a.csv

# Verify exact SHA-256 checksum:
sha256sum outputs/phantom11_v5a.csv
# Expected: 6acdd06563420e135f154cff29194260f187410aacc5888fe071ccc4e9bfd6c4

# Run complete automated test suite (66 tests):
python -m unittest discover -s tests -p "test_*.py"
```

---
*PHANTOM-11 Engineering Team — More Than A Hack 2027*
