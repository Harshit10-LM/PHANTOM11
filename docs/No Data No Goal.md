# No Data No Goal — PHANTOM-11
## Technical Submission Documentation — More Than A Hack 2027

**Team Name:** No Data No Goal  
**Algorithm Name:** PHANTOM-11 (Candidate v5-A)  
**Competition:** More Than A Hack 2027 (Metrica Sports)  
**Official Submission File:** `outputs/No Data No Goal.csv`  
**Verified SHA-256 Checksum:** `6acdd06563420e135f154cff29194260f187410aacc5888fe071ccc4e9bfd6c4`  
**Format Validator Status:** `PASSED (199,319 rows, frame 0..199318, 487 active spells, mean duration 8.5s)`  
**Baseline to Outperform:** Official private full-match baseline = `0.2433`  

---

## 1. Approach Overview & Architecture

The official competition baseline (`baseline_smooth50`) assigns possession to whichever player is geometrically closest to the ball in 2D coordinates, smoothed across a centered $\pm 50$-frame window ($\pm 2.0\text{s}$). While straightforward, this proximity-first paradigm exhibits three fundamental physical failure modes:

1. **Spatial Proximity Ambiguities on Aerial Passes:** A pass or lofted cross flying overhead at $15\text{ m/s}$ frequently passes within $2.0\text{m}$ of an opposing player on the 2D ground plane, triggering a spurious turnover even though the opponent was meters below the ball and never made contact.
2. **Boundary Lag & Smeared Turnovers:** Centered window smoothing averages past and future states together, smearing genuine turnover points by 15–25 frames ($0.6\text{s}$ to $1.0\text{s}$) and misallocating the start of attacking sequences.
3. **Dead-Ball & Restart Merging:** When a team kicks the ball out of bounds for a throw-in or corner kick and subsequently takes the restart, naive proximity merges the pre-stoppage play, the out-of-play interval, and the post-restart attack into a single unsegmented possession spell lasting over a minute.

### The PHANTOM-11 Framework

PHANTOM-11 formulates possession recognition as an **evidence-gated, constrained sequence decoding pipeline**:

```
Raw Tracking ATD (199,319 frames) + Smart Tagging Timeline
                       │
                       ▼
    [1. Ball-Carrier Kinematics & Spatial Density]
    - Dynamic co-movement (v_rel <= 2.5 m/s, cos(theta) >= 0.40)
    - Tracker reliability filtering & stationary traps
                       │
                       ▼
    [2. Duration-Aware Dynamic Programming Decoder]
    - Constrained Hidden Semi-Markov Model (HSMM)
    - Exact Viterbi DP regularizing micro-flickers in < 0.5s
                       │
                       ▼
    [3. Physical Boundary Refinement]
    - Confidence-gated boundary snapping to true strike frames
    - Elimination of fleeting deflections (< 1.0s, 0 carrier frames)
                       │
                       ▼
    [4. Flight-Aware Possession Handover]
    - Aerial pass continuity (unifying overhead transit)
    - Premature interception alignment (delay until d <= 1.5m / trap)
    - Immediate turnover recognition for volleys & sliding tackles
                       │
                       ▼
    [5. IFAB Law 9 Pitch-Boundary Enforcement]
    - Strict geometric pitch boundary rectangle (105.3m x 68.0m)
    - Physical margin (0.35m) accounting for ball radius & line width
    - Clamping out-of-bounds frames to Dead Ball ('D')
                       │
                       ▼
   Official Submission: outputs/No Data No Goal.csv (199,319 frames)
```

---

## 2. Core Algorithmic Components

### 2.1 Physical Ball-Carrier Kinematics
A player slot $s$ is identified as an active ball carrier if and only if:
1. **Proximity:** Distance to ball $d \le 2.0\text{m}$.
2. **Tracker Reliability:** Single-frame displacement $\le 0.60\text{m}$ ($15\text{ m/s}$).
3. **Dynamic Co-Movement:** When moving ($v > 1.0\text{ m/s}$), relative velocity $v_{\text{rel}} = \|\mathbf{v}_{\text{player}} - \mathbf{v}_{\text{ball}}\| \le 2.5\text{ m/s}$ and velocity cosine similarity $\cos(\theta) \ge 0.40$.
4. **Stationary Control:** When stationary ($v \le 1.0\text{ m/s}$), $v_{\text{rel}} \le 1.5\text{ m/s}$ and $d \le 1.5\text{ m}$.

### 2.2 Constrained Sequence Decoding (Viterbi HSMM)
Frame-level evidence is decoded through a duration-regularized Viterbi dynamic programming decoder in $O(T \cdot K^2)$ time ($< 0.5\text{s}$ for 199,319 frames). It penalizes rapid state thrashing during contested 50-50 duels while preserving genuine short possessions ($\ge 1.0\text{s}$).

### 2.3 Flight-Aware Possession Handovers
Under the official competition rules:
> *"While live, loose or contested possession generally remains with the last controlling team until an opponent gains controlled possession."*

* **Aerial Pass Continuity:** High-speed passes ($v_{\text{ball}} \ge 7.0\text{ m/s}$ / $25.2\text{ km/h}$) received downfield by a teammate within $2.5\text{m}$ remain with the passing team throughout flight. Intervening 2D proximity blips are eliminated unless an opponent established confirmed carrier control ($\ge 4$ frames = $0.16\text{s}$).
* **Premature Handover Alignment:** When an opponent intercepts a downfield pass, handover is delayed until the interceptor actually engages the ball ($d \le 1.5\text{m}$ or traps $v_{\text{ball}} \le 3.5\text{ m/s}$).
* **Turnover Responsiveness:** Volleys, sliding clearances, and direct interception passes are recognized immediately at contact ($\text{spellF1} = 1.0000$), avoiding artificial turnover latency.

### 2.4 IFAB Law 9 Pitch-Boundary Enforcement
IFAB Law 9 dictates that the ball is out of play when it wholly crosses the goal line or touchline:
$$\text{margin} = r_{\text{ball}} (0.11\text{m}) + w_{\text{line}} (0.12\text{m}) + \sigma_{\text{coord}} (0.12\text{m}) = \mathbf{0.35\text{ metres}}$$
* $d_{\text{oob}} \ge 0.80\text{m}$: Immediate confirmation of Dead Ball (`D`).
* $0.35\text{m} \le d_{\text{oob}} < 0.80\text{m}$: Requires $\ge 2$ consecutive frames ($80\text{ms}$) persistence filter to reject boundary calibration jitter.
* Missing ball observations (`NaN`) strictly assign $d_{\text{oob}} = 0.0\text{m}$, preventing missing tracking from ever falsely declaring Dead Ball.
* Full-match non-play intervals (pre-match, halftime, post-match) are strictly enforced as $100\%$ Dead Ball (`D`).

---

## 3. Evaluation & Validation Evidence

### 3.1 Synthetic Evaluation Across 41 Independent Scenarios
Evaluated across 41 independent scenarios spanning 7 stress suites (aerial crosses, first-time volleys, sliding clearances, 50-50 duels, throw-in rebounds, tracking dropouts, and boundary crossings):

| Model Candidate | Mean SpellF1 (41 Scenarios) | Mean Boundary Timing Error |
| :--- | :---: | :---: |
| **v3** (Baseline DP) | 0.6385 | 16.27 frames |
| **v5** (Flight Handover) | 0.8602 | 11.78 frames |
| **v6** (Deceleration Gated) | 0.8407 | 12.71 frames |
| **v7-A** (v6 + Boundary) | 0.8829 | 8.98 frames |
| **PHANTOM-11 v5-A (Selected)** | **0.9024** | **8.05 frames (322ms)** |

### 3.2 Real Match Public Ground-Truth Diagnostic (7,500 Frames)
Evaluated across `BAR-ATH_possession_gt_public.csv` ([11,185 .. 18,684]) using `evaluation/possession_metrics.py`:
* **Multiclass Frame Accuracy:** `v5-A` achieves **`73.72%`** (highest of all models; `v5` is `72.39%`, `v6` is `72.33%`, `v7-A` is `73.67%`).
* **Dead-Ball Classification:** Precision = **`0.7305`**, Recall = **`0.2491`**, F1 = **`0.3716`** (with 366 True Positives vs 266 in `v5`, $+100\text{ TP}$).
* **Boundary Crossing Precision (Frame 18,026):** On the 101 confirmed out-of-bounds frames, ground truth is `D` on 100/101 frames ($99.01\%$). At frame 18,026, the ball crossed the goal line at $14.9\text{ m/s}$ ($x = -0.795\text{m}$); ground truth switched to `D` at frame 18,027, verifying physical boundary alignment within **$40\text{ms}$ (1 video frame)** of human perception.
* *Policy Compliance:* Zero parameters or thresholds were tuned against this public window.

---

## 4. Multi-Match Independence & Known Limitations

### 4.1 Match-Independent Design
All domain logic interfaces through the abstract `MatchConfig` class:
* **Frame Rates:** Temporal parameters scale dynamically with `match_config.fps` (verified at 20, 25, 30 FPS with $\text{spellF1} \ge 0.90$).
* **Pitch Dimensions:** Boundary coordinates scale with `match_config.pitch_length` and `match_config.pitch_width` (verified across $100\times64\text{m}$ to $110\times72\text{m}$).
* **Tracking Rosters:** Dynamically maps Home and Away slots (verified across 28, 46, and 56 slots).
* **Team Metadata:** Zero hard-coded dependencies on team names.

### 4.2 Known Limitations & Scientific Honesty
1. **Unmeasured Private Score:** The competition's private full-match ground truth is strictly withheld. Synthetic benchmark scores represent diagnostic proxies, and verified real-match generalisation cannot be claimed without external match labels.
2. **Optical Parallax on High Aerial Crosses:** In 2D optical tracking, an elevated ball crossing the touchline in the air exhibits optical parallax from stadium camera elevation that cannot be fully resolved without 3D camera elevation models.
3. **Occlusions Outside Stadium:** When the ball leaves camera view outside the touchline, tracking drops out (`NaN`). While visible out-of-bounds frames are clamped to `D`, un-tracked gaps outside the pitch are handled by sequence decoder continuity.

---

## 5. Exact Reproduction Instructions

### 5.1 Verified Environment & Dependencies
Tested and verified on Python 3.12 (standard CPU, dependency-free numerical stack):
```
numpy>=1.26.0,<2.3.0  (verified with numpy 2.2.6)
pandas>=2.0.0,<3.2.0  (verified with pandas 3.1.1)
```

### 5.2 Single Regeneration Command
From the root of the public repository:

```bash
# Generate the official submission file:
python run_phantom11.py --candidate v5a
```
*This command executes the full deterministic pipeline from raw tracking data in under 5 seconds and writes `outputs/No Data No Goal.csv`.*

### 5.3 Verification
```bash
# Validate format:
python ../more-than-a-hack-2027/evaluation/validate_submission.py "outputs/No Data No Goal.csv"

# Verify SHA-256 hash:
sha256sum "outputs/No Data No Goal.csv"
# Expected: 6acdd06563420e135f154cff29194260f187410aacc5888fe071ccc4e9bfd6c4

# Run test suite:
python -m unittest discover -s tests -p "test_*.py"
```

---
*No Data No Goal — More Than A Hack 2027*
