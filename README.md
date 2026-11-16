# PHANTOM-11 — No Data No Goal
### Evidence-Aware Possession Sequence Decoding for Football Tracking Data

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hackathon: More Than A Hack 2027](https://img.shields.io/badge/Competition-More%20Than%20A%20Hack%202027-orange.svg)](https://github.com/metrica-sports/more-than-a-hack-2027)

**Official Team Name:** No Data No Goal  
**Algorithm:** PHANTOM-11 (Primary Candidate: `v5-A`)  
**Competition:** More Than A Hack 2027 (Metrica Sports)  
**Submission Deliverables:**
* **Official Submission CSV:** `outputs/No Data No Goal.csv` (and `outputs/phantom11_v5a.csv`)
* **Verified SHA-256 Checksum:** `6acdd06563420e135f154cff29194260f187410aacc5888fe071ccc4e9bfd6c4`
* **Format Validator Status:** `PASSED (199,319 rows, frame 0..199318, 487 active spells, mean 8.5s)`
* **Official Technical Paper:** [docs/No Data No Goal.md](docs/No%20Data%20No%20Goal.md)

---

## 1. Problem: The Pitfalls of Spatial Proximity

In optical football tracking data, the standard baseline (`baseline_smooth50`) assigns possession to whichever player is geometrically closest to the ball in 2D coordinates, smoothed across a centered $\pm 50$-frame window ($\pm 2.0\text{s}$).

While simple, this proximity-first paradigm exhibits three critical physical failures:
1. **Spatial Proximity Collapses on Aerial Passes:** A lofted pass or cross travelling overhead at $15\text{ m/s}$ frequently passes within $2.0\text{m}$ of an opposing player on the 2D ground plane, triggering spurious turnover blips even though the opponent was meters below the ball and never made contact.
2. **Boundary Lag & Smeared Turnovers:** Centered window smoothing averages past and future states, smearing genuine turnover points by 15–25 frames ($0.6\text{s}$ to $1.0\text{s}$) and misallocating the start of attacking sequences.
3. **Dead-Ball & Restart Merging:** When a team kicks the ball out of bounds for a throw-in or corner kick and subsequently takes the restart, naive proximity merges the pre-stoppage play, the out-of-play interval, and the post-restart attack into a single unsegmented possession spell lasting over a minute.

---

## 2. Our Solution: Evidence-Aware Temporal Decoding

Developed by team **No Data No Goal**, **PHANTOM-11** reformulates possession recognition as an **evidence-gated, constrained sequence decoding pipeline**.

Instead of trusting raw 2D Euclidean distances, PHANTOM-11 evaluates physical ball-player relative velocities, velocity vector alignment, spatial density spheres, and pitch geometry. It solves temporal continuity as a constrained Hidden Semi-Markov Model (HSMM) via exact Viterbi dynamic programming in **$< 0.5\text{ seconds}$**, completely eliminating micro-flickers while preserving genuine short possessions ($\ge 1.0\text{s}$).

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

## 3. Core Innovations

### 1. Ball-Carrier Kinematics & Spatial Evidence
Identifies genuine ball-carrying and dribbling based on relative speed ($v_{\text{rel}} \le 2.5\text{ m/s}$) and velocity vector cosine similarity ($\cos(\theta) \ge 0.40$), preventing high-speed passes flying past opponents from being misclassified as turnovers.

### 2. Hidden Semi-Markov Model with Constrained Viterbi Decoding
Solves global temporal continuity via exact Viterbi dynamic programming ($O(T \cdot K^2)$). It penalizes rapid state thrashing during contested 50-50 duels while preserving genuine short possessions ($\ge 1.0\text{s}$).

### 3. Confidence-Gated Possession Boundary Refinement
Snaps possession change points from lagging window boundaries directly to the physical strike or tackle frame, eliminating fleeting deflections ($< 1.0\text{s}$, 0 carrier frames) without control.

### 4. Flight-Aware Possession Handover
* **Aerial Pass Continuity:** Unifies long passes in flight ($v_{\text{ball}} \ge 7.0\text{ m/s}$ / $25.2\text{ km/h}$) received downfield by a teammate within $2.5\text{m}$, eliminating spurious mid-flight opponent blips.
* **Premature Handover Alignment:** Delays turnover recognition until the interceptor actually engages the ball ($d \le 1.5\text{m}$ or traps $v_{\text{ball}} \le 3.5\text{ m/s}$).
* **Turnover Responsiveness:** Volleys, sliding clearances, and direct interception passes are recognized immediately at contact ($\text{spellF1} = 1.0000$).

### 5. IFAB Law 9 Pitch-Boundary Detection & Dead-Ball Enforcement
Enforces pitch geometry ($105.3\text{m} \times 68.0\text{m}$) with a calibrated $0.35\text{m}$ physical buffer (ball radius + line width + optical uncertainty). Eliminates out-of-bounds false active possessions during throw-in and corner-kick preparations ($99.01\%$ agreement with human ground truth on the public evaluation window).

---

## 4. Multi-Version Algorithmic Evolution

| Version | Description | Active Spells | Mean Duration | Micro-Spells (<0.5s) | Direct Turnovers |
|---|---|---:|---:|---:|---:|
| **v0** | Baseline (Nearest Player + 50-frame Smoothing) | 669 | 7.64s | 68 | 528 |
| **v1** | Conservative Missing-Ball Continuity + Halftime Clamping | 579 | 7.45s | 60 | 455 |
| **v2** | Physical Movement Kinematics & Team Geometry | 629 | 6.86s | 95 | 505 |
| **v3** | Duration-Aware Sequence Decoder (Baseline Candidate) | 493 | 8.71s | 4 | 356 |
| **v4** | Confidence-Gated Physical Boundary Refinement | 471 | 9.07s | 4 | 344 |
| **v5** | Flight-Aware Possession Handover & Aerial Continuity | 431 | 9.53s | 4 | 321 |
| **v5-A** | **IFAB Law 9 Pitch-Boundary Clamping (Primary Submission)** | **487** | **8.53s** | **4** | **321** |

---

## 5. Reproducibility: How to Run

### 1. Environment Setup
PHANTOM-11 requires standard Python 3.11 or 3.12 (standard CPU, zero GPU dependencies).

```bash
# Clone the repository
git clone https://github.com/Harshit10-LM/PHANTOM11.git
cd PHANTOM11

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*Verified dependencies: `numpy==2.2.6`, `pandas==3.1.1` (pinned ranges: `numpy>=1.26.0,<2.3.0`, `pandas>=2.0.0,<3.2.0`).*

### 2. Obtain Competition Code
Fetch the public official competition assets (`games.py`, starter scripts) at the pinned commit:
```bash
python scripts/fetch_official_repo.py
```

### 3. Place Competition Data
Place authorized match tracking files into `data/` or provide `--data-dir PATH`:
* `BAR-ATH_tactical_atd.txt`
* `BAR-ATH_tactical_atd.xml`
* `BAR-ATH_tactical_smart_tagging.xml`

### 4. Single-Command Reproduction
To execute the complete deterministic pipeline from scratch and regenerate the official submission file:

```bash
python run_phantom11.py --candidate v5a
```
*Pipeline finishes in under 5 seconds on standard CPU and writes `outputs/No Data No Goal.csv`.*

---

## 6. Output Validation & Automated Tests

### Official Submission Format Validation
Run the competition's official validator:
```bash
python ../more-than-a-hack-2027/evaluation/validate_submission.py "outputs/No Data No Goal.csv"
```
**Expected Validator Output:**
```
OK    No Data No Goal.csv
      199319 rows, frame 0..199318

      label counts
        H    59351   29.8%
        A    44467   22.3%
        D    95501   47.9%
        X        0    0.0%

      reports 487 possessions of mean 8.5 s

Valid.
```

### Checksum Verification
```bash
sha256sum "outputs/No Data No Goal.csv"
# Expected: 6acdd06563420e135f154cff29194260f187410aacc5888fe071ccc4e9bfd6c4
```

### Automated Regression Test Suite
Run all 66 unit, regression, and cross-match generalisation tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*Expected: `Ran 66 tests in ~0.3s - OK`.*

---

## 7. Evaluation & Benchmark Findings

### Independent Synthetic Benchmark (41 Scenarios Across 7 Suites)
Evaluated across 41 independent scenarios (aerial passes, volleys, duels, throw-ins, clearances, tracking drops, and boundaries):

| Candidate | Mean SpellF1 (41 Scenarios) | Mean Boundary Timing Error |
| :--- | :---: | :---: |
| **v3** (Baseline DP) | 0.6385 | 16.27 frames |
| **v5** (Flight Handover) | 0.8602 | 11.78 frames |
| **v6** (Deceleration Gated) | 0.8407 | 12.71 frames |
| **v7-A** (v6 + Boundary) | 0.8829 | 8.98 frames |
| **PHANTOM-11 v5-A (Selected)** | **0.9024** | **8.05 frames (322ms)** |

### Public Ground-Truth Diagnostic (7,500 Frames)
Evaluated across `data/BAR-ATH_possession_gt_public.csv` ([11,185 .. 18,684]):
* **Multiclass Frame Accuracy:** `v5-A` achieves **`73.72%`** (highest overall; `v5` is `72.39%`, `v6` is `72.33%`).
* **Dead-Ball Classification:** Precision = **`0.7305`**, Recall = **`0.2491`**, F1 = **`0.3716`** (with 366 True Positives vs 266 in `v5`, $+100\text{ TP}$).
* **Boundary Precision (Frame 18,026):** On the 101 confirmed out-of-bounds frames, ground truth is `D` on 100/101 frames ($99.01\%$). At frame 18,026, the ball crossed the goal line at $14.9\text{ m/s}$ ($x = -0.795\text{m}$); ground truth switched to `D` at frame 18,027, verifying physical boundary alignment within **$40\text{ms}$ (1 video frame)** of human perception.
* *Policy Compliance:* Zero parameters or thresholds were tuned against this public window.

---

## 8. Limitations & Scientific Scope

1. **Unobserved Private Ground-Truth Score:** The competition's private full-match ground truth remains strictly unobserved. Synthetic benchmark scores ($0.9024$) represent diagnostic proxies under controlled scenarios, not official private competition scores.
2. **Official Baseline to Beat:** The official published baseline to outperform is **`0.2433` on private full-match spellF1**. We make no premature claims of leaderboard ranking or top-20 qualification.
3. **Camera Elevation Parallax:** High aerial balls crossing the touchline in 2D tracking exhibit optical parallax from stadium camera elevation that cannot be fully modeled without 3D camera elevation metadata.
4. **Out-of-Stadium Occlusions:** Tracking dropouts (`NaN`) outside the pitch are handled by sequence continuity once the ball leaves camera view.

---

## 9. Technical Documentation & Reports

Detailed technical documentation and audit reports are preserved in `docs/`:
* [docs/No Data No Goal.md](docs/No%20Data%20No%20Goal.md): Complete technical paper prepared for the competition submission form.
* [docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md): Comprehensive technical reference paper.
* [docs/STEP18_FINAL_GENERALISATION_AUDIT.md](docs/STEP18_FINAL_GENERALISATION_AUDIT.md): Final generalisation, fragmentation, and public GT audit.
* [docs/STEP17_FINAL_CANDIDATE_COMPARISON.md](docs/STEP17_FINAL_CANDIDATE_COMPARISON.md): Step 17 comparative candidate evaluation.

---

## 10. Team & Licensing

* **Team Name:** **No Data No Goal**
* **Algorithm:** **PHANTOM-11**
* **Competition:** More Than A Hack 2027 (Metrica Sports)
* **Code License:** [MIT License](LICENSE)
* **Data Notice:** Match assets are licensed under Metrica Sports competition terms and are **not** redistributed here.
