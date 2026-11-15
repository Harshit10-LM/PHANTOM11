# PHANTOM-11: Evidence-Aware Possession Sequence Decoding
## Technical Documentation — More Than A Hack 2027

**Team:** PHANTOM-11  
**Competition:** More Than A Hack 2027 (Metrica Sports)  
**Match:** FC Barcelona vs Athletic Club (`BAR-ATH`), LaLiga EA Sports 2024-25, Matchday 2  
**Final Submission Model:** PHANTOM-11 v3 (`phantom11_v3.csv`)  
**Verified Checksum:** `289be9ad33748690c6197a4659060a459858e344c78e84dbfdcc51a373952d62`  

---

## 1. Approach Overview: Why Physical Evidence and Sequence Decoding?

The official hackathon baseline (`baseline_smooth50`) establishes possession by finding the player closest to the ball in 2D normalized coordinates and applying a centered majority-vote window of $\pm 50$ frames ($\pm 2.0$ seconds). While simple, this creates three structural failure modes:

1. **Spatial Proximity Artifacts:** A pass travelling at 15 m/s across midfield that flies within 2.5m of an opponent registers the opponent as "nearest", triggering an artificial turnover even though the opponent never touched the ball.
2. **Smoothing Boundary Lag:** A centered 50-frame window averages 2 seconds of the past and future together. When a genuine turnover or tackle occurs, the preceding team's label bleeds into the new possession for 15–25 frames, mislabeling the start of possessions.
3. **Micro-Island Fragmentation:** If a frame-level detector greedily overrides individual frames where an opponent touches or challenges for the ball, it introduces isolated 2-to-6 frame blips. Because an opponent blip splits an ongoing possession into two, it artificially inflates possession counts and penalizes temporal IoU matching.

### Our Solution
PHANTOM-11 formulates possession inference as an **evidence-aware, constrained sequence decoding problem**:
- **Frame-Level Evidence:** Measures ball-player relative kinematics (speed difference and velocity vector alignment) to detect true ball-carrying / dribbling, spatial density around the ball, and tactical soft priors.
- **Dynamic Transition Costs:** Transition penalties are modulated by physical change-point readiness (ball kicks, carrier transfers, tracking visibility edges).
- **Duration-Aware Dynamic Programming:** Solved via an exact Viterbi dynamic programming decoder in $0.5$ seconds, regularizing unsupported micro-islands while strictly preserving genuine short possessions ($\ge 1.0\text{s}$).

---

## 2. Match Assets Used & Architectural Progression

| Asset Name | Usage in PHANTOM-11 | Role in Inference |
|---|---|---|
| `BAR-ATH_tactical_atd.txt` | **Primary** | Full-match tracking coordinates for ball and 46 player slots. |
| `BAR-ATH_tactical_atd.xml` | **Primary** | FIFA/EPTS slot metadata mapping track slots to Home (0..23) vs Away (24..45). |
| `BAR-ATH_tactical_smart_tagging.xml` | **Supporting** | Soft priors for tactical phases (`BUILD UP`, `PROGRESSION`, etc.) and set pieces. |
| `BAR-ATH_tactical_homography.csv` | **Inspected, Dropped** | Verified during Step 3; tactical ATD is already in pitch-plane metric coordinates. |
| `BAR-ATH_tactical.mp4` | **Dropped** | Full visual decoding was unnecessary; tracking coordinates provided complete physical evidence. |
| Broadcast assets | **Dropped** | Tactical camera provided complete pitch view without broadcast pan/zoom occlusions. |

### Incremental Model Evolution
- **v0 (Baseline):** Reconstructed the official nearest-player baseline with 50-frame centered smoothing ($669$ possession spells).
- **v1 (Continuity):** Introduced conservative missing-ball possession continuity for gaps $\le 6.0\text{s}$ with bilateral same-team control and displacement $\le 30\text{m}$. Clamped halftime and non-play boundaries to Dead Ball (`D`). ($579$ spells).
- **v2 (Tactical Intelligence):** Extracted relative kinematics, active carrier detection, and team spatial density spheres ($1.5\text{m}, 3\text{m}, 6\text{m}$). Applied conservative carrier overrides, shifting lagging turnover boundaries to the true moment of takeover ($629$ spells).
- **v3 (Sequence Decoder):** Implemented the constrained Viterbi sequence decoder with physical transition costs and soft duration regularization. Reduced unsupported micro-possessions from 95 down to 4 while preserving genuine possessions ($493$ spells, mean duration $8.71\text{s}$).

---

## 3. Training Data & Anti-Memorization Policy

- **Ground-Truth Usage:** **Zero.** No models were trained or tuned on the provided match data, whether full-match or the public 5-minute window (`gt/BAR-ATH_possession_gt.csv`).
- **External Datasets:** None. PHANTOM-11 relies entirely on deterministic physical kinematics, spatial geometry, and rule-based dynamic programming.
- **Self-Supervision & Determinism:** All features and thresholds are derived from physical principles (human sprint maximums, ball aerodynamics, football rules), ensuring complete generalizability to any subsequent match.

---

## 4. Possession Rules & Decision Logic

### A. Active Ball-Carrying / Dribbling
A player slot $s$ is designated as actively carrying the ball if and only if:
1. Distance to ball $d \le 2.0\text{ m}$.
2. Tracker is reliable (single-frame displacement $\le 0.60\text{ m}$, equivalent to $\le 15\text{ m/s}$).
3. When moving ($> 1.0\text{ m/s}$): relative speed $v_{\text{rel}} \le 2.5\text{ m/s}$ and velocity cosine similarity $\cos(\theta) \ge 0.40$.
4. When stationary: $v_{\text{rel}} \le 1.5\text{ m/s}$ and $d \le 1.5\text{ m}$.

### B. Contested Duels & 50-50 Balls
- When both the nearest Home and Away players are within $2.5\text{ m}$ of the ball and the spatial margin $|d_{A, 1} - d_{H, 1}| \le 1.0\text{ m}$, the frame is flagged as `CONTESTED_DUEL`.
- Reliability drops to $\le 0.35$.
- In the sequence decoder, contested duels maintain the established possession state rather than thrashing back and forth between teams.

### C. Loose Balls & Passes in Transit
- When both teams are $> 3.0\text{ m}$ away from the ball and ball speed is high ($> 5.0\text{ m/s}$), the ball is flagged as `LOOSE_BALL` / pass in flight.
- Opponents who are briefly closer to the passing vector but have high relative velocity ($v_{\text{rel}} > 4.0\text{ m/s}$) are prevented from taking possession.

### D. Missing-Ball Gaps & Dead Ball Stoppages
- Gaps $> 6.0\text{ s}$ ($> 150$ frames) or overlapping Smart Tagging `SET PIECES` / `GOALS` are classified as Dead Ball (`D`).
- Short tracking dropouts ($\le 150$ frames) where the same team controlled the ball before and after, ball displacement is plausible ($\le 30\text{ m}$), and average ball speed is physically valid ($\le 25\text{ m/s}$), are bridged as continuous possession.

### E. Match Boundary Integrity
- Pre-match (`0..11,184`), Halftime (`90,652..110,807`), and Post-match (`190,198..199,318`) are dynamically derived from official match constants and strictly enforced as Dead Ball (`D`). 100% of the 20,156 halftime frames are preserved as `D`.

---

## 5. Comparative Evaluation & Diagnostics

```
Metric                           v0           v1           v2           v3 (Final)  
---------------------------------------------------------------------------------
Total Possessions                669          579          629          493         
Mean Duration                    7.64s        7.45s        6.86s        8.71s       
Median Duration                  4.72s        4.76s        4.12s        5.64s       
Micro-Possessions (< 0.5s)       68           60           95           4           
Direct Turnovers (H <-> A)       528          455          505          356         
Dead Ball Frames (D)             71,520       91,456       91,456       91,957      
Home Frames (H)                  72,957       61,454       61,482       61,241      
Away Frames (A)                  54,842       46,409       46,381       46,121      
```

### Transition Table (v2 $\rightarrow$ v3)
- **Total changed frames:** 750 frames (0.376%)
  - $A \to D$: 275 frames
  - $H \to D$: 226 frames
  - $H \to A$: 132 frames
  - $A \to H$: 117 frames
- Micro-possessions ($< 0.5\text{s}$) plummeted from **95 to 4**, eliminating temporal jitter and artificial spell splits.

---

## 6. Official Validation & Compliance

- Evaluated via `evaluation/validate_submission.py`:
  ```
  OK    phantom11_v3.csv
        199319 rows, frame 0..199318
        label counts: H 61241 (30.7%), A 46121 (23.1%), D 91957 (46.1%), X 0 (0.0%)
        reports 493 possessions of mean 8.7 s
  ```
- Checked row count: Exactly 199,319 rows.
- Frame alignment: Monotonic 0 to 199,318.
- Valid labels: Only `H`, `A`, `D` (no NaNs, no illegal strings).

---

## 7. Match Configuration & Generalisation Architecture

To ensure PHANTOM-11 is not artificially locked to a single fixture, all match-specific constants have been decoupled into a dedicated `MatchConfig` architecture (`phantom11/match_config.py`):

1. **Decoupled Parameters:** Pitch dimensions $(L, W)$, frame rate (FPS), match frame count $N$, playing blocks (kickoffs, half-times, full-time whistle), club names, and tracking slot counts are parameterized rather than hardcoded.
2. **Dynamic Slot Scaling:** The data loader and spatial geometry engine dynamically adapt to any number of tracking slots ($S$) read from EPTS XML metadata.
3. **Smart Tagging Optionality:** Tactical tag evidence gracefully degrades to neutral priors when annotations are absent for a match.
4. **Validation Guardrails:** `MatchConfig.validate()` verifies asset integrity and raises explicit errors for unsupported formats.
5. **Generalisation Scope:** While the mathematical core (kinematic carrier detection and Viterbi sequence regularization) is fully match-agnostic, empirical validation to date has been conducted on `BAR-ATH`. Subsequent datasets require independent metadata calibration and format compatibility verification.

---

## 8. Limitations & Future Work

1. **Lack of Z-Dimension in 2D Tracking:** High aerial balls (long lofted goal-kicks or crosses) appear close to players in 2D space. With 3D coordinates or video pose estimation, heading duels could be disambiguated with even higher precision.
2. **Fleeting Touch Intentionality:** When a defender deflects a pass without establishing control, human annotators may debate whether that touch constitutes a new possession spell. PHANTOM-11 requires established control before initiating a new spell.
3. **Multi-Camera Integration:** Integrating broadcast video to resolve occlusions near stadium touchlines where tactical tracking loses the ball could further improve dead-ball boundary sharpness.

---

## 9. Exact Reproduction Command

```bash
# In an environment with numpy and pandas:
python run_phantom11.py --data-dir /path/to/data --output-dir outputs

# Verifies with SHA-256:
shasum -a 256 outputs/phantom11_v3.csv
# Expected: 289be9ad33748690c6197a4659060a459858e344c78e84dbfdcc51a373952d62
```
