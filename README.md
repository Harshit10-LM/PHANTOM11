# PHANTOM-11: Evidence-Aware Possession Sequence Decoding

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hackathon: More Than A Hack 2027](https://img.shields.io/badge/Competition-More%20Than%20A%20Hack%202027-orange.svg)](https://github.com/metrica-sports/more-than-a-hack-2027)

**PHANTOM-11** is an evidence-aware football possession prediction and sequence decoding system developed for the **More Than A Hack 2027** competition organized by Metrica Sports.

Instead of relying solely on spatial 2D proximity (who happens to be closest to the ball), PHANTOM-11 extracts physical team-control evidence (ball-player co-movement, team geometry, tactical phase priors) and solves possession inference as a constrained Hidden Semi-Markov dynamic programming sequence decoding problem.

---

## Key Features

1. **Physical Ball-Carrier Detection:** Identifies genuine ball-carrying / dribbling based on relative speed ($v_{\text{rel}} \le 2.5\text{ m/s}$) and velocity alignment ($\cos(\theta) \ge 0.40$), eliminating false turnovers when passes fly past opponents.
2. **Conservative Missing-Ball Continuity:** Distinguishes brief visual occlusions ($\le 6.0\text{s}$) from genuine dead-ball stoppages using bilateral boundary control and displacement plausibility.
3. **Tactical Soft Priors:** Leverages Metrica Smart Tagging phases (attacking vs. defending vs. restarts) as probabilistic soft priors without hard-coding labels.
4. **Duration-Aware Dynamic Programming:** Formulated as a constrained Hidden Semi-Markov Model (HSMM) solved with Viterbi DP in **0.5 seconds** ($O(T \cdot K^2)$), unifying micro-flickers while preserving genuine short possessions.
5. **Deterministic & CPU-Only:** Zero stochastic neural networks, zero GPU requirements. Reproduces exact byte-for-byte predictions in under 5 seconds.

---

## Multi-Version Evolution

| Version | Description | Total Possessions | Mean Duration | Micro-Possessions (<0.5s) | Direct Turnovers |
|---|---|---:|---:|---:|---:|
| **v0** | Baseline (Nearest Player + 50-frame Smoothing) | 669 | 7.64s | 68 | 528 |
| **v1** | Conservative Missing-Ball Continuity + Halftime Clamping | 579 | 7.45s | 60 | 455 |
| **v2** | Physical Movement Kinematics & Team Geometry | 629 | 6.86s | 95 | 505 |
| **v3** | **Duration-Aware Sequence Decoder (Final Candidate)** | **493** | **8.71s** | **4** | **356** |

---

## Directory Structure

```
PHANTOM11-public/
├── README.md                  # Project overview and reproduction guide
├── LICENSE                    # MIT open-source license
├── requirements.txt           # Pinned dependencies (numpy, pandas)
├── run_phantom11.py           # Unified reproduction entry point
├── decoder_config.json        # Decoder hyperparameters and SI units
├── phantom11/                 # Source package
│   ├── __init__.py
│   ├── match_config.py        # Reusable MatchConfig structure & validation
│   ├── path_resolver.py       # Portable official repo & data resolver
│   ├── pipeline.py            # End-to-end multi-stage pipeline
│   ├── movement_features.py   # Ball/player kinematics & carrier detection
│   ├── team_geometry.py       # Permutation-invariant team density spheres
│   ├── tactical_tags.py       # Smart Tagging soft evidence parser
│   ├── control_evidence.py    # Composite control scores & reason codes
│   ├── boundary_evidence.py   # Physical change points & readiness
│   ├── sequence_decoder.py    # Viterbi DP with duration regularization
│   ├── continuity.py          # Missing-ball continuity algorithm
│   └── load_tracking.py       # Structured ATD tracking loader
├── tests/                     # 34 synthetic unit & regression tests
│   ├── test_continuity.py     # 6 synthetic continuity tests
│   ├── test_evidence.py       # 11 synthetic evidence & carrier tests
│   ├── test_decoder.py        # 12 synthetic sequence decoder tests
│   └── test_match_config.py   # 5 match configuration & validation tests
├── docs/                      # Technical documentation
│   ├── TECHNICAL_DOCUMENT.md  # Official submission technical paper
│   ├── MATCH_GENERALISATION_AUDIT.md # Match decoupling & generalisation report
│   ├── STEP6_REPORT.md        # Tactical evidence report
│   └── STEP7_REPORT.md        # Sequence decoder report
└── scripts/
    ├── setup_environment.sh   # One-command fresh environment installer
    └── fetch_official_repo.py # Fetches public official code at pinned commit
```

---

## Getting Started (Fresh Machine Setup)

### 1. Clone this repository

```bash
git clone https://github.com/Harshit10-LM/PHANTOM11.git
cd PHANTOM11
```

### 2. Set up Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Obtain Official Code Dependency

PHANTOM-11 uses the official competition asset loading specifications (`games.py`).
To fetch the public MIT-licensed official repository at the pinned commit (`f8f2629`):

```bash
python scripts/fetch_official_repo.py
```
*(This downloads public code only; no licensed match data is acquired).*

### 4. Place the Authorized Competition Dataset

Authorized hackathon participants obtain the licensed dataset bundle (`BAR-ATH`) through the official competition portal. Unpack the files into either:
- `data/tactical/` inside this project, OR
- A custom path passed via `--data-dir` or `DATA_DIR` environment variable.

Expected files:
```
BAR-ATH_tactical_atd.txt
BAR-ATH_tactical_atd.xml
BAR-ATH_tactical_homography.csv
BAR-ATH_tactical_smart_tagging.xml
```

---

## Running Tests

All unit tests run on synthetic in-memory data and require no licensed match assets:

```bash
python -m unittest discover tests
```
*Expected: 34 tests pass in ~0.6 seconds (6 continuity, 11 evidence & carrier, 12 sequence decoder, 5 match configuration).*

---

## Multi-Match Generalisation & Architecture Readiness

PHANTOM-11 is designed with a clear separation between **match configuration** and **possession prediction algorithms**:

1. **Current Verified Baseline:** Operation is verified and benchmarked against the official competition fixture: **FC Barcelona vs Athletic Club (`BAR-ATH`)**.
2. **Decoupled Architecture:** The mathematical modules (carrier co-movement kinematics, permutation-invariant team geometry, and the duration-regularized Viterbi dynamic programming decoder) operate on abstract metric coordinates and state emission matrices, making them fundamentally match-agnostic.
3. **`MatchConfig` Abstraction:** Match-specific parameters (match identifier, club names, dataset location, frame count, FPS, pitch dimensions, playing/non-play intervals, and tracking slot mappings) are cleanly encapsulated in `phantom11.match_config.MatchConfig`.
4. **Prerequisites for Additional Matches:**
   - Other match datasets must provide their own metadata, camera synchronization, and frame offset alignments (or use unified EPTS tracking counters).
   - Pitch dimensions and frame rate must be calibrated in the `MatchConfig`.
   - Third-party formats require compatible parsers (e.g., standard FIFA/EPTS raw tracking).
5. **Honest Scope & No Universal Compatibility Claim:**
   - Cross-match prediction quality has **not yet been empirically validated** against ground truth outside of the authorized competition match.
   - We do not claim universal, out-of-the-box compatibility with arbitrary raw formats without appropriate configuration.
6. **Future Roadmap (Innovation Challenge):**
   - The upcoming Innovation Challenge extension will leverage this modular configuration to support multi-match tactical analysis, team-style profiling, and comparative possession benchmarking across diverse fixtures and leagues.

---

## Reproducing the Submission

To run the complete deterministic pipeline and generate `phantom11_v3.csv`:

```bash
python run_phantom11.py
```

Or specify custom directories:
```bash
python run_phantom11.py --data-dir /path/to/data --output-dir outputs
```

The output file will be written to:
`outputs/phantom11_v3.csv`

### Validating the Output

Run the official competition validator:

```bash
python vendor/more-than-a-hack-2027/evaluation/validate_submission.py outputs/phantom11_v3.csv
```

Expected output:
```
OK    phantom11_v3.csv
      199319 rows, frame 0..199318
      label counts
        H    61241   30.7%
        A    46121   23.1%
        D    91957   46.1%
        X        0    0.0%
      reports 493 possessions of mean 8.7 s
```

### Checksum Verification

The final generated submission has the verified SHA-256 hash:
```
289be9ad33748690c6197a4659060a459858e344c78e84dbfdcc51a373952d62  phantom11_v3.csv
```

---

## Limitations

1. **Single-Camera Occlusions:** In broadcast-style occlusions where the ball is hidden for $>6$ seconds without set-piece tags, the continuity module conservatively defaults to Dead Ball (`D`).
2. **Deflection Nuances:** Fleeting deflections that do not alter the overall team possession trajectory are ignored by the sequence regularizer; intentional micro-touches without established control are intentionally treated as non-possessions.
3. **Aerial Trajectories:** 2D tracking lacks vertical z-coordinates; high-speed aerial crosses passing over a player may occasionally register low relative 2D distance.

---

## Licensing & Attribution

- Source code in this repository is licensed under the [MIT License](LICENSE).
- Built for **More Than A Hack 2027** organized by Metrica Sports.
- Match assets (tracking data, tactical video, smart tagging) are licensed separately under Metrica Sports competition terms and are **not** redistributed here.
