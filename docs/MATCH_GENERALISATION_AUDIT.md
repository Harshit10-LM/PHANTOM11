# PHANTOM-11: Match Generalisation and Dependency Audit

**Project:** PHANTOM-11 (More Than A Hack 2027)  
**Date:** November 2026  
**Status:** Generalisation-Readiness Audit Complete  

---

## 1. Executive Summary

PHANTOM-11 was initially engineered and verified against the official More Than A Hack 2027 competition dataset: FC Barcelona vs. Athletic Club (`BAR-ATH`). 

Before releasing the public repository and preparing for the Innovation Challenge, we conducted an audit of all match-specific assumptions across the codebase, identifying:
1. Which components are **already fully general and match-agnostic**.
2. Which components contain **hardcoded match assumptions** that should be decoupled via a clean configuration layer.
3. What information is required to run PHANTOM-11 on a new football match.

---

## 2. Inventory of Match-Specific Assumptions

### 2.1 Club Identifiers and Team Names
* **Current State:**
  - Official constants in `games.py` specify `GAME = "BAR-ATH"`, with teams Barcelona (Home = 0) and Athletic Club (Away = 1).
  - In `phantom11/tactical_tags.py`, `TEAM_MAP` contains hardcoded lowercase string aliases: `{"barcelona": 0, "atletic club": 1, "athletic club": 1}`.
  - In `phantom11/path_resolver.py`, `resolve_data_dir()` explicitly looks for `BAR-ATH_tactical_atd.txt`.
* **Impact on Reusability:**
  - Running on another match (e.g., `RMA-ATM` or `MCI-LIV`) would cause Smart Tagging team strings to not map to 0/1, falling back to `-1` (unassigned) and muting tactical prior evidence.
  - Data directory resolver would fail if the filename doesn't start with `BAR-ATH`.
* **Remedy:**
  - Introduce `MatchConfig.home_team` and `MatchConfig.away_team`.
  - Construct `team_map` dynamically by normalizing team names from match metadata.
  - Parameterize `path_resolver.py` to accept any match stem while retaining `BAR-ATH` as default.

---

### 2.2 Frame Count ($N = 199,319$)
* **Current State:**
  - $N = 199,319$ is the specific length of the tactical camera video for `BAR-ATH` (~132.8 minutes at 25 fps).
  - In `phantom11/continuity.py`, default `NON_PLAY_INTERVALS` hardcodes frame `199318` as the final frame.
  - In `phantom11/pipeline.py` and `predict_v3.py`, array sizes are allocated based on `n_frames` passed from `load_full_tracking()`.
* **Impact on Reusability:**
  - Matches with extra time, differing stoppage time, or truncated video files have different frame counts.
* **Remedy:**
  - Read $N$ directly from `MatchConfig.n_frames` (derived from tracking metadata / video frame count).
  - Dynamically derive non-play intervals using `n_frames` rather than a static literal.

---

### 2.3 Frame Rate (FPS = 25.0)
* **Current State:**
  - Tactical tracking and video are assumed to run at 25.0 Hz.
  - Velocity and acceleration thresholds in `phantom11/movement_features.py` use `DEFAULT_FPS = 25.0`.
  - The sequence decoder duration penalty in `phantom11/sequence_decoder.py` uses `min_frames = 12` (~0.48s at 25 fps).
  - Maximum continuity gap in `phantom11/continuity.py` is `MAX_CONTINUITY_GAP_FRAMES = 150` (6.0s at 25 fps).
* **Impact on Reusability:**
  - Most optical and EPTS systems operate at 25 fps (FIFA standard). However, broadcast tracking or Hawk-Eye systems may provide 29.97, 30, or 50 fps data.
* **Remedy:**
  - Store `fps` in `MatchConfig` and `decoder_config.json`.
  - Convert all temporal thresholds dynamically from physical seconds ($T_{\text{sec}}$) to frame counts ($N_{\text{frames}} = \text{round}(T_{\text{sec}} \times \text{FPS})$).

---

### 2.4 Pitch Dimensions
* **Current State:**
  - `BAR-ATH` pitch is $105.3\text{ m} \times 68.0\text{ m}$ (Olympic Stadium Lluís Companys).
  - In `phantom11/continuity.py`, the default argument is `pitch=(105.3, 68.0)`.
  - Normalised tracking coordinates $[0, 1]$ in EPTS raw files are multiplied by $(L, W)$ to produce physical metric coordinates.
* **Impact on Reusability:**
  - Pitches vary in length (100–110m) and width (64–75m). Assuming fixed pitch dimensions slightly scales Euclidean distances and velocities.
* **Remedy:**
  - Store `pitch: Tuple[float, float]` in `MatchConfig` (read from metadata or competition constants).

---

### 2.5 Tracking Slot Count ($S = 46$)
* **Current State:**
  - In `phantom11/load_tracking.py`, arrays were allocated with fixed shape `(N, 46, 2)`.
  - In `phantom11/movement_features.py`, `team_geometry.py`, and `control_evidence.py`, slot count was already dynamic (`n_slots = players_xy.shape[1]`).
* **Impact on Reusability:**
  - Another match might have 40, 50, or 60 tracking slots depending on substitutions and warmup tracklets.
* **Remedy:**
  - Determine `n_slots` dynamically from `len(team_of_track)` parsed from the EPTS metadata XML file.

---

### 2.6 Match Boundaries (Kickoffs, Halftime, Full Time)
* **Current State:**
  - `BAR-ATH` has kickoff at frame 11,185, halftime 90,652..110,807, and full time at 190,197.
  - `phantom11/boundary_evidence.py::derive_non_play_intervals(game)` computes these intervals dynamically from `games.half_video_blocks(game)`.
  - `phantom11/continuity.py` still contained a fallback constant `NON_PLAY_INTERVALS`.
* **Impact on Reusability:**
  - Every football match has unique kickoff and whistle timestamps.
* **Remedy:**
  - Derive non-play intervals directly from `MatchConfig.playing_intervals` or official game block definitions.

---

### 2.7 Video-to-Tracking Frame Alignment
* **Current State:**
  - For `BAR-ATH`, `games.py` documents per-half offsets: `half 1: 1702057`, `half 2: 1702058`.
  - ATD frame IDs are mapped to video frames via `video_frame = frame_id - 1` (since raw ATD starts at frame 1 for video frame 0).
* **Impact on Reusability:**
  - If tracking data is not pre-synchronised to video frame 0, an explicit synchronization offset table per half is required.
* **Remedy:**
  - Explicitly document synchronization requirements in `MatchConfig`.

---

### 2.8 Smart Tagging Annotation Taxonomy
* **Current State:**
  - `phantom11/tactical_tags.py` groups event codes into attacking, defending, and stoppage categories (e.g. `BUILD UP`, `DEFENSIVE TRANSITION`, `SET PIECES`).
* **Impact on Reusability:**
  - These codes represent standard Metrica Sports Smart Tagging taxonomy. However, third-party matches may lack Smart Tagging entirely or use StatsBomb / Opta event vocabularies.
* **Remedy:**
  - Treat Smart Tagging as an **optional asset**. If absent, the module cleanly emits neutral priors ($0.0$) without throwing errors or breaking the pipeline.

---

## 3. Generalisation Status Summary Table

| Component | Status Before Audit | Refactoring Action | Reusability After |
|---|---|---|---|
| **Sequence Decoder (DP)** | Fully Agnostic | None required (uses abstract state costs) | 100% Match Agnostic |
| **Control Evidence Math** | Fully Agnostic | Slot count already dynamic via `.shape[1]` | 100% Match Agnostic |
| **Ball Kinematics** | Parameterized | Pass `fps` from `MatchConfig` | 100% Match Agnostic |
| **Data Loader** | Hardcoded 46 slots | Allocate `(N, n_slots, 2)` from metadata | Fully Dynamic |
| **Team Mapping** | Hardcoded strings | Dynamic dictionary from `MatchConfig` | Fully Dynamic |
| **Pitch Scaling** | Hardcoded default | Injected via `MatchConfig.pitch` | Fully Dynamic |
| **Match Boundaries** | Partial constants | Injected via `MatchConfig.non_play_intervals` | Fully Dynamic |
| **Path Resolver** | Hardcoded BAR-ATH | Parameterized stem search with fallback | Match Agnostic |

---

## 4. Conclusion & Next Steps

The mathematical core of PHANTOM-11 (the geometric possession scoring and the Viterbi dynamic programming decoder) is **completely match-agnostic**. The only coupling lay in data loading, default parameters, and team string lookups.

Introducing a unified `MatchConfig` cleanly isolates all match-specific metadata without modifying the underlying possession prediction algorithms.
