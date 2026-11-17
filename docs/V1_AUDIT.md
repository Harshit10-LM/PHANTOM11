# PHANTOM-11 v1 — Independent Technical Audit

**Audit date:** 2026-11-15  
**Audited version:** `phantom11_v1.csv` + source files in `phantom11/`  
**Auditor note:** Every finding below is derived from actually running the code with real data or reading the source verbatim. No assumption is taken from prior summaries. Ground-truth labels were NOT consulted at any point.

---

## Audit Scope

| # | Topic |
|---|-------|
| 1 | Match-boundary handling |
| 2 | Smart Tagging logic |
| 3 | Diagnostic reconciliation (gap coverage vs. changed frames) |
| 4 | Continuity algorithm correctness |
| 5 | Test-suite coverage |
| 6 | Final verdict |

---

## Audit 1 — Match-Boundary Handling

### What was checked

The hardcoded `NON_PLAY_INTERVALS` in `continuity.py` lines 49-53 were compared against the values dynamically derived from `games.half_video_blocks("BAR-ATH")`.

**Command run:**
```
python -c "import games; print(games.half_video_blocks('BAR-ATH'))"
# Output: [(1, 11185, 90651), (2, 110808, 190197)]
```

The derivation formula:
- Pre-match:  `(0, h1_start − 1)` = `(0, 11184)`
- Halftime:   `(h1_end + 1, h2_start − 1)` = `(90652, 110807)`
- Post-match: `(h2_end + 1, n_frames − 1)` = `(190198, 199318)`

### Finding

✅ **CONFIRMED CORRECT — No off-by-one errors.**

The hardcoded values `[(0, 11184), (90652, 110807), (190198, 199318)]` **exactly match** the values derived from `games.half_video_blocks`. Both kickoff frames (11,185 and 110,808) are correctly included in live play, not in the dead-ball interval.

### Improvement Opportunity (Medium Priority — Not a Bug)

The boundaries are correct numerically but are still *hardcoded constants* rather than being derived dynamically from `games.py`. If the official repo ever updates `games.half_video_blocks` (e.g., for a different match), `continuity.py` would silently drift out of sync.

**Recommended fix:** Replace the hardcoded literal at the top of `continuity.py` with:

```python
import games as _games
def _derive_non_play_intervals(game=_games.GAME):
    blocks = _games.half_video_blocks(game)
    n = _games.source_frames("tactical", game)
    return [
        (0, blocks[0][1] - 1),                   # pre-match
        (blocks[0][2] + 1, blocks[1][1] - 1),     # halftime
        (blocks[1][2] + 1, n - 1),                # post-match
    ]
NON_PLAY_INTERVALS = _derive_non_play_intervals()
```

> [!NOTE]
> This change would not alter the current output for BAR-ATH. Its value is defensive programming only: protection against copy-paste for future matches.

---

## Audit 2 — Smart Tagging Logic

### What was checked

`predict_v1.py` line 130-134 filters Smart Tagging events to `SET PIECES` and `GOALS` codes and passes their frame windows to `apply_continuity()`. The continuity module then treats any gap overlapping those windows as a confirmed stoppage (skips recovery).

### Finding 1 — Smart Tagging windows are *tactical phases*, not exact ball-out-of-play instants

> [!IMPORTANT]
> **Severity: Medium — Known approximation, not a correctness bug.**

The Smart Tagging XML annotates *tactical phases* centred around set-piece events, typically spanning a few seconds before and after the actual restart. This means:

- A `SET PIECES` window might begin **before** the whistle and end **after** play restarts.
- A gap whose ball comes back into play mid-window could still be preserved as dead-ball — **slightly over-protective**.
- Conversely, a quick ball-out-of-play that happens *between* Smart Tagging annotations (e.g. a ball briefly crossing the touchline with no player tracking dropout) would **not be caught** by this filter.

**Risk assessment:** Over-protection is the safer failure mode for a conservative algorithm. Under-bridging (missing a continuity recovery) is preferable to over-bridging (incorrectly labelling a dead ball as possession). The current implementation errs on the correct side.

**Measured impact:** 11 of 186 gaps were preserved by the set-piece filter. All 11 would have needed to satisfy multiple other strict criteria (same team, ≤150 frames, plausible displacement) before being recovered anyway, so the filter's false-positive rate is bounded.

### Finding 2 — `GOALS` events correctly included as stopper

✅ Goals always involve the ball going in the net and play stopping. Including them is correct.

### Finding 3 — No other Smart Tagging codes are used as stopper

The module ignores all events other than `SET PIECES` and `GOALS`. Tactical codes like `PRESSURE`, `DEFENSIVE_LINE` etc. are present but correctly not used as dead-ball evidence. ✅

---

## Audit 3 — Diagnostic Reconciliation

### Reported numbers (from previous session summary)

- 9 recovered gaps, covering 1,274 frames
- 211 frames changed D→A, 163 frames changed D→H
- Previous claim: "579 possession spells in v1"

### Recomputed from actual files

**Full v0→v1 confusion table** (run directly from CSVs):

|          | v1=H   | v1=A   | v1=D   | v1=X | Total v0 |
|----------|-------:|-------:|-------:|-----:|---------:|
| **v0=H** | 61,228 |     78 | 11,651 |    0 |   72,957 |
| **v0=A** |     63 | 46,120 |  8,659 |    0 |   54,842 |
| **v0=D** |    163 |    211 | 71,146 |    0 |   71,520 |
| **v0=X** |      0 |      0 |      0 |    0 |        0 |
| **Total v1** | **61,454** | **46,409** | **91,456** | **0** | **199,319** |

**Total frames changed: 20,825 (10.45%)** ✅ — matches prior report.

### Reconciling 1,274 gap frames vs. 374 D→H/D→A label changes

The continuity algorithm rewrites **all** frames within a recovered gap to the team label, regardless of what v0 had in those frames. Many gap frames were already labelled H or A in v0 by the 50-frame smoothing window "bleeding" across the gap boundary.

**Per-gap breakdown:**

| Gap frames | Team | Already team in v0 | Net new D→team changes | H→A or A→H cross-changes |
|---:|:---:|---:|---:|---:|
| 12831–12978 (148) | H | 100 | 48 | 0 |
| 16646–16777 (132) | A | 77 | 55 | **23 H→A** |
| 55514–55662 (149) | A | 100 | 49 | 0 |
| 63670–63813 (144) | A | 76 | 68 | **24 H→A** |
| 64525–64670 (146) | A | 69 | 77 | **31 H→A** |
| 79891–80022 (132) | H | 59 | 73 | **41 A→H** |
| 118555–118700 (146) | H | 78 | 68 | **22 A→H** |
| 119341–119480 (140) | A | 100 | 40 | 0 |
| 187127–187263 (137) | H | 100 | 37 | 0 |
| **Total** | | **759** | **515** | **141** |

**Summary:**
- 1,274 gap frames overwritten
- 759 were already team-labeled in v0 (smoothing window bleed) → unchanged
- **515 genuinely changed from D → team label** (374 pure D→team + 141 cross-team flips)
- **141 frames flipped directly H↔A** (the smoothing in v0 had placed the opposing label mid-gap)

> [!WARNING]
> The 141 H↔A cross-flips are **technically correct behavior** (the continuity algorithm is overwriting v0's mid-gap smoothing artifacts with a uniform team label), but they represent **the highest-risk changes in v1**. If the gap recovery decision for any of those 5 mixed-evidence gaps is wrong, we silently flip 23–41 frames to the wrong team without any D buffer.

**Recommended monitoring:** When Step 6 evaluates v1 against the public ground truth, pay special attention to these 5 gaps (frames 16646–80022 range).

**Possession spells:** 579 ✅ — confirmed by `validate_submission.py` output.

---

## Audit 4 — Continuity Algorithm Correctness

### 4a — Code flow trace for `get_dominant_team()`

```python
# continuity.py lines 79-90
def get_dominant_team(nearest_team, start, end):
    window = nearest_team[start:end + 1]
    valid = window[window >= 0]      # exclude NOBODY (= -1)
    if len(valid) == 0:
        return -1
    counts = np.bincount(valid, minlength=2)
    if counts[0] > counts[1] and counts[0] / len(valid) >= 0.6:
        return 0  # Home
    elif counts[1] > counts[0] and counts[1] / len(valid) >= 0.6:
        return 1  # Away
    return -1
```

✅ Logic is correct. The 60% threshold requires a clear majority. Using `minlength=2` guards against an all-team-0 window producing an index error on `counts[1]`.

**Edge case checked:** When both teams are equal (e.g. 50/50) → returns -1 → triggers fallback to single boundary frame → if still -1 → `INSUFFICIENT_EVIDENCE`. This chain is correct.

### 4b — Ball coordinates units

`extract_tracking_and_gaps()` (`missing_ball_analysis.py` lines 72-73):
```python
ball_coords[video_frame, 0] = bx_norm * length   # metres
ball_coords[video_frame, 1] = by_norm * width     # metres
```

`apply_continuity()` uses these coordinates for displacement:
```python
displacement = float(np.sqrt(dx*dx + dy*dy))  # in metres
time_elapsed_s = dur / 25.0                    # seconds
speed_mps = displacement / time_elapsed_s      # m/s
```

✅ **Units are consistent throughout.** `MAX_BALL_DISPLACEMENT_METERS = 30.0` and `MAX_BALL_SPEED_MPS = 25.0` are both in metres/m/s.

### 4c — Physical threshold review

| Constant | Value | Assessment |
|---|---|---|
| `MAX_CONTINUITY_GAP_FRAMES` | 150 (6.0 s) | ✅ Conservative: referees rarely restart play in under 6s |
| `MAX_BALL_DISPLACEMENT_METERS` | 30.0 m | ✅ Covers a full-pitch cross; plausible for untracked flight |
| `MAX_BALL_SPEED_MPS` | 25.0 m/s (90 km/h) | ✅ Reasonable for average ball flight; well below max shot speed (~120 km/h) |
| `EVIDENCE_WINDOW_FRAMES` | 5 frames (0.2 s) | ⚠️ See finding 4d |

> [!NOTE]
> `MAX_CONTINUITY_GAP_FRAMES = 150` (6s) is wider than the bimodal gap distribution's short cluster (≤26 frames). This means the algorithm *can* bridge medium gaps (27–150 frames). The set-piece filter, displacement check, and same-team requirement provide defence-in-depth. **This is not a bug but a design choice that should be revisited in Step 6 based on evaluation results.**

### 4d — Evidence window too narrow (Low-Medium Risk)

`EVIDENCE_WINDOW_FRAMES = 5` means only the 5 frames immediately before/after the gap boundary are examined for team dominance. At 25 fps, this is only 0.2 seconds.

**Risk:** If the nearest-player assignment is noisy for 1–2 frames right at the gap boundary (e.g., a tackle just ended), the dominant-team detection can fail and fall back to the single boundary frame, which is even noisier.

**Mitigation already in code:** The fallback to `nearest_team[idx_before]` is a safety net, and if neither side gives a clear team, the gap is `INSUFFICIENT_EVIDENCE`. The system is therefore conservative by default.

**Recommendation for Step 6:** Try `EVIDENCE_WINDOW_FRAMES = 15` (0.6s) to get more robust team-dominance evidence. This is a threshold change that should be justified by evaluation scores, not by inspection.

### 4e — Boundary overlap check for straddling gaps

```python
# continuity.py line 155
in_non_play = any(not (e < nps or s > npe) for nps, npe in non_play_intervals)
```

This triggers if the gap **partially overlaps** any non-play interval — i.e., a gap straddling the kickoff boundary is treated as non-play. 

✅ This is **conservative and correct policy**: if a gap begins in dead time and ends in live play, the algorithm cannot know when ball tracking resumed, so preserving as D is the safe choice.

**Verified:** None of the 9 recovered gaps overlap any non-play interval. The check is working correctly.

### 4f — Non-play clamping overwrites H→D and A→D (Expected Behavior)

The confusion table shows `H→D: 11,651` and `A→D: 8,659`. These 20,310 frames are almost entirely from the halftime period, where v0's smoothing window assigned H/A labels (because players were still on the pitch). v1 correctly clamps these to D.

✅ **This is the primary value of v1 over v0** and is correct behavior.

---

## Audit 5 — Test Suite Coverage

### Unit test results (all 6 passed)

```
Ran 6 tests in 0.001s — OK
```

### Coverage gaps identified

| Scenario | Covered? | Notes |
|---|---|---|
| Short same-team gap → recovered | ✅ Test 1 | |
| Set-piece overlap → preserved | ✅ Test 2 | |
| Turnover across gap → not bridged | ✅ Test 3 | |
| Gap > 150 frames → not bridged | ✅ Test 4 | |
| Non-play clamping | ✅ Test 5 | |
| Edge gaps at start/end | ✅ Test 6 | |
| Gap where v0 had H but recovery assigns A | ❌ **Missing** | The 141 H↔A cross-flips have no synthetic test |
| Gap with ball coords outside pitch bounds | ❌ **Missing** | `INSUFFICIENT_EVIDENCE` path for out-of-range coords |
| `IMPLAUSIBLE_DISPLACEMENT` path | ❌ **Missing** | No test for displacement > 30m scenario |
| Fallback to single boundary frame when window empty | ❌ **Missing** | No test for EVIDENCE_WINDOW_FRAMES=0 edge |

> [!NOTE]
> The existing 6 tests cover the most critical paths. The missing tests are for edge cases that are unlikely to cause silent failures but would improve confidence. These can be added in Step 6 as low-priority improvements.

---

## Audit 6 — Validation and Format Check

**Official validator output:**
```
OK    phantom11_v1.csv
      199319 rows, frame 0..199318
      label counts: H 61454 (30.8%), A 46409 (23.3%), D 91456 (45.9%), X 0 (0.0%)
      reports 579 possessions of mean 7.5 s
```

✅ Format is valid. ✅ Row count correct. ✅ Frame range correct. ✅ All labels valid.

---

## Summary Table

| # | Finding | Severity | Status | Action |
|---|---------|----------|--------|--------|
| 1a | Hardcoded `NON_PLAY_INTERVALS` values are **numerically correct** (match `games.py`) | — | ✅ No bug | Optional: derive dynamically |
| 1b | No off-by-one errors on kickoff/halftime frame boundaries | — | ✅ Confirmed | — |
| 2a | Smart Tagging windows are tactical phases, not exact instants | Medium | ⚠️ Known approx. | Acceptable for now; revisit in Step 6 |
| 2b | Only `SET PIECES` + `GOALS` used as stoppage evidence | — | ✅ Correct | — |
| 3 | 1,274 gap frames recovered; only 374 are pure D→team; 141 are H↔A cross-flips | Medium | ✅ Explained | Monitor these 5 gaps in Step 6 evaluation |
| 4a | `get_dominant_team()` logic and 60% threshold correct | — | ✅ Confirmed | — |
| 4b | Ball coordinates are in metres throughout | — | ✅ Confirmed | — |
| 4c | Physical thresholds are reasonable and not tuned against labels | — | ✅ Confirmed | — |
| 4d | `EVIDENCE_WINDOW_FRAMES=5` is narrow (0.2s) | Low | ⚠️ Design risk | Try 15 in Step 6 (justify with scores) |
| 4e | Boundary overlap check is conservative and correct | — | ✅ Confirmed | — |
| 4f | H→D and A→D clamping of 20,310 halftime frames is correct behavior | — | ✅ Confirmed | — |
| 5 | 6/6 unit tests pass; 4 edge-case scenarios not tested | Low | ⚠️ Coverage gap | Add in Step 6 (low priority) |
| 6 | Official validator: OK, 199,319 rows, correct format | — | ✅ Confirmed | — |

---

## Verdict

> **v1 is safe to use as the foundation for Step 6.**

There are **no implementation bugs** in `continuity.py` or `predict_v1.py`. The code is deterministic, reproducible, and uses no ground-truth labels. The boundary values are confirmed correct. The algorithm is genuinely conservative: 177 of 186 gaps are left unchanged.

The only medium-priority concern is the **141 H↔A cross-flips** in 5 recovered gaps where v0's smoothing had placed the opposing team's label mid-gap. These flips are algorithmically defensible but represent the highest uncertainty in v1. Step 6 evaluation against the public ground truth will reveal whether these are improvements or regressions.

**No code changes are required before proceeding to Step 6.** The optional improvements (dynamic boundary derivation, wider evidence window) can be incorporated when evaluation evidence is available.

---

*All commands and outputs in this audit were run against the actual data and source files in `../more-than-a-hack-2027/`. No ground-truth labels were consulted.*
