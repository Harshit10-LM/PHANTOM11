#!/usr/bin/env python3
"""PHANTOM-11: Synthetic Unit Tests for Continuity Module

Step 5 of More Than A Hack 2027:
All data generated in this test suite is strictly SYNTHETIC and isolated.
Tests the deterministic continuity rules under 6 specific scenarios:
1. Short same-team missing-ball gap -> Recovered as continuous possession
2. Genuine dead-ball stoppage -> Preserved as Dead Ball (D)
3. Opposing-team controlled turnover -> Preserved (no false bridge across turnover)
4. Long uncertain missing interval -> Preserved as stoppage / ambiguous
5. Kickoff and halftime boundaries -> Strictly clamped to Dead Ball (D)
6. Missing tracking at start/end of match -> Gracefully handled without errors
"""

import unittest
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from phantom11.continuity import apply_continuity
except ImportError:
    from continuity import apply_continuity


class TestConservativeContinuity(unittest.TestCase):
    """Synthetic unit tests for phantom11.continuity module."""

    def setUp(self):
        # Base synthetic match timeline: 500 frames
        self.n = 500
        self.pitch = (105.3, 68.0)
        # Synthetic non-play intervals: 0..50 (pre-match) and 200..250 (halftime)
        self.non_play = [(0, 50), (200, 250), (450, 499)]

    def create_synthetic_base(self):
        """Helper to create baseline synthetic arrays."""
        v0 = np.full(self.n, "D", dtype="<U1")
        ball_present = np.ones(self.n, dtype=bool)
        ball_coords = np.full((self.n, 2), 50.0, dtype=np.float32)
        nearest_team = np.full(self.n, 0, dtype=np.int8)  # Default Home
        return v0, ball_present, ball_coords, nearest_team

    def test_1_short_same_team_missing_ball_gap(self):
        """Scenario 1: Short same-team missing-ball gap (SYNTHETIC DATA).
        A 10-frame gap in active play (frames 100..109) where Team H had ball before and after.
        Ball displaced only 5 meters.
        Expected: Continuity bridges the gap with 'H'.
        """
        v0, ball_present, ball_coords, nearest_team = self.create_synthetic_base()
        # Active play region
        v0[90:120] = "H"
        # 10-frame gap at 100..109
        ball_present[100:110] = False
        ball_coords[100:110] = np.nan
        # Pre-gap position: (50.0, 30.0), Team H (0)
        ball_coords[99] = [50.0, 30.0]
        nearest_team[95:100] = 0
        # Post-gap position: (53.0, 32.0), Team H (0) -> disp = sqrt(3^2 + 2^2) = 3.6m
        ball_coords[110] = [53.0, 32.0]
        nearest_team[110:115] = 0

        # Simulate that baseline v0 had dropped to 'D' inside the gap
        v0[100:110] = "D"

        v1, diag = apply_continuity(
            v0, ball_present, ball_coords, nearest_team,
            set_pieces_intervals=[], pitch=self.pitch, non_play_intervals=self.non_play
        )

        self.assertEqual(diag["decisions"]["RECOVER_CONTINUITY"], 1)
        self.assertTrue(np.all(v1[100:110] == "H"))
        print("  [Pass] Test 1: Short same-team missing-ball gap successfully bridged to H.")

    def test_2_genuine_dead_ball_stoppage(self):
        """Scenario 2: Genuine dead-ball stoppage overlapping set piece (SYNTHETIC DATA).
        A missing gap (frames 120..140) overlaps a Smart Tagging throw-in.
        Expected: Set piece protection prevents bridging; preserved as 'D'.
        """
        v0, ball_present, ball_coords, nearest_team = self.create_synthetic_base()
        v0[120:141] = "D"
        ball_present[120:141] = False
        ball_coords[120:141] = np.nan
        nearest_team[115:120] = 0
        nearest_team[141:146] = 0

        # Throw-in interval at 118..145
        set_pieces = [(118, 145)]

        v1, diag = apply_continuity(
            v0, ball_present, ball_coords, nearest_team,
            set_pieces_intervals=set_pieces, pitch=self.pitch, non_play_intervals=self.non_play
        )

        self.assertEqual(diag["decisions"]["PRESERVE_SET_PIECE"], 1)
        self.assertEqual(diag["decisions"]["RECOVER_CONTINUITY"], 0)
        self.assertTrue(np.all(v1[120:141] == "D"))
        print("  [Pass] Test 2: Set-piece stoppage successfully preserved as Dead Ball.")

    def test_3_opposing_team_controlled_turnover(self):
        """Scenario 3: Opposing-team controlled turnover (SYNTHETIC DATA).
        A 6-frame gap (frames 150..155) where Team H had ball before, but Team A gained it after.
        Expected: Turnover detected; not bridged with either team, baseline preserved.
        """
        v0, ball_present, ball_coords, nearest_team = self.create_synthetic_base()
        ball_present[150:156] = False
        ball_coords[150:156] = np.nan
        # Team H before
        nearest_team[145:150] = 0
        # Team A after
        nearest_team[156:161] = 1

        v0[150:153] = "H"
        v0[153:156] = "A"

        v1, diag = apply_continuity(
            v0, ball_present, ball_coords, nearest_team,
            set_pieces_intervals=[], pitch=self.pitch, non_play_intervals=self.non_play
        )

        self.assertEqual(diag["decisions"]["AMBIGUOUS_TURNOVER"], 1)
        self.assertEqual(diag["decisions"]["RECOVER_CONTINUITY"], 0)
        # Baseline unchanged
        self.assertTrue(np.array_equal(v1[150:156], v0[150:156]))
        print("  [Pass] Test 3: Opposing-team turnover left unmerged (baseline preserved).")

    def test_4_long_uncertain_missing_interval(self):
        """Scenario 4: Long uncertain missing interval (SYNTHETIC DATA).
        A 200-frame gap (frames 260..459) > MAX_CONTINUITY_GAP_FRAMES (150).
        Expected: Exceeds duration limit; not bridged, preserved as stoppage.
        """
        v0, ball_present, ball_coords, nearest_team = self.create_synthetic_base()
        ball_present[260:440] = False
        ball_coords[260:440] = np.nan
        v0[260:440] = "D"
        nearest_team[255:260] = 0
        nearest_team[440:445] = 0

        v1, diag = apply_continuity(
            v0, ball_present, ball_coords, nearest_team,
            set_pieces_intervals=[], pitch=self.pitch, non_play_intervals=self.non_play
        )

        self.assertEqual(diag["decisions"]["EXCEEDS_MAX_DURATION"], 1)
        self.assertEqual(diag["decisions"]["RECOVER_CONTINUITY"], 0)
        self.assertTrue(np.all(v1[260:440] == "D"))
        print("  [Pass] Test 4: Long uncertain gap preserved as stoppage.")

    def test_5_kickoff_and_halftime_boundaries(self):
        """Scenario 5: Kickoff and halftime boundaries (SYNTHETIC DATA).
        Synthetic pre-kickoff (0..50) and halftime (200..250) contain fake H/A labels.
        Expected: Boundary filter clamps all non-play frames to 'D'.
        """
        v0, ball_present, ball_coords, nearest_team = self.create_synthetic_base()
        # Erroneous active labels in halftime
        v0[200:251] = "H"
        # Erroneous active labels in pre-match
        v0[0:51] = "A"

        v1, diag = apply_continuity(
            v0, ball_present, ball_coords, nearest_team,
            set_pieces_intervals=[], pitch=self.pitch, non_play_intervals=self.non_play
        )

        self.assertTrue(np.all(v1[0:51] == "D"))
        self.assertTrue(np.all(v1[200:251] == "D"))
        self.assertGreater(diag["clamped_non_play_frames"], 0)
        print("  [Pass] Test 5: Pre-match and halftime boundaries strictly clamped to D.")

    def test_6_missing_tracking_at_start_and_end(self):
        """Scenario 6: Missing tracking at beginning and end of recording (SYNTHETIC DATA).
        Frames 0..10 and 490..499 have NaN coordinates.
        Expected: Handled safely without index out of bounds, clamped to 'D'.
        """
        v0, ball_present, ball_coords, nearest_team = self.create_synthetic_base()
        ball_present[0:10] = False
        ball_present[490:500] = False
        ball_coords[0:10] = np.nan
        ball_coords[490:500] = np.nan

        # Should execute cleanly without IndexError
        v1, diag = apply_continuity(
            v0, ball_present, ball_coords, nearest_team,
            set_pieces_intervals=[], pitch=self.pitch, non_play_intervals=self.non_play
        )

        self.assertEqual(len(v1), self.n)
        self.assertTrue(np.all(v1[0:10] == "D"))
        self.assertTrue(np.all(v1[490:500] == "D"))
        print("  [Pass] Test 6: Boundary edges with missing tracking handled safely.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SYNTHETIC UNIT TESTS FOR PHANTOM-11 CONTINUITY MODULE")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConservativeContinuity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
