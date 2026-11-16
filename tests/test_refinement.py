#!/usr/bin/env python3
"""Synthetic Unit & Robustness Test Suite for PHANTOM-11 Boundary Refinement (v4)

Step 9 of More Than A Hack 2027:
- Tests confidence-gated boundary refinement on synthetic scenarios:
  1. Brief opponent deflection (unifies into continuous possession).
  2. Successful opponent interception (sustained carrier, preserved intact).
  3. Same-team completed pass (flying past opponent, preserved).
  4. Long missing-ball interval during continued play (bridged safely).
  5. Genuine throw-in stoppage (preserved as Dead Ball D).
  6. Restart followed by new possession (snapped to kick).
  7. Two genuine quick turnovers (both preserved).
  8. Noisy player tracking identity changes / slot jumps (handled robustly).
  9. Second-half pre-kickoff / halftime dead-ball interval (strictly clamped to D).
  10. Robustness to simulated tracking dropouts.
  11. Robustness to Gaussian coordinate noise.
- Strictly synthetic in-memory data. Zero hand-labels from the competition match.
"""

import unittest
import numpy as np

# We will test against the refinement logic
from phantom11.boundary_refinement import refine_possession_boundaries


class TestBoundaryRefinement(unittest.TestCase):
    """Synthetic test suite for confidence-gated possession boundary refinement."""

    def setUp(self):
        self.fps = 25.0
        self.pitch = (105.3, 68.0)

    def test_01_brief_opponent_deflection_unified(self):
        """Scenario 1: 15-frame opponent deflection without carrier is unified into H."""
        # 100 frames total: 40 frames H, 15 frames A (no carrier), 45 frames H
        n = 100
        labels = np.array(["H"] * 40 + ["A"] * 15 + ["H"] * 45)
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[:40] = True
        is_carrier_h[55:] = True
        # No carrier for A during 40..54 (just a deflection touch)

        ball_present = np.ones(n, dtype=bool)
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        d_h1 = np.full(n, 1.0, dtype=np.float32)
        d_a1 = np.full(n, 3.0, dtype=np.float32)
        d_a1[40:55] = 1.2  # momentarily close

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            fps=self.fps,
        )

        # The deflection should be unified into H
        self.assertTrue(np.all(refined == "H"))

    def test_02_successful_interception_preserved(self):
        """Scenario 2: Genuine short interception with active carrier (20 frames) is PRESERVED."""
        # 40 frames H, 20 frames A (with active carrier), 40 frames H
        n = 100
        labels = np.array(["H"] * 40 + ["A"] * 20 + ["H"] * 40)
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[:40] = True
        is_carrier_a[40:60] = True  # genuine carrier control
        is_carrier_h[60:] = True

        ball_present = np.ones(n, dtype=bool)
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        d_h1 = np.full(n, 3.0, dtype=np.float32)
        d_a1 = np.full(n, 3.0, dtype=np.float32)
        d_h1[:40] = 0.8
        d_a1[40:60] = 0.8
        d_h1[60:] = 0.8

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            fps=self.fps,
        )

        # The 20 frames of A must NOT be erased
        self.assertEqual(list(refined[40:60]), ["A"] * 20)

    def test_03_same_team_completed_pass_preserved(self):
        """Scenario 3: Pass flying past opponent without opponent carrier remains H."""
        n = 80
        labels = np.array(["H"] * n)
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[:20] = True
        is_carrier_h[60:] = True

        ball_present = np.ones(n, dtype=bool)
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        d_h1 = np.full(n, 2.0, dtype=np.float32)
        d_a1 = np.full(n, 1.5, dtype=np.float32)  # opponent near in flight

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            fps=self.fps,
        )

        self.assertTrue(np.all(refined == "H"))

    def test_04_short_dead_ball_dropout_bridged(self):
        """Scenario 4: Brief 15-frame tracking dropout inside H is bridged to H."""
        n = 100
        labels = np.array(["H"] * 40 + ["D"] * 15 + ["H"] * 45)
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[:40] = True
        is_carrier_h[55:] = True

        ball_present = np.ones(n, dtype=bool)
        ball_present[40:55] = False  # ball missing
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        ball_xy[40:55] = np.nan
        d_h1 = np.full(n, 1.0, dtype=np.float32)
        d_a1 = np.full(n, 4.0, dtype=np.float32)

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            stoppage_support=np.zeros(n, dtype=np.float32),
            fps=self.fps,
        )

        self.assertTrue(np.all(refined == "H"))

    def test_05_genuine_throw_in_stoppage_preserved(self):
        """Scenario 5: 15-second genuine stoppage with set piece tag remains D."""
        n = 375  # 15s at 25 fps
        labels = np.array(["D"] * n)
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        ball_present = np.zeros(n, dtype=bool)
        ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
        d_h1 = np.full(n, 5.0, dtype=np.float32)
        d_a1 = np.full(n, 5.0, dtype=np.float32)
        stoppage_support = np.full(n, 0.8, dtype=np.float32)

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            stoppage_support=stoppage_support,
            fps=self.fps,
        )

        self.assertTrue(np.all(refined == "D"))

    def test_06_restart_snapping_to_kick(self):
        """Scenario 6: D -> H restart with player walking up (stationary ball) is snapped to kick."""
        n = 50
        # Frames 0..29 are D (player walking up, ball stationary at 2.8m).
        # At frame 30, player arrives (1.0m) and kicks.
        labels = np.array(["D"] * 20 + ["H"] * 30)  # v3 prematurely labeled H at frame 20
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[30:] = True

        ball_present = np.ones(n, dtype=bool)
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        d_h1 = np.full(n, 3.0, dtype=np.float32)
        d_h1[30:] = 0.8
        d_a1 = np.full(n, 8.0, dtype=np.float32)

        ball_speed = np.zeros(n, dtype=np.float32)
        ball_speed[30:] = 4.0

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            ball_speed=ball_speed,
            non_play_intervals=[],
            fps=self.fps,
        )

        # Before frame 30, player is 3.0m away and ball stationary -> should be D
        self.assertEqual(list(refined[:30]), ["D"] * 30)
        # From frame 30 onwards, player controls -> H
        self.assertEqual(list(refined[30:]), ["H"] * 20)

    def test_07_two_genuine_quick_turnovers_preserved(self):
        """Scenario 7: H (2s) -> A (1.5s) -> H (2s) both with confirmed carriers are preserved."""
        # 50 frames H, 37 frames A, 50 frames H
        n = 137
        labels = np.array(["H"] * 50 + ["A"] * 37 + ["H"] * 50)
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[:50] = True
        is_carrier_a[50:87] = True
        is_carrier_h[87:] = True

        ball_present = np.ones(n, dtype=bool)
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        d_h1 = np.full(n, 1.0, dtype=np.float32)
        d_h1[50:87] = 4.0
        d_a1 = np.full(n, 4.0, dtype=np.float32)
        d_a1[50:87] = 1.0

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            fps=self.fps,
        )

        self.assertEqual(list(refined[:50]), ["H"] * 50)
        self.assertEqual(list(refined[50:87]), ["A"] * 37)
        self.assertEqual(list(refined[87:]), ["H"] * 50)

    def test_08_noisy_tracker_jump_handled(self):
        """Scenario 8: Slot position jump does not trigger an artificial turnover."""
        n = 60
        labels = np.array(["H"] * n)
        is_carrier_h = np.ones(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[30] = False  # 1-frame glitch

        ball_present = np.ones(n, dtype=bool)
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        d_h1 = np.full(n, 0.8, dtype=np.float32)
        d_a1 = np.full(n, 4.0, dtype=np.float32)

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            fps=self.fps,
        )

        self.assertTrue(np.all(refined == "H"))

    def test_09_halftime_strictly_clamped_to_d(self):
        """Scenario 9: Halftime interval is strictly protected as D."""
        n = 100
        labels = np.array(["H"] * 30 + ["H"] * 40 + ["A"] * 30)
        non_play = [(30, 69)]  # 40 frames of halftime
        is_carrier_h = np.ones(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)

        ball_present = np.ones(n, dtype=bool)
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        d_h1 = np.full(n, 0.5, dtype=np.float32)
        d_a1 = np.full(n, 5.0, dtype=np.float32)

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=non_play,
            fps=self.fps,
        )

        self.assertEqual(list(refined[30:70]), ["D"] * 40)

    def test_10_robustness_to_artificial_dropouts(self):
        """Scenario 10: 5% random single-frame tracking dropouts do not fragment possessions."""
        np.random.seed(42)
        n = 200
        labels = np.array(["H"] * 100 + ["A"] * 100)
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[:100] = True
        is_carrier_a[100:] = True

        # Drop 5% of frames
        dropouts = np.random.rand(n) < 0.05
        is_carrier_h[dropouts] = False
        is_carrier_a[dropouts] = False

        ball_present = ~dropouts
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        d_h1 = np.full(n, 1.0, dtype=np.float32)
        d_h1[100:] = 4.0
        d_a1 = np.full(n, 4.0, dtype=np.float32)
        d_a1[100:] = 1.0

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            fps=self.fps,
        )

        # Still exactly two spells: H (100) and A (100)
        self.assertEqual(list(refined[:100]), ["H"] * 100)
        self.assertEqual(list(refined[100:]), ["A"] * 100)

    def test_11_robustness_to_coordinate_noise(self):
        """Scenario 11: Gaussian noise (+/- 0.3m) on player distances maintains stability."""
        np.random.seed(42)
        n = 150
        labels = np.array(["H"] * 75 + ["A"] * 75)
        is_carrier_h = np.zeros(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        is_carrier_h[:75] = True
        is_carrier_a[75:] = True

        ball_present = np.ones(n, dtype=bool)
        ball_xy = np.full((n, 2), 50.0, dtype=np.float32)
        noise = np.random.normal(0.0, 0.15, n).astype(np.float32)
        d_h1 = np.full(n, 1.0, dtype=np.float32) + noise
        d_h1[75:] += 3.0
        d_a1 = np.full(n, 4.0, dtype=np.float32) + noise
        d_a1[75:] -= 3.0

        refined = refine_possession_boundaries(
            labels=labels,
            ball_present=ball_present,
            ball_xy=ball_xy,
            is_carrier_h=is_carrier_h,
            is_carrier_a=is_carrier_a,
            d_h1=d_h1,
            d_a1=d_a1,
            non_play_intervals=[],
            fps=self.fps,
        )

        self.assertEqual(list(refined[:75]), ["H"] * 75)
        self.assertEqual(list(refined[75:]), ["A"] * 75)


if __name__ == "__main__":
    unittest.main()
