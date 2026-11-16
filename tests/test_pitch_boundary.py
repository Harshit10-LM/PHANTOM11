#!/usr/bin/env python3
"""Unit tests for phantom11.pitch_boundary module (IFAB Law 9 Out-of-Bounds Detection)."""

import unittest
import numpy as np
from phantom11.pitch_boundary import (
    compute_out_of_bounds_distance,
    detect_out_of_bounds_mask,
    apply_pitch_boundary_clamping,
)


class TestPitchBoundary(unittest.TestCase):
    def setUp(self):
        # Tracking coordinates are [0, 105.3] in x and [0, 68.0] in y
        self.pitch = (105.3, 68.0)
        self.length = 105.3
        self.width = 68.0

    def test_ball_deep_inside_pitch(self):
        ball_xy = np.array([[52.65, 34.0], [10.0, 10.0], [90.0, 50.0]], dtype=np.float32)
        ball_present = np.ones(3, dtype=bool)
        d_oob = compute_out_of_bounds_distance(ball_xy, ball_present, pitch=self.pitch)
        self.assertTrue(np.all(d_oob == 0.0))

    def test_ball_on_line_within_margin(self):
        # Line boundary is at y = 68.0m and x = 105.3m. Ball at y = 68.15m is within 0.35m buffer
        ball_xy = np.array([[50.0, 68.15], [105.45, 30.0]], dtype=np.float32)
        ball_present = np.ones(2, dtype=bool)
        d_oob = compute_out_of_bounds_distance(ball_xy, ball_present, pitch=self.pitch)
        self.assertAlmostEqual(d_oob[0], 0.15, places=2)
        self.assertAlmostEqual(d_oob[1], 0.15, places=2)

        # With default margin=0.35m, these should NOT trigger out-of-bounds mask
        mask = detect_out_of_bounds_mask(ball_xy, ball_present, pitch=self.pitch, margin=0.35)
        self.assertFalse(np.any(mask))

    def test_ball_clearly_out_of_bounds(self):
        # Ball at y = 69.5m (1.5m outside touchline)
        ball_xy = np.array([
            [50.0, 69.5],
            [50.0, 69.6],
            [50.0, 69.7],
        ], dtype=np.float32)
        ball_present = np.ones(3, dtype=bool)
        d_oob = compute_out_of_bounds_distance(ball_xy, ball_present, pitch=self.pitch)
        self.assertTrue(np.all(d_oob >= 1.5))

        mask = detect_out_of_bounds_mask(ball_xy, ball_present, pitch=self.pitch, margin=0.35, min_consecutive_frames=2)
        self.assertTrue(np.all(mask))

    def test_missing_ball_nan(self):
        ball_xy = np.array([[np.nan, np.nan], [120.0, 34.0]], dtype=np.float32)
        ball_present = np.array([False, True], dtype=bool)
        d_oob = compute_out_of_bounds_distance(ball_xy, ball_present, pitch=self.pitch)
        self.assertEqual(d_oob[0], 0.0)
        self.assertAlmostEqual(d_oob[1], 14.7, places=1)

    def test_clamping_to_dead_ball(self):
        # Frames: 0=inside, 1=inside, 2=outside (69.5m), 3=outside (69.5m), 4=inside
        ball_xy = np.array([
            [50.0, 34.0],
            [50.0, 34.0],
            [50.0, 69.5],  # OOB (>0.8m deep)
            [50.0, 69.5],  # OOB
            [50.0, 34.0],
        ], dtype=np.float32)
        ball_present = np.ones(5, dtype=bool)
        labels = np.array(["H", "H", "H", "H", "H"], dtype="<U1")
        
        clamped = apply_pitch_boundary_clamping(
            labels=labels,
            ball_xy=ball_xy,
            ball_present=ball_present,
            pitch=self.pitch,
            margin=0.35,
        )
        expected = np.array(["H", "H", "D", "D", "H"], dtype="<U1")
        np.testing.assert_array_equal(clamped, expected)

    def test_isolated_single_frame_shallow_spike_ignored(self):
        # 1 isolated frame OOB between 0.35m and 0.8m (e.g. y = 68.5m, d_oob = 0.5m)
        ball_xy = np.array([
            [50.0, 34.0],
            [50.0, 68.5],  # isolated shallow OOB (0.5m)
            [50.0, 34.0],
        ], dtype=np.float32)
        ball_present = np.ones(3, dtype=bool)
        labels = np.array(["H", "H", "H"], dtype="<U1")
        
        clamped = apply_pitch_boundary_clamping(
            labels=labels,
            ball_xy=ball_xy,
            ball_present=ball_present,
            pitch=self.pitch,
            margin=0.35,
        )
        # Should remain H because single shallow frame is filtered
        np.testing.assert_array_equal(clamped, labels)


if __name__ == "__main__":
    unittest.main()
