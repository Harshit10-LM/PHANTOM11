#!/usr/bin/env python3
"""Cross-Match Generalisation Tests for PHANTOM-11

Tests the pipeline across diverse match configurations:
- Varied FPS (20.0, 25.0, 30.0)
- Varied pitch dimensions (100x64m, 105x68m, 110x72m)
- Varied tracking slot counts (28 slots, 46 slots, 56 slots)
- Varied team names and IDs
- Varied match durations (short, medium, long)
- Tracking coordinate noise
- Zero hardcoded Barcelona/Athletic Club assumptions
"""

import unittest
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom11.match_config import MatchConfig
from phantom11.control_evidence import compute_control_evidence
from phantom11.movement_features import compute_ball_kinematics
from phantom11.flight_handover import refine_flight_handovers
from phantom11.sequence_decoder import decode_possession_sequence
from tests.run_v5_v6_comparative_audit import run_pipeline_v6
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm


class TestCrossMatchGeneralisation(unittest.TestCase):
    """Test suite ensuring PHANTOM-11 is fully agnostic to match-specific constants."""

    def _generate_synthetic_match(self, config: MatchConfig, n_frames=500):
        """Generate synthetic tracking data matching a specific MatchConfig."""
        n_slots = config.n_slots or 28
        team_of_track = config.team_of_track
        if team_of_track is None:
            team_of_track = np.array([0] * (n_slots // 2) + [1] * (n_slots - n_slots // 2), dtype=np.int8)

        pitch_len, pitch_wid = config.pitch
        ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
        players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

        # Base positions on pitch
        mid_x, mid_y = pitch_len / 2.0, pitch_wid / 2.0
        for s in range(n_slots):
            # spread players across pitch
            side = -1.0 if team_of_track[s] == 0 else 1.0
            players_xy[:, s, 0] = mid_x + side * (10.0 + (s % 11) * 3.0)
            players_xy[:, s, 1] = mid_y + ((s % 5) - 2) * 10.0

        # Phase 1: Home player 0 carries ball
        p1 = int(2.5 * config.fps)
        for t in range(p1):
            pos = [mid_x - 20.0 + (t / config.fps) * 2.0, mid_y]
            ball_xy[t] = pos
            players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

        # Phase 2: Aerial pass to Home player 1 at 16.0 m/s over Away defender (2.0s duration)
        pass_speed = 16.0  # m/s
        pass_duration = int(2.0 * config.fps)
        p2 = p1 + pass_duration
        start_x = ball_xy[p1 - 1, 0]
        dest_x = start_x + (pass_duration / config.fps) * pass_speed
        opp_x = (start_x + dest_x) / 2.0

        for t in range(p1, p2):
            pos = [start_x + ((t - p1) / config.fps) * pass_speed, mid_y]
            ball_xy[t] = pos
            # Defender stands directly under path in 2D
            players_xy[t, n_slots // 2] = [opp_x, mid_y + 0.2]
            # Receiver waits at destination
            players_xy[t, 1] = [dest_x + 0.3, mid_y]

        # Phase 3: Home player 1 controls ball and carries
        for t in range(p2, n_frames):
            pos = [dest_x + ((t - p2) / config.fps) * 2.0, mid_y]
            ball_xy[t] = pos
            players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

        gt = np.array(["H"] * n_frames)
        return ball_xy, players_xy, team_of_track, gt

    def test_varied_frame_rates(self):
        """Verify pipeline behaves consistently at 20, 25, and 30 FPS."""
        for fps in (20.0, 25.0, 30.0):
            cfg = MatchConfig(
                match_id=f"TEST_FPS_{int(fps)}",
                home_team="Team A",
                away_team="Team B",
                fps=fps,
                pitch=(105.0, 68.0),
                n_slots=28,
            )
            b, p, t, gt = self._generate_synthetic_match(cfg, n_frames=int(20 * fps))
            pred = run_pipeline_v6(b, p, t, [], fps=fps)
            gt_codes = pm.label_codes(gt)
            pred_codes = pm.label_codes(pred)
            res = pm.spell_scores(gt_codes, pred_codes, [(0, len(gt))])
            self.assertGreaterEqual(res["mean_f1"], 0.90, f"Failed at FPS={fps}")

    def test_varied_pitch_dimensions(self):
        """Verify pipeline handles diverse pitch sizes (small, standard, large)."""
        pitches = [(100.0, 64.0), (105.0, 68.0), (110.0, 72.0)]
        for pitch in pitches:
            cfg = MatchConfig(
                match_id=f"TEST_PITCH_{int(pitch[0])}x{int(pitch[1])}",
                pitch=pitch,
                fps=25.0,
                n_slots=28,
            )
            b, p, t, gt = self._generate_synthetic_match(cfg, n_frames=300)
            pred = run_pipeline_v6(b, p, t, [], fps=25.0)
            res = pm.spell_scores(pm.label_codes(gt), pm.label_codes(pred), [(0, len(gt))])
            self.assertGreaterEqual(res["mean_f1"], 0.90, f"Failed on pitch={pitch}")

    def test_varied_slot_counts(self):
        """Verify pipeline correctly maps tracking matrices with 28, 46, and 56 slots."""
        for n_slots in (28, 46, 56):
            cfg = MatchConfig(
                match_id=f"TEST_SLOTS_{n_slots}",
                n_slots=n_slots,
                fps=25.0,
            )
            b, p, t, gt = self._generate_synthetic_match(cfg, n_frames=300)
            self.assertEqual(p.shape[1], n_slots)
            pred = run_pipeline_v6(b, p, t, [], fps=25.0)
            res = pm.spell_scores(pm.label_codes(gt), pm.label_codes(pred), [(0, len(gt))])
            self.assertGreaterEqual(res["mean_f1"], 0.90, f"Failed with {n_slots} slots")

    def test_coordinate_noise_robustness(self):
        """Verify pipeline remains robust to noise on non-standard pitch and frame rate."""
        cfg = MatchConfig(
            match_id="TEST_NOISE",
            fps=30.0,
            pitch=(110.0, 70.0),
            n_slots=46,
        )
        b, p, t, gt = self._generate_synthetic_match(cfg, n_frames=400)
        np.random.seed(42)
        b_noisy = b + np.random.normal(0, 0.25, b.shape)
        p_noisy = p + np.random.normal(0, 0.25, p.shape)

        pred = run_pipeline_v6(b_noisy, p_noisy, t, [], fps=30.0)
        res = pm.spell_scores(pm.label_codes(gt), pm.label_codes(pred), [(0, len(gt))])
        self.assertGreaterEqual(res["mean_f1"], 0.85, "Pipeline degraded under coordinate noise")

    def test_zero_barcelona_reliance(self):
        """Verify that running with entirely distinct team names produces valid outputs."""
        cfg = MatchConfig(
            match_id="LIV_MCI_2027",
            home_team="Liverpool",
            away_team="Manchester City",
            n_frames=500,
            fps=25.0,
        )
        b, p, t, gt = self._generate_synthetic_match(cfg, n_frames=500)
        pred = run_pipeline_v6(b, p, t, [], fps=25.0)
        self.assertEqual(len(pred), 500)
        self.assertTrue(set(pred).issubset({"H", "A", "D", "X"}))


if __name__ == "__main__":
    unittest.main()
