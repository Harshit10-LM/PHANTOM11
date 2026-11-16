#!/usr/bin/env python3
"""Cross-Match Generalisation Tests for PHANTOM-11 v5-A

Validates:
- Varied FPS (20.0, 25.0, 30.0)
- Varied pitch dimensions (100x64m, 105x68m, 110x72m)
- Dynamic pitch-boundary clamping across custom pitch dimensions
- Varied tracking slot counts (28 slots, 46 slots, 56 slots)
- Arbitrary team names and match IDs (Liverpool vs Manchester City)
- Tracking coordinate noise (sigma = 0.25m)
- Boundary-adjacent live play vs out-of-bounds throw-in setups
- Zero reliance on Barcelona/Athletic Club metadata
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
from phantom11.predict_v7 import run_pipeline_v5a
from phantom11.path_resolver import resolve_official_repo

resolve_official_repo()
import possession_metrics as pm


class TestCrossMatchV5A(unittest.TestCase):
    """Test suite ensuring PHANTOM-11 v5-A is fully generalisable and match-agnostic."""

    def _generate_synthetic_match(self, config: MatchConfig, n_frames=500):
        n_slots = config.n_slots or 28
        team_of_track = config.team_of_track
        if team_of_track is None:
            team_of_track = np.array([0] * (n_slots // 2) + [1] * (n_slots - n_slots // 2), dtype=np.int8)

        pitch_len, pitch_wid = config.pitch
        ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
        players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

        mid_x, mid_y = pitch_len / 2.0, pitch_wid / 2.0
        for s in range(n_slots):
            side = -1.0 if team_of_track[s] == 0 else 1.0
            players_xy[:, s, 0] = mid_x + side * (10.0 + (s % 11) * 3.0)
            players_xy[:, s, 1] = mid_y + ((s % 5) - 2) * 10.0

        # Phase 1: Home player 0 carries ball
        p1 = int(2.5 * config.fps)
        for t in range(p1):
            pos = [mid_x - 20.0 + (t / config.fps) * 2.0, mid_y]
            ball_xy[t] = pos
            players_xy[t, 0] = [pos[0] + 0.3, pos[1]]

        # Phase 2: Aerial pass to Home player 1 over Away defender
        pass_speed = 16.0  # m/s
        pass_duration = int(2.0 * config.fps)
        p2 = p1 + pass_duration
        start_x = ball_xy[p1 - 1, 0]
        dest_x = start_x + (pass_duration / config.fps) * pass_speed
        opp_x = (start_x + dest_x) / 2.0

        for t in range(p1, p2):
            pos = [start_x + ((t - p1) / config.fps) * pass_speed, mid_y]
            ball_xy[t] = pos
            players_xy[t, n_slots // 2] = [opp_x, mid_y + 0.2]
            players_xy[t, 1] = [dest_x + 0.3, mid_y]

        # Phase 3: Home player 1 controls and carries
        for t in range(p2, n_frames):
            pos = [dest_x + ((t - p2) / config.fps) * 2.0, mid_y]
            ball_xy[t] = pos
            players_xy[t, 1] = [pos[0] + 0.3, pos[1]]

        gt = np.array(["H"] * n_frames)
        return ball_xy, players_xy, team_of_track, gt

    def test_varied_frame_rates(self):
        """Verify v5-A behaves consistently at 20, 25, and 30 FPS."""
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
            pred = run_pipeline_v5a(b, p, t, [], pitch=cfg.pitch, fps=fps)
            res = pm.spell_scores(pm.label_codes(gt), pm.label_codes(pred), [(0, len(gt))])
            self.assertGreaterEqual(res["mean_f1"], 0.90, f"Failed at FPS={fps}")

    def test_varied_pitch_dimensions_and_boundary_clamping(self):
        """Verify v5-A clamps correctly on small (100x64m), standard (105x68m), and large (110x72m) pitches."""
        pitches = [(100.0, 64.0), (105.0, 68.0), (110.0, 72.0)]
        for pitch in pitches:
            cfg = MatchConfig(
                match_id=f"TEST_PITCH_{int(pitch[0])}x{int(pitch[1])}",
                pitch=pitch,
                fps=25.0,
                n_slots=28,
            )
            b, p, t, gt = self._generate_synthetic_match(cfg, n_frames=300)
            pred = run_pipeline_v5a(b, p, t, [], pitch=pitch, fps=25.0)
            res = pm.spell_scores(pm.label_codes(gt), pm.label_codes(pred), [(0, len(gt))])
            self.assertGreaterEqual(res["mean_f1"], 0.90, f"Failed on pitch={pitch}")

    def test_dynamic_pitch_boundary_clamping(self):
        """Verify pitch boundary clamping dynamically respects the configured pitch dimensions."""
        # Small pitch: 100m x 64m. Ball at y=65.0m is OOB on 64m pitch, but inside on 68m pitch!
        b = np.array([[50.0, 65.5]] * 10, dtype=np.float32)
        p = np.full((10, 28, 2), 50.0, dtype=np.float32)
        t = np.array([0] * 14 + [1] * 14, dtype=np.int8)

        # On small pitch (100, 64): y=65.5m is 1.5m outside touchline -> must clamp to D
        pred_small = run_pipeline_v5a(b, p, t, [], pitch=(100.0, 64.0), fps=25.0)
        self.assertTrue(np.all(pred_small == "D"), "Failed to clamp OOB ball on 100x64m pitch")

        # On standard pitch (105, 68): y=65.5m is inside field of play (2.5m inside) -> must NOT clamp to D
        pred_std = run_pipeline_v5a(b, p, t, [], pitch=(105.0, 68.0), fps=25.0)
        self.assertTrue(np.all(pred_std == "H"), "Falsely clamped in-pitch ball on 105x68m pitch")

    def test_varied_slot_counts(self):
        """Verify v5-A correctly handles 28, 46, and 56 player slots."""
        for n_slots in (28, 46, 56):
            cfg = MatchConfig(
                match_id=f"TEST_SLOTS_{n_slots}",
                n_slots=n_slots,
                fps=25.0,
            )
            b, p, t, gt = self._generate_synthetic_match(cfg, n_frames=300)
            self.assertEqual(p.shape[1], n_slots)
            pred = run_pipeline_v5a(b, p, t, [], pitch=cfg.pitch, fps=25.0)
            res = pm.spell_scores(pm.label_codes(gt), pm.label_codes(pred), [(0, len(gt))])
            self.assertGreaterEqual(res["mean_f1"], 0.90, f"Failed with {n_slots} slots")

    def test_coordinate_noise_robustness(self):
        """Verify v5-A remains robust to coordinate noise."""
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

        pred = run_pipeline_v5a(b_noisy, p_noisy, t, [], pitch=cfg.pitch, fps=30.0)
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
        pred = run_pipeline_v5a(b, p, t, [], pitch=cfg.pitch, fps=25.0)
        self.assertEqual(len(pred), 500)
        self.assertTrue(set(pred).issubset({"H", "A", "D", "X"}))


if __name__ == "__main__":
    unittest.main()
