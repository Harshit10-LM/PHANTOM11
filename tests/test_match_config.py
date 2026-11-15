#!/usr/bin/env python3
"""Unit tests for PHANTOM-11 MatchConfig and multi-match generalisation."""

import unittest
import numpy as np

from phantom11.match_config import MatchConfig


class TestMatchConfig(unittest.TestCase):
    """Test suite for MatchConfig configuration, validation and derivation."""

    def test_01_from_official_game_bar_ath(self):
        """Verify MatchConfig derives official BAR-ATH metadata exactly."""
        cfg = MatchConfig.from_official_game("BAR-ATH")
        self.assertEqual(cfg.match_id, "BAR-ATH")
        self.assertEqual(cfg.home_team, "Barcelona")
        self.assertEqual(cfg.away_team, "Athletic Club")
        self.assertEqual(cfg.n_frames, 199319)
        self.assertEqual(cfg.fps, 25.0)
        self.assertEqual(cfg.pitch, (105.3, 68.0))
        self.assertEqual(len(cfg.playing_intervals), 2)
        self.assertEqual(cfg.playing_intervals[0], (11185, 90651))
        self.assertEqual(cfg.playing_intervals[1], (110808, 190197))
        self.assertEqual(cfg.non_play_intervals, [(0, 11184), (90652, 110807), (190198, 199318)])

    def test_02_team_map_variants(self):
        """Verify team mapping correctly resolves home and away variants."""
        cfg = MatchConfig(
            match_id="BAR-ATH",
            home_team="Barcelona",
            away_team="Athletic Club",
        )
        tmap = cfg.get_team_map()
        self.assertEqual(tmap["barcelona"], 0)
        self.assertEqual(tmap["atletic club"], 1)
        self.assertEqual(tmap["athletic club"], 1)
        self.assertEqual(tmap.get("unknown_team", -1), -1)

    def test_03_validation_valid(self):
        """Verify validation passes on well-formed configuration."""
        cfg = MatchConfig(
            match_id="TEST-MATCH",
            home_team="Team A",
            away_team="Team B",
            n_frames=150000,
            fps=25.0,
            pitch=(105.0, 68.0),
        )
        is_valid, issues = cfg.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    def test_04_validation_invalid_frames_and_pitch(self):
        """Verify validation catches negative frames and unphysical pitch."""
        cfg = MatchConfig(
            match_id="BAD-MATCH",
            n_frames=-10,
            fps=0.0,
            pitch=(20.0, 10.0),
        )
        is_valid, issues = cfg.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("frame count" in s.lower() for s in issues))
        self.assertTrue(any("frame rate" in s.lower() for s in issues))
        self.assertTrue(any("pitch" in s.lower() for s in issues))

    def test_05_validation_unsupported_format(self):
        """Verify unsupported data format is flagged clearly."""
        cfg = MatchConfig(
            match_id="UNSUPPORTED-MATCH",
            data_format="PROPRIETARY_BINARY_XYZ",
        )
        is_valid, issues = cfg.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("unsupported data format" in s.lower() for s in issues))


if __name__ == "__main__":
    unittest.main()
