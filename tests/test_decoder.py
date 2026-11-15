#!/usr/bin/env python3
"""PHANTOM-11: Synthetic Unit Tests for Sequence Decoder

Step 7 of More Than A Hack 2027:
All data generated in this suite is strictly SYNTHETIC and isolated.
Tests the actual production sequence decoder across 12 specific scenarios:
 1. Long uninterrupted Barcelona possession.
 2. Long uninterrupted Athletic possession.
 3. A genuine short interception lasting less than two seconds (1.2s / 30 frames).
 4. One-frame false team switch (H H H A H H H -> unified to H).
 5. Multiple noisy H/A switches during a single continuous possession.
 6. A genuine dead-ball stoppage.
 7. A restart followed by the same team's possession.
 8. Missing-ball intervals with reliable evidence on both sides.
 9. A long missing-ball interval with uncertain ownership.
10. Contested loose ball with low-confidence observations.
11. Prematch and halftime boundaries strictly enforced as D.
12. A genuine rapid series of turnovers (2s each) preserved as distinct spells.
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
    from phantom11.sequence_decoder import decode_possession_sequence, DEFAULT_CONFIG
except ImportError:
    from sequence_decoder import decode_possession_sequence, DEFAULT_CONFIG


class TestSequenceDecoder(unittest.TestCase):
    """Synthetic unit tests for PHANTOM-11 Step 7 sequence decoder."""

    def setUp(self):
        self.fps = 25.0
        self.config = DEFAULT_CONFIG.copy()

    def create_synthetic_arrays(self, n):
        v2_labels = np.full(n, "H", dtype="<U1")
        score_h = np.ones(n, dtype=np.float32)
        score_a = np.zeros(n, dtype=np.float32)
        reliability = np.ones(n, dtype=np.float32)
        ball_present = np.ones(n, dtype=bool)
        is_carrier_h = np.ones(n, dtype=bool)
        is_carrier_a = np.zeros(n, dtype=bool)
        stoppage_support = np.zeros(n, dtype=np.float32)
        turnover_readiness_ha = np.full(n, 0.10, dtype=np.float32)
        turnover_readiness_ah = np.full(n, 0.10, dtype=np.float32)
        stoppage_readiness = np.full(n, 0.10, dtype=np.float32)
        restart_readiness = np.full(n, 0.10, dtype=np.float32)
        in_non_play = np.zeros(n, dtype=bool)

        return (
            v2_labels, score_h, score_a, reliability, ball_present,
            is_carrier_h, is_carrier_a, stoppage_support,
            turnover_readiness_ha, turnover_readiness_ah,
            stoppage_readiness, restart_readiness, in_non_play
        )

    def test_1_long_barcelona_possession(self):
        """Scenario 1: Long uninterrupted Barcelona possession (100 frames)."""
        n = 100
        args = self.create_synthetic_arrays(n)
        v3 = decode_possession_sequence(*args, config=self.config)
        self.assertTrue(np.all(v3 == "H"))
        print("  [Pass] Test 1: Long uninterrupted Barcelona possession decoded cleanly.")

    def test_2_long_athletic_possession(self):
        """Scenario 2: Long uninterrupted Athletic possession (100 frames)."""
        n = 100
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        v2[:] = "A"
        sh[:] = 0.0
        sa[:] = 1.0
        ch[:] = False
        ca[:] = True
        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3 == "A"))
        print("  [Pass] Test 2: Long uninterrupted Athletic possession decoded cleanly.")

    def test_3_genuine_short_interception(self):
        """Scenario 3: Genuine short interception lasting 1.2s (30 frames).
        Athletic intercepts between frame 40 and 70 (30 frames), carrying the ball.
        Expected: A distinct 30-frame possession spell of 'A' is preserved.
        """
        n = 100
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        # Frames 40..69: Athletic intercepts
        v2[40:70] = "A"
        sh[40:70] = 0.0
        sa[40:70] = 1.0
        ch[40:70] = False
        ca[40:70] = True
        tha[40] = 0.90  # physical turnover readiness
        tah[70] = 0.90

        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3[:40] == "H"))
        self.assertTrue(np.all(v3[40:70] == "A"))
        self.assertTrue(np.all(v3[70:] == "H"))
        print("  [Pass] Test 3: Genuine 1.2s short interception correctly preserved.")

    def test_4_one_frame_false_switch(self):
        """Scenario 4: One-frame false team switch (frame 50 is 'A' in v2).
        Expected: Unified to 'H' (micro-island eliminated).
        """
        n = 100
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        v2[50] = "A"
        ca[50] = True
        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3 == "H"))
        print("  [Pass] Test 4: One-frame false team switch successfully unified.")

    def test_5_multiple_noisy_switches(self):
        """Scenario 5: Multiple noisy switches (3-frame flickers: H H A A A H H A A H H).
        Expected: Filtered out, decoded as continuous 'H'.
        """
        n = 50
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        v2[15:18] = "A"
        v2[25:27] = "A"
        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3 == "H"))
        print("  [Pass] Test 5: Noisy micro-flickers unified into coherent possession.")

    def test_6_genuine_dead_ball_stoppage(self):
        """Scenario 6: Genuine dead-ball stoppage (frames 40..60, ball missing, set-piece tag).
        Expected: Decoded as 'D'.
        """
        n = 100
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        v2[40:60] = "D"
        bp[40:60] = False
        ch[40:60] = False
        st[40:60] = 0.60
        sr[40] = 0.80
        rr[60] = 0.80

        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3[:40] == "H"))
        self.assertTrue(np.all(v3[40:60] == "D"))
        self.assertTrue(np.all(v3[60:] == "H"))
        print("  [Pass] Test 6: Genuine dead-ball stoppage correctly decoded as D.")

    def test_7_restart_same_team(self):
        """Scenario 7: Dead-ball restart followed by same team's possession.
        D runs frames 10..30, then Home resumes.
        Expected: D runs 10..30, then H resumes cleanly.
        """
        n = 60
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        v2[10:30] = "D"
        bp[10:30] = False
        ch[10:30] = False
        rr[30] = 0.90
        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3[10:30] == "D"))
        self.assertTrue(np.all(v3[30:] == "H"))
        print("  [Pass] Test 7: Restart followed by same team possession preserved.")

    def test_8_missing_ball_reliable_continuity(self):
        """Scenario 8: Short missing-ball interval with continuous H control on both sides.
        Expected: Live play preserved across dropout.
        """
        n = 60
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        bp[20:30] = False
        ch[20:30] = False
        # v2 is already H (bridged by continuity)
        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3 == "H"))
        print("  [Pass] Test 8: Short missing-ball tracking dropout preserved as continuous H.")

    def test_9_long_missing_ball_uncertainty(self):
        """Scenario 9: Long missing-ball interval (frames 20..80) with v2 as D.
        Expected: Preserved as D.
        """
        n = 100
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        v2[20:80] = "D"
        bp[20:80] = False
        ch[20:80] = False
        sh[20:80] = 0.0
        rel[20:80] = 0.05
        sr[20] = 0.70
        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3[20:80] == "D"))
        print("  [Pass] Test 9: Long uncertain missing-ball interval preserved as Dead Ball.")

    def test_10_contested_loose_ball(self):
        """Scenario 10: Contested loose ball with low-confidence observations.
        Expected: Model does not jump randomly between teams.
        """
        n = 50
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        sh[10:30] = 0.30
        sa[10:30] = 0.30
        ch[10:30] = False
        ca[10:30] = False
        rel[10:30] = 0.25
        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        # Should stay stable as H (previous state) without thrashing
        self.assertTrue(np.all(v3[10:30] == "H"))
        print("  [Pass] Test 10: Contested loose ball remains stable without rapid switching.")

    def test_11_prematch_and_halftime(self):
        """Scenario 11: Match boundaries (pre-match 0..10 and halftime 40..60).
        Even if erroneous carrier signals exist, non-play must strictly be 'D'.
        """
        n = 80
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        inp[0:10] = True
        inp[40:60] = True
        # Try to inject fake carrier in halftime
        ch[45:50] = True
        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3[0:10] == "D"))
        self.assertTrue(np.all(v3[40:60] == "D"))
        self.assertTrue(np.all(v3[10:40] == "H"))
        print("  [Pass] Test 11: Pre-match and halftime boundaries strictly enforced as D.")

    def test_12_rapid_series_of_genuine_turnovers(self):
        """Scenario 12: Genuine rapid series of turnovers (2s each = 50 frames).
        H (0..49) -> A (50..99) -> H (100..149).
        Expected: All 3 spells correctly preserved.
        """
        n = 150
        (v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp) = self.create_synthetic_arrays(n)
        # Frames 50..99: Away turnover
        v2[50:100] = "A"
        sh[50:100] = 0.0
        sa[50:100] = 1.0
        ch[50:100] = False
        ca[50:100] = True
        tha[50] = 0.90
        tah[100] = 0.90

        v3 = decode_possession_sequence(v2, sh, sa, rel, bp, ch, ca, st, tha, tah, sr, rr, inp, config=self.config)
        self.assertTrue(np.all(v3[:50] == "H"))
        self.assertTrue(np.all(v3[50:100] == "A"))
        self.assertTrue(np.all(v3[100:] == "H"))
        print("  [Pass] Test 12: Series of genuine rapid turnovers (2s each) preserved intact.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SYNTHETIC UNIT TESTS FOR PHANTOM-11 SEQUENCE DECODER")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceDecoder)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
