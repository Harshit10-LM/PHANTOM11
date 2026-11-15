#!/usr/bin/env python3
"""PHANTOM-11: Reusable Match Configuration Structure

Separates match-specific metadata, dimensions, and asset locations from
possession prediction algorithms.

Supports:
- Automatic derivation from official competition constants (games.py)
- Configuration of prospective future matches with compatible tracking
- Explicit validation and error reporting for unsupported datasets
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure path resolver is available
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

try:
    from phantom11.path_resolver import resolve_official_repo, resolve_data_dir
except ImportError:
    try:
        from path_resolver import resolve_official_repo, resolve_data_dir
    except ImportError:
        resolve_official_repo, resolve_data_dir = None, None


@dataclass
class MatchConfig:
    """Configuration and metadata specification for a football match."""

    match_id: str = "BAR-ATH"
    home_team: str = "Barcelona"
    away_team: str = "Athletic Club"
    data_dir: Optional[str] = None
    n_frames: int = 199319
    fps: float = 25.0
    pitch: Tuple[float, float] = (105.3, 68.0)
    n_slots: Optional[int] = None
    team_of_track: Optional[np.ndarray] = None
    playing_intervals: List[Tuple[int, int]] = field(default_factory=list)
    non_play_intervals: List[Tuple[int, int]] = field(default_factory=list)
    data_format: str = "FIFA_EPTS_RAW"
    available_assets: Tuple[str, ...] = ("video", "atd_meta", "atd_raw", "homography", "smart_tagging")
    offsets: Dict[int, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_official_game(cls, game: str = "BAR-ATH", data_dir: Optional[str] = None) -> "MatchConfig":
        """Construct a MatchConfig directly from official games.py constants.

        This uses the hackathon's official constants as the single source of truth.
        """
        if resolve_official_repo:
            resolve_official_repo()
        import games
        import baseline

        if game not in games.GAMES:
            raise KeyError(
                f"Game '{game}' not found in official games.py registry. "
                f"Available games: {sorted(games.GAMES.keys())}"
            )

        game_entry = games.GAMES[game]
        n_frames = games.source_frames("tactical", game=game)
        pitch = tuple(game_entry["pitch"])
        fps = float(game_entry.get("fps", 25.0))
        offsets = dict(game_entry.get("offsets", {}))

        # Playing intervals from half video blocks
        half_blocks = games.half_video_blocks(game)
        playing_intervals = [(b[1], b[2]) for b in half_blocks]

        # Non-play intervals: pre-match, halftime, post-match
        h1_start, h1_end = playing_intervals[0]
        h2_start, h2_end = playing_intervals[1]
        non_play_intervals = [
            (0, h1_start - 1),
            (h1_end + 1, h2_start - 1),
            (h2_end + 1, n_frames - 1),
        ]

        # Resolve asset paths and metadata if data is present
        resolved_data = data_dir or (resolve_data_dir() if resolve_data_dir else None)
        team_of_track = None
        home_name, away_name = "Home", "Away"
        n_slots = None

        if resolved_data:
            try:
                paths = games.asset_paths(game=game, source="tactical", data_dir=resolved_data)
                meta_path = paths.get("atd_meta")
                if meta_path and os.path.isfile(meta_path):
                    team_of_track = baseline.team_of_each_track(meta_path)
                    n_slots = len(team_of_track)
                    club_names = baseline.team_names(meta_path)
                    if club_names and len(club_names) == 2:
                        home_name, away_name = club_names[0], club_names[1]
            except Exception:
                # Fall back to known names for official BAR-ATH
                if game == "BAR-ATH":
                    home_name, away_name = "Barcelona", "Athletic Club"

        if game == "BAR-ATH":
            home_name, away_name = "Barcelona", "Athletic Club"

        available_assets = games.SOURCES.get("tactical", {}).get(
            "assets", ("video", "atd_meta", "atd_raw", "homography", "smart_tagging")
        )

        return cls(
            match_id=game,
            home_team=home_name,
            away_team=away_name,
            data_dir=resolved_data,
            n_frames=n_frames,
            fps=fps,
            pitch=pitch,
            n_slots=n_slots,
            team_of_track=team_of_track,
            playing_intervals=playing_intervals,
            non_play_intervals=non_play_intervals,
            data_format="FIFA_EPTS_RAW",
            available_assets=tuple(available_assets),
            offsets=offsets,
            metadata={"source": "official_games_py", "game": game},
        )

    def get_team_map(self) -> Dict[str, int]:
        """Construct normalized team name mapping to team indices (0=Home, 1=Away)."""
        mapping: Dict[str, int] = {}

        # Home team variations
        home_clean = self.home_team.strip().lower()
        mapping[home_clean] = 0
        for token in home_clean.split():
            if len(token) > 3:
                mapping[token] = 0

        # Away team variations
        away_clean = self.away_team.strip().lower()
        mapping[away_clean] = 1
        for token in away_clean.split():
            if len(token) > 3:
                mapping[token] = 1

        # Specific known variants for Barcelona vs Athletic Club
        if self.match_id == "BAR-ATH" or "barcelona" in mapping:
            mapping["barcelona"] = 0
            mapping["barca"] = 0
            mapping["atletic club"] = 1  # official spelling in XML tag
            mapping["athletic club"] = 1
            mapping["athletic"] = 1

        return mapping

    def get_non_play_intervals(self) -> List[Tuple[int, int]]:
        """Return non-play intervals (pre-match, halftime, post-match)."""
        if self.non_play_intervals:
            return self.non_play_intervals

        if self.playing_intervals and len(self.playing_intervals) >= 2:
            h1_start, h1_end = self.playing_intervals[0]
            h2_start, h2_end = self.playing_intervals[1]
            return [
                (0, max(0, h1_start - 1)),
                (h1_end + 1, max(h1_end + 1, h2_start - 1)),
                (h2_end + 1, max(h2_end + 1, self.n_frames - 1)),
            ]

        # If no intervals known, return empty list
        return []

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration integrity and check for unsupported parameters.

        Returns (is_valid, list_of_issues).
        """
        issues = []
        if not self.match_id:
            issues.append("Missing match_id.")
        if self.n_frames <= 0:
            issues.append(f"Invalid frame count: {self.n_frames}.")
        if self.fps <= 0.0 or self.fps > 120.0:
            issues.append(f"Unusual frame rate: {self.fps} fps.")
        if len(self.pitch) != 2 or self.pitch[0] <= 50.0 or self.pitch[1] <= 30.0:
            issues.append(f"Invalid pitch dimensions: {self.pitch}.")
        if self.data_format not in ("FIFA_EPTS_RAW", "CSV_METRIC", "GENERIC_COORDINATE"):
            issues.append(f"Unsupported data format: '{self.data_format}'. Supported formats: FIFA_EPTS_RAW.")

        return len(issues) == 0, issues
