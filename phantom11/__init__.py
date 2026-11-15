"""PHANTOM-11: Tactical Intelligence and Team Possession Decoding Package.

Step 8A: Clean, portable package structure for More Than A Hack 2027.
"""

import sys
import os

# Automatically resolve official dependency paths on import
try:
    from phantom11.path_resolver import resolve_official_repo
    try:
        resolve_official_repo()
    except Exception:
        pass
except ImportError:
    pass

__version__ = "3.0.0"
__author__ = "PHANTOM-11 Team"

try:
    from phantom11.match_config import MatchConfig
except ImportError:
    pass

