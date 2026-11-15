#!/usr/bin/env python3
"""PHANTOM-11: Portable Path and Dependency Resolver

Resolves the location of:
1. Official Hackathon Repository (games.py, starter/, evaluation/).
2. Match Data Bundle (data/tactical/).

Supports running from:
- Within the official clone
- As a standalone public clone with a local or sibling official repo
- Fresh clone where the public official code is fetched at pinned commit f8f2629
"""

import os
import sys
import subprocess

PINNED_OFFICIAL_COMMIT = "f8f2629ff1e0985f81d7e59d8413678514ee1530"
OFFICIAL_REPO_URL = "https://github.com/metrica-sports/more-than-a-hack-2027.git"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def resolve_official_repo(explicit_path=None, auto_fetch=False):
    """Find the official repository directory and add it to sys.path."""
    candidate_paths = []
    if explicit_path:
        candidate_paths.append(explicit_path)

    # Environment variables
    for env_var in ["OFFICIAL_DIR", "HACKATHON_DIR", "MORE_THAN_A_HACK_DIR"]:
        val = os.environ.get(env_var)
        if val:
            candidate_paths.append(val)

    # Standard relative paths
    candidate_paths.extend([
        # Sibling directory in parent folder
        os.path.join(os.path.dirname(PROJECT_ROOT), "more-than-a-hack-2027"),
        # Subdirectory in project
        os.path.join(PROJECT_ROOT, "vendor", "more-than-a-hack-2027"),
        os.path.join(PROJECT_ROOT, ".official_repo"),
        # Parent directory if running inside the official repo
        PROJECT_ROOT,
        os.path.dirname(PROJECT_ROOT),
    ])

    for path in candidate_paths:
        if path and os.path.isdir(path):
            games_file = os.path.join(path, "games.py")
            if os.path.isfile(games_file):
                official_root = os.path.abspath(path)
                starter_dir = os.path.join(official_root, "starter")
                eval_dir = os.path.join(official_root, "evaluation")
                for d in [official_root, starter_dir, eval_dir]:
                    if d not in sys.path:
                        sys.path.insert(0, d)
                return official_root

    # Auto-fetch public MIT code at pinned commit if requested
    if auto_fetch:
        target_dir = os.path.join(PROJECT_ROOT, "vendor", "more-than-a-hack-2027")
        print(f"Official code not found locally. Fetching public repo at {PINNED_OFFICIAL_COMMIT[:7]}...")
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        subprocess.run(["git", "clone", OFFICIAL_REPO_URL, target_dir], check=True)
        subprocess.run(["git", "-C", target_dir, "checkout", PINNED_OFFICIAL_COMMIT], check=True)
        return resolve_official_repo(explicit_path=target_dir, auto_fetch=False)

    raise FileNotFoundError(
        "Could not find the official 'more-than-a-hack-2027' repository.\n"
        "Please provide its path using --official-dir or set the HACKATHON_DIR environment variable,\n"
        "or run 'python scripts/fetch_official_repo.py' to acquire the public codebase."
    )


def resolve_data_dir(explicit_path=None, official_root=None, match_stem="BAR-ATH"):
    """Find the tactical match data directory containing tracking files."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    for env_var in ["DATA_DIR", "TACTICAL_DATA_DIR"]:
        val = os.environ.get(env_var)
        if val:
            candidates.append(val)

    if official_root:
        candidates.append(os.path.join(official_root, "data", "tactical"))
        candidates.append(os.path.join(official_root, "data"))

    candidates.extend([
        os.path.join(PROJECT_ROOT, "data", "tactical"),
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(os.path.dirname(PROJECT_ROOT), "more-than-a-hack-2027", "data", "tactical"),
    ])

    expected_file = f"{match_stem}_tactical_atd.txt" if match_stem else "BAR-ATH_tactical_atd.txt"

    for path in candidates:
        if path and os.path.isdir(path):
            # Check if this is the 'data' directory containing 'tactical/'
            if os.path.isfile(os.path.join(path, "tactical", expected_file)):
                return os.path.abspath(path)
            # Check if this is directly the 'tactical' directory
            if os.path.basename(path) == "tactical" and os.path.isfile(os.path.join(path, expected_file)):
                return os.path.dirname(os.path.abspath(path))
            if os.path.isfile(os.path.join(path, expected_file)):
                return os.path.dirname(os.path.abspath(path))

    # Fallback: check for any *_tactical_atd.txt file
    for path in candidates:
        if path and os.path.isdir(path):
            tact_dir = os.path.join(path, "tactical") if os.path.isdir(os.path.join(path, "tactical")) else path
            atd_files = [f for f in os.listdir(tact_dir) if f.endswith("_tactical_atd.txt")]
            if atd_files:
                if os.path.basename(tact_dir) == "tactical":
                    return os.path.dirname(os.path.abspath(tact_dir))
                return os.path.abspath(path)

    raise FileNotFoundError(
        f"Could not find tactical match data directory containing '{expected_file}'.\n"
        "Please specify its path using --data-dir or set the DATA_DIR environment variable."
    )
