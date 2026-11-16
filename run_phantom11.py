#!/usr/bin/env python3
"""PHANTOM-11: Official Execution Entry Point for More Than A Hack 2027

Usage:
    python run_phantom11.py [--candidate v5a] [--data-dir PATH] [--output-dir PATH] [--official-dir PATH]

By default, automatically locates the tactical match data and official repository,
executes the full deterministic pipeline, and writes our leading candidate:
    outputs/phantom11_v5a.csv
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phantom11.pipeline import run_pipeline
from phantom11.path_resolver import resolve_official_repo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PHANTOM-11 Prediction Generator")
    parser.add_argument("--candidate", "--version", default="v5a", choices=["v0", "v1", "v2", "v3", "v4", "v5", "v5a"], help="Model candidate version to generate (default: v5a)")
    parser.add_argument("--game", "--match-id", default="BAR-ATH", dest="game", help="Match identifier (default: BAR-ATH)")
    parser.add_argument("--data-dir", default=None, help="Directory containing tactical data files")
    parser.add_argument("--output-dir", default=None, help="Directory to save generated CSV files")
    parser.add_argument("--official-dir", default=None, help="Path to official more-than-a-hack-2027 repository")
    args = parser.parse_args()

    if args.official_dir:
        resolve_official_repo(args.official_dir)

    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, game=args.game, candidate=args.candidate)
