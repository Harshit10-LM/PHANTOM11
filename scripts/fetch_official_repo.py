#!/usr/bin/env python3
"""Fetch the official More Than A Hack 2027 repository at pinned commit.

Downloads the public, MIT-licensed code and documentation only.
Does NOT download any licensed match assets.
"""

import os
import sys
import subprocess

PINNED_COMMIT = "f8f2629ff1e0985f81d7e59d8413678514ee1530"
REPO_URL = "https://github.com/metrica-sports/more-than-a-hack-2027.git"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TARGET_DIR = os.path.join(PROJECT_ROOT, "vendor", "more-than-a-hack-2027")


def fetch():
    if os.path.isdir(os.path.join(TARGET_DIR, ".git")):
        print(f"Official repository already present at {TARGET_DIR}")
        subprocess.run(["git", "-C", TARGET_DIR, "checkout", PINNED_COMMIT], check=True)
        print(f"Verified at pinned commit {PINNED_COMMIT[:7]}")
        return

    print(f"Cloning official repository from {REPO_URL}...")
    os.makedirs(os.path.dirname(TARGET_DIR), exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL, TARGET_DIR], check=True)
    subprocess.run(["git", "-C", TARGET_DIR, "checkout", PINNED_COMMIT], check=True)
    print(f"Successfully checked out official repo at {PINNED_COMMIT[:7]}")


if __name__ == "__main__":
    fetch()
