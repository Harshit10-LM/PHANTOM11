#!/usr/bin/env bash
set -e

echo "=============================================================================="
echo "PHANTOM-11 Environment Setup (More Than A Hack 2027)"
echo "=============================================================================="

# Detect Python
PYTHON=${PYTHON:-python3}

echo "Using Python: $($PYTHON --version)"

# Create virtual environment if absent
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    $PYTHON -m venv .venv
fi

# Activate
source .venv/bin/activate

# Upgrade pip and install requirements
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Fetch official repository if absent
if [ ! -f "games.py" ] && [ ! -d "vendor/more-than-a-hack-2027" ]; then
    echo "Fetching public official repository code (pinned commit f8f2629)..."
    python scripts/fetch_official_repo.py
fi

echo "=============================================================================="
echo "Setup complete! Run tests with: python -m unittest discover tests"
echo "Generate predictions with: python run_phantom11.py"
echo "=============================================================================="
