#!/usr/bin/env bash
# =============================================================================
# Local Run Script
# Runs the WebUI using the upstream TRELLIS.2 environment
# =============================================================================
set -euo pipefail

TRELLIS_DIR="../TRELLIS.2"
VENV_DIR="${TRELLIS_DIR}/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Error: Could not find TRELLIS.2 virtual environment at $VENV_DIR"
    exit 1
fi

echo "🚀 Starting TRELLIS.2 WebUI using upstream environment..."

# Activate the upstream venv
source "${VENV_DIR}/bin/activate"

# Fallback model path if not set in environment
export TRELLIS_MODEL_PATH="${TRELLIS_MODEL_PATH:-/home/matatrata/AI/Models/TRELLIS.2-4B}"

# Run the app
exec python app.py
