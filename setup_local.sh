#!/usr/bin/env bash
# =============================================================================
# Local Development Setup Script
# Links the upstream TRELLIS.2 environment into the trellis2-webui
# =============================================================================
set -euo pipefail

TRELLIS_DIR="../TRELLIS.2"
VENV_DIR="${TRELLIS_DIR}/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Error: Could not find TRELLIS.2 virtual environment at $VENV_DIR"
    echo "Please ensure TRELLIS.2 is cloned and built at $TRELLIS_DIR"
    exit 1
fi

echo "✅ Found TRELLIS.2 environment at $VENV_DIR"
echo "Installing trellis2-webui into the TRELLIS.2 environment..."

# Activate the upstream venv
source "${VENV_DIR}/bin/activate"

# Install webui dependencies (fastapi, uvicorn, etc.)
uv pip install -e . --no-build-isolation

echo ""
echo "========================================"
echo "✅ Local setup complete!"
echo "To run the WebUI locally:"
echo ""
echo "  source ${VENV_DIR}/bin/activate"
echo "  TRELLIS_MODEL_PATH=/path/to/models/TRELLIS.2-4B python app.py"
echo "========================================"
