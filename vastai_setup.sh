#!/usr/bin/env bash
# =============================================================================
# TRELLIS.2 WebUI — Vast.ai Provisioning Script (Two-Phase Install)
# =============================================================================
# Usage: Set this script's raw GitHub URL as the "on-start script" in your
#        vast.ai template.  Image: vastai/base-image:cuda-13.2.0-auto
#
# Phase 1: Clone & build TRELLIS.2 (upstream) with all CUDA extensions
# Phase 2: Clone & install trellis2-webui (this repo) on top
#
# All state lives under /workspace/ (persists across stop/start).
# First boot: ~15-25 min. Subsequent boots: ~30 s.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via vast.ai env vars
# ---------------------------------------------------------------------------
TRELLIS_REPO="${TRELLIS_REPO_URL:-https://github.com/microsoft/TRELLIS.2.git}"
TRELLIS_BRANCH="${TRELLIS_REPO_BRANCH:-main}"
WEBUI_REPO="${WEBUI_REPO_URL:-https://github.com/YOUR_USER/trellis2-webui.git}"
WEBUI_BRANCH="${WEBUI_REPO_BRANCH:-main}"

TRELLIS_DIR="/workspace/TRELLIS.2"
WEBUI_DIR="/workspace/trellis2-webui"
MODEL_DIR="/workspace/models/TRELLIS.2-4B"
VENV_DIR="${TRELLIS_DIR}/.venv"
LOG_FILE="/workspace/setup.log"
PORT="${TRELLIS_PORT:-8000}"

HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
exec > >(tee -a "$LOG_FILE") 2>&1
echo ""
echo "========================================"
echo "  TRELLIS.2 WebUI Setup — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/8] Installing system packages..."
apt-get update -qq
apt-get install -y -qq libjpeg-dev libgl1-mesa-glx git curl > /dev/null 2>&1

if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt-get install -y -qq nodejs > /dev/null 2>&1
fi
echo "  Node.js: $(node --version)"

# ---------------------------------------------------------------------------
# 2. Install uv
# ---------------------------------------------------------------------------
echo "[2/8] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
echo "  uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Clone TRELLIS.2 (upstream)
# ---------------------------------------------------------------------------
echo "[3/8] Setting up TRELLIS.2..."
if [ -d "${TRELLIS_DIR}/.git" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$TRELLIS_DIR"
    git pull --ff-only || true
    git submodule update --init --recursive
else
    echo "  Cloning ${TRELLIS_REPO} (branch: ${TRELLIS_BRANCH})..."
    git clone -b "$TRELLIS_BRANCH" "$TRELLIS_REPO" "$TRELLIS_DIR" --recursive
    cd "$TRELLIS_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Phase 1: Build TRELLIS.2 venv + CUDA extensions
# ---------------------------------------------------------------------------
MARKER="${VENV_DIR}/.vastai_installed"
echo "[4/8] Phase 1 — Python environment + CUDA extensions..."

if [ -f "$MARKER" ]; then
    echo "  Already built (found marker). Skipping."
else
    cd "$TRELLIS_DIR"
    uv venv --python 3.10 "$VENV_DIR"

    export VIRTUAL_ENV="$VENV_DIR"
    export PATH="${VENV_DIR}/bin:$PATH"

    echo "  Installing PyTorch cu130..."
    uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130

    echo "  Installing build tools..."
    uv pip install ninja packaging psutil setuptools wheel

    echo "  Compiling CUDA extensions (~10-15 min)..."
    uv sync --no-build-isolation

    echo "  Installing NVIDIA runtime packages..."
    uv pip install \
        nvidia-cublas-cu12 \
        nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cudnn-cu12 \
        nvidia-cufft-cu12 \
        nvidia-curand-cu12 \
        nvidia-cusolver-cu12 \
        nvidia-cusparse-cu12 \
        nvidia-cusparselt-cu12 \
        nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 \
        nvidia-nvtx-cu12

    touch "$MARKER"
    echo "  ✅ Phase 1 complete."
fi

# Activate venv for remaining steps
export VIRTUAL_ENV="$VENV_DIR"
export PATH="${VENV_DIR}/bin:$PATH"

# ---------------------------------------------------------------------------
# 5. Clone trellis2-webui (this repo)
# ---------------------------------------------------------------------------
echo "[5/8] Setting up trellis2-webui..."
if [ -d "${WEBUI_DIR}/.git" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$WEBUI_DIR"
    git pull --ff-only || true
else
    echo "  Cloning ${WEBUI_REPO} (branch: ${WEBUI_BRANCH})..."
    git clone -b "$WEBUI_BRANCH" "$WEBUI_REPO" "$WEBUI_DIR"
    cd "$WEBUI_DIR"
fi

# ---------------------------------------------------------------------------
# 6. Phase 2: Install webui dependencies into the shared venv
# ---------------------------------------------------------------------------
echo "[6/8] Phase 2 — Installing WebUI dependencies..."
cd "$WEBUI_DIR"
uv pip install -e .

# ---------------------------------------------------------------------------
# 7. Download model
# ---------------------------------------------------------------------------
echo "[7/8] Checking model..."
if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    echo "  Model already downloaded."
else
    echo "  Downloading TRELLIS.2-4B from HuggingFace (~16 GB)..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B', local_dir='${MODEL_DIR}'${HF_TOKEN:+, token='${HF_TOKEN}'})"
    echo "  ✅ Model downloaded."
fi

# ---------------------------------------------------------------------------
# 8. Build frontend + start server
# ---------------------------------------------------------------------------
echo "[8/8] Building frontend and starting server..."
cd "${WEBUI_DIR}/webui"
if [ -d "dist" ] && [ "dist/index.html" -nt "index.html" ]; then
    echo "  Frontend already built."
else
    npm ci --silent 2>/dev/null || npm install --silent
    npm run build
    echo "  ✅ Frontend built."
fi

cd "$WEBUI_DIR"

export TRELLIS_MODEL_PATH="$MODEL_DIR"
export ATTN_BACKEND="sdpa"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OPENCV_IO_ENABLE_OPENEXR="1"

echo ""
echo "========================================"
echo "  TRELLIS.2 WebUI starting on port ${PORT}"
echo "  Model: ${MODEL_DIR}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"
echo ""

exec python -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
