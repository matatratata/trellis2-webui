# trellis2-webui

Web UI for [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) — image-to-3D generation.

This is a standalone web application that uses TRELLIS.2 as a dependency. It provides a browser-based interface for generating 3D assets from images with full PBR materials.

## Features

- **Single Image → 3D** — upload one image, get a 3D model
- **Multi-View → 3D** — upload 2-6 views for higher accuracy
- **Re-Texturing** — apply new textures to existing meshes
- **Interactive Viewer** — preview renders + Three.js 3D viewer
- **GLB Export** — download production-ready GLB files
- **Quality Presets** — fast / balanced / high / max with surface-type tuning

## Architecture

```
trellis2-webui/          ← This repo
├── app.py               ← FastAPI backend (uses trellis2 as a library)
├── webui/               ← Vite frontend (HTML/JS/CSS)
├── assets/hdri/         ← HDRI environment maps for rendering
├── vastai_setup.sh      ← Vast.ai provisioning script
└── pyproject.toml

TRELLIS.2/               ← Upstream repo (installed separately)
├── trellis2/            ← Python package (pipelines, models, etc.)
├── o-voxel/             ← CUDA extension (submodule)
└── pyproject.toml       ← Declares all CUDA extension deps
```

The WebUI imports from `trellis2` (pipelines, renderers, utils) and `o_voxel` (mesh post-processing). These are installed from the upstream TRELLIS.2 repo via a two-phase install process.

## Quick Start (Vast.ai)

See [vastai_template.md](vastai_template.md) for full deployment instructions.

**TL;DR:** Create a template with `vastai/base-image:cuda-13.2.0-auto` and set `vastai_setup.sh` as the on-start script.

## Local Development

### Prerequisites

- NVIDIA GPU with ≥ 24 GB VRAM
- CUDA Toolkit 12.4+ or 13.x
- Python 3.10+
- Node.js 18+

### Setup

```bash
# 1. Clone & build TRELLIS.2 (with CUDA extensions)
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
cd TRELLIS.2
uv venv && uv sync --no-build-isolation

# 2. Clone this repo and install into the same venv
cd ..
git clone https://github.com/YOUR_USER/trellis2-webui.git
cd trellis2-webui
source ../TRELLIS.2/.venv/bin/activate
pip install -e .

# 3. Download the model
huggingface-cli download microsoft/TRELLIS.2-4B --local-dir ~/models/TRELLIS.2-4B

# 4. Build frontend
cd webui && npm install && npm run build && cd ..

# 5. Run
TRELLIS_MODEL_PATH=~/models/TRELLIS.2-4B python app.py
```

Open http://localhost:8000

### Frontend Development

For hot-reload during UI development:

```bash
# Terminal 1: backend
TRELLIS_MODEL_PATH=~/models/TRELLIS.2-4B python app.py

# Terminal 2: frontend dev server (proxies /api to backend)
cd webui && npm run dev
```

Open http://localhost:5173

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRELLIS_MODEL_PATH` | `/workspace/models/TRELLIS.2-4B` | Model directory |
| `TRELLIS_PORT` | `8000` | Server port |
| `ATTN_BACKEND` | `sdpa` | Attention backend (`sdpa` or `flash_attn`) |

## License

MIT — see [LICENSE](LICENSE).

TRELLIS.2 and some dependencies (nvdiffrast, nvdiffrec) have their own licenses.
