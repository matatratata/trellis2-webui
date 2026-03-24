# TRELLIS.2 WebUI — Vast.ai Deployment Guide

## Quick Start

1. **Create a template** on [vast.ai](https://cloud.vast.ai/templates/):

   | Setting | Value |
   |---|---|
   | **Docker Image** | `vastai/base-image:cuda-13.2.0-auto` |
   | **On-start Script** | Raw URL of `vastai_setup.sh` from this repo |
   | **Docker Options** | `-p 8000:8000` |
   | **Launch Mode** | `ssh` |
   | **Disk Space** | ≥ 60 GB |

2. **Search for an instance** — filter by:
   - GPU: RTX 3090 / RTX 4090 / A6000 / A100 (≥ 24 GB VRAM)
   - CUDA ≥ 12.4
   - Disk: ≥ 60 GB

3. **Launch** — first boot takes ~15-25 min, cached restarts ~30 s.

4. **Access** at `http://<instance-ip>:8000`

## How It Works

The `vastai_setup.sh` script runs a **two-phase install**:

| Phase | What | Time (first) | Cached? |
|-------|------|--------------|---------|
| **1** | Clone TRELLIS.2, build venv, compile 6 CUDA extensions | ~10-15 min | ✅ marker file |
| **2** | Clone trellis2-webui, `pip install -e .` | ~30 s | ✅ git pull |
| | Download model (~16 GB) | ~5-10 min | ✅ if exists |
| | Build frontend | ~15 s | ✅ if newer |
| | Start server | instant | — |

All state lives under `/workspace/` which **persists across stop/start**.

## Environment Variables

Set these in the vast.ai template:

| Variable | Default | Description |
|---|---|---|
| `TRELLIS_REPO_URL` | `https://github.com/microsoft/TRELLIS.2.git` | Upstream TRELLIS.2 repo |
| `WEBUI_REPO_URL` | *(must set)* | This repo's URL |
| `WEBUI_REPO_BRANCH` | `main` | Branch to use |
| `TRELLIS_PORT` | `8000` | Server port |
| `HF_TOKEN` | *(empty)* | HuggingFace token (if needed) |

## Monitoring

```bash
tail -f /workspace/setup.log
```

## Troubleshooting

### CUDA compilation fails
```bash
# SSH in, remove marker, restart
rm /workspace/TRELLIS.2/.venv/.vastai_installed
# Then restart the instance
```

### Disk space
Full install needs ~50 GB: venv (~8 GB) + model (~16 GB) + repos (~2 GB) + OS (~20 GB).

### Port not accessible
Ensure `-p 8000:8000` is in Docker Options and port 8000 is in Direct Port settings.
