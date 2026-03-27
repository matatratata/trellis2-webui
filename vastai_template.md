# TRELLIS.2 WebUI — Vast.ai Deployment Guide

## Quick Start

1. **Create a template** on [vast.ai](https://cloud.vast.ai/templates/):

   | Setting | Value |
   |---|---|
   | **Docker Image** | `vastai/base-image:cuda-13.2.0-auto` |
   | **Launch Mode** | `jupyter_direc` (Jupyter Direct HTTPS) |
   | **Disk Space** | ≥ 60 GB |

2. **Set environment variables** (in template or account settings):

   ```
   -e PROVISIONING_SCRIPT="https://raw.githubusercontent.com/matatratata/trellis2-webui/main/vastai_setup.sh"
   -e ENABLE_HTTPS=true
   ```

   *(Put the raw URL of `vastai_setup.sh` from your fork/repo)*

   **Optional overrides:**

   | Variable | Default | Description |
   |---|---|---|
   | `TRELLIS_REPO_URL` | `https://github.com/microsoft/TRELLIS.2.git` | Upstream TRELLIS.2 repo |
   | `WEBUI_REPO_URL` | `https://github.com/matatratata/trellis2-webui.git` | Your trellis2-webui repo URL |
   | `WEBUI_REPO_BRANCH` | `main` | Branch to use |
   | `TRELLIS_PORT` | `8000` | Server port |
   | `HF_TOKEN` | *(empty)* | HuggingFace token (gated models) |

   > **🔒 HF_TOKEN:** TRELLIS.2 is a gated model. Set `HF_TOKEN` in your [Vast.ai account settings](https://cloud.vast.ai/account/) (Environment Variables section), **not** in the template — this keeps it out of shared/published templates and git.

3. **Docker options** — add the WebUI port:

   ```
   -p 8000:8000
   ```

   The base template already exposes ports for Jupyter (8080), Instance Portal (1111), Syncthing (8384), and Tensorboard (6006). Just add 8000 for the WebUI.

4. **Search for an instance** — filter by:
   - GPU: RTX 3090 / RTX 4090 / A6000 / A100 (≥ 24 GB VRAM)
   - CUDA ≥ 12.4
   - Disk: ≥ 60 GB

5. **Launch** — first boot takes ~15-25 min, cached restarts ~30 s.

6. **Access:**
   - **TRELLIS.2 WebUI** — `https://<instance-ip>:8000` or via Instance Portal
   - **Jupyter Lab** — click the "Jupyter" button (direct HTTPS, fast access + built-in terminal & file browser)
   - **SSH** — click the "SSH" button for terminal access with tmux
   - **Instance Portal** — click "Open" to manage all services

> **💡 HTTPS:** The `jupyter_direc` launch mode uses direct HTTPS. Install the [Vast.ai TLS certificate](https://docs.vast.ai/instances/jupyter) once in your browser to avoid warnings. This gives you fast, encrypted access to Jupyter + all proxied services.

---

## How It Works

The `PROVISIONING_SCRIPT` env var tells the base image to download and run `vastai_setup.sh` after Supervisor starts (boot phase 9). This means all built-in services (Jupyter, SSH, Syncthing, Tensorboard) are already running while our script installs TRELLIS.2 in the background.

The script runs a **two-phase install**:

| Phase | What | Time (first) | Cached? |
|-------|------|--------------|---------|
| **1** | Clone TRELLIS.2, build venv, compile 6 CUDA extensions | ~10-15 min | ✅ marker file |
| **2** | Clone trellis2-webui, `pip install -e .` | ~30 s | ✅ git pull |
| | Download model (~16 GB) | ~5-10 min | ✅ if exists |
| | Build frontend | ~15 s | ✅ if newer |
| | Register & start WebUI as Supervisor service | instant | — |

All state lives under `/workspace/` which **persists across stop/start**.

The WebUI is registered as a Supervisor service (`trellis-webui`) so it auto-restarts on crash and its logs are accessible through the Instance Portal.

---

## Monitoring

```bash
# Watch the provisioning progress
tail -f /workspace/setup.log

# Check service status
supervisorctl status

# View WebUI logs
supervisorctl tail -f trellis-webui
```

---

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
Ensure `-p 8000:8000` is in Docker Options and port 8000 is listed in Direct Port settings.

### Re-run provisioning
```bash
rm /.provisioning_complete
# Then restart the instance
```
