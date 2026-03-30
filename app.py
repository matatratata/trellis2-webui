import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"

import asyncio
import gc
import uuid
import shutil
import json
import io
import base64
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse import config as sparse_config
sparse_config.ATTN = 'sdpa'  # Force sdpa — flash_attn not installed
from trellis2.pipelines import Trellis2ImageTo3DPipeline, Trellis2TexturingPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
TMP_DIR = APP_DIR / "tmp_webui"
WEBUI_DIR = APP_DIR / "webui" / "dist"
ASSETS_DIR = APP_DIR / "assets"
MAX_SEED = 100000  # Cap random seeds at 100k for all stages

# Model path: env var > default location for vast.ai
MODEL_PATH = os.environ.get(
    'TRELLIS_MODEL_PATH',
    '/workspace/models/TRELLIS.2-4B'
)

# Use second GPU for mesh processing if available, otherwise same GPU
GENERATION_DEVICE = "cuda:0"
MESH_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

# Global pipeline references (initialized on startup)
pipeline: Trellis2ImageTo3DPipeline = None  # type: ignore
tex_pipeline: Trellis2TexturingPipeline = None  # type: ignore
envmap: dict = {}

# In-memory state store for CPU numpy arrays (session_id -> dict)
latent_store: dict = {}

# Progress tracking
progress_store: dict = {}  # task_id -> {stage, step, total, done}

# GPU lock — only one generation at a time
gpu_lock = asyncio.Lock()


def move_pipeline_to(p, device):
    """Deep inspection tool to safely move ALL models attached to a pipeline.
    Handles nn.ModuleDict (which isinstance(dict) returns False for)."""
    if p is None: return

    p._device = device

    if hasattr(p, 'to'):
        try: p.to(device)
        except Exception: pass

    if hasattr(p, 'models'):
        models = p.models
        # Support for both standard dicts AND PyTorch nn.ModuleDict
        if isinstance(models, dict) or hasattr(models, 'values'):
            for m in models.values():
                if hasattr(m, 'to'):
                    try: m.to(device)
                    except Exception: pass

    # Sweep internal attributes for custom attached PyTorch modules
    for attr_name in dir(p):
        if attr_name.startswith('_'): continue
        try:
            val = getattr(p, attr_name)
            if isinstance(val, torch.nn.Module):
                if hasattr(val, 'to'):
                    try: val.to(device)
                    except Exception: pass
        except Exception:
            pass


def offload_sampling_models():
    """Offload flow models and conditioning model to CPU, keeping only decoders active."""
    sampling_models = [
        'sparse_structure_flow_model', 'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024', 'tex_slat_flow_model_512', 'tex_slat_flow_model_1024',
    ]
    for p in [pipeline, tex_pipeline]:
        if p is None: continue
        if hasattr(p, 'models') and (isinstance(p.models, dict) or hasattr(p.models, 'keys')):
            for name in sampling_models:
                if name in p.models:
                    try: p.models[name].to('cpu')
                    except Exception: pass
        for attr in ['image_cond_model', 'rembg_model']:
            if hasattr(p, attr):
                m = getattr(p, attr)
                if m is not None and hasattr(m, 'to'):
                    try: m.to('cpu')
                    except Exception: pass
    gc.collect()
    torch.cuda.empty_cache()


def force_offload_all():
    """Forces absolutely everything to CPU to give to_glb maximum VRAM."""
    move_pipeline_to(pipeline, 'cpu')
    move_pipeline_to(tex_pipeline, 'cpu')

    for v in envmap.values():
        if hasattr(v, 'image') and hasattr(v.image, 'to'):
            v.image = v.image.to('cpu')
        if hasattr(v, '_nvdiffrec_envlight'):
            delattr(v, '_nvdiffrec_envlight')

    gc.collect()
    torch.cuda.empty_cache()


def restore_envmaps():
    """Restores envmaps to the GPU for rendering."""
    for v in envmap.values():
        if hasattr(v, 'image') and hasattr(v.image, 'to'):
            v.image = v.image.to(GENERATION_DEVICE)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TRELLIS.2 WebUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_pil_image(file: UploadFile) -> Image.Image:
    data = file.file.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")

def get_session_dir(session_id: str) -> Path:
    d = TMP_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def make_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"

def render_preview_images(mesh, envmap_dict) -> dict:
    mesh.simplify(16777216)
    images = render_utils.render_snapshot(
        mesh, resolution=1024, r=2, fov=36, nviews=8, envmap=envmap_dict
    )
    result = {}
    for key, frames in images.items():
        result[key] = []
        for frame in frames:
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            result[key].append(f"data:image/png;base64,{b64}")
    return result

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global pipeline, tex_pipeline, envmap
    TMP_DIR.mkdir(exist_ok=True)

    print(f"Loading model from: {MODEL_PATH}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_PATH)
    move_pipeline_to(pipeline, GENERATION_DEVICE)
    force_offload_all()

    try:
        tex_pipeline = Trellis2TexturingPipeline.from_pretrained(
            MODEL_PATH, config_file="texturing_pipeline.json"
        )
        move_pipeline_to(tex_pipeline, GENERATION_DEVICE)
        force_offload_all()
    except Exception as e:
        print(f"⚠ Texturing pipeline not available: {e}")
        traceback.print_exc()
        tex_pipeline = None

    envmap_data = {}
    hdri_dir = ASSETS_DIR / "hdri"
    for name in ['forest', 'sunset', 'courtyard']:
        exr_path = hdri_dir / f"{name}.exr"
        if exr_path.exists():
            envmap_data[name] = EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            ))
    envmap = envmap_data
    force_offload_all()

    print(f"✅ TRELLIS.2 WebUI ready")
    print(f"   Generation on: {GENERATION_DEVICE} ({torch.cuda.get_device_name(GENERATION_DEVICE)})")
    if torch.cuda.device_count() > 1:
        print(f"   Mesh export on: {MESH_DEVICE} ({torch.cuda.get_device_name(MESH_DEVICE)})")


@app.on_event("shutdown")
async def shutdown():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Progress SSE
# ---------------------------------------------------------------------------
@app.get("/api/progress/{task_id}")
async def progress_stream(task_id: str):
    async def event_generator():
        while True:
            info = progress_store.get(task_id, {})
            data = json.dumps(info)
            yield f"data: {data}\n\n"
            if info.get("done"):
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# API: Single Image → 3D
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = Form(0),
    randomize_seed: bool = Form(True),
    resolution: str = Form("1024"),
    ss_guidance_strength: float = Form(7.5),
    ss_guidance_rescale: float = Form(0.7),
    ss_sampling_steps: int = Form(12),
    ss_rescale_t: float = Form(5.0),
    shape_guidance_strength: float = Form(7.5),
    shape_guidance_rescale: float = Form(0.5),
    shape_sampling_steps: int = Form(12),
    shape_rescale_t: float = Form(3.0),
    tex_guidance_strength: float = Form(1.0),
    tex_guidance_rescale: float = Form(0.0),
    tex_sampling_steps: int = Form(12),
    tex_rescale_t: float = Form(3.0),
):
    task_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    if randomize_seed:
        seed = int(np.random.randint(0, MAX_SEED))

    pil_image = load_pil_image(image)
    pil_image = pipeline.preprocess_image(pil_image)
    pipeline_type = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}.get(resolution, "1024_cascade")

    progress_store[task_id] = {"stage": "Starting", "step": 0, "total": 100, "done": False}

    async with gpu_lock:
        try:
            progress_store[task_id] = {"stage": "Conditioning image", "step": 10, "total": 100, "done": False}

            ss_params = {"steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength, "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t}
            shape_params = {"steps": shape_sampling_steps, "guidance_strength": shape_guidance_strength, "guidance_rescale": shape_guidance_rescale, "rescale_t": shape_rescale_t}
            tex_params = {"steps": tex_sampling_steps, "guidance_strength": tex_guidance_strength, "guidance_rescale": tex_guidance_rescale, "rescale_t": tex_rescale_t}

            def do_generation():
                import torch as _torch
                import gc
                _torch.manual_seed(seed)

                with _torch.no_grad():
                    force_offload_all()
                    # Let low_vram mode handle per-model loading — do NOT bulk-load
                    pipeline._device = GENERATION_DEVICE

                    cond_512 = pipeline.get_cond([pil_image], 512)
                    cond_1024 = pipeline.get_cond([pil_image], 1024) if pipeline_type != '512' else None

                    progress_store[task_id] = {"stage": "Sparse structure", "step": 20, "total": 100, "done": False}
                    ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
                    coords = pipeline.sample_sparse_structure(cond_512, ss_res, 1, ss_params)

                    progress_store[task_id] = {"stage": "Generating shape", "step": 30, "total": 100, "done": False}
                    if pipeline_type == '512':
                        shape_slat = pipeline.sample_shape_slat(cond_512, pipeline.models['shape_slat_flow_model_512'], coords, shape_params)
                        res = 512
                    elif pipeline_type == '1024':
                        shape_slat = pipeline.sample_shape_slat(cond_1024, pipeline.models['shape_slat_flow_model_1024'], coords, shape_params)
                        res = 1024
                    elif pipeline_type == '1024_cascade':
                        shape_slat, res = pipeline.sample_shape_slat_cascade(cond_512, cond_1024, pipeline.models['shape_slat_flow_model_512'], pipeline.models['shape_slat_flow_model_1024'], 512, 1024, coords, shape_params)
                    elif pipeline_type == '1536_cascade':
                        shape_slat, res = pipeline.sample_shape_slat_cascade(cond_512, cond_1024, pipeline.models['shape_slat_flow_model_512'], pipeline.models['shape_slat_flow_model_1024'], 512, 1536, coords, shape_params)

                    progress_store[task_id] = {"stage": "Generating texture", "step": 55, "total": 100, "done": False}
                    tex_cond = cond_512 if pipeline_type == '512' else cond_1024
                    tex_model = pipeline.models['tex_slat_flow_model_512'] if pipeline_type == '512' else pipeline.models['tex_slat_flow_model_1024']
                    tex_slat = pipeline.sample_tex_slat(tex_cond, tex_model, shape_slat, tex_params)

                    # Free ALL sampling models + intermediate tensors before decode
                    del cond_512, cond_1024, tex_cond, coords
                    force_offload_all()
                    pipeline._device = GENERATION_DEVICE

                    progress_store[task_id] = {"stage": "Decoding mesh", "step": 75, "total": 100, "done": False}
                    out_mesh = pipeline.decode_latent(shape_slat, tex_slat, res)
                    mesh = out_mesh[0]

                    progress_store[task_id] = {"stage": "Rendering preview", "step": 80, "total": 100, "done": False}
                    restore_envmaps()
                    previews = render_preview_images(mesh, envmap)

                    # ✅ Extract fully decoded raw tensors right here!
                    # The decoder is bypassed entirely in /extract endpoints.
                    state = {
                        'verts': mesh.vertices.detach().cpu().clone(),
                        'faces': mesh.faces.detach().cpu().clone(),
                        'attrs': mesh.attrs.detach().cpu().clone(),
                        'coords': mesh.coords.detach().cpu().clone(),
                        'pbr_layout': pipeline.pbr_attr_layout,
                        'res': res,
                    }

                    del shape_slat, tex_slat, out_mesh, mesh
                    gc.collect()
                    _torch.cuda.empty_cache()

                force_offload_all()
                return previews, state

            previews, state = await asyncio.get_event_loop().run_in_executor(None, do_generation)
            latent_store[session_id] = state

            progress_store[task_id] = {"stage": "Done", "step": 100, "total": 100, "done": True}

            return JSONResponse({
                "task_id": task_id,
                "session_id": session_id,
                "seed": seed,
                "previews": previews,
            })

        except Exception as e:
            progress_store[task_id] = {"stage": f"Error: {str(e)}", "step": 0, "total": 100, "done": True}
            traceback.print_exc()
            force_offload_all()
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Multi-View → 3D
# ---------------------------------------------------------------------------
@app.post("/api/generate-multiview")
async def generate_multiview(
    request: Request,
    seed: int = Form(0),
    randomize_seed: bool = Form(True),
    resolution: str = Form("1024"),
    ss_guidance_strength: float = Form(7.5),
    ss_guidance_rescale: float = Form(0.7),
    ss_sampling_steps: int = Form(12),
    ss_rescale_t: float = Form(5.0),
    shape_guidance_strength: float = Form(7.5),
    shape_guidance_rescale: float = Form(0.5),
    shape_sampling_steps: int = Form(12),
    shape_rescale_t: float = Form(3.0),
    tex_guidance_strength: float = Form(1.0),
    tex_guidance_rescale: float = Form(0.0),
    tex_sampling_steps: int = Form(12),
    tex_rescale_t: float = Form(3.0),
):
    task_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    if randomize_seed:
        seed = int(np.random.randint(0, MAX_SEED))

    form = await request.form()
    image_files = []
    for key, value in form.multi_items():
        if key.startswith("image"):
            data = await value.read()
            pil = Image.open(io.BytesIO(data)).convert("RGBA")
            pil = pipeline.preprocess_image(pil)
            image_files.append(pil)

    pipeline_type = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}.get(resolution, "1024_cascade")
    progress_store[task_id] = {"stage": "Starting", "step": 0, "total": 100, "done": False}

    async with gpu_lock:
        try:
            progress_store[task_id] = {"stage": "Generating 3D (multi-view)", "step": 10, "total": 100, "done": False}

            def do_multiview():
                import torch as _torch
                import gc
                with _torch.no_grad():
                    force_offload_all()
                    # Let low_vram mode handle per-model loading — do NOT bulk-load
                    pipeline._device = GENERATION_DEVICE

                    outputs, latents = pipeline.run_multiview(
                        image_files, seed=seed, preprocess_image=False,
                        sparse_structure_sampler_params={"steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength, "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t},
                        shape_slat_sampler_params={"steps": shape_sampling_steps, "guidance_strength": shape_guidance_strength, "guidance_rescale": shape_guidance_rescale, "rescale_t": shape_rescale_t},
                        tex_slat_sampler_params={"steps": tex_sampling_steps, "guidance_strength": tex_guidance_strength, "guidance_rescale": tex_guidance_rescale, "rescale_t": tex_rescale_t},
                        pipeline_type=pipeline_type, return_latent=True,
                    )

                    progress_store[task_id] = {"stage": "Rendering preview", "step": 80, "total": 100, "done": False}
                    mesh = outputs[0]
                    restore_envmaps()
                    previews = render_preview_images(mesh, envmap)
                    res = latents[2]

                    # ✅ Extract fully decoded raw tensors here!
                    state = {
                        'verts': mesh.vertices.detach().cpu().clone(),
                        'faces': mesh.faces.detach().cpu().clone(),
                        'attrs': mesh.attrs.detach().cpu().clone(),
                        'coords': mesh.coords.detach().cpu().clone(),
                        'pbr_layout': pipeline.pbr_attr_layout,
                        'res': res,
                    }

                    del outputs, latents, mesh
                    gc.collect()
                    _torch.cuda.empty_cache()

                force_offload_all()
                return previews, state

            previews, state = await asyncio.get_event_loop().run_in_executor(None, do_multiview)
            latent_store[session_id] = state

            progress_store[task_id] = {"stage": "Done", "step": 100, "total": 100, "done": True}

            return JSONResponse({
                "task_id": task_id, "session_id": session_id, "seed": seed, "previews": previews, "num_views": len(image_files)
            })

        except Exception as e:
            progress_store[task_id] = {"stage": f"Error: {str(e)}", "step": 0, "total": 100, "done": True}
            traceback.print_exc()
            force_offload_all()
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Extract GLB
# ---------------------------------------------------------------------------
@app.post("/api/extract-glb")
async def extract_glb(
    session_id: str = Form(...),
    decimation_target: int = Form(500000),
    texture_size: int = Form(2048),
):
    if session_id not in latent_store:
        raise HTTPException(status_code=404, detail="Session not found. Generate a model first.")

    state = latent_store[session_id]
    session_dir = get_session_dir(session_id)

    async with gpu_lock:
        try:
            def do_export():
                import torch
                import gc

                # Purge absolutely everything to guarantee 31GB is available for to_glb
                with torch.no_grad():
                    force_offload_all()

                dev = MESH_DEVICE
                with torch.no_grad(), torch.cuda.device(dev):
                    allocated = torch.cuda.memory_allocated(dev) / 1e9
                    print(f"\n[MEMORY] Before to_glb on {dev}: {allocated:.2f} GB allocated.")

                    verts = state['verts'].to(dev)
                    faces = state['faces'].to(dev)
                    attrs = state['attrs'].to(dev)
                    coords = state['coords'].to(dev)

                    print(f"[GLB] Exporting on {dev}, decimation={decimation_target}, tex_size={texture_size}")

                    try:
                        glb = o_voxel.postprocess.to_glb(
                            vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                            attr_layout=state['pbr_layout'], grid_size=state['res'], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=decimation_target, texture_size=texture_size,
                            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
                        )
                    except Exception as e:
                        if "out of memory" in str(e).lower() or "oom" in str(e).lower() or "allocate" in str(e).lower():
                            print(f"⚠ OOM detected in to_glb with remesh=True! Retrying with remesh=False (Fallback Mode)...")
                            # Wipe the failed tensors and free memory
                            del verts, faces, attrs, coords
                            gc.collect()
                            torch.cuda.empty_cache()

                            # Reload pure CPU tensors to GPU for the retry
                            verts = state['verts'].to(dev)
                            faces = state['faces'].to(dev)
                            attrs = state['attrs'].to(dev)
                            coords = state['coords'].to(dev)

                            # Retry without the memory-intensive cumesh remeshing pass
                            glb = o_voxel.postprocess.to_glb(
                                vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                                attr_layout=state['pbr_layout'], grid_size=state['res'], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                decimation_target=decimation_target, texture_size=texture_size,
                                remesh=False, use_tqdm=True,
                            )
                        else:
                            raise e

                del verts, faces, attrs, coords
                gc.collect()
                torch.cuda.empty_cache()

                ts = make_timestamp()
                glb_path = session_dir / f"sample_{ts}.glb"
                glb.export(str(glb_path))

                del glb
                gc.collect()
                return str(glb_path)

            glb_path = await asyncio.get_event_loop().run_in_executor(None, do_export)
            restore_envmaps()

            return FileResponse(glb_path, media_type="model/gltf-binary", filename=Path(glb_path).name)

        except Exception as e:
            force_offload_all()
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Extract OBJ (ZIP with .obj + .mtl + textures)
# ---------------------------------------------------------------------------
@app.post("/api/extract-obj")
async def extract_obj(
    session_id: str = Form(...),
    decimation_target: int = Form(500000),
    texture_size: int = Form(2048),
):
    if session_id not in latent_store:
        raise HTTPException(status_code=404, detail="Session not found. Generate a model first.")

    state = latent_store[session_id]
    session_dir = get_session_dir(session_id)

    async with gpu_lock:
        try:
            def do_export():
                import torch
                import gc

                # Purge absolutely everything to guarantee 31GB is available for to_glb
                with torch.no_grad():
                    force_offload_all()

                dev = MESH_DEVICE
                with torch.no_grad(), torch.cuda.device(dev):
                    allocated = torch.cuda.memory_allocated(dev) / 1e9
                    print(f"\n[MEMORY] Before to_glb on {dev}: {allocated:.2f} GB allocated.")

                    verts = state['verts'].to(dev)
                    faces = state['faces'].to(dev)
                    attrs = state['attrs'].to(dev)
                    coords = state['coords'].to(dev)

                    print(f"[OBJ] Exporting on {dev}, decimation={decimation_target}, tex_size={texture_size}")
                    try:
                        textured_mesh = o_voxel.postprocess.to_glb(
                            vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                            attr_layout=state['pbr_layout'], grid_size=state['res'], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=decimation_target, texture_size=texture_size,
                            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
                        )
                    except Exception as e:
                        if "out of memory" in str(e).lower() or "oom" in str(e).lower() or "allocate" in str(e).lower():
                            print(f"⚠ OOM detected in to_glb with remesh=True! Retrying with remesh=False (Fallback Mode)...")
                            del verts, faces, attrs, coords
                            gc.collect()
                            torch.cuda.empty_cache()

                            verts = state['verts'].to(dev)
                            faces = state['faces'].to(dev)
                            attrs = state['attrs'].to(dev)
                            coords = state['coords'].to(dev)

                            textured_mesh = o_voxel.postprocess.to_glb(
                                vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                                attr_layout=state['pbr_layout'], grid_size=state['res'], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                decimation_target=decimation_target, texture_size=texture_size,
                                remesh=False, use_tqdm=True,
                            )
                        else:
                            raise e

                del verts, faces, attrs, coords
                gc.collect()
                torch.cuda.empty_cache()

                ts = make_timestamp()
                obj_dir = session_dir / f"obj_{ts}"
                obj_dir.mkdir(exist_ok=True)

                obj_path = obj_dir / "model.obj"
                textured_mesh.export(str(obj_path), file_type='obj', include_texture=True)

                del textured_mesh
                gc.collect()

                zip_path = session_dir / f"model_{ts}.zip"
                with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                    for f in obj_dir.iterdir():
                        zf.write(str(f), f.name)

                shutil.rmtree(str(obj_dir), ignore_errors=True)
                return str(zip_path)

            zip_path = await asyncio.get_event_loop().run_in_executor(None, do_export)
            restore_envmaps()

            return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name)

        except Exception as e:
            force_offload_all()
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Convert GLB to OBJ (no GPU required)
# ---------------------------------------------------------------------------
@app.post("/api/convert-to-obj")
async def convert_to_obj(glb_file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = get_session_dir(session_id)

    try:
        glb_data = await glb_file.read()
        glb_tmp = session_dir / "input.glb"
        glb_tmp.write_bytes(glb_data)

        def do_convert():
            mesh = trimesh.load(str(glb_tmp))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.to_mesh()

            ts = make_timestamp()
            obj_dir = session_dir / f"obj_{ts}"
            obj_dir.mkdir(exist_ok=True)

            obj_path = obj_dir / "model.obj"
            mesh.export(str(obj_path), file_type='obj', include_texture=True)

            zip_path = session_dir / f"model_{ts}.zip"
            with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in obj_dir.iterdir():
                    zf.write(str(f), f.name)

            shutil.rmtree(str(obj_dir), ignore_errors=True)
            return str(zip_path)

        zip_path = await asyncio.get_event_loop().run_in_executor(None, do_convert)
        return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Re-Texturing
# ---------------------------------------------------------------------------
@app.post("/api/texturing")
async def texturing(
    mesh_file: UploadFile = File(...),
    image: UploadFile = File(...),
    seed: int = Form(11456),
    randomize_seed: bool = Form(False),
    resolution: str = Form("1024"),
    texture_size: int = Form(2048),
    tex_guidance_strength: float = Form(1.0),
    tex_guidance_rescale: float = Form(0.0),
    tex_sampling_steps: int = Form(12),
    tex_rescale_t: float = Form(3.0),
):
    if tex_pipeline is None:
        raise HTTPException(status_code=501, detail="Texturing pipeline not available")

    if randomize_seed:
        seed = int(np.random.randint(0, MAX_SEED))

    session_id = str(uuid.uuid4())
    session_dir = get_session_dir(session_id)

    mesh_data = await mesh_file.read()
    suffix = Path(mesh_file.filename or "mesh.glb").suffix
    mesh_tmp = session_dir / f"input_mesh{suffix}"
    mesh_tmp.write_bytes(mesh_data)

    mesh_obj = trimesh.load(str(mesh_tmp))
    if isinstance(mesh_obj, trimesh.Scene):
        mesh_obj = mesh_obj.to_mesh()

    pil_image = load_pil_image(image)
    pil_image = tex_pipeline.preprocess_image(pil_image)

    async with gpu_lock:
        try:
            def do_texture():
                import torch
                import gc
                with torch.no_grad():
                    force_offload_all()
                    tex_pipeline._device = GENERATION_DEVICE

                    output = tex_pipeline.run(
                        mesh_obj, pil_image, seed=seed, preprocess_image=False,
                        tex_slat_sampler_params={"steps": tex_sampling_steps, "guidance_strength": tex_guidance_strength, "guidance_rescale": tex_guidance_rescale, "rescale_t": tex_rescale_t},
                        resolution=int(resolution), texture_size=texture_size,
                    )

                    ts = make_timestamp()
                    glb_path = session_dir / f"textured_{ts}.glb"
                    output.export(str(glb_path))

                    del output
                    gc.collect()
                    torch.cuda.empty_cache()

                force_offload_all()
                return str(glb_path)

            glb_path = await asyncio.get_event_loop().run_in_executor(None, do_texture)
            return FileResponse(glb_path, media_type="model/gltf-binary", filename=Path(glb_path).name)

        except Exception as e:
            force_offload_all()
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Example images list
# ---------------------------------------------------------------------------
@app.get("/api/examples")
async def list_examples():
    examples_dir = ASSETS_DIR / "example_image"
    if not examples_dir.exists():
        return {"examples": []}
    files = sorted([f.name for f in examples_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')])
    return {"examples": files}


@app.get("/api/examples/{filename}")
async def get_example(filename: str):
    filepath = ASSETS_DIR / "example_image" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(filepath))


# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------
if WEBUI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEBUI_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def index():
        return {"message": "Frontend not built. Run 'npm run build' in webui/ or use 'npm run dev' for development."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    port = int(os.environ.get("TRELLIS_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
