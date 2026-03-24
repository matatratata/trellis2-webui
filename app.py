import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"

import asyncio
import uuid
import shutil
import json
import io
import base64
import traceback
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
MAX_SEED = np.iinfo(np.int32).max

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

# In-memory state store for latent results (session_id -> state_dict)
latent_store: dict = {}

# Progress tracking
progress_store: dict = {}  # task_id -> {stage, step, total, done}

# GPU lock — only one generation at a time
gpu_lock = asyncio.Lock()

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
def pack_state(latents):
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy().tolist(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy().tolist(),
        'coords': shape_slat.coords.cpu().numpy().tolist(),
        'res': res,
    }


def unpack_state(state: dict):
    shape_slat = SparseTensor(
        feats=torch.tensor(state['shape_slat_feats'], dtype=torch.float32).cuda(),
        coords=torch.tensor(state['coords'], dtype=torch.int32).cuda(),
    )
    tex_slat = shape_slat.replace(torch.tensor(state['tex_slat_feats'], dtype=torch.float32).cuda())
    return shape_slat, tex_slat, state['res']


def load_pil_image(file: UploadFile) -> Image.Image:
    """Read an UploadFile into a PIL Image."""
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
    """Render snapshot images of a mesh and return as base64 dict."""
    mesh.simplify(16777216)  # nvdiffrast limit
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
    pipeline.cuda()

    try:
        tex_pipeline = Trellis2TexturingPipeline.from_pretrained(
            MODEL_PATH, config_file="texturing_pipeline.json"
        )
        tex_pipeline.cuda()
    except Exception:
        print("⚠ Texturing pipeline not available (missing texturing_pipeline.json)")
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
            progress_store[task_id] = {"stage": "Generating 3D", "step": 10, "total": 100, "done": False}

            outputs, latents = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipeline.run(
                    pil_image,
                    seed=seed,
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "guidance_strength": ss_guidance_strength,
                        "guidance_rescale": ss_guidance_rescale,
                        "rescale_t": ss_rescale_t,
                    },
                    shape_slat_sampler_params={
                        "steps": shape_sampling_steps,
                        "guidance_strength": shape_guidance_strength,
                        "guidance_rescale": shape_guidance_rescale,
                        "rescale_t": shape_rescale_t,
                    },
                    tex_slat_sampler_params={
                        "steps": tex_sampling_steps,
                        "guidance_strength": tex_guidance_strength,
                        "guidance_rescale": tex_guidance_rescale,
                        "rescale_t": tex_rescale_t,
                    },
                    pipeline_type=pipeline_type,
                    return_latent=True,
                )
            )

            progress_store[task_id] = {"stage": "Rendering preview", "step": 80, "total": 100, "done": False}

            mesh = outputs[0]
            previews = await asyncio.get_event_loop().run_in_executor(
                None, lambda: render_preview_images(mesh, envmap)
            )

            # Store latents for later GLB extraction
            state = pack_state(latents)
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

    # Parse multipart form to get all image files
    form = await request.form()
    image_files = []
    for key, value in form.multi_items():
        if key.startswith("image"):
            data = await value.read()
            pil = Image.open(io.BytesIO(data)).convert("RGBA")
            pil = pipeline.preprocess_image(pil)
            image_files.append(pil)

    if len(image_files) < 2:
        raise HTTPException(status_code=400, detail="Multi-view requires at least 2 images")
    if len(image_files) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 images for multi-view")

    pipeline_type = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}.get(resolution, "1024_cascade")

    progress_store[task_id] = {"stage": "Starting", "step": 0, "total": 100, "done": False}

    async with gpu_lock:
        try:
            progress_store[task_id] = {"stage": "Generating 3D (multi-view)", "step": 10, "total": 100, "done": False}

            outputs, latents = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipeline.run_multiview(
                    image_files,
                    seed=seed,
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "guidance_strength": ss_guidance_strength,
                        "guidance_rescale": ss_guidance_rescale,
                        "rescale_t": ss_rescale_t,
                    },
                    shape_slat_sampler_params={
                        "steps": shape_sampling_steps,
                        "guidance_strength": shape_guidance_strength,
                        "guidance_rescale": shape_guidance_rescale,
                        "rescale_t": shape_rescale_t,
                    },
                    tex_slat_sampler_params={
                        "steps": tex_sampling_steps,
                        "guidance_strength": tex_guidance_strength,
                        "guidance_rescale": tex_guidance_rescale,
                        "rescale_t": tex_rescale_t,
                    },
                    pipeline_type=pipeline_type,
                    return_latent=True,
                )
            )

            progress_store[task_id] = {"stage": "Rendering preview", "step": 80, "total": 100, "done": False}

            mesh = outputs[0]
            previews = await asyncio.get_event_loop().run_in_executor(
                None, lambda: render_preview_images(mesh, envmap)
            )

            state = pack_state(latents)
            latent_store[session_id] = state

            progress_store[task_id] = {"stage": "Done", "step": 100, "total": 100, "done": True}

            return JSONResponse({
                "task_id": task_id,
                "session_id": session_id,
                "seed": seed,
                "previews": previews,
                "num_views": len(image_files),
            })

        except Exception as e:
            progress_store[task_id] = {"stage": f"Error: {str(e)}", "step": 0, "total": 100, "done": True}
            traceback.print_exc()
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
            shape_slat, tex_slat, res = unpack_state(state)
            mesh = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pipeline.decode_latent(shape_slat, tex_slat, res)[0]
            )

            # Free cuda:0 to release VRAM (important at 1536 resolution)
            pipeline.cpu()
            if tex_pipeline is not None:
                tex_pipeline.cpu()
            # Move envmaps off GPU temporarily
            for k, v in envmap.items():
                v.image = v.image.cpu()
                if hasattr(v, '_nvdiffrec_envlight'):
                    del v._nvdiffrec_envlight
            del shape_slat, tex_slat
            import gc; gc.collect()
            torch.cuda.empty_cache()

            def do_export():
                # Move mesh data to the mesh-processing GPU
                dev = MESH_DEVICE
                verts = mesh.vertices.to(dev) if hasattr(mesh.vertices, 'to') else mesh.vertices
                faces = mesh.faces.to(dev) if hasattr(mesh.faces, 'to') else mesh.faces
                attrs = mesh.attrs.to(dev) if hasattr(mesh.attrs, 'to') else mesh.attrs
                coords = mesh.coords.to(dev) if hasattr(mesh.coords, 'to') else mesh.coords

                print(f"[GLB] Exporting on {dev}, decimation={decimation_target}, tex_size={texture_size}")
                with torch.cuda.device(dev):
                    glb = o_voxel.postprocess.to_glb(
                        vertices=verts,
                        faces=faces,
                        attr_volume=attrs,
                        coords=coords,
                        attr_layout=pipeline.pbr_attr_layout,
                        grid_size=res,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target=decimation_target,
                        texture_size=texture_size,
                        remesh=True,
                        remesh_band=1,
                        remesh_project=0,
                        use_tqdm=True,
                    )
                ts = make_timestamp()
                glb_path = session_dir / f"sample_{ts}.glb"
                glb.export(str(glb_path))
                torch.cuda.empty_cache()
                return str(glb_path)

            glb_path = await asyncio.get_event_loop().run_in_executor(None, do_export)

            # Restore pipeline and envmaps to cuda:0
            pipeline.cuda()
            if tex_pipeline is not None:
                tex_pipeline.cuda()
            for k, v in envmap.items():
                v.image = v.image.cuda()

            return FileResponse(glb_path, media_type="model/gltf-binary", filename=Path(glb_path).name)

        except Exception as e:
            try:
                pipeline.cuda()
                if tex_pipeline is not None:
                    tex_pipeline.cuda()
                for k, v in envmap.items():
                    v.image = v.image.cuda()
            except Exception:
                pass
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Re-Texturing
# ---------------------------------------------------------------------------
@app.post("/api/texturing")
async def texturing(
    mesh_file: UploadFile = File(...),
    image: UploadFile = File(...),
    seed: int = Form(0),
    randomize_seed: bool = Form(True),
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

    # Save uploaded mesh to temp file
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
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: tex_pipeline.run(
                    mesh_obj,
                    pil_image,
                    seed=seed,
                    preprocess_image=False,
                    tex_slat_sampler_params={
                        "steps": tex_sampling_steps,
                        "guidance_strength": tex_guidance_strength,
                        "guidance_rescale": tex_guidance_rescale,
                        "rescale_t": tex_rescale_t,
                    },
                    resolution=int(resolution),
                    texture_size=texture_size,
                )
            )

            ts = make_timestamp()
            glb_path = session_dir / f"textured_{ts}.glb"
            output.export(str(glb_path))
            torch.cuda.empty_cache()

            return FileResponse(str(glb_path), media_type="model/gltf-binary", filename=glb_path.name)

        except Exception as e:
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
# Try dist first (production build), then fall back to dev serving
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
