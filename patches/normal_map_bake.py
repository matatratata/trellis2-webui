"""
Patch: Bake normal map texture in o_voxel.postprocess.to_glb()

The original to_glb() computes vertex normals and uses them as per-vertex
attributes on the output trimesh, but does NOT bake them into a normal map
texture. This means the normalTexture field on the PBR material is always
None, and OBJ exports have no normal map.

This patch adds GPU-accelerated normal map baking using the existing
nvdiffrast UV rasterization pipeline (same quality as basecolor/metallic/
roughness), with proper coordinate-system conversion and seam inpainting.

The patch targets the o-voxel source BEFORE pip install. It is applied by
vastai_setup.sh during provisioning.

Can also be applied to an already-installed copy by passing the site-packages
path to o_voxel as the argument.
"""
import sys
import os


def patch(ovoxel_dir):
    target = os.path.join(ovoxel_dir, 'o_voxel', 'postprocess.py')
    if not os.path.exists(target):
        print(f"  [SKIP] {target} not found")
        return

    src = open(target).read()
    changed = False

    # ── Patch 1: Bake normals into texture after attribute sampling ──
    # Find the spot right after the grid_sample_3d call where attrs are filled
    old_after_attrs = (
        "    )\n"
        "    if use_tqdm:\n"
        "        pbar.update(1)\n"
        "    if verbose:\n"
        "        print(\"Done\")\n"
        "    \n"
        "    # --- Texture Post-Processing & Material Construction ---"
    )

    new_after_attrs = (
        "    )\n"
        "    \n"
        "    # --- Bake vertex normals into a normal-map texture ---\n"
        "    # Interpolate per-vertex normals across UV-space, same rasterization as attrs\n"
        "    normal_map_tex = dr.interpolate(out_normals.unsqueeze(0), rast, out_faces)[0][0]\n"
        "    # Apply the same coordinate-system swap done to vertex normals below (Y<->Z, -Y)\n"
        "    normal_map_tex = normal_map_tex.clone()\n"
        "    normal_map_tex_y = normal_map_tex[..., 1].clone()\n"
        "    normal_map_tex_z = normal_map_tex[..., 2].clone()\n"
        "    normal_map_tex[..., 1] = normal_map_tex_z\n"
        "    normal_map_tex[..., 2] = -normal_map_tex_y\n"
        "    # Re-normalise after interpolation\n"
        "    nrm_len = normal_map_tex.norm(dim=-1, keepdim=True).clamp(min=1e-8)\n"
        "    normal_map_tex = normal_map_tex / nrm_len\n"
        "    # Remap [-1,1] -> [0,255]\n"
        "    normal_map_np = np.clip((normal_map_tex.cpu().numpy() * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)\n"
        "\n"
        "    if use_tqdm:\n"
        "        pbar.update(1)\n"
        "    if verbose:\n"
        "        print(\"Done\")\n"
        "    \n"
        "    # --- Texture Post-Processing & Material Construction ---"
    )

    if "# --- Bake vertex normals into a normal-map" in src:
        print("  [SKIP] Normal map bake already applied")
    elif old_after_attrs in src:
        src = src.replace(old_after_attrs, new_after_attrs)
        changed = True
        print("  Patched to_glb: added normal map baking via nvdiffrast")
    else:
        print("  [WARN] Could not find attribute sampling pattern to patch")

    # ── Patch 2: Inpaint the normal map alongside other textures ──
    old_inpaint = (
        "    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]\n"
        "    \n"
        "    # Create PBR material"
    )

    new_inpaint = (
        "    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]\n"
        "    # Inpaint normal map (default flat normal = (128,128,255) for empty regions)\n"
        "    normal_map_np[~mask] = [128, 128, 255]  # Set empty to flat normal before inpaint\n"
        "    normal_map_np = cv2.inpaint(normal_map_np, mask_inv, 3, cv2.INPAINT_TELEA)\n"
        "    \n"
        "    # Create PBR material (with normal map)"
    )

    if "Inpaint normal map" in src:
        print("  [SKIP] Normal map inpaint already applied")
    elif old_inpaint in src:
        src = src.replace(old_inpaint, new_inpaint)
        changed = True
        print("  Patched to_glb: added normal map inpainting")
    else:
        print("  [WARN] Could not find inpainting pattern to patch")

    # ── Patch 3: Add normalTexture to PBR material constructor ──
    old_material = (
        "        metallicFactor=1.0,\n"
        "        roughnessFactor=1.0,\n"
        "        alphaMode=alpha_mode,"
    )

    new_material = (
        "        metallicFactor=1.0,\n"
        "        roughnessFactor=1.0,\n"
        "        normalTexture=Image.fromarray(normal_map_np),\n"
        "        alphaMode=alpha_mode,"
    )

    if "normalTexture=Image.fromarray(normal_map_np)" in src:
        print("  [SKIP] normalTexture already in PBR material")
    elif old_material in src:
        src = src.replace(old_material, new_material)
        changed = True
        print("  Patched to_glb: added normalTexture to PBR material")
    else:
        print("  [WARN] Could not find PBR material pattern to patch")

    if changed:
        open(target, 'w').write(src)
        print("  ✅ Normal map bake patches applied to postprocess.py")
    else:
        print("  No changes made")


if __name__ == '__main__':
    ovoxel_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/extensions/o-voxel'
    patch(ovoxel_dir)
