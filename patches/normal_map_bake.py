"""
Patch: Bake TANGENT-SPACE normal map in o_voxel.postprocess.to_glb()

The original to_glb() computes vertex normals and uses them as per-vertex
attributes on the output trimesh, but does NOT bake them into a normal map
texture. This means the normalTexture field on the PBR material is always
None, and OBJ exports have no normal map.

This patch adds GPU-accelerated tangent-space normal map baking:
  1. Captures rast_db derivatives from nvdiffrast rasterization
  2. Computes area-weighted vertex normals on the original high-res mesh
  3. Builds a TBN basis per-texel from UV-space position derivatives
  4. Projects high-res normals into tangent space for a correct blue-dominant map

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

    # ── Patch 1: Capture rast_db in chunked rasterization loop ──
    old_rast_init = (
        "    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)\n"
        "    \n"
        "    # Rasterize in chunks to save memory\n"
        "    for i in range(0, out_faces.shape[0], 100000):\n"
        "        rast_chunk, _ = dr.rasterize(\n"
    )

    new_rast_init = (
        "    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)\n"
        "    rast_deriv = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)\n"
        "    \n"
        "    # Rasterize in chunks to save memory\n"
        "    for i in range(0, out_faces.shape[0], 100000):\n"
        "        rast_chunk, rast_db_chunk = dr.rasterize(\n"
    )

    if "rast_deriv" in src:
        print("  [SKIP] rast_deriv already present")
    elif old_rast_init in src:
        src = src.replace(old_rast_init, new_rast_init)
        # Also add rast_deriv accumulation after rast accumulation
        old_rast_acc = "        rast = torch.where(mask_chunk, rast_chunk, rast)\n"
        new_rast_acc = (
            "        rast = torch.where(mask_chunk, rast_chunk, rast)\n"
            "        rast_deriv = torch.where(mask_chunk, rast_db_chunk, rast_deriv)\n"
        )
        src = src.replace(old_rast_acc, new_rast_acc)
        changed = True
        print("  Patched rasterization loop: capturing rast_db derivatives")
    else:
        print("  [WARN] Could not find rasterization init pattern to patch")

    # ── Patch 2: Bake tangent-space normals after attribute sampling ──
    old_after_attrs = (
        "    )\n"
        "    if use_tqdm:\n"
        "        pbar.update(1)\n"
        "    if verbose:\n"
        "        print(\"Done\")\n"
        "    \n"
        "    # --- Texture Post-Processing & Material Construction ---"
    )

    tangent_space_bake = (
        "    )\n"
        "    \n"
        "    # --- Bake TANGENT-SPACE normal map ---\n"
        "    # Compute area-weighted vertex normals for the original high-res mesh,\n"
        "    # then project into a per-texel tangent frame built from UV derivatives.\n"
        "    _v0, _v1, _v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]\n"
        "    _fn = torch.cross(_v1 - _v0, _v2 - _v0, dim=-1)\n"
        "    orig_vnormals = torch.zeros_like(vertices)\n"
        "    orig_vnormals.scatter_add_(0, faces[:, 0].unsqueeze(1).expand(-1, 3), _fn)\n"
        "    orig_vnormals.scatter_add_(0, faces[:, 1].unsqueeze(1).expand(-1, 3), _fn)\n"
        "    orig_vnormals.scatter_add_(0, faces[:, 2].unsqueeze(1).expand(-1, 3), _fn)\n"
        "    orig_vnormals = orig_vnormals / orig_vnormals.norm(dim=-1, keepdim=True).clamp(min=1e-8)\n"
        "    del _v0, _v1, _v2, _fn\n"
        "    _tri_nrm = orig_vnormals[faces[face_id.long()]]\n"
        "    hires_valid = (_tri_nrm * uvw.unsqueeze(-1)).sum(dim=1)\n"
        "    hires_valid = hires_valid / hires_valid.norm(dim=-1, keepdim=True).clamp(min=1e-8)\n"
        "    del _tri_nrm, orig_vnormals\n"
        "    hires_nrm = torch.zeros(texture_size, texture_size, 3, device='cuda')\n"
        "    hires_nrm[mask] = hires_valid\n"
        "    del hires_valid\n"
        "    simp_nrm = dr.interpolate(out_normals.unsqueeze(0), rast, out_faces)[0][0]\n"
        "    simp_nrm = simp_nrm / simp_nrm.norm(dim=-1, keepdim=True).clamp(min=1e-8)\n"
        "    _, pos_db = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces, rast_db=rast_deriv, diff_attrs='all')\n"
        "    T_raw = pos_db[0, ..., 0::2]\n"
        "    B_raw = pos_db[0, ..., 1::2]\n"
        "    del pos_db\n"
        "    N = simp_nrm\n"
        "    T = T_raw - (T_raw * N).sum(-1, keepdim=True) * N\n"
        "    T = T / T.norm(dim=-1, keepdim=True).clamp(min=1e-8)\n"
        "    B = torch.cross(N, T, dim=-1)\n"
        "    hand = (B_raw * B).sum(-1, keepdim=True).sign()\n"
        "    hand[hand == 0] = 1.0\n"
        "    B = B * hand\n"
        "    del T_raw, B_raw, hand\n"
        "    ts = torch.stack([(hires_nrm * T).sum(-1), (hires_nrm * B).sum(-1), (hires_nrm * N).sum(-1)], dim=-1)\n"
        "    ts = ts / ts.norm(dim=-1, keepdim=True).clamp(min=1e-8)\n"
        "    del T, B, N, simp_nrm, hires_nrm\n"
        "    ts[~mask] = torch.tensor([0.0, 0.0, 1.0], device='cuda')\n"
        "    normal_map_np = np.clip((ts.cpu().numpy() * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)\n"
        "    del ts\n"
        "\n"
        "    if use_tqdm:\n"
        "        pbar.update(1)\n"
        "    if verbose:\n"
        "        print(\"Done\")\n"
        "    \n"
        "    # --- Texture Post-Processing & Material Construction ---"
    )

    if "# --- Bake TANGENT-SPACE normal map ---" in src:
        print("  [SKIP] Tangent-space normal map bake already applied")
    elif "# --- Bake vertex normals into a normal-map" in src:
        print("  [WARN] Old world-space normal map patch detected — manual update needed")
    elif old_after_attrs in src:
        src = src.replace(old_after_attrs, tangent_space_bake)
        changed = True
        print("  Patched to_glb: added tangent-space normal map baking")
    else:
        print("  [WARN] Could not find attribute sampling pattern to patch")

    # ── Patch 3: Inpaint the normal map alongside other textures ──
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

    # ── Patch 4: Add normalTexture to PBR material constructor ──
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
        print("  ✅ Tangent-space normal map patches applied to postprocess.py")
    else:
        print("  No changes made")


if __name__ == '__main__':
    ovoxel_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/extensions/o-voxel'
    patch(ovoxel_dir)
