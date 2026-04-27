"""
Patch: Bake TANGENT-SPACE normal map in o_voxel.postprocess.to_glb()

Replaces analytically flat derivatives with smooth vertex tangents (MikkTSpace)
and fixes random dual-contouring winding noise using hemisphere alignment.

Improvements over v1:
  - Angle-weighted tangent accumulation (matches MikkTSpace weighting)
  - Iterative dilation + inpaint for robust seam padding
"""
import sys
import os
import re

def patch(ovoxel_dir):
    target = os.path.join(ovoxel_dir, 'o_voxel', 'postprocess.py')
    if not os.path.exists(target):
        print(f"  [SKIP] {target} not found")
        return
    src = open(target).read()
    changed = False

    # 1. Unify High-Res Mesh Winding (Fixes normal cancellation / rainbow noise)
    fill_holes_pattern = r"(mesh\.fill_holes\(max_hole_perimeter=3e-2\)\n)(.*?vertices, faces = mesh\.read\(\))"
    new_fill_holes = r"\1    mesh.remove_duplicate_faces()\n    mesh.repair_non_manifold_edges()\n    mesh.unify_face_orientations()\n\2"
    if "mesh.unify_face_orientations()" not in src[:src.find("vertices, faces = mesh.read()")]:
        src = re.sub(fill_holes_pattern, new_fill_holes, src, flags=re.DOTALL)
        changed = True

    # 2. Ensure simplified mesh is unified before unwrapping
    unwrap_pattern = r"(\s+)(out_vertices, out_faces, out_uvs, out_vmaps = mesh\.uv_unwrap\()"
    new_unwrap = r"\1mesh.unify_face_orientations()\n\1\2"
    if "mesh.unify_face_orientations()\n    out_vertices, out_faces" not in src:
        src = re.sub(unwrap_pattern, new_unwrap, src)
        changed = True

    # 3. Clean up the old rast_db tracking that ate VRAM and caused flat polygons
    if "rast_deriv" in src:
        src = re.sub(r"    rast_deriv = torch\.zeros.*?\n", "", src)
        src = src.replace("rast_chunk, rast_db_chunk = dr.rasterize(", "rast_chunk, _ = dr.rasterize(")
        src = re.sub(r"        rast_deriv = torch\.where\(mask_chunk, rast_db_chunk, rast_deriv\)\n", "", src)
        changed = True

    # 4. Smooth Tangent-Space Baking Block (angle-weighted MikkTSpace tangents)
    tangent_space_bake = r"""\1
    # --- Bake TANGENT-SPACE normal map (Smooth MikkTSpace-style, angle-weighted) ---
    _v0, _v1, _v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    _fn = torch.cross(_v1 - _v0, _v2 - _v0, dim=-1)
    orig_vnormals = torch.zeros_like(vertices)
    orig_vnormals.scatter_add_(0, faces[:, 0].unsqueeze(1).expand(-1, 3), _fn)
    orig_vnormals.scatter_add_(0, faces[:, 1].unsqueeze(1).expand(-1, 3), _fn)
    orig_vnormals.scatter_add_(0, faces[:, 2].unsqueeze(1).expand(-1, 3), _fn)
    orig_vnormals = orig_vnormals / orig_vnormals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    del _v0, _v1, _v2, _fn
    
    _tri_nrm = orig_vnormals[faces[face_id.long()]]
    hires_valid = (_tri_nrm * uvw.unsqueeze(-1)).sum(dim=1)
    hires_valid = hires_valid / hires_valid.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    del _tri_nrm, orig_vnormals
    
    hires_nrm = torch.zeros(texture_size, texture_size, 3, device='cuda', dtype=torch.float32)
    hires_nrm[mask] = hires_valid
    del hires_valid
    
    # Calculate angle-weighted smooth tangents at vertices (MikkTSpace-style)
    _v0, _v1, _v2 = out_vertices[out_faces[:, 0]], out_vertices[out_faces[:, 1]], out_vertices[out_faces[:, 2]]
    _uv0, _uv1, _uv2 = out_uvs[out_faces[:, 0]], out_uvs[out_faces[:, 1]], out_uvs[out_faces[:, 2]]
    dp1, dp2 = _v1 - _v0, _v2 - _v0
    duv1, duv2 = _uv1 - _uv0, _uv2 - _uv0
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    r = torch.where(det.abs() > 1e-8, 1.0 / det, torch.zeros_like(det))
    face_T = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * r.unsqueeze(1)
    face_B = (dp2 * duv1[:, 0:1] - dp1 * duv2[:, 0:1]) * r.unsqueeze(1)
    # Angle weights per corner vertex (MikkTSpace uses angle, not area)
    _e01, _e02 = _v1 - _v0, _v2 - _v0
    _e12, _e10 = _v2 - _v1, _v0 - _v1
    _e20, _e21 = _v0 - _v2, _v1 - _v2
    _cos0 = (_e01 * _e02).sum(-1) / (_e01.norm(dim=-1) * _e02.norm(dim=-1)).clamp(min=1e-8)
    _cos1 = (_e12 * _e10).sum(-1) / (_e12.norm(dim=-1) * _e10.norm(dim=-1)).clamp(min=1e-8)
    _cos2 = (_e20 * _e21).sum(-1) / (_e20.norm(dim=-1) * _e21.norm(dim=-1)).clamp(min=1e-8)
    _ang0 = torch.acos(_cos0.clamp(-1.0, 1.0)).unsqueeze(1)
    _ang1 = torch.acos(_cos1.clamp(-1.0, 1.0)).unsqueeze(1)
    _ang2 = torch.acos(_cos2.clamp(-1.0, 1.0)).unsqueeze(1)
    del _e01, _e02, _e12, _e10, _e20, _e21, _cos0, _cos1, _cos2
    del _v0, _v1, _v2, _uv0, _uv1, _uv2, dp1, dp2, duv1, duv2, det, r
    
    out_T, out_B = torch.zeros_like(out_vertices), torch.zeros_like(out_vertices)
    out_T.scatter_add_(0, out_faces[:, 0].unsqueeze(1).expand(-1, 3), face_T * _ang0)
    out_T.scatter_add_(0, out_faces[:, 1].unsqueeze(1).expand(-1, 3), face_T * _ang1)
    out_T.scatter_add_(0, out_faces[:, 2].unsqueeze(1).expand(-1, 3), face_T * _ang2)
    out_B.scatter_add_(0, out_faces[:, 0].unsqueeze(1).expand(-1, 3), face_B * _ang0)
    out_B.scatter_add_(0, out_faces[:, 1].unsqueeze(1).expand(-1, 3), face_B * _ang1)
    out_B.scatter_add_(0, out_faces[:, 2].unsqueeze(1).expand(-1, 3), face_B * _ang2)
    del face_T, face_B, _ang0, _ang1, _ang2
    out_T = out_T / out_T.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    out_B = out_B / out_B.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Interpolate smooth frame at each texel
    simp_N = dr.interpolate(out_normals.unsqueeze(0), rast, out_faces)[0][0]
    simp_T = dr.interpolate(out_T.unsqueeze(0), rast, out_faces)[0][0]
    simp_B = dr.interpolate(out_B.unsqueeze(0), rast, out_faces)[0][0]
    del out_T, out_B
    simp_N = simp_N / simp_N.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Safe Gram-Schmidt Orthogonalization
    T = simp_T - (simp_T * simp_N).sum(-1, keepdim=True) * simp_N
    T_norm = T.norm(dim=-1, keepdim=True)
    T = torch.where(T_norm > 1e-8, T / T_norm, torch.tensor([1.0, 0.0, 0.0], device='cuda', dtype=torch.float32))
    B = simp_B - (simp_B * simp_N).sum(-1, keepdim=True) * simp_N - (simp_B * T).sum(-1, keepdim=True) * T
    B_norm = B.norm(dim=-1, keepdim=True)
    B = torch.where(B_norm > 1e-8, B / B_norm, torch.cross(simp_N, T, dim=-1))
    
    # Hemispherical alignment: forces high-res normal to point outwards
    flip = (hires_nrm * simp_N).sum(dim=-1, keepdim=True).sign()
    flip = torch.where(flip == 0, torch.ones_like(flip), flip)
    hires_nrm = hires_nrm * flip
    
    # Map to tangent space (negating B compensates for the UV V-flip during export)
    ts = torch.stack([(hires_nrm * T).sum(-1), -(hires_nrm * B).sum(-1), (hires_nrm * simp_N).sum(-1)], dim=-1)
    ts = ts / ts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ts[~mask] = torch.tensor([0.0, 0.0, 1.0], device='cuda', dtype=torch.float32)
    normal_map_np = np.clip((ts.cpu().numpy() * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    \2"""

    if "# --- Bake TANGENT-SPACE normal map (Smooth MikkTSpace-style" not in src:
        # Replaces everything from the trilinear grid_sample output to the PBR construction
        bake_pattern = r"(mode='trilinear',\n\s+\)\n).*?(    # --- Texture Post-Processing & Material Construction ---)"
        new_src = re.sub(bake_pattern, tangent_space_bake, src, flags=re.DOTALL)
        if new_src != src:
            src = new_src
            changed = True
            print("  Patched to_glb: upgraded to robust smooth tangent-space baking")

    # 5. Iterative dilation + Inpaint Normal Map + Add to Material
    # Use iterative dilation (8 texels) before inpaint for robust seam padding.
    # Even for OBJ, renderers like Blender/Substance filter at texel boundaries.
    inpaint_pattern = r"(alpha = cv2\.inpaint.*?)\n(\s+# Create PBR material)"
    new_inpaint = (
        r"\1\n"
        r"    normal_map_np[~mask] = [128, 128, 255]\n"
        r"    _nmap_orig = normal_map_np.copy()\n"
        r"    _dilate_kern = np.ones((3, 3), dtype=np.uint8)\n"
        r"    for _ in range(4):\n"
        r"        normal_map_np = cv2.dilate(normal_map_np, _dilate_kern, iterations=1)\n"
        r"        normal_map_np[mask] = _nmap_orig[mask]\n"
        r"    normal_map_np = cv2.inpaint(normal_map_np, (~mask & (normal_map_np[..., 2] == 255)).astype(np.uint8), 3, cv2.INPAINT_TELEA)\n"
        r"    del _nmap_orig, _dilate_kern\n"
        r"\2"
    )
    if "normal_map_np[~mask] = [128, 128, 255]" not in src:
        src = re.sub(inpaint_pattern, new_inpaint, src)
        changed = True

    material_pattern = r"(roughnessFactor=1\.0,\n)(\s+)(alphaMode=alpha_mode,)"
    new_material = r"\1\2normalTexture=Image.fromarray(normal_map_np),\n\2\3"
    if "normalTexture=Image.fromarray(normal_map_np)" not in src:
        src = re.sub(material_pattern, new_material, src)
        changed = True

    if changed:
        open(target, 'w').write(src)
        print("    Tangent-space normal map patches successfully updated in postprocess.py")
    else:
        print("  No changes made (already up to date).")

if __name__ == '__main__':
    ovoxel_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/extensions/o-voxel'
    patch(ovoxel_dir)
