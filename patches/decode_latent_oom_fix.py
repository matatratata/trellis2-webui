"""
Patch: OOM fix for decode_latent / fill_holes in
trellis2/pipelines/trellis2_image_to_3d.py

The decoder leaves large intermediate tensors (subs) alive on the GPU
while cumesh's fill_holes() tries to allocate. Combined with conditioning
tensors (cond_512, cond_1024, coords) still in scope during run()/
run_multiview(), this easily saturates 48 GB cards.

This patch:
  1. Frees `subs` + flushes CUDA cache before fill_holes in decode_latent()
  2. Wraps fill_holes() in try/except so OOM is non-fatal (cosmetic op)
  3. Frees conditioning tensors before decode_latent() in run()

Applied by vastai_setup.sh during provisioning.
"""
import sys
import os


def patch(trellis_dir):
    target = os.path.join(trellis_dir, 'trellis2', 'pipelines', 'trellis2_image_to_3d.py')
    if not os.path.exists(target):
        print(f"  [SKIP] {target} not found")
        return

    src = open(target).read()

    changed = False

    # ── Patch 1: decode_latent — free subs + wrap fill_holes ──
    old_decode = (
        "        meshes, subs = self.decode_shape_slat(shape_slat, resolution)\n"
        "        tex_voxels = self.decode_tex_slat(tex_slat, subs)\n"
        "        out_mesh = []\n"
        "        for m, v in zip(meshes, tex_voxels):\n"
        "            m.fill_holes()"
    )
    new_decode = (
        "        meshes, subs = self.decode_shape_slat(shape_slat, resolution)\n"
        "        tex_voxels = self.decode_tex_slat(tex_slat, subs)\n"
        "\n"
        "        # [OOM fix] Free decoder intermediates before cumesh fill_holes\n"
        "        del subs\n"
        "        import gc as _gc; _gc.collect()\n"
        "        torch.cuda.empty_cache()\n"
        "\n"
        "        out_mesh = []\n"
        "        for m, v in zip(meshes, tex_voxels):\n"
        "            try:\n"
        "                m.fill_holes()\n"
        "            except Exception as _e:\n"
        "                print(f'⚠ Warning: cumesh fill_holes failed (OOM?): {_e}')"
    )

    if "# [OOM fix] Free decoder intermediates" in src:
        print("  [SKIP] decode_latent OOM fix already applied")
    elif old_decode in src:
        src = src.replace(old_decode, new_decode)
        changed = True
        print("  Patched decode_latent: free subs + safe fill_holes")
    else:
        print("  [WARN] Could not find decode_latent pattern to patch")

    # ── Patch 2: run() — free conds before decode_latent ──
    old_run_tail = (
        "        torch.cuda.empty_cache()\n"
        "        out_mesh = self.decode_latent(shape_slat, tex_slat, res)\n"
        "        if return_latent:\n"
        "            return out_mesh, (shape_slat, tex_slat, res)\n"
        "        else:\n"
        "            return out_mesh"
    )
    new_run_tail = (
        "        # [OOM fix] Free conditioning tensors before memory-intensive decode\n"
        "        try:\n"
        "            del cond_512, cond_1024, coords\n"
        "        except NameError:\n"
        "            pass\n"
        "        import gc as _gc; _gc.collect()\n"
        "        torch.cuda.empty_cache()\n"
        "        out_mesh = self.decode_latent(shape_slat, tex_slat, res)\n"
        "        if return_latent:\n"
        "            return out_mesh, (shape_slat, tex_slat, res)\n"
        "        else:\n"
        "            return out_mesh"
    )

    if "# [OOM fix] Free conditioning tensors" in src:
        print("  [SKIP] run() OOM fix already applied")
    elif old_run_tail in src:
        # Replace ALL occurrences (run + run_multiview share the same tail)
        src = src.replace(old_run_tail, new_run_tail)
        changed = True
        count = src.count("# [OOM fix] Free conditioning tensors")
        print(f"  Patched run()/run_multiview(): free conds before decode ({count} occurrence(s))")
    else:
        print("  [WARN] Could not find run() tail pattern to patch")

    if changed:
        open(target, 'w').write(src)
        print("  ✅ OOM fixes applied to trellis2_image_to_3d.py")
    else:
        print("  No changes made")


if __name__ == '__main__':
    trellis_dir = sys.argv[1] if len(sys.argv) > 1 else '/workspace/TRELLIS.2'
    patch(trellis_dir)
