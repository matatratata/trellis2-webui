"""
Patch: DINOv3ViTModel layer attribute compatibility
for trellis2/modules/image_feature_extractor.py

In transformers >= 5.x, DINOv3ViTModel wraps layers inside a
DINOv3ViTEncoder (self.model), so the layer list moved from
  model.layer  ->  model.model.layer

This patch makes extract_features() work with both old and new layouts.
Applied by vastai_setup.sh during provisioning.
"""
import sys
import os


def patch(trellis_dir):
    target = os.path.join(trellis_dir, 'trellis2', 'modules', 'image_feature_extractor.py')
    if not os.path.exists(target):
        print(f"  [SKIP] {target} not found")
        return

    src = open(target).read()

    # The safe replacement block
    new = (
        "# Compat: resolve layer list across transformers versions\n"
        "        _layers = getattr(self.model, 'layer', None)\n"
        "        if _layers is None:\n"
        "            _inner = getattr(self.model, 'model', self.model)\n"
        "            _layers = getattr(_inner, 'layer', None) or getattr(_inner, 'layers', None)\n"
        "        assert _layers is not None, f'Cannot find layer list in DINOv3 model: {type(self.model)}'\n"
        "        for i, layer_module in enumerate(_layers):"
    )

    # Already has the safe version?
    if "resolve layer list across transformers versions" in src:
        print("  [SKIP] already patched with safe version")
        return

    # Try original code first
    old_original = "for i, layer_module in enumerate(self.model.layer):"
    # Try broken patch (eager getattr fallback)
    old_broken = "_layers = getattr(self.model, 'layer', None) or getattr(self.model.model, 'layer', self.model.model.layers)"

    if old_original in src:
        src = src.replace(old_original, new)
    elif old_broken in src:
        # Replace everything from the broken compat comment through the for loop
        import re
        src = re.sub(
            r'# Compat:.*?\n\s+_layers = getattr\(self\.model.*?layers\)\n\s+for i, layer_module in enumerate\(_layers\):',
            new,
            src,
            flags=re.DOTALL,
        )
    else:
        print("  [SKIP] no known pattern found in image_feature_extractor.py")
        return

    open(target, 'w').write(src)
    print("  Patched image_feature_extractor.py for DINOv3 layer compat")


if __name__ == '__main__':
    trellis_dir = sys.argv[1] if len(sys.argv) > 1 else '/workspace/TRELLIS.2'
    patch(trellis_dir)
