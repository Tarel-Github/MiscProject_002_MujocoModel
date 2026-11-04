"""Step 1 (MuJoCo Model Studio env): dump the procedural face as raw arrays for Blender.

    python examples/blender/export_face_npz.py face.npz
Blink/eye-wide are bone motions in the procedural rig; here they are baked into shape keys
so the Blender asset demonstrates the pure shape-key (blend shape) workflow.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np  # noqa: E402

from mjstudio.library.face import build_face_rig  # noqa: E402

rig = build_face_rig("face", n_theta=104, n_phi=80)
keys = {}
for p in ("smile", "blink_left", "blink_right", "eye_wide_left", "eye_wide_right", "brow_raise_left",
          "brow_raise_right", "brow_lower_left", "brow_lower_right", "mouth_smile_left", "mouth_smile_right",
          "mouth_frown_left", "mouth_frown_right", "lips_part", "lip_pucker", "cheek_puff_left", "cheek_puff_right"):
    rig.reset()
    rig.set(p, 1.0)
    keys[p] = rig.local_vertices() - rig.V0
rig.reset()
np.savez_compressed(sys.argv[1], V=rig.V0, F=rig.F, jaw=rig.jaw_w, names=np.array(list(keys)),
                    **{f"k{i}": d for i, d in enumerate(keys.values())})
print("vertices", len(rig.V0), "faces", len(rig.F), "shape keys", len(keys))
