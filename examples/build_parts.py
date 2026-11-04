"""Build the anatomical library parts and save each as a standalone MJCF part file.

    python examples/build_parts.py        -> examples/parts/*.xml
Each file is a normal MuJoCo model (open it in the editor or in MuJoCo's viewer)
and can be assembled into other models with File > Import part.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import mujoco  # noqa: E402

from mjstudio.library import anatomy as A  # noqa: E402

OUT = HERE / "parts"
OUT.mkdir(exist_ok=True)
PARTS = {
    "hand_left": (A.build_hand, dict(side="left")),
    "hand_right": (A.build_hand, dict(side="right")),
    "foot_left": (A.build_foot, dict(side="left")),
    "foot_right": (A.build_foot, dict(side="right")),
    "foot_left_detailed_toes": (A.build_foot, dict(side="left", toes="detailed")),
    "head_eyes_jaw": (A.build_head, dict(jaw_slide=True)),
}
for name, (fn, kw) in PARTS.items():
    doc = A.make_part(fn, name, **kw)
    path = OUT / f"{name}.xml"
    doc.save(path)
    m = mujoco.MjModel.from_xml_path(str(path))
    print(f"{path.name:32s} bodies={m.nbody - 1:3d} joints={m.njnt:3d} actuators={m.nu:3d} sensors={m.nsensor:3d}")
