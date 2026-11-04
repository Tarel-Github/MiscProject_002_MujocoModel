"""Step 2 (Blender / bpy): build the face in Blender and export it with Blender's glTF exporter.

    blender -b -P examples/blender/make_face_glb.py -- face.npz face_blender.glb
    (or with the 'bpy' module: python make_face_glb.py face.npz face_blender.glb)
Blender conventions: Z up, the character faces -Y (left = +X). Bones 'head' and 'jaw'
carry the skin weights; shape keys become glTF morph targets.
"""
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
src, dst = argv[0], argv[1]
z = np.load(src)
V_mj, F, jaw = z["V"], z["F"], z["jaw"]
names = [str(n) for n in z["names"]]


def to_blender(P):          # MuJoCo head frame (x fwd, y left, z up) -> Blender (x left, y back, z up)
    P = np.asarray(P, float)
    return np.c_[P[:, 1], -P[:, 0], P[:, 2]]


bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("Face")
mesh.from_pydata(to_blender(V_mj).tolist(), [], F.tolist())
for poly in mesh.polygons:          # smooth shading: the exporter then shares vertices between faces
    poly.use_smooth = True
mesh.update()
obj = bpy.data.objects.new("Face", mesh)
bpy.context.scene.collection.objects.link(obj)
obj.shape_key_add(name="Basis")
for i, n in enumerate(names):
    sk = obj.shape_key_add(name=n)
    co = to_blender(V_mj + z[f"k{i}"])
    sk.data.foreach_set("co", co.astype(np.float32).ravel())
# armature with head + jaw bones (jaw hinge at the temporomandibular joint)
arm_data = bpy.data.armatures.new("Rig")
arm = bpy.data.objects.new("Rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
hb = arm_data.edit_bones.new("head")
hb.head, hb.tail = (0, 0, -0.09), (0, 0, 0.12)
jb = arm_data.edit_bones.new("jaw")
jb.head, jb.tail = (0, -0.005, -0.035), (0, -0.075, -0.07)
jb.parent = hb
bpy.ops.object.mode_set(mode="OBJECT")
obj.parent = arm
mod = obj.modifiers.new("Armature", "ARMATURE")
mod.object = arm
vg_h = obj.vertex_groups.new(name="head")
vg_j = obj.vertex_groups.new(name="jaw")
for i, w in enumerate(jaw):
    if w < 1.0:
        vg_h.add([i], float(1.0 - w), "REPLACE")
    if w > 0.0:
        vg_j.add([i], float(w), "REPLACE")
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_morph=True, export_skins=True,
                          export_morph_normal=False, export_yup=True, export_apply=False)
print("exported", dst, "with", len(names), "shape keys")
