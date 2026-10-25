import bpy, glob, os, mathutils
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "cleargrasp-3d-models", "obj")
for f in sorted(glob.glob(root + "/*.obj")):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=f)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    mins = [min(v.co[i] for o in objs for v in o.data.vertices) for i in range(3)]
    maxs = [max(v.co[i] for o in objs for v in o.data.vertices) for i in range(3)]
    print("MODEL", os.path.basename(f), "parts", len(objs), "verts", sum(len(o.data.vertices) for o in objs), "dims", [round(b - a, 4) for a, b in zip(mins, maxs)], "min", [round(m, 3) for m in mins])
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "METAL"
prefs.get_devices()
print("DEVICES", [(d.name, d.type, d.use) for d in prefs.devices])
