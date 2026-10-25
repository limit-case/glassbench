"""Procedural ClearGrasp-like tabletop scenes, rendered headless with Cycles (Metal).

Run: blender -b -P tools/simulation/render_scene.py -- --variant d415|d435|l515 --seeds 0 1 2 --out synth/out [--ir-twin]

Per scene (synth/out/<variant>/<seed>/):
  rgb.png                visible light, transparent objects as glass / clear plastic
  gt.exr                 multilayer EXR from the opaque twin (all transparent objects swapped to opaque diffuse):
                         depth (Z), instance index (1 = transparent object, 2 = opaque distractor, 3 = table/wall), normal
  ir_left.png, ir_right.png   (stereo variants) projector-lit IR pair with the glass objects, rectified, parallel cameras
  ir_left_twin.png, ir_right_twin.png   (--ir-twin) same pair with the opaque twin objects
  behind.exr             (l515) depth with the transparent objects removed (what a glass-blind ToF would see)
  meta.json              seed, intrinsics, baseline, objects, materials, lights, timings
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import bpy
import mathutils
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MODELS = sorted(glob.glob(ROOT + "/datasets/cleargrasp-3d-models/obj/*.obj"))
PATTERN = ROOT + "/data/configs/dots.png"
RES = (640, 360)

# Sensor geometry. Stereo baselines: D415 55 mm, D435 50 mm (Intel D400 datasheet 337029-017 rev 023, Table 3-49).
# HFOV values used (69.4 / 85.2 deg) came from product pages; the datasheet's depth FOVs are 65 / 87 deg (69.4 is the D415 colour FOV);
# see data/fov_check.csv for the sensitivity (--hfov overrides). L515: intrinsics from datasets/transcg/camera_intrinsics (1280x720), scaled.
VARIANTS = {
    "d415": dict(kind="stereo", hfov_deg=69.4, baseline=0.055),
    "d435": dict(kind="stereo", hfov_deg=85.2, baseline=0.050),
    "l515": dict(kind="tof", fx=902.403, fy=902.803, cx=650.961, cy=365.441, full_w=1280, full_h=720),
}


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=ROOT + "/synth/out")
    ap.add_argument("--ir-twin", action="store_true", help="also render the IR pair of the opaque-twin scene")
    ap.add_argument("--res", type=int, nargs=2, default=list(RES))
    ap.add_argument("--rgb-samples", type=int, default=64)
    ap.add_argument("--ir-samples", type=int, default=96)
    ap.add_argument("--no-rgb", action="store_true")
    ap.add_argument("--hfov", type=float, default=None, help="override the stereo depth HFOV in degrees (FOV sensitivity check)")
    return ap.parse_args(argv)


# ------------------------------------------------------------------ helpers
def setup_engine(res):
    sc = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "METAL"
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.cycles.max_bounces = 16
    sc.cycles.transmission_bounces = 16
    sc.cycles.glossy_bounces = 8
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.caustics_reflective = True
    sc.cycles.caustics_refractive = True
    sc.cycles.use_light_tree = True
    return sc


def principled(name, base=(0.8, 0.8, 0.8, 1), rough=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = base
    b.inputs["Roughness"].default_value = rough
    return m, b


def table_material(rng):
    m, b = principled("table", rough=rng.uniform(0.4, 0.9))
    nt = m.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    s = rng.uniform(2, 14)
    mp.inputs["Scale"].default_value = (s, s * rng.uniform(0.3, 1.0), s)
    kind = rng.choice(["noise", "wave", "checker", "voronoi"])
    tex = nt.nodes.new({"noise": "ShaderNodeTexNoise", "wave": "ShaderNodeTexWave",
                        "checker": "ShaderNodeTexChecker", "voronoi": "ShaderNodeTexVoronoi"}[kind])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    c0 = [rng.uniform(0.02, 0.9) for _ in range(3)]
    c1 = [min(1.0, max(0.0, c + rng.uniform(-0.3, 0.3))) for c in c0]
    ramp.color_ramp.elements[0].color = (*c0, 1)
    ramp.color_ramp.elements[1].color = (*c1, 1)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    out = tex.outputs["Color"] if kind == "checker" else tex.outputs["Distance" if kind == "voronoi" else "Fac"]
    nt.links.new(out, ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m, kind


def glass_material(rng, plastic):
    tint = [rng.uniform(0.85, 1.0) for _ in range(3)]
    if plastic:
        m, b = principled("plastic", (*tint, 1), rng.uniform(0.05, 0.12))
        b.inputs["IOR"].default_value = rng.uniform(1.40, 1.50)
    else:
        m, b = principled("glass", (*tint, 1), rng.uniform(0.0, 0.03))
        b.inputs["IOR"].default_value = rng.uniform(1.45, 1.52)
    b.inputs["Transmission Weight"].default_value = 1.0
    return m


def twin_material():
    return principled("twin", (0.5, 0.5, 0.5, 1), 0.8)[0]


def new_obj(name, mesh):
    o = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(o)
    return o


def look_at(obj, target, roll=0.0):
    d = target - obj.location
    q = d.to_track_quat("-Z", "Y")
    q = q @ mathutils.Quaternion((0, 0, 1), roll)
    obj.rotation_euler = q.to_euler()


def set_world(sc, rng, strength_scale=1.0):
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    if rng.random() < 0.5:
        g = rng.uniform(0.05, 0.6)
        bg.inputs["Color"].default_value = (g, g, g * rng.uniform(0.9, 1.1), 1)
        bg.inputs["Strength"].default_value = 1.0 * strength_scale
        kind = "uniform"
    else:
        sky = nt.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(rng.uniform(10, 70))
        sky.sun_rotation = math.radians(rng.uniform(0, 360))
        nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = rng.uniform(0.4, 1.2) * strength_scale
        kind = "sky"
    sc.world = w
    return w, kind


def add_projector(cam, baseline, name="proj"):
    li = bpy.data.lights.new(name, "SPOT")
    li.energy = 40.0
    li.spot_size = math.radians(100)
    li.spot_blend = 0.0
    li.shadow_soft_size = 0.004
    li.cycles.is_caustics_light = True  # MNEE: refracted pattern through the glass
    li.use_nodes = True
    nt = li.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    im = nt.nodes.new("ShaderNodeTexImage")
    em = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputLight")
    im.image = bpy.data.images.load(PATTERN)
    im.extension = "CLIP"
    im.interpolation = "Linear"
    mp.inputs["Scale"].default_value = (0.5, 0.5, 1)
    mp.inputs["Location"].default_value = (0.5, 0.5, 0)
    nt.links.new(tc.outputs["Normal"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], im.inputs["Vector"])
    nt.links.new(im.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    o = new_obj(name, li)
    o.parent = cam
    o.location = (baseline / 2, -0.01, 0)  # between the imagers, slightly below the optical axis
    return o


def render_to(path, fmt="PNG", exr_passes=False, samples=64, denoise=False, filter_width=None, bw=False):
    sc = bpy.context.scene
    vl = bpy.context.view_layer
    sc.cycles.samples = samples
    sc.cycles.use_denoising = denoise
    if denoise:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.filter_width = filter_width if filter_width is not None else 1.5
    vl.use_pass_z = exr_passes
    vl.use_pass_normal = exr_passes
    vl.use_pass_object_index = exr_passes
    s = sc.render.image_settings
    if fmt == "EXR":
        s.file_format = "OPEN_EXR_MULTILAYER"
        s.color_depth = "32"
    else:
        s.file_format = "PNG"
        s.color_mode = "BW" if bw else "RGB"
        s.color_depth = "8"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def auto_exposure(sc, out, target=0.15):
    """Camera auto-exposure: quick linear float render, exposure (stops) so the median linear luminance is ~ target."""
    rp = sc.render.resolution_percentage
    sc.render.resolution_percentage = 25
    sc.view_settings.exposure = 0.0
    tmp = out + "/_exp.exr"
    sc.cycles.samples, sc.cycles.use_denoising = 8, False
    sc.cycles.filter_width = 1.5
    st = sc.render.image_settings
    st.file_format, st.color_depth = "OPEN_EXR", "32"
    sc.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    sc.render.resolution_percentage = rp
    img = bpy.data.images.load(tmp)
    img.colorspace_settings.name = "Linear Rec.2020" if False else img.colorspace_settings.name
    px = np.array(img.pixels[:]).reshape(-1, 4)[:, :3]
    bpy.data.images.remove(img)
    os.remove(tmp)
    lum = px @ np.array([0.2126, 0.7152, 0.0722])
    return float(np.clip(math.log2(target / max(float(np.median(lum)), 1e-6)), -14, 6))


# ------------------------------------------------------------------ one scene
def make_scene(args, seed, cfg):
    t0 = time.time()
    timings = {}
    rng = random.Random(seed)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = setup_engine(tuple(args.res))
    W, H = args.res

    # table (+ optional backdrop wall)
    bpy.ops.mesh.primitive_plane_add(size=3.0)
    table = bpy.context.object
    table.name = "table"
    tm, tex_kind = table_material(rng)
    table.data.materials.append(tm)
    table.pass_index = 3

    # transparent objects
    n_tr = rng.randint(1, 5)
    objs, info, trans, opaque = [], [], [], []
    for i in range(n_tr):
        path = rng.choice(MODELS)
        bpy.ops.wm.obj_import(filepath=path)
        o = bpy.context.selected_objects[0]
        o.name = f"trans{i}"
        plastic = rng.random() < 0.4
        o.data.materials.clear()
        o.data.materials.append(glass_material(rng, plastic))
        o.pass_index = 1
        trans.append(o)
        info.append(dict(model=os.path.basename(path), plastic=plastic))
    for i in range(rng.randint(0, 3)):
        kind = rng.choice(["cube", "cylinder", "sphere"])
        s = rng.uniform(0.03, 0.08)
        if kind == "cube":
            bpy.ops.mesh.primitive_cube_add(size=s)
        elif kind == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=s / 2, depth=s * rng.uniform(1, 2))
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=s / 2)
        o = bpy.context.object
        o.name = f"opaque{i}"
        m, _ = principled("opq", (rng.random(), rng.random(), rng.random(), 1), rng.uniform(0.2, 0.9))
        o.data.materials.append(m)
        o.pass_index = 2
        opaque.append(o)
    objs = trans + opaque
    for o in objs:
        o.location = (rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(0.04, 0.25))
        o.rotation_euler = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
        bpy.context.view_layer.update()
    # rigid-body drop onto the (passive) table, then freeze the final transforms
    bpy.ops.rigidbody.world_add()
    bpy.context.view_layer.objects.active = table
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    t1 = time.time()
    sc.rigidbody_world.point_cache.frame_end = 135
    for o in objs:
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
        bpy.ops.rigidbody.object_add(type="ACTIVE")
        o.rigid_body.collision_shape = "CONVEX_HULL"
        o.rigid_body.mass = 0.1
        o.rigid_body.friction = 0.9
        o.rigid_body.restitution = 0.1
        o.rigid_body.linear_damping = 0.6
        o.rigid_body.angular_damping = 0.8
        o.select_set(False)
    for f in range(1, 131):
        sc.frame_set(f)
    final = {o: o.matrix_world.copy() for o in objs}
    bpy.ops.rigidbody.world_remove()
    for o, mw in final.items():
        o.matrix_world = mw
    sc.frame_set(1)
    timings["physics"] = time.time() - t1

    # objects that rolled away from the cluster (or off the table) are removed
    allp = np.array([[o.location.x, o.location.y] for o in trans])  # the cluster is anchored on the transparent objects
    med = np.median(allp, axis=0)
    anchor = allp[np.argmin(np.hypot(allp[:, 0] - med[0], allp[:, 1] - med[1]))]  # object nearest the median
    far = lambda o: o.location.z < -0.02 or np.hypot(o.location.x - anchor[0], o.location.y - anchor[1]) > 0.25
    gone = [o for o in trans + opaque if far(o)]
    gone_ids = {o.as_pointer() for o in gone}
    info = [i for o, i in zip(trans, info) if o.as_pointer() not in gone_ids]
    trans = [o for o in trans if o.as_pointer() not in gone_ids]
    opaque = [o for o in opaque if o.as_pointer() not in gone_ids]
    for o in gone:
        bpy.data.objects.remove(o)
    if not trans:
        return None
    objs = trans + opaque
    pts = np.array([[o.location.x, o.location.y] for o in objs])
    target = mathutils.Vector((float(np.median(pts[:, 0])), float(np.median(pts[:, 1])), 0.04))

    # camera
    cam = new_obj("cam", bpy.data.cameras.new("cam"))
    sc.camera = cam
    cd = cam.data
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    if cfg["kind"] == "stereo":
        f_px = (W / 2) / math.tan(math.radians(cfg["hfov_deg"]) / 2)
        cx, cy = W / 2, H / 2
    else:
        sx = W / cfg["full_w"]
        f_px, cx, cy = cfg["fx"] * sx, cfg["cx"] * sx, cfg["cy"] * sx
    cd.lens = f_px * 36.0 / W
    cd.shift_x = (cx - W / 2) / W
    cd.shift_y = -(cy - H / 2) / W
    dist = rng.uniform(0.4, 0.8)
    elev = math.radians(rng.uniform(30, 75))
    az = rng.uniform(0, 2 * math.pi)
    cam.location = target + mathutils.Vector((dist * math.cos(elev) * math.cos(az), dist * math.cos(elev) * math.sin(az), dist * math.sin(elev)))
    look_at(cam, target, roll=math.radians(rng.uniform(-3, 3)))
    bpy.context.view_layer.update()
    K = dict(fx=f_px, fy=f_px if cfg["kind"] == "stereo" else cfg["fy"] * W / cfg["full_w"], cx=cx, cy=cy, width=W, height=H)

    # backdrop wall behind the cluster (as seen from the camera), 70% of scenes
    wall = None
    if rng.random() < 0.7:
        away = mathutils.Vector((target.x - cam.location.x, target.y - cam.location.y, 0)).normalized()
        bpy.ops.mesh.primitive_plane_add(size=4.0)
        wall = bpy.context.object
        wall.location = target + away * rng.uniform(0.6, 1.4)
        wall.location.z = 0.5
        wall.rotation_euler = (math.pi / 2, 0, math.atan2(away.y, away.x) - math.pi / 2)
        wm, _ = table_material(rng)
        wall.data.materials.append(wm)
        wall.pass_index = 3

    # MNEE caustics for the projector: glass casts, everything else receives
    for o in trans:
        o.cycles.is_caustics_caster = True
    recv = [table] + ([wall] if wall else []) + [o for o in objs if o not in trans]
    for o in recv:
        o.cycles.is_caustics_receiver = True

    # visible lights
    vis_lights = []
    for i in range(rng.randint(1, 3)):
        li = bpy.data.lights.new(f"area{i}", "AREA")
        li.energy = rng.uniform(30, 300)
        li.size = rng.uniform(0.2, 1.0)
        lo = new_obj(f"area{i}", li)
        lo.location = (target.x + rng.uniform(-0.6, 0.6), target.y + rng.uniform(-0.6, 0.6), rng.uniform(0.5, 1.5))
        look_at(lo, target)
        vis_lights.append(lo)
    world, world_kind = set_world(sc, rng)

    ir_world = bpy.data.worlds.new("ir_world")  # small uniform ambient IR instead of the visible sky
    ir_world.use_nodes = True
    ir_world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1)

    def set_visible_light(on):
        for lo in vis_lights:
            lo.hide_render = not on
        sc.world = world if on else ir_world

    glass_mats = {o: o.data.materials[0] for o in trans}
    twin = twin_material()

    def use_twin(on):
        for o in trans:
            o.data.materials.clear()
            o.data.materials.append(twin if on else glass_mats[o])

    out = os.path.join(args.out, args.variant, f"{seed:05d}")
    os.makedirs(out, exist_ok=True)
    proj = None
    if cfg["kind"] == "stereo":
        proj = add_projector(cam, cfg["baseline"])
        proj.hide_render = True

    # 1. RGB (glass)
    if not args.no_rgb:
        t = time.time()
        use_twin(False)
        set_visible_light(True)
        exposure = auto_exposure(sc, out)
        sc.view_settings.exposure = exposure
        render_to(out + "/rgb.png", samples=args.rgb_samples, denoise=True)
        sc.view_settings.exposure = 0.0
        timings["rgb"] = time.time() - t

    # 2. opaque-twin ground truth: depth, instance index, normal
    t = time.time()
    use_twin(True)
    render_to(out + "/gt.exr", fmt="EXR", exr_passes=True, samples=1, filter_width=0.01)
    timings["gt"] = time.time() - t

    # 3. ToF: depth with the transparent objects removed
    if cfg["kind"] == "tof":
        t = time.time()
        for o in trans:
            o.hide_render = True
        render_to(out + "/behind.exr", fmt="EXR", exr_passes=True, samples=1, filter_width=0.01)
        for o in trans:
            o.hide_render = False
        timings["behind"] = time.time() - t

    # 4. stereo IR pair (projector only, ambient IR small)
    baseline = None
    if cfg["kind"] == "stereo":
        baseline = cfg["baseline"]
        set_visible_light(False)
        proj.hide_render = False
        right = new_obj("cam_r", bpy.data.cameras.new("cam_r"))
        right.data = cam.data
        right.parent = cam
        right.location = (baseline, 0, 0)
        for twin_on, suffix in ([(False, "")] + ([(True, "_twin")] if args.ir_twin else [])):
            use_twin(twin_on)
            t = time.time()
            sc.camera = cam
            render_to(f"{out}/ir_left{suffix}.png", samples=args.ir_samples, denoise=False, bw=True)
            sc.camera = right
            render_to(f"{out}/ir_right{suffix}.png", samples=args.ir_samples, denoise=False, bw=True)
            timings["ir" + suffix] = time.time() - t
        sc.camera = cam

    meta = dict(cam_matrix_world=[list(r) for r in cam.matrix_world], exposure=None if args.no_rgb else exposure, seed=seed, variant=args.variant, intrinsics=K, baseline=baseline, texture=tex_kind, wall=bool(wall),
                world=world_kind, n_lights=len(vis_lights), objects=info, n_opaque=len(opaque),
                camera_dist=dist, camera_elev_deg=math.degrees(elev), timings=timings, total=time.time() - t0)
    with open(out + "/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    args = parse()
    cfg = VARIANTS[args.variant]
    if args.hfov:
        cfg = dict(cfg, hfov_deg=args.hfov)
    for seed in args.seeds:
        t = time.time()
        meta = make_scene(args, seed, cfg)
        print(f"SCENE variant={args.variant} seed={seed} seconds={time.time() - t:.1f}", "invalid" if meta is None else json.dumps(meta["timings"]), flush=True)


main()
