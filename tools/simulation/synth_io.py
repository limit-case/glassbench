"""Load rendered synthetic scenes (tools/simulation/render_scene.py) as dicts compatible with the real-data loaders."""
import json
import os

import cv2
import numpy as np
import OpenEXR

MAX_DEPTH = 10.0  # beyond this the ray missed the scene


def read_exr(path):
    ch = OpenEXR.File(path).channels()
    return {k.replace("ViewLayer.", ""): v.pixels for k, v in ch.items()}


def load_gt(d):
    e = read_exr(os.path.join(d, "gt.exr"))
    z = e["Depth.Z"].astype(np.float32)
    z[~np.isfinite(z) | (z <= 0) | (z > MAX_DEPTH)] = 0
    normal = np.stack([e["Normal.X"], e["Normal.Y"], e["Normal.Z"]], -1)
    return z, e["IndexOB.X"].astype(np.int32), normal


def stereo_overlap(depth_gt, fx, baseline, margin=4):
    """Left-image pixels that have a counterpart in the right image (x - disparity >= margin). Real stereo GT is
    invalid in the left strip with no overlap, so it is excluded from the GT-valid set for stereo variants."""
    h, w = depth_gt.shape
    xs = np.arange(w)[None, :].astype(np.float32)
    disp = np.where(depth_gt > 0, fx * baseline / np.maximum(depth_gt, 1e-6), 0)
    return (xs - disp) >= margin


def load_scene(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z, idx, normal = load_gt(d)
    s = dict(id=os.path.basename(os.path.dirname(d)) + "/" + os.path.basename(d), meta=meta, K=meta["intrinsics"],
             depth_gt=z, gt_valid=z > 0, mask=idx == 1, index=idx, normal=normal)
    if meta["baseline"]:
        s["gt_valid"] = s["gt_valid"] & stereo_overlap(z, meta["intrinsics"]["fx"], meta["baseline"])
    rgb = os.path.join(d, "rgb.png")
    if os.path.exists(rgb):
        s["rgb"] = cv2.cvtColor(cv2.imread(rgb), cv2.COLOR_BGR2RGB)
    for k in ("ir_left", "ir_right", "ir_left_twin", "ir_right_twin"):
        p = os.path.join(d, k + ".png")
        if os.path.exists(p):
            s[k] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    bp = os.path.join(d, "behind.exr")
    if os.path.exists(bp):
        zb = read_exr(bp)["Depth.Z"].astype(np.float32)
        zb[~np.isfinite(zb) | (zb <= 0) | (zb > MAX_DEPTH)] = 0
        s["depth_behind"] = zb
    return s
