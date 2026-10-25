"""Loaders for ClearGrasp (real) and TransCG. Depth is returned in meters, float32; invalid = 0."""
import glob
import os

import cv2
import numpy as np
import OpenEXR

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")
CLEARGRASP_ROOT = os.path.join(DATA, "cleargrasp-dataset-test-val")
CLEARGRASP_SPLITS = ["real-test/d415", "real-test/d435", "real-val/d435"]
TRANSCG_ROOT = os.path.join(DATA, "transcg")

# TransCG stores depth as uint16 PNG; metadata.json has no scale. Verified against object
# pose z (check_transcg_pose.py, see UNITS.md): D435 is mm, L515 is 0.25 mm units.
# depth and depth-gt share a scale.
TRANSCG_DEPTH_SCALE = {"D435": 1000.0, "L515": 4000.0}
TRANSCG_CAMERAS = {"D435": "1", "L515": "2"}


def _read_exr_depth(path):
    """EXR depth is stored as 3 identical float16 channels; take one. NaN/inf/negative -> 0."""
    ch = OpenEXR.File(path).channels()
    px = next(iter(ch.values())).pixels
    d = px[..., 0] if px.ndim == 3 else px
    d = d.astype(np.float32)
    d[~np.isfinite(d) | (d < 0)] = 0.0
    return d


def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _read_depth_png(path, scale):
    return _read_png(path).astype(np.float32) / scale


def list_cleargrasp(splits=CLEARGRASP_SPLITS):
    """Return [(camera, split, sample_id)] for the ClearGrasp real splits."""
    out = []
    for split in splits:
        cam = os.path.basename(split)
        files = sorted(glob.glob(os.path.join(CLEARGRASP_ROOT, split, "*-transparent-rgb-img.jpg")))
        for f in files:
            sid = os.path.basename(f).split("-")[0]
            out.append((cam, split, sid))
    return out


def load_cleargrasp(split, sample_id):
    base = os.path.join(CLEARGRASP_ROOT, split, sample_id)
    mask = _read_png(base + "-mask.png")
    if mask.ndim == 3:
        mask = mask[..., 0]
    depth_gt = _read_exr_depth(base + "-opaque-depth-img.exr")
    return {
        "camera": os.path.basename(split),
        "id": f"{split}/{sample_id}",
        "rgb": _read_rgb(base + "-transparent-rgb-img.jpg"),
        "depth_raw": _read_exr_depth(base + "-transparent-depth-img.exr"),
        "depth_gt": depth_gt,
        "gt_valid": depth_gt > 0,
        "mask": mask > 0,
    }


def _scene_key(path):
    scene, view = path.split(os.sep)[-2:]
    return int(scene.replace("scene", "")), int(view) if view.isdigit() else view


def list_transcg(camera):
    """Return sorted view directories under datasets/transcg/scene*/<view> that have the camera's files."""
    idx = TRANSCG_CAMERAS[camera]
    dirs = glob.glob(os.path.join(TRANSCG_ROOT, "scene*", "*", f"depth{idx}-gt.png"))
    return sorted((os.path.dirname(d) for d in dirs), key=_scene_key)


def load_transcg(view_dir, camera):
    idx = TRANSCG_CAMERAS[camera]
    p = lambda name: os.path.join(view_dir, name)
    scale = TRANSCG_DEPTH_SCALE[camera]
    depth_gt = _read_depth_png(p(f"depth{idx}-gt.png"), scale)
    return {
        "camera": camera,
        "id": "/".join(view_dir.split(os.sep)[-2:]),
        "rgb": _read_rgb(p(f"rgb{idx}.png")),
        "depth_raw": _read_depth_png(p(f"depth{idx}.png"), scale),
        "depth_gt": depth_gt,
        "gt_valid": depth_gt > 0,
        "mask": _read_png(p(f"depth{idx}-gt-mask.png")) > 0,
    }


def _depth_stats(d):
    v = d[d > 0]
    if v.size == 0:
        return "no valid pixels"
    return (f"min={v.min():.3f} max={v.max():.3f} median={np.median(v):.3f} "
            f"zero={100 * (d == 0).mean():.1f}%")


def describe(s):
    print(f"[{s['camera']}] {s['id']}")
    print(f"  rgb {s['rgb'].shape} {s['rgb'].dtype} | depth_raw {s['depth_raw'].shape} "
          f"{s['depth_raw'].dtype} | depth_gt {s['depth_gt'].shape} | mask {s['mask'].shape}")
    print(f"  raw: {_depth_stats(s['depth_raw'])}")
    print(f"  gt : {_depth_stats(s['depth_gt'])}")
    print(f"  mask coverage={100 * s['mask'].mean():.1f}% | gt_valid={100 * s['gt_valid'].mean():.1f}%")


if __name__ == "__main__":
    n = 3
    cg = list_cleargrasp()
    for cam, split in [("d415", "real-test/d415"), ("d435", "real-test/d435"), ("d435", "real-val/d435")]:
        items = [x for x in cg if x[1] == split]
        print(f"== ClearGrasp {split}: {len(items)} samples")
        for _, sp, sid in items[:n]:
            describe(load_cleargrasp(sp, sid))
    for cam in TRANSCG_CAMERAS:
        views = list_transcg(cam)
        print(f"== TransCG {cam}: {len(views)} samples")
        for v in views[:n]:
            describe(load_transcg(v, cam))
