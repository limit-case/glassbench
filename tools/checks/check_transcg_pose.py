"""Check TransCG pose units / depth scale: GT depth near the projected object center vs pose z."""
import sys
import glob
import os
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402

N_VIEWS = 20
SEED = 0
HALF = 7  # window half-size in px
T21 = np.load(os.path.join(loader.TRANSCG_ROOT, "T_camera2_camera1.npy"))
K = {c: np.load(glob.glob(os.path.join(loader.TRANSCG_ROOT, "camera_intrinsics", f"{i}-camIntrinsics-*.npy"))[0])
     for c, i in loader.TRANSCG_CAMERAS.items()}


def window_depth(depth, K, p):
    """Median of valid depth in a window around the projection of camera-frame point p; None if off-image/empty."""
    if p[2] <= 0:
        return None
    u = int(round(K[0, 0] * p[0] / p[2] + K[0, 2]))
    v = int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))
    h, w = depth.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    win = depth[max(v - HALF, 0):v + HALF + 1, max(u - HALF, 0):u + HALF + 1]
    win = win[win > 0]
    return float(np.median(win)) if win.size else None


def report(name, ratios, skipped):
    r = np.array(ratios)
    if r.size == 0:
        print(f"{name}: no usable poses ({skipped} skipped)")
        return
    p10, p50, p90 = np.percentile(r, [10, 50, 90])
    print(f"{name}: n={r.size} (skipped {skipped}) ratio median={p50:.3f} p10={p10:.3f} p90={p90:.3f} "
          f"min={r.min():.3f} max={r.max():.3f}")


def main():
    rng = random.Random(SEED)
    l515_views = set(loader.list_transcg("L515"))
    views = [v for v in loader.list_transcg("D435")
             if v in l515_views and glob.glob(os.path.join(v, "corrected_pose", "*.npy"))]
    views = rng.sample(views, N_VIEWS)
    res = {k: ([], [0]) for k in ["D435 gt", "L515 gt (T)", "L515 raw (T)", "L515 gt (T^-1)", "L515 raw (T^-1)"]}
    zs = []
    for vd in views:
        d435 = loader.load_transcg(vd, "D435")
        l515 = loader.load_transcg(vd, "L515")
        for f in glob.glob(os.path.join(vd, "corrected_pose", "*.npy")):
            pose = np.load(f)
            t = pose[:3, 3]
            zs.append(t[2])
            tests = [("D435 gt", d435["depth_gt"], "D435", t)]
            for tag, T in [("T", T21), ("T^-1", np.linalg.inv(T21))]:
                pl = (T @ np.append(t, 1))[:3]
                tests += [(f"L515 gt ({tag})", l515["depth_gt"], "L515", pl),
                          (f"L515 raw ({tag})", l515["depth_raw"], "L515", pl)]
            for key, depth, cam, p in tests:
                d = window_depth(depth, K[cam], p)
                if d is None:
                    res[key][1][0] += 1
                else:
                    res[key][0].append(d / p[2])
    print(f"{len(views)} views, {len(zs)} poses, D435 pose z: median={np.median(zs):.3f} "
          f"min={min(zs):.3f} max={max(zs):.3f}")
    for key, (r, sk) in res.items():
        report(key, r, sk[0])


if __name__ == "__main__":
    main()
