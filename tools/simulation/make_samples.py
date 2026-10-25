"""Write a small, self-describing sample of the synthetic data (3 scenes per sensor model) to examples/sample/.
Depth maps are uint16 PNG in millimetres (0 = invalid / no return)."""
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import stereo  # noqa: E402
import synth_io  # noqa: E402
import tof  # noqa: E402

OUT = paths.SAMPLE
FROZEN = json.load(open(paths.config("calibrated.json")))


def mm(d):
    return np.clip(np.round(d * 1000), 0, 65535).astype(np.uint16)


def raw_both(s, variant):
    if variant in ("d415", "d435"):
        base = stereo.load_params(variant)
        pr = FROZEN["stereo"]["params"]
        cal = dict(base, uniqueness=int(pr["uniqueness"]), disp12=int(pr["lr_tol"]), texture_min=float(pr["texture_min"]))
        f = lambda p: stereo.sgbm_depth(s["ir_left"], s["ir_right"], s["K"]["fx"], s["meta"]["baseline"], p)
        return f(base), f(cal)
    pr = FROZEN["tof"]["params"]
    seed = s["meta"]["seed"]
    return tof.tof_depth(s, seed=seed), tof.tof_depth(s, seed=seed, g=float(pr["g"]), t_glass=float(pr["t_glass"]), a_min=float(pr["a_min"]))


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    for variant in ("d415", "d435", "l515"):
        n = 0
        for d in sorted(glob.glob(f"{paths.SYNTH}/out/{variant}/*")):
            s = synth_io.load_scene(d)
            if s["mask"].mean() < 0.01:
                continue
            o = f"{OUT}/{variant}/{os.path.basename(d)}"
            os.makedirs(o)
            shutil.copy(d + "/rgb.png", o)
            for k in ("ir_left", "ir_right"):
                if os.path.exists(f"{d}/{k}.png"):
                    shutil.copy(f"{d}/{k}.png", o)
            cv2.imwrite(o + "/gt_depth_mm.png", mm(s["depth_gt"]))
            cv2.imwrite(o + "/gt_valid.png", (s["gt_valid"] * 255).astype(np.uint8))
            cv2.imwrite(o + "/mask.png", (s["mask"] * 255).astype(np.uint8))
            if "depth_behind" in s:
                cv2.imwrite(o + "/depth_behind_mm.png", mm(s["depth_behind"]))
            un, ca = raw_both(s, variant)
            cv2.imwrite(o + "/raw_uncalibrated_mm.png", mm(un))
            cv2.imwrite(o + "/raw_calibrated_mm.png", mm(ca))
            meta = json.load(open(d + "/meta.json"))
            meta.pop("timings", None)
            json.dump(meta, open(o + "/meta.json", "w"), indent=1)
            n += 1
            if n == 3:
                break
    print("done")
