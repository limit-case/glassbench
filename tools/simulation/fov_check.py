"""Sensitivity of the simulated D415-like results to the depth HFOV: paired seeds 0-19 at 69.4 deg (used) vs 65 deg (Intel datasheet).
Run from the repo root after rendering synth/fovcheck (see docs/provenance.md)."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import stereo  # noqa: E402
import synth_io  # noqa: E402
from models_eval import evaluate_image  # noqa: E402

FROZEN = json.load(open(paths.config("calibrated.json")))


def shares(root, params):
    T = 0
    for seed in range(20):
        s = synth_io.load_scene(f"{root}/d415/{seed:05d}")
        dep = stereo.sgbm_depth(s["ir_left"], s["ir_right"], s["K"]["fx"], s["meta"]["baseline"], params)
        s["depth_raw"] = dep
        T = T + evaluate_image(s, {"raw": dep}, has_control=False)["raw", "mask", "rel5%"]["T"]
    col = T.sum(0)
    n = col.sum()
    return dict(missing=100 * col[0] / n, correct=100 * col[1] / n, wrong=100 * col[2] / n, wrong_among_returned=100 * col[2] / (col[1] + col[2]), pixels=int(n))


base = stereo.load_params("d415")
pr = FROZEN["stereo"]["params"]
cal = dict(base, uniqueness=int(pr["uniqueness"]), disp12=int(pr["lr_tol"]), texture_min=float(pr["texture_min"]))
rows = []
for setting, p in (("uncalibrated", base), ("calibrated", cal)):
    for hfov, root in ((69.4, paths.SYNTH + "/out"), (65.0, paths.SYNTH + "/fovcheck")):
        rows.append(dict(setting=setting, hfov_deg=hfov, **shares(root, p)))
with open(paths.data("fov_check.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
