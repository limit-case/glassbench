"""Tune StereoSGBM on opaque-twin IR pairs (tuning scenes, seeds 1000+), target: ClearGrasp d415 non-mask (~99% correct at 5%)."""
import os
import glob
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import synth_io  # noqa: E402
import stereo  # noqa: E402

variant = sys.argv[1]
scenes = [synth_io.load_scene(d) for d in sorted(glob.glob(f"{paths.SYNTH}/tune/{variant}/*"))]
nd = {"d415": 96, "d435": 64}[variant]


def score(p):
    c = m = w = n = 0
    for s in scenes:
        K = s["K"]
        d = stereo.sgbm_depth(s["ir_left_twin"], s["ir_right_twin"], K["fx"], s["meta"]["baseline"], p)
        v = s["gt_valid"]
        g = s["depth_gt"][v]
        dd = d[v]
        ok = dd > 0
        cor = ok & (np.abs(dd - g) <= 0.05 * g)
        c += cor.sum(); m += (~ok).sum(); w += (ok & ~cor).sum(); n += v.sum()
    return 100 * c / n, 100 * m / n, 100 * w / n


res = []
for block, uniq, sw, md in itertools.product([5, 7, 9, 11], [5, 10, 15], [50, 150], [8, 16]):
    p = dict(stereo.DEFAULTS, num_disp=nd, block=block, uniqueness=uniq, speckle_window=sw, min_disp=md)
    res.append((score(p), p))
res.sort(key=lambda r: -r[0][0] + 5 * r[0][2])
for (c, m, w), p in res[:8]:
    print(f"correct {c:5.1f} missing {m:5.1f} wrong {w:5.1f}", {k: p[k] for k in ("block", "uniqueness", "speckle_window", "min_disp")})
