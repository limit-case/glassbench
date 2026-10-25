"""Sanity-check depth scale: raw/GT ratio on non-transparent, GT-valid, raw>0 pixels."""
import sys
import os
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402

N = 30
SEED = 0


def cameras():
    cg = loader.list_cleargrasp()
    yield "CG d415", [lambda x=x: loader.load_cleargrasp(x[1], x[2]) for x in cg if x[0] == "d415"]
    yield "CG d435", [lambda x=x: loader.load_cleargrasp(x[1], x[2]) for x in cg if x[0] == "d435"]
    for cam in loader.TRANSCG_CAMERAS:
        views = loader.list_transcg(cam)
        yield f"TransCG {cam}", [lambda v=v, cam=cam: loader.load_transcg(v, cam) for v in views]


def main():
    rng = random.Random(SEED)
    for name, loaders in cameras():
        picks = rng.sample(loaders, min(N, len(loaders)))
        ratios, raws, gts = [], [], []
        for load in picks:
            s = load()
            sel = ~s["mask"] & s["gt_valid"] & (s["depth_raw"] > 0)
            raw, gt = s["depth_raw"][sel], s["depth_gt"][sel]
            ratios.append(raw / gt)
            raws.append(raw)
            gts.append(gt)
        r, raw, gt = map(np.concatenate, (ratios, raws, gts))
        p10, p50, p90 = np.percentile(r, [10, 50, 90])
        print(f"{name} ({len(picks)} samples, {r.size} px): ratio median={p50:.4f} p10={p10:.4f} p90={p90:.4f} "
              f"| median raw={np.median(raw):.4f} m, median GT={np.median(gt):.4f} m")


if __name__ == "__main__":
    main()
