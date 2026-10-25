"""Figures for the simulation study: example scenes per sensor model and real-vs-synthetic side-by-side panels."""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import synth_io  # noqa: E402
from figure1 import buckets, load_by_id, render  # noqa: E402
from synth_eval import raw_depth

OUT = paths.FIGURES + "/synth"


def make(variant, d):
    s = synth_io.load_scene(d)
    s["depth_raw"] = raw_depth(s, variant)
    return s


def mix(s):
    m = s["mask"] & s["gt_valid"]
    b = buckets(s, 0.05)[m]
    return np.array([(b == 1).mean(), (b == 2).mean(), (b == 3).mean()]) if m.sum() else None, int(m.sum())


def cap(label, s):
    m, n = mix(s)
    return f"{label}; 5% tolerance. Inside the transparent-object mask: {100*m[2]:.0f}% confidently wrong, {100*m[1]:.0f}% missing, {100*m[0]:.0f}% correct."


def examples(variant, k):
    """First k scenes (by seed) whose transparent mask covers at least 1% of the image."""
    out = []
    for d in sorted(glob.glob(f"{paths.SYNTH}/out/{variant}/*")):
        s = synth_io.load_scene(d)
        if s["mask"].mean() >= 0.01:
            out.append(d)
        if len(out) == k:
            break
    return out


def stack(paths, out):
    ims = [cv2.imread(p) for p in paths]
    w = min(i.shape[1] for i in ims)
    ims = [cv2.resize(i, (w, int(i.shape[0] * w / i.shape[1]))) for i in ims]
    cv2.imwrite(out, np.vstack(ims))


def main():
    os.makedirs(OUT, exist_ok=True)
    for variant, k in (("d415", 2), ("d435", 2), ("l515", 4)):
        for d in examples(variant, k):
            s = make(variant, d)
            render(s, f"{OUT}/example_{variant}_{os.path.basename(d)}.png", cap(f"Synthetic {variant}-like, scene {os.path.basename(d)}", s), dpi=130)
    # real vs synthetic: synthetic scene whose in-mask bucket mix is closest to the real example
    pairs = [("CG d415", "real-test/d415/000000075", "d415", "figure1_alt"), ("TransCG L515", "scene2/148", "l515", "figure1")]
    for cam, cid, variant, name in pairs:
        real = load_by_id(cam, cid)
        rm, _ = mix(real)
        best, bd = None, 9
        for d in sorted(glob.glob(f"{paths.SYNTH}/out/{variant}/*")):
            s = synth_io.load_scene(d)
            if s["mask"].mean() < 0.01:
                continue
            s["depth_raw"] = raw_depth(s, variant)
            sm, _ = mix(s)
            dist = float(np.abs(sm - rm).sum())
            if dist < bd:
                best, bd = (d, s), dist
        d, s = best
        render(real, f"{OUT}/_real_{name}.png", cap(f"REAL {cam} {cid}", real), dpi=130)
        render(s, f"{OUT}/_synth_{name}.png", cap(f"SYNTHETIC {variant}-like scene {os.path.basename(d)} (closest in-mask bucket mix)", s), dpi=130)
        stack([f"{OUT}/_real_{name}.png", f"{OUT}/_synth_{name}.png"], f"{OUT}/real_vs_synth_{variant}.png")
        os.remove(f"{OUT}/_real_{name}.png")
        os.remove(f"{OUT}/_synth_{name}.png")
        print("pair", variant, os.path.basename(d), "mix distance", round(bd, 3))


if __name__ == "__main__":
    main()
