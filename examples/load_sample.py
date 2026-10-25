"""Load the open synthetic sample and print its three-bucket table.

Reads every scene under examples/sample/, scores the simulated raw depth against the ground truth on the
transparent-object mask (GT-valid pixels, 5% tolerance, pooled over the scenes of one sensor), and prints
correct / missing / confidently wrong shares for the uncalibrated and the calibrated simulator settings.
The rule is the one in docs/measurement.md. Needs only numpy and opencv-python.

    python examples/load_sample.py [--tol rel5|abs1cm]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")


def read_mm(path):
    """uint16 millimetre PNG -> float metres; 0 stays 0 (no surface, or no return)."""
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0


def load_scene(folder):
    """One sample scene as a dict of arrays in metres, plus its meta.json."""
    scene = {
        "meta": json.load(open(os.path.join(folder, "meta.json"))),
        "rgb": cv2.cvtColor(cv2.imread(os.path.join(folder, "rgb.png")), cv2.COLOR_BGR2RGB),
        "gt": read_mm(os.path.join(folder, "gt_depth_mm.png")),
        "gt_valid": cv2.imread(os.path.join(folder, "gt_valid.png"), cv2.IMREAD_UNCHANGED) > 0,
        "mask": cv2.imread(os.path.join(folder, "mask.png"), cv2.IMREAD_UNCHANGED) > 0,
        "raw_uncalibrated": read_mm(os.path.join(folder, "raw_uncalibrated_mm.png")),
        "raw_calibrated": read_mm(os.path.join(folder, "raw_calibrated_mm.png")),
    }
    return scene


def buckets(raw, gt, sel, tol):
    """Counts of missing / correct / confidently wrong (and wrong that read too far) over the pixels in sel."""
    raw, gt = raw[sel], gt[sel]
    missing = raw == 0
    correct = ~missing & (np.abs(raw - gt) <= tol(gt))
    wrong = ~missing & ~correct
    return np.array([raw.size, missing.sum(), correct.sum(), wrong.sum(), (wrong & (raw > gt)).sum()])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tol", choices=["rel5", "abs1cm"], default="rel5")
    tols = {"rel5": lambda gt: 0.05 * gt, "abs1cm": lambda gt: 0.01 + 0 * gt}
    args = ap.parse_args()
    tol = tols[args.tol]

    print(f"{'sensor':<7}{'scenes':>7}{'setting':>15}{'correct %':>11}{'missing %':>11}{'wrong %':>9}{'too far %':>11}")
    for sensor in ("d415", "d435", "l515"):
        folders = sorted(glob.glob(os.path.join(SAMPLE, sensor, "*")))
        scenes = [load_scene(f) for f in folders]
        for setting in ("raw_uncalibrated", "raw_calibrated"):
            n, miss, cor, wr, far = sum(
                buckets(s[setting], s["gt"], s["mask"] & s["gt_valid"], tol) for s in scenes
            )
            print(f"{sensor:<7}{len(scenes):>7}{setting[4:]:>15}{100 * cor / n:>11.1f}{100 * miss / n:>11.1f}"
                  f"{100 * wr / n:>9.1f}{100 * far / max(wr, 1):>11.1f}")


if __name__ == "__main__":
    main()
