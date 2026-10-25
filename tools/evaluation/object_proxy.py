"""Per-object consequence proxy (analysis only, no new data).

For each real group and each connected component of the GT transparent mask (>= MIN_AREA px), take the median raw - GT error over the
component's returned pixels (raw > 0, GT-valid) and count how many objects are off by more than 2 cm and more than 5 cm.
The 2 cm / 5 cm thresholds are illustrative parallel-jaw / suction placement tolerances only; this is NOT a grasping experiment.
Touching transparent objects merge into one component (the masks do not separate instances). Writes object_proxy.csv."""
import sys
import os
import csv
import re
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402

MIN_AREA = 200        # px, size of a component
MIN_RETURNED = 20     # px, an object "returns depth" if at least this many of its pixels have a raw value
GROUPS = ["CG d415", "CG d435 test", "CG d435 val", "TransCG D435", "TransCG L515"]


def jobs():
    out = []
    for cam, split, sid in loader.list_cleargrasp():
        g = "CG d415" if cam == "d415" else ("CG d435 test" if "test" in split else "CG d435 val")
        out.append((g, "cg", split, sid))
    for c in ("D435", "L515"):
        out += [(f"TransCG {c}", "tc", v, c) for v in loader.list_transcg(c)]
    return out


def work(j):
    g, kind, a, b = j
    s = loader.load_cleargrasp(a, b) if kind == "cg" else loader.load_transcg(a, b)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(s["mask"].astype(np.uint8), connectivity=8)
    res = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        m = (lab == i) & s["gt_valid"]
        ret = m & (s["depth_raw"] > 0)
        frac = float(ret.sum() / max(m.sum(), 1))        # share of the object's GT-valid pixels that have a raw value
        if ret.sum() < MIN_RETURNED:
            res.append((g, np.nan, frac))                # no usable return: a visible failure
        else:
            res.append((g, 100 * float(np.median(s["depth_raw"][ret] - s["depth_gt"][ret])), frac))   # signed error, cm
    return res


if __name__ == "__main__":
    allr = []
    with Pool(3) as p:
        for r in p.imap(work, jobs(), chunksize=16):
            allr += r
    rows = []
    for g in GROUPS:
        e = np.array([x[1] for x in allr if x[0] == g])
        ret = e[~np.isnan(e)]
        a = np.abs(ret)
        fr = np.array([x[2] for x in allr if x[0] == g])
        rows.append(dict(group=g, objects=len(e), objects_with_return=len(ret), no_return_pct=100 * (len(e) - len(ret)) / len(e),
                         mostly_missing_pct=100 * (fr < 0.5).mean(),
                         off_gt2cm_pct=100 * (a > 2).mean(), off_gt5cm_pct=100 * (a > 5).mean(),
                         off_gt5cm_too_far_pct=100 * ((ret > 5).sum() / max((a > 5).sum(), 1)),
                         median_abs_err_cm=float(np.median(a)), median_signed_err_cm=float(np.median(ret))))
    with open(paths.data("object_proxy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
