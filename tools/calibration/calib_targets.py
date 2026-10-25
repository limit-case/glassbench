"""Real calibration-set targets only (CG d435 val; TransCG L515 train scenes 1,2,3,4,6,9,10). Held-out sets are NOT touched here."""
import csv
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402
from models_eval import evaluate_image  # noqa: E402
from three_bucket import hist_quantile  # noqa: E402

TRAIN_SCENES = {1, 2, 3, 4, 6, 9, 10}


def job_list(which):
    if which == "stereo":
        return [("cg", split, sid) for cam, split, sid in loader.list_cleargrasp() if split == "real-val/d435"]
    return [("tc", v) for v in loader.list_transcg("L515") if int(re.search(r"scene(\d+)", v).group(1)) in TRAIN_SCENES]


def work(j):
    s = loader.load_cleargrasp(j[1], j[2]) if j[0] == "cg" else loader.load_transcg(j[1], "L515")
    ev = evaluate_image(s, {"raw": s["depth_raw"]}, has_control=False)
    d = ev["raw", "mask", "rel5%"]
    return d["T"], d["wrong_far"], d["hist"]


def summarize(sensor, name, res):
    T = sum(r[0] for r in res)
    far = sum(r[1] for r in res)
    hist = sum(r[2] for r in res)
    col = T.sum(0)
    n = int(col.sum())
    return dict(sensor=sensor, set=name, images=len(res), pixels=n, correct=100 * col[1] / n, missing=100 * col[0] / n, wrong=100 * col[2] / n,
                wrong_among_returned=100 * col[2] / (col[1] + col[2]), wrong_far_share=100 * far / col[2],
                err_cm_med=hist_quantile(hist, 0.5), err_cm_p90=hist_quantile(hist, 0.9))


if __name__ == "__main__":
    rows = []
    with Pool(3) as p:
        for sensor, name in (("stereo", "real CG d435 val"), ("tof", "real TransCG L515 train scenes")):
            res = p.map(work, job_list(sensor), chunksize=8)
            rows.append(summarize(sensor, name, res))
            print(rows[-1], flush=True)
    with open(paths.data("calibration_targets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
