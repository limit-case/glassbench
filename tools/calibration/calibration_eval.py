"""Held-out evaluation after the calibration parameters were frozen (synth/configs/calibrated.json).
Real held-out: CG d435 test, CG d415, TransCG L515 test scenes 5, 7, 8 (all views).
Simulated held-out: fresh scenes (seeds 5000+), evaluated with the uncalibrated and the calibrated settings."""
import csv
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402
import stereo  # noqa: E402
import synth_io  # noqa: E402
import tof  # noqa: E402
from models_eval import evaluate_image  # noqa: E402
from three_bucket import hist_quantile  # noqa: E402

TEST_SCENES = {5, 7, 8}
PAIRS = [  # (sensor, simulated variant, real held-out name)
    ("stereo", "d435", "real CG d435 test"),
    ("stereo", "d415", "real CG d415 test"),
    ("tof", "l515", "real TransCG L515 test scenes"),
]


def real_jobs(name):
    if name == "real CG d435 test":
        return [("cg", sp, sid) for cam, sp, sid in loader.list_cleargrasp() if sp == "real-test/d435"]
    if name == "real CG d415 test":
        return [("cg", sp, sid) for cam, sp, sid in loader.list_cleargrasp() if sp == "real-test/d415"]
    return [("tc", v) for v in loader.list_transcg("L515") if int(re.search(r"scene(\d+)", v).group(1)) in TEST_SCENES]


def real_work(j):
    s = loader.load_cleargrasp(j[1], j[2]) if j[0] == "cg" else loader.load_transcg(j[1], "L515")
    ev = evaluate_image(s, {"raw": s["depth_raw"]}, has_control=j[0] == "cg")
    return {k: (v["T"], v["wrong_far"], v["hist"]) for k, v in ev.items()}


def pooled(results):
    """results: list of {(model, region, tol): (T, far, hist)} -> {(region, tol): summary}"""
    acc = {}
    for r in results:
        for (model, region, tol), (T, far, hist) in r.items():
            a = acc.setdefault((region, tol), [0, 0, 0, []])
            a[0] = a[0] + T
            a[1] += far
            a[2] = a[2] + hist
            a[3].append(T.sum(0))
    out = {}
    for k, (T, far, hist, per) in acc.items():
        col = T.sum(0)
        n = col.sum()
        out[k] = dict(images=len(per), pixels=int(n), correct=100 * col[1] / n, missing=100 * col[0] / n, wrong=100 * col[2] / n,
                      wrong_among_returned=100 * col[2] / (col[1] + col[2]), wrong_far_share=100 * far / col[2] if col[2] else float("nan"),
                      err_cm_med=hist_quantile(hist, 0.5), err_cm_p90=hist_quantile(hist, 0.9))
    return out


def sim_settings(sensor, variant, frozen):
    if sensor == "stereo":
        base = stereo.load_params(variant)
        pr = frozen["stereo"]["params"]
        cal = dict(base, uniqueness=int(pr["uniqueness"]), disp12=int(pr["lr_tol"]), texture_min=float(pr["texture_min"]))
        return {"sim uncalibrated": base, "sim calibrated": cal}
    pr = frozen["tof"]["params"]
    return {"sim uncalibrated": {}, "sim calibrated": dict(g=float(pr["g"]), t_glass=float(pr["t_glass"]), a_min=float(pr["a_min"]))}


def sim_results(sensor, variant, frozen):
    settings = sim_settings(sensor, variant, frozen)
    res = {k: [] for k in settings}
    for mp in sorted(glob.glob(f"{paths.SYNTH}/fresh/{variant}/*/meta.json")):
        s = synth_io.load_scene(os.path.dirname(mp))
        for name, p in settings.items():
            if sensor == "stereo":
                dep = stereo.sgbm_depth(s["ir_left"], s["ir_right"], s["K"]["fx"], s["meta"]["baseline"], p)
            else:
                dep = tof.tof_depth(s, seed=s["meta"]["seed"], **p)
            s["depth_raw"] = dep
            ev = evaluate_image(s, {"raw": dep}, has_control=True)
            res[name].append({k: (v["T"], v["wrong_far"], v["hist"]) for k, v in ev.items()})
    return {k: pooled(v) for k, v in res.items()}


def loss(a, b):
    return abs(a["correct"] - b["correct"]) + abs(a["missing"] - b["missing"]) + abs(a["wrong"] - b["wrong"])


if __name__ == "__main__":
    frozen = json.load(open(paths.config("calibrated.json")))
    rows = []
    summary = {}
    with Pool(3) as p:
        for sensor, variant, real_name in PAIRS:
            real = pooled(p.map(real_work, real_jobs(real_name), chunksize=4))
            sims = sim_results(sensor, variant, frozen)
            summary[real_name] = (real, sims)
            for src, table in [(real_name, real)] + list(sims.items()):
                for (region, tol), r in sorted(table.items(), key=lambda kv: (kv[0][0] != "mask", kv[0][1])):
                    ref = real.get((region, tol))
                    rows.append(dict(sensor=sensor, pair=real_name, source=src if src == real_name else f"{src} ({variant}-like)", region=region, tol=tol,
                                     **{k: r[k] for k in ("images", "pixels", "correct", "missing", "wrong", "wrong_among_returned", "wrong_far_share", "err_cm_med", "err_cm_p90")},
                                     loss_vs_real="" if src == real_name or ref is None else loss(r, ref)))
    with open(paths.data("calibration_heldout.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'pair':<32}{'source':<34}{'imgs':>5} {'correct':>8}{'missing':>8}{'wrong':>7}{'w/ret':>7}{'far%':>7}{'err med/p90':>13}{'loss':>7}")
    for r in rows:
        if r["region"] == "mask" and r["tol"] == "rel5%":
            print(f"{r['pair']:<32}{r['source']:<34}{r['images']:>5} {float(r['correct']):8.1f}{float(r['missing']):8.1f}{float(r['wrong']):7.1f}"
                  f"{float(r['wrong_among_returned']):7.1f}{float(r['wrong_far_share']):7.1f}{float(r['err_cm_med']):7.1f}/{float(r['err_cm_p90']):.1f}"
                  f"{'' if r['loss_vs_real'] == '' else '%7.1f' % float(r['loss_vs_real'])}")
