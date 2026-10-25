"""Three-bucket evaluation of simulated sensors vs the real-sensor results (writes synth_vs_real.csv)."""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import stereo  # noqa: E402
import synth_io  # noqa: E402
import tof  # noqa: E402
from models_eval import evaluate_image  # noqa: E402
from three_bucket import hist_quantile  # noqa: E402

OUT = paths.SYNTH + "/out"
SENSORS = {"d415": "stereo", "d435": "stereo", "l515": "tof"}
REAL_REF = {"d415": "CG d415", "d435": "CG d435 test", "l515": "TransCG L515 (test scenes)"}
REAL_ALL = {**{v: k for k, v in REAL_REF.items()}, "CG d435 val": "d435"}  # real-val is an extra d435 reference
NAMES = ["missing", "correct", "wrong"]


def raw_depth(s, variant, twin=False):
    if SENSORS[variant] == "stereo":
        l, r = (s["ir_left_twin"], s["ir_right_twin"]) if twin else (s["ir_left"], s["ir_right"])
        return stereo.sgbm_depth(l, r, s["K"]["fx"], s["meta"]["baseline"], variant=variant)
    return tof.tof_depth(s, seed=s["meta"]["seed"])


def summarize(label, source, sensor, acc, imgs):
    rows = []
    for (region, tol), a in sorted(acc.items(), key=lambda kv: (kv[0][0] != "mask", kv[0][1])):
        col = a["T"].sum(0)
        n = int(col.sum())
        per = np.array(imgs[region, tol], float)
        ok = per.sum(1) > 0
        med = lambda c: float(np.median(100 * per[ok, c] / per[ok].sum(1)))
        rows.append(dict(sensor=sensor, source=source, camera=label, region=region, tol=tol, images=int(ok.sum()), pixels=n,
                         correct=100 * col[1] / n, missing=100 * col[0] / n, wrong=100 * col[2] / n,
                         wrong_among_returned=100 * col[2] / max(col[1] + col[2], 1),
                         wrong_far_share=100 * a["wrong_far"] / col[2] if col[2] else float("nan"),
                         err_cm_med=hist_quantile(a["hist"], 0.5), err_cm_p90=hist_quantile(a["hist"], 0.9),
                         correct_img_med=med(1), missing_img_med=med(0), wrong_img_med=med(2)))
    return rows


def evaluate_variant(variant, twin=False):
    acc, imgs = {}, {}
    dirs = sorted(glob.glob(f"{OUT}/{variant}/*/meta.json"))
    for mp in dirs:
        s = synth_io.load_scene(os.path.dirname(mp))
        if twin and "ir_left_twin" not in s:
            continue
        s["depth_raw"] = raw_depth(s, variant, twin)
        ev = evaluate_image(s, {"raw": s["depth_raw"]}, has_control=True)
        for (model, region, tol), d in ev.items():
            a = acc.setdefault((region, tol), dict(T=0, wrong_far=0, hist=0))
            a["T"] = a["T"] + d["T"]
            a["wrong_far"] += d["wrong_far"]
            a["hist"] = a["hist"] + d["hist"]
            imgs.setdefault((region, tol), []).append(d["T"].sum(0))
    return acc, imgs, len(dirs)


def real_rows():
    rows = []
    for r in csv.DictReader(open(paths.data("models_summary.csv"))):
        if r["model"] == "raw" and r["camera"] in REAL_ALL:
            sensor = REAL_ALL[r["camera"]]
            c, w = float(r["correct"]), float(r["wrong"])
            rows.append(dict(sensor=sensor, source="real", camera=r["camera"], region=r["region"], tol=r["tol"], images=r["images"],
                             pixels=r["pixels"], correct=c, missing=float(r["missing"]), wrong=w, wrong_among_returned=100 * w / (c + w),
                             wrong_far_share=float(r["wrong_far_share"]), err_cm_med=float(r["err_cm_med"]), err_cm_p90=float(r["err_cm_p90"]),
                             correct_img_med=float(r["correct_img_med"]), missing_img_med=float(r["missing_img_med"]), wrong_img_med=float(r["wrong_img_med"])))
    return rows


if __name__ == "__main__":
    variants = sys.argv[1:] or ["d415", "d435", "l515"]
    rows = real_rows()
    for v in variants:
        acc, imgs, n = evaluate_variant(v)
        rows += summarize(f"synthetic {v}-like ({n} scenes)", "synthetic", v, acc, imgs)
        if SENSORS[v] == "stereo":
            acc, imgs, n = evaluate_variant(v, twin=True)
            rows += summarize(f"synthetic {v}-like, opaque twin ({n} scenes)", "synthetic_opaque_twin", v, acc, imgs)
    with open(paths.data("synth_vs_real.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'camera':<44}{'region':<8}{'tol':<7}{'imgs':>5} {'correct':>8}{'missing':>8}{'wrong':>7}{'w/ret':>7}{'far%':>7}{'err med/p90':>14}")
    for r in rows:
        if r["tol"] != "rel5%":
            continue
        print(f"{r['camera']:<44}{r['region']:<8}{r['tol']:<7}{int(float(r['images'])):>5} {float(r['correct']):8.1f}{float(r['missing']):8.1f}{float(r['wrong']):7.1f}"
              f"{float(r['wrong_among_returned']):7.1f}{float(r['wrong_far_share']):7.1f}{float(r['err_cm_med']):8.1f}/{float(r['err_cm_p90']):.1f}")
