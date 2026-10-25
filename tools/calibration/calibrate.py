"""Grid-search calibration of the simulated sensors against the real calibration sets (see docs/provenance.md).
Reads only the calibration-set targets (calibration_targets.csv). Held-out sets are never touched here."""
import csv
import glob
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import stereo  # noqa: E402
import synth_io  # noqa: E402
import tof  # noqa: E402

STEREO_GRID = dict(uniqueness=[10, 20, 30, 40, 50], lr_tol=[0, 1, 2, 4], texture_min=[0, 3, 6, 10, 15, 20, 30])
TOF_GRID = dict(g=[1, 2, 4, 8, 16, 32], t_glass=[1.0, 0.8, 0.6, 0.4, 0.2], a_min=[0, 0.05, 0.1, 0.15, 0.2, 0.3])
N_TWIN = 40
N_TOF = 100


def targets():
    return {r["sensor"]: {k: float(r[k]) for k in ("correct", "missing", "wrong")} for r in csv.DictReader(open(paths.data("calibration_targets.csv")))}


def buckets_counts(d, g):
    ok = d > 0
    cor = ok & (np.abs(d - g) <= 0.05 * g)
    return np.array([int((~ok).sum()), int(cor.sum()), int((ok & ~cor).sum())])  # missing, correct, wrong


def pct(c):
    return 100 * c / c.sum()


def stereo_search(tg):
    scenes = sorted(glob.glob(paths.SYNTH + "/out/d435/*/meta.json"))
    base = stereo.load_params("d435")
    Ts = STEREO_GRID["texture_min"]
    rows = []
    data = []
    for mp in scenes:
        s = synth_io.load_scene(os.path.dirname(mp))
        data.append(dict(l=s["ir_left"], r=s["ir_right"], tex=stereo.texture_std(s["ir_left"]), m=s["mask"] & s["gt_valid"], g=s["depth_gt"], fx=s["K"]["fx"], b=s["meta"]["baseline"],
                         tw=("ir_left_twin" in s) and s))
    twins = [d for d in data if d["tw"]][:N_TWIN]
    t0 = time.time()
    for u, lr in itertools.product(STEREO_GRID["uniqueness"], STEREO_GRID["lr_tol"]):
        p = dict(base, uniqueness=u, disp12=lr, texture_min=0)
        cm = {T: np.zeros(3, np.int64) for T in Ts}
        co = {T: np.zeros(3, np.int64) for T in Ts}
        for d in data:
            dep = stereo.sgbm_depth(d["l"], d["r"], d["fx"], d["b"], p)
            for T in Ts:
                dd = np.where(d["tex"] >= T, dep, 0)
                cm[T] += buckets_counts(dd[d["m"]], d["g"][d["m"]])
        for d in twins:
            s = d["tw"]
            dep = stereo.sgbm_depth(s["ir_left_twin"], s["ir_right_twin"], d["fx"], d["b"], p)
            tex = stereo.texture_std(s["ir_left_twin"])
            v = s["gt_valid"]
            for T in Ts:
                dd = np.where(tex >= T, dep, 0)
                co[T] += buckets_counts(dd[v], s["depth_gt"][v])
        for T in Ts:
            m, c, w = pct(cm[T])
            oc = pct(co[T])[1]
            loss = abs(c - tg["correct"]) + abs(m - tg["missing"]) + abs(w - tg["wrong"])
            rows.append(dict(sensor="stereo", uniqueness=u, lr_tol=lr, texture_min=T, g="", t_glass="", a_min="", correct=c, missing=m, wrong=w,
                             wrong_among_returned=100 * w / (c + w) if c + w else float("nan"), loss=loss, opaque_correct=oc, valid=oc >= 98.0))
        print(f"stereo u={u} lr={lr} done {time.time()-t0:.0f}s", flush=True)
    return rows


def tof_search(tg):
    scenes = sorted(glob.glob(paths.SYNTH + "/out/l515/*/meta.json"))[:N_TOF]
    data = [synth_io.load_scene(os.path.dirname(p)) for p in scenes]
    rows = []
    for g, t, a in itertools.product(TOF_GRID["g"], TOF_GRID["t_glass"], TOF_GRID["a_min"]):
        cm = np.zeros(3, np.int64)
        for s in data:
            d = tof.tof_depth(s, seed=s["meta"]["seed"], g=g, t_glass=t, a_min=a)
            m = s["mask"] & s["gt_valid"]
            cm += buckets_counts(d[m], s["depth_gt"][m])
        mi, c, w = pct(cm)
        loss = abs(c - tg["correct"]) + abs(mi - tg["missing"]) + abs(w - tg["wrong"])
        rows.append(dict(sensor="tof", uniqueness="", lr_tol="", texture_min="", g=g, t_glass=t, a_min=a, correct=c, missing=mi, wrong=w,
                         wrong_among_returned=100 * w / (c + w) if c + w else float("nan"), loss=loss, opaque_correct="", valid=True))
    return rows


if __name__ == "__main__":
    tg = targets()
    which = sys.argv[1:] or ["tof", "stereo"]
    rows = []
    if "tof" in which:
        rows += tof_search(tg["tof"])
        print("tof done", flush=True)
    if "stereo" in which:
        rows += stereo_search(tg["stereo"])
    with open(paths.data("calibration_search.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    frozen = {}
    for sensor, keys in (("stereo", ["uniqueness", "lr_tol", "texture_min"]), ("tof", ["g", "t_glass", "a_min"])):
        rs = [r for r in rows if r["sensor"] == sensor and r["valid"]]
        if not rs:
            continue
        best = min(rs, key=lambda r: r["loss"])
        base = [r for r in rows if r["sensor"] == sensor and all(float(r[k]) == v for k, v in zip(keys, {"stereo": [10, 1, 0], "tof": [1, 1.0, 0]}[sensor]))][0]
        frozen[sensor] = dict(params={k: best[k] for k in keys}, calibration_loss=best["loss"], uncalibrated_loss=base["loss"],
                              calibration_result={k: best[k] for k in ("correct", "missing", "wrong", "wrong_among_returned")},
                              uncalibrated_result={k: base[k] for k in ("correct", "missing", "wrong", "wrong_among_returned")}, target=tg[sensor])
        print(sensor, json.dumps(frozen[sensor], indent=1))
    json.dump(frozen, open(paths.config("calibrated_proposal.json"), "w"), indent=1)
