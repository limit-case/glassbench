"""docs/media/figures/synth/calibrated_vs_real.png: real vs uncalibrated sim vs calibrated sim, calibration sets and held-out sets."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402

COL = {"correct": "tab:green", "missing": "0.6", "wrong": "tab:red"}


def main(scale=1.0, out=None, compact=False):
    out = out or paths.FIGURES + "/synth/calibrated_vs_real.png"
    tg = {r["sensor"]: r for r in csv.DictReader(open(paths.data("calibration_targets.csv")))}
    fz = json.load(open(paths.config("calibrated.json")))
    cs = json.load(open(paths.config("calibrated.json")))
    # calibration-set results (from the search), held-out results (from calibration_heldout.csv)
    search = {(r["sensor"], r["uniqueness"], r["lr_tol"], r["texture_min"], r["g"], r["t_glass"], r["a_min"]): r for r in csv.DictReader(open(paths.data("calibration_search.csv")))}

    def pick(sensor, params, base):
        keys = ["uniqueness", "lr_tol", "texture_min"] if sensor == "stereo" else ["g", "t_glass", "a_min"]
        def match(r, p):
            return r["sensor"] == sensor and all(float(r[k]) == float(p[k]) for k in keys)
        rows = list(csv.DictReader(open(paths.data("calibration_search.csv"))))
        return [r for r in rows if match(r, params)][0]
    stereo_base = dict(uniqueness=10, lr_tol=1, texture_min=0)
    tof_base = dict(g=1, t_glass=1.0, a_min=0)
    panels = []
    for sensor, title, tname, base in (("stereo", "Stereo: calibration set\n(real CG d435 val)", "real CG d435 val", stereo_base), ("tof", "ToF: calibration set\n(real L515 train scenes)", "real TransCG L515 train scenes", tof_base)):
        t = tg[sensor]
        un = pick(sensor, base, base)
        ca = pick(sensor, fz[sensor]["params"], base)
        panels.append((title, [("real", t), ("sim uncal.", un), ("sim cal.", ca)], f"loss {float(un['loss']):.1f} -> {float(ca['loss']):.1f}"))
    held = list(csv.DictReader(open(paths.data("calibration_heldout.csv"))))
    for pair, title in (("real CG d435 test", "Stereo: held-out\nreal CG d435 test"), ("real CG d415 test", "Stereo: held-out\nreal CG d415 test"), ("real TransCG L515 test scenes", "ToF: held-out\nreal L515 test scenes")):
        rs = [r for r in held if r["pair"] == pair and r["region"] == "mask" and r["tol"] == "rel5%"]
        real = [r for r in rs if r["source"] == pair][0]
        un = [r for r in rs if "uncalibrated" in r["source"]][0]
        ca = [r for r in rs if "calibrated" in r["source"] and "uncalibrated" not in r["source"]][0]
        panels.append((title, [("real", real), ("sim uncal.", un), ("sim cal.", ca)], f"loss {float(un['loss_vs_real']):.1f} -> {float(ca['loss_vs_real']):.1f}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(9.2, 3.4) if compact else (3.4 * len(panels) * scale, 4.6 * scale), sharey=True)
    for ax, (title, bars, lab) in zip(axes, panels):
        for i, (name, r) in enumerate(bars):
            bottom = 0
            for k in ("correct", "missing", "wrong"):
                v = float(r[k])
                ax.bar(i, v, bottom=bottom, color=COL[k], width=0.7, label=k if (ax is axes[0] and i == 0) else None)
                if v > 6:
                    ax.text(i, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white")
                bottom += v
        ax.set_xticks(range(3))
        ax.set_xticklabels([b[0].replace("sim ", "").replace("uncal.", "uncal") for b in bars] if compact else [b[0] for b in bars], fontsize=8 if compact else 9)
        ax.set_title(title, fontsize=8 if compact else 10)
        ax.set_xlabel(lab.replace("loss ", "loss ") + " pp", fontsize=7.5 if compact else 9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of mask pixels" if compact else "% of GT-valid mask pixels (5% tolerance)", fontsize=8 if compact else 10)
    if compact:
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.42), ncol=3)
        fig.suptitle("Calibrated simulation vs real (transparent mask)", fontsize=12)
        fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
