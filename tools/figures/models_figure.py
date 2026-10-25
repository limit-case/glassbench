"""Bucket maps (5% tolerance) for the Figure 1 / alt images: RGB, raw sensor, Model A, Model B."""
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import models_eval as me  # noqa: E402
from figure1 import BUCKET_CMAP, load_by_id  # noqa: E402
from three_bucket import TOLS  # noqa: E402

IMAGES = [("TransCG L515", "scene2/148", "figure1"), ("CG d415", "real-test/d415/000000075", "figure1_alt")]
PANELS = [("raw", "Raw sensor"), ("A_rel_aligned", "Model A: Depth Anything V2-S\n(RGB only, global scale+shift)"),
          ("A_rel_local", "Model A, local alignment\n(fit on ring around mask)"),
          ("B_dfnet", "Model B: DFNet\n(RGB + raw depth)")]


def bucket_map(s, pred):
    b = me.bucket_codes(pred, s["depth_gt"], TOLS["rel5%"])
    out = np.select([b == 1, b == 2, b == 0], [1, 3, 2], 0).astype(np.uint8)
    out[~s["gt_valid"]] = 0
    return out


def main():
    pf = me.Predictors()
    os.makedirs(paths.FIGURES + "/models", exist_ok=True)
    fig_all, axs_all = plt.subplots(len(IMAGES), 5, figsize=(25, 4.6 * len(IMAGES)))
    for r, (cam, cid, name) in enumerate(IMAGES):
        s = load_by_id(cam, cid)
        preds, _ = pf(s)
        for fig, axs in ((fig_all, axs_all[r]), (None, None)):
            if fig is None:
                fig, axs = plt.subplots(1, 5, figsize=(25, 4.6))
            axs[0].imshow(s["rgb"])
            axs[0].set_title(f"RGB: {cam} {cid}", fontsize=11)
            for ax, (key, title) in zip(axs[1:], PANELS):
                m = bucket_map(s, preds[key])
                ax.imshow(m, cmap=BUCKET_CMAP, vmin=0, vmax=3, interpolation="nearest")
                inside = m[s["mask"] & s["gt_valid"]]
                ax.set_title(f"{title}\nin mask: {100*(inside==3).mean():.0f}% wrong, {100*(inside==2).mean():.0f}% missing, "
                             f"{100*(inside==1).mean():.0f}% correct", fontsize=10)
            for ax in axs:
                ax.contour(s["mask"], levels=[0.5], colors="cyan", linewidths=0.7)
                ax.axis("off")
            if fig is not fig_all:
                fig.legend(handles=[Patch(color="tab:green", label="correct (5%)"), Patch(color="tab:red", label="confidently wrong"),
                                    Patch(color="0.55", label="missing"), Patch(facecolor="white", edgecolor="0.5", label="no GT")],
                           loc="lower center", ncol=4, frameon=False)
                fig.tight_layout(rect=(0, 0.06, 1, 1))
                fig.savefig(f"{paths.FIGURES}/models/{name}_models.png", dpi=130)
                plt.close(fig)
    fig_all.legend(handles=[Patch(color="tab:green", label="correct (5%)"), Patch(color="tab:red", label="confidently wrong"),
                            Patch(color="0.55", label="missing"), Patch(facecolor="white", edgecolor="0.5", label="no GT")],
                   loc="lower center", ncol=4, frameon=False)
    fig_all.tight_layout(rect=(0, 0.04, 1, 1))
    fig_all.savefig(paths.FIGURES + "/models/models_comparison.png", dpi=130)


if __name__ == "__main__" and __import__("sys").argv[1:] != ["report"]:
    main()


def report_figure(out=os.path.join(paths.ROOT, "report", "figs", "models_report.png")):
    """Compact 4-column version for the report: RGB, raw sensor, Model A (local alignment), Model B (DFNet)."""
    pf = me.Predictors()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = [("raw", "Raw sensor"), ("A_rel_local", "Model A (RGB only)"), ("B_dfnet", "Model B (DFNet)")]
    fig, axs = plt.subplots(len(IMAGES), 4, figsize=(11, 2.9 * len(IMAGES)))
    for r, (cam, cid, name) in enumerate(IMAGES):
        s = load_by_id(cam, cid)
        preds, _ = pf(s)
        axs[r][0].imshow(s["rgb"])
        axs[r][0].set_title(f"RGB ({cam})", fontsize=11)
        for ax, (key, title) in zip(axs[r][1:], cols):
            mp = bucket_map(s, preds[key])
            ax.imshow(mp, cmap=BUCKET_CMAP, vmin=0, vmax=3, interpolation="nearest")
            inside = mp[s["mask"] & s["gt_valid"]]
            ax.set_title(f"{title}\nwrong {100*(inside==3).mean():.0f}% / missing {100*(inside==2).mean():.0f}% / correct {100*(inside==1).mean():.0f}%", fontsize=10)
        for ax in axs[r]:
            ax.contour(s["mask"], levels=[0.5], colors="cyan", linewidths=0.7)
            ax.axis("off")
    fig.legend(handles=[Patch(color="tab:green", label="correct (5%)"), Patch(color="tab:red", label="confidently wrong"),
                        Patch(color="0.55", label="missing"), Patch(facecolor="white", edgecolor="0.5", label="no GT")],
               loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=200)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "report":
    report_figure()
