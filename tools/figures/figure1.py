"""Publication figures: Figure 1 (see-through failure) and its stereo-camera contrast (glass goes missing)."""
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402
from error_maps import BUCKET_CMAP, buckets

TOL = 0.05


def per_image(cam):
    out = []
    with open(paths.data("three_bucket_per_image.csv")) as f:
        for r in csv.DictReader(f):
            if r["camera"] == cam and r["region"] == "mask" and r["tol"] == "rel5%" and int(r["n"]):
                n = int(r["n"])
                out.append(dict(id=r["id"], n=n, **{k: int(r[k]) / n for k in ("missing", "correct", "wrong")}))
    return out


def load_by_id(cam, cid):
    if cam.startswith("CG"):
        split, sid = cid.rsplit("/", 1)
        return loader.load_cleargrasp(split, sid)
    scene, view = cid.split("/")
    return loader.load_transcg(os.path.join(loader.TRANSCG_ROOT, scene, view), cam.split()[-1])


def render(s, path, caption, dpi=300, scale=1.0, compact=False, show_caption=True):
    # color range from raw and GT depth near the transparent objects, so the glass is not washed out by the background
    near = cv2.dilate(s["mask"].astype(np.uint8), np.ones((81, 81), np.uint8)) > 0
    vals = np.concatenate([s["depth_gt"][near & s["gt_valid"]], s["depth_raw"][near & (s["depth_raw"] > 0)]])
    vmin, vmax = np.percentile(vals, [1, 99])
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("black")
    fig = plt.figure(figsize=(16 * scale, (2.9 if show_caption else 2.3) if compact else 4.8 * scale), dpi=dpi)
    gs = fig.add_gridspec(1, 4, wspace=0.04, left=0.01, right=0.99, top=0.92 if compact else 0.94, bottom=(0.38 if show_caption else 0.30) if compact else 0.32)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes[0].imshow(s["rgb"])
    tf = 12 if compact else 11
    axes[0].set_title("RGB", fontsize=tf)
    axes[1].imshow(np.ma.masked_equal(s["depth_raw"], 0), cmap=cm, vmin=vmin, vmax=vmax)
    axes[1].set_title("Raw depth" if compact else "Raw sensor depth (black = no return)", fontsize=tf)
    im = axes[2].imshow(np.ma.masked_equal(s["depth_gt"], 0), cmap=cm, vmin=vmin, vmax=vmax)
    axes[2].set_title("Ground-truth depth" if compact else "Ground-truth depth (black = invalid)", fontsize=tf)
    cax = fig.add_axes([0.035, (0.235 if show_caption else 0.16), 0.2, 0.04] if compact else [0.34, 0.21, 0.32, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("depth (m), raw and GT panels" if compact else "depth (m), shared by raw and GT panels", fontsize=9 if compact else 10)
    if compact:
        cb.ax.tick_params(labelsize=9)
    axes[3].imshow(buckets(s, TOL), cmap=BUCKET_CMAP, vmin=0, vmax=3, interpolation="nearest")
    axes[3].set_title("Raw vs. GT (5%)" if compact else "Raw vs. ground truth, per pixel", fontsize=tf)
    for ax in axes:
        ax.contour(s["mask"], levels=[0.5], colors="cyan", linewidths=0.7)
        ax.axis("off")
    labels = (["correct", "confidently wrong", "missing", "no GT", "mask"] if compact else
              ["correct (within 5% of GT)", "confidently wrong (> 5% off)", "missing (no return)", "no GT", "transparent-object mask"])
    fig.legend(handles=[Patch(color="tab:green", label=labels[0]), Patch(color="tab:red", label=labels[1]), Patch(color="0.55", label=labels[2]),
                        Patch(facecolor="white", edgecolor="0.5", label=labels[3]), Patch(facecolor="none", edgecolor="cyan", label=labels[4])],
               loc="center left" if compact else "lower center", ncol=5, frameon=False, fontsize=10 if compact else 10, bbox_to_anchor=(0.30, 0.235 if show_caption else 0.16) if compact else (0.5, 0.05), columnspacing=1.2 if compact else 1.6)
    if show_caption:
        fig.text(0.5, 0.0 if compact else 0.008, caption, ha="center", fontsize=9 if compact else 10)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if compact:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    else:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)


def caption(cam, s, cid):
    m = s["mask"] & s["gt_valid"]
    b = buckets(s, TOL)[m]
    return (f"{cam}, {cid}; 5% tolerance. Inside the transparent-object mask: "
            f"{100 * (b == 3).mean():.0f}% confidently wrong, {100 * (b == 2).mean():.0f}% missing, {100 * (b == 1).mean():.0f}% correct.")


if __name__ == "__main__":
    if sys.argv[1:] == ["candidates"]:
        picks = [("TransCG L515", "scene3/232"), ("TransCG L515", "scene2/148"),
                 ("CG d415", "real-test/d415/000000075"), ("CG d415", "real-test/d415/000000034")]
        for cam, cid in picks:
            s = load_by_id(cam, cid)
            render(s, f"{paths.FIGURES}/candidates/{cam.replace(' ', '_')}_{cid.replace('/', '_')}.png", caption(cam, s, cid), dpi=60)
            print(cam, cid, caption(cam, s, cid))
    elif sys.argv[1:] == ["final"]:
        for cam, cid, path in [("TransCG L515", "scene2/148", paths.FIGURES + "/figure1.png"),
                               ("CG d415", "real-test/d415/000000075", paths.FIGURES + "/figure1_alt.png")]:
            s = load_by_id(cam, cid)
            render(s, path, caption(cam, s, cid))
            print(path, caption(cam, s, cid))
