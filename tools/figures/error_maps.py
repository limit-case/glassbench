"""Figures: CG d435 non-mask outliers, and 4-panel bucket maps (5% tolerance) for 2 images per camera."""
import sys
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402

SEED = 0
BUCKET_CMAP = ListedColormap(["white", "tab:green", "0.55", "tab:red"])  # 0 gt-invalid, 1 correct, 2 missing, 3 wrong


def buckets(s, tol_rel=0.05):
    raw, gt, valid = s["depth_raw"], s["depth_gt"], s["gt_valid"]
    b = np.zeros(raw.shape, np.uint8)
    correct = (raw > 0) & (np.abs(raw - gt) <= tol_rel * gt)
    b[valid & correct] = 1
    b[valid & (raw == 0)] = 2
    b[valid & (raw > 0) & ~correct] = 3
    return b


def depth_panel(ax, d, vmin, vmax, title):
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("black")
    ax.imshow(np.ma.masked_equal(d, 0), cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)


def draw(s, path, last_panel, suptitle):
    gt = s["depth_gt"][s["gt_valid"]]
    vmin, vmax = np.percentile(gt, [2, 98])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    axes[0].imshow(s["rgb"])
    axes[0].set_title("RGB", fontsize=9)
    depth_panel(axes[1], s["depth_raw"], vmin, vmax, "raw depth (black = 0)")
    depth_panel(axes[2], s["depth_gt"], vmin, vmax, "GT depth (black = invalid)")
    last_panel(axes[3], s)
    for ax in axes:
        ax.contour(s["mask"], levels=[0.5], colors="cyan", linewidths=0.6)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def error_panel(ax, s):
    err = 100 * (s["depth_raw"] - s["depth_gt"])
    err = np.ma.masked_where(~s["gt_valid"] | (s["depth_raw"] == 0), np.clip(err, -10, 10))
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("0.8")
    im = ax.imshow(err, cmap=cm, vmin=-10, vmax=10)
    ax.set_title("raw - GT, cm (clipped +-10; gray = missing/invalid)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03)


def bucket_panel(ax, s):
    ax.imshow(buckets(s), cmap=BUCKET_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title("5% buckets: green correct / gray missing / red wrong / white no GT", fontsize=9)


def load_id(cid):
    """Load by loader id: 'real-test/d435/000000000' or 'scene1/1'."""
    if cid.startswith("real-"):
        split, sid = cid.rsplit("/", 1)
        return loader.load_cleargrasp(split, sid)
    raise ValueError(cid)


def outliers():
    with open(paths.data("three_bucket_per_image.csv")) as f:
        rows = [r for r in csv.DictReader(f)
                if r["camera"].startswith("CG d435") and r["region"] == "nonmask" and r["tol"] == "abs1cm" and int(r["n"])]
    for r in rows:
        r["rate"] = int(r["wrong"]) / int(r["n"])
    top = sorted(rows, key=lambda r: -r["rate"])[:5]
    print("CG d435 top-5 non-mask wrong rate (abs 1 cm):")
    for i, r in enumerate(top, 1):
        print(f"  {i}. {r['id']}  wrong={100 * r['rate']:.1f}%  missing={100 * int(r['missing']) / int(r['n']):.1f}%")
        s = load_id(r["id"])
        name = r["id"].replace("/", "_")
        draw(s, f"{paths.FIGURES}/cg_d435_outliers/rank{i}_{name}.png", error_panel,
             f"{r['id']}  non-mask wrong (1 cm): {100 * r['rate']:.1f}%")


def error_maps():
    rng = random.Random(SEED)
    cg = loader.list_cleargrasp()
    pools = {
        "CG d415": [lambda x=x: loader.load_cleargrasp(x[1], x[2]) for x in cg if x[0] == "d415"],
        "CG d435": [lambda x=x: loader.load_cleargrasp(x[1], x[2]) for x in cg if x[0] == "d435"],
    }
    for cam in loader.TRANSCG_CAMERAS:
        v = loader.list_transcg(cam)
        pools[f"TransCG {cam}"] = [lambda v=v, cam=cam: loader.load_transcg(v, cam) for v in v]
    for name, loads in pools.items():
        got = 0
        for load in rng.sample(loads, len(loads)):
            s = load()
            if (s["mask"] & s["gt_valid"]).sum() < 2000:
                continue
            draw(s, f"{paths.FIGURES}/error_maps/{name.replace(' ', '_')}_{s['id'].replace('/', '_')}.png", bucket_panel,
                 f"{name}  {s['id']}")
            got += 1
            if got == 2:
                break


if __name__ == "__main__":
    outliers()
    error_maps()
