"""Evaluate existing depth models against the three-bucket scheme and compare with the raw sensor.

Model A: Depth Anything V2 Small (RGB only), relative (scale+shift aligned on non-mask GT-valid pixels)
         and metric-indoor (as is, and scale+shift aligned).
Model B: TransCG DFNet (RGB + raw depth), official inference procedure.
Weights live in ~/glassbench_models/ (outside the repo).
"""
import argparse
import csv
import json
import os
import random
import re
import sys

HF_HOME = os.path.expanduser("~/glassbench_models/hf")
os.environ.setdefault("HF_HOME", HF_HOME)

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import NearestNDInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402
from three_bucket import DIFF_EDGES, TOLS, hist_quantile, jobs  # noqa: E402

MODELS_DIR = os.path.expanduser("~/glassbench_models")
DFNET_REPO = os.path.join(MODELS_DIR, "transcg")
DFNET_CKPT = os.path.join(MODELS_DIR, "dfnet.pth")
DA2_REL = "depth-anything/Depth-Anything-V2-Small-hf"
DA2_METRIC = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SEED = 0
N_TRANSCG = 300
MODEL_NAMES = ["raw", "A_rel_aligned", "A_metric", "A_metric_aligned", "A_rel_local", "A_metric_local", "B_dfnet"]
RING_PX = 40  # local-alignment variant: GT-valid pixels within this many px of the mask, outside it
MAX_DEPTH_FOR_DISP = 10.0  # disparity floor 1/10 m when inverting aligned relative output


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- predictors
class DA2:
    def __init__(self, repo):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo).eval()
        self.dev = device()
        self.model.to(self.dev)

    @torch.no_grad()
    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        inputs = self.proc(images=Image.fromarray(rgb), return_tensors="pt").to(self.dev)
        out = self.model(**inputs)
        post = self.proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]["predicted_depth"]
        return post.float().cpu().numpy()


class DFNetRunner:
    """Official TransCG inference (inference.py), re-implemented without its CUDA/config plumbing."""

    def __init__(self, coeff=3.0, image_size=(320, 240), dmin=0.3, dmax=1.0):
        sys.path.insert(0, DFNET_REPO)
        from models.DFNet import DFNet
        self.dev = device()
        self.model = DFNet(in_channels=4, hidden_channels=64, L=5, k=12)
        # The checkpoint pickle only references benign globals (OrderedDict, torch tensor rebuild,
        # numpy scalar/dtype, _codecs.encode); it was audited with pickletools before loading.
        ck = torch.load(DFNET_CKPT, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", "", 1): v for k, v in ck["model_state_dict"].items()}
        self.model.load_state_dict(sd)
        self.epoch = ck.get("epoch")
        self.model.eval().to(self.dev)
        self.coeff, self.image_size, self.dmin, self.dmax = coeff, image_size, dmin, dmax

    @torch.no_grad()
    def __call__(self, rgb, depth):
        h, w = depth.shape
        rgb_r = cv2.resize(rgb.astype(np.float32), self.image_size, interpolation=cv2.INTER_NEAREST)
        d = cv2.resize(depth.astype(np.float32), self.image_size, interpolation=cv2.INTER_NEAREST)
        d = np.where(d < self.dmin, 0, d)
        d = np.where(d > self.dmax, 0, d)
        d[np.isnan(d)] = 0
        avail = d[d > 0]
        mu = avail.mean() if avail.size else 0
        std = avail.std() if avail.size else 1
        d = np.where(d < mu - self.coeff * std, 0, d)
        d = np.where(d > mu + self.coeff * std, 0, d)
        pts = np.where(d > 0)
        if pts[0].size:
            d = NearestNDInterpolator(np.transpose(pts), d[pts])(*np.indices(d.shape))
        nmin = d.min() - 0.5 * d.std() - 1e-6
        nmax = d.max() + 0.5 * d.std() + 1e-6
        d = (d - nmin) / (nmax - nmin)
        rgb_t = torch.FloatTensor((rgb_r / 255.0).transpose(2, 0, 1)).to(self.dev).unsqueeze(0)
        d_t = torch.FloatTensor(d).to(self.dev).unsqueeze(0)
        res = self.model(rgb_t, d_t).squeeze(0).cpu().numpy()
        res = res * (nmax - nmin) + nmin
        return cv2.resize(res, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- alignment
def _lsq(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _ref_pixels(s, rng_seed=0, cap=200_000, ring=False):
    """GT-valid pixels outside the transparent mask (for TransCG, GT == raw there).
    ring=True: only those within RING_PX of the mask (falls back to all if fewer than 1000)."""
    ref = s["gt_valid"] & ~s["mask"]
    if ring:
        near = cv2.dilate(s["mask"].astype(np.uint8), np.ones((2 * RING_PX + 1, 2 * RING_PX + 1), np.uint8)) > 0
        if (ref & near).sum() >= 1000:
            ref = ref & near
    idx = np.flatnonzero(ref.ravel())
    if idx.size > cap:
        idx = np.random.default_rng(rng_seed).choice(idx, cap, replace=False)
    return idx


def align_relative(pred_disp, s, ring=False):
    """Fit gt_disparity ~ a*pred + b on reference pixels, return depth = 1/disparity, a, and ref RMSE (m)."""
    idx = _ref_pixels(s, ring=ring)
    gt = s["depth_gt"].ravel()[idx].astype(np.float64)
    a, b = _lsq(pred_disp.ravel()[idx].astype(np.float64), 1.0 / gt)
    depth = (1.0 / np.maximum(a * pred_disp + b, 1.0 / MAX_DEPTH_FOR_DISP)).astype(np.float32)
    rmse = float(np.sqrt(np.mean((depth.ravel()[idx] - gt) ** 2)))
    return depth, a, rmse


def align_metric(pred, s, ring=False):
    idx = _ref_pixels(s, ring=ring)
    gt = s["depth_gt"].ravel()[idx].astype(np.float64)
    a, b = _lsq(pred.ravel()[idx].astype(np.float64), gt)
    depth = (a * pred + b).astype(np.float32)
    return depth, a, float(np.sqrt(np.mean((depth.ravel()[idx] - gt) ** 2)))


class Predictors:
    def __init__(self):
        self.rel = DA2(DA2_REL)
        self.metric = DA2(DA2_METRIC)
        self.dfnet = DFNetRunner()

    def __call__(self, s):
        """Return ({model: depth in meters}, {alignment diagnostics})."""
        out, diag = {"raw": s["depth_raw"]}, {}
        rel = self.rel(s["rgb"])
        out["A_rel_aligned"], a, rmse = align_relative(rel, s)
        diag["A_rel_aligned"] = dict(slope=a, ref_rmse_cm=100 * rmse)
        met = self.metric(s["rgb"]).astype(np.float32)
        out["A_metric"] = met
        out["A_metric_aligned"], a, rmse = align_metric(met, s)
        diag["A_metric_aligned"] = dict(slope=a, ref_rmse_cm=100 * rmse)
        out["A_rel_local"], a, rmse = align_relative(rel, s, ring=True)
        diag["A_rel_local"] = dict(slope=a, ref_rmse_cm=100 * rmse)
        out["A_metric_local"], a, rmse = align_metric(met, s, ring=True)
        diag["A_metric_local"] = dict(slope=a, ref_rmse_cm=100 * rmse)
        out["B_dfnet"] = self.dfnet(s["rgb"], s["depth_raw"])
        return out, diag


# ---------------------------------------------------------------- evaluation
def bucket_codes(pred, gt, tol_fn):
    """0 missing (pred <= 0 or not finite), 1 correct, 2 confidently wrong."""
    ok = np.isfinite(pred) & (pred > 0)
    correct = ok & (np.abs(np.where(ok, pred, 0) - gt) <= tol_fn(gt))
    return np.where(~ok, 0, np.where(correct, 1, 2)).astype(np.int8)


def evaluate_image(s, preds, has_control):
    """{(model, region, tol): dict(T=3x3 raw-bucket x model-bucket, wrong_far, still_far, hist)}"""
    gt = s["depth_gt"]
    regions = {"mask": s["mask"] & s["gt_valid"]}
    if has_control:
        regions["nonmask"] = ~s["mask"] & s["gt_valid"]
    out = {}
    for tol, tol_fn in TOLS.items():
        raw_b = bucket_codes(s["depth_raw"], gt, tol_fn)
        for model, pred in preds.items():
            mb = raw_b if model == "raw" else bucket_codes(pred, gt, tol_fn)
            for region, m in regions.items():
                rb, mm = raw_b[m].astype(np.int64), mb[m].astype(np.int64)
                T = np.bincount(rb * 3 + mm, minlength=9).reshape(3, 3)
                p, g = pred[m], gt[m]
                far = (mm == 2) & (p > g)
                out[model, region, tol] = dict(
                    T=T, wrong_far=int(far.sum()), still_far=int((far & (rb == 2)).sum()),
                    hist=np.histogram(100 * (p[mm == 2] - g[mm == 2]), DIFF_EDGES)[0].astype(np.int32))
    return out


def group_labels(label, sample_id, scene_split):
    if label.startswith("TransCG"):
        sp = scene_split[int(re.match(r"scene(\d+)", sample_id).group(1))]
        return [f"{label} (all sampled)", f"{label} ({sp} scenes)"]
    return [label]


def sample_jobs():
    rng = random.Random(SEED)
    all_jobs = list(jobs())
    out = [j for j in all_jobs if j[1] == "cg"]
    for cam in loader.TRANSCG_CAMERAS:
        js = [j for j in all_jobs if j[0] == f"TransCG {cam}"]
        out += rng.sample(js, N_TRANSCG)
    return out


def scene_splits():
    return {i: json.load(open(os.path.join(loader.TRANSCG_ROOT, f"scene{i}", "metadata.json")))["split"]
            for i in range(1, 11)}


def load_job(j):
    label, kind, args = j
    return loader.load_cleargrasp(*args) if kind == "cg" else loader.load_transcg(*args)


def run(limit=None):
    preds_fn = Predictors()
    print(f"device={device()} DFNet checkpoint epoch={preds_fn.dfnet.epoch}", flush=True)
    split = scene_splits()
    js = sample_jobs()
    if limit:
        js = js[:limit] + [j for j in js if j[0].startswith("TransCG")][:limit]
    acc, per_img, align = {}, {}, {}
    for i, j in enumerate(js):
        s = load_job(j)
        preds, diag = preds_fn(s)
        ev = evaluate_image(s, preds, has_control=j[1] == "cg")
        for g in group_labels(j[0], s["id"], split):
            for key, d in ev.items():
                a = acc.setdefault((key[0], g, key[1], key[2]), dict(T=0, wrong_far=0, still_far=0, hist=0))
                a["T"] = a["T"] + d["T"]
                a["wrong_far"] += d["wrong_far"]
                a["still_far"] += d["still_far"]
                a["hist"] = a["hist"] + d["hist"]
                per_img.setdefault((key[0], g, key[1], key[2]), []).append(d["T"].sum(0))
            for m, dg in diag.items():
                align.setdefault((m, g), []).append(dg)
        if i % 50 == 0:
            print(f"{i}/{len(js)}", flush=True)
    return acc, per_img, align


def write_outputs(acc, per_img, align):
    names = ["missing", "correct", "wrong"]
    summ, trans = [], []
    for key in sorted(acc, key=lambda k: (k[1], k[2] != "mask", k[3], MODEL_NAMES.index(k[0]))):
        model, group, region, tol = key
        a = acc[key]
        col = a["T"].sum(0)  # model-bucket totals
        n = int(col.sum())
        imgs = np.array(per_img[key], dtype=float)
        valid = imgs.sum(1) > 0
        med = lambda c: float(np.median(100 * imgs[valid, c] / imgs[valid].sum(1))) if valid.any() else float("nan")
        summ.append(dict(
            model=model, camera=group, region=region, tol=tol, images=int(valid.sum()), pixels=n,
            correct=100 * col[1] / n, missing=100 * col[0] / n, wrong=100 * col[2] / n,
            correct_img_med=med(1), missing_img_med=med(0), wrong_img_med=med(2),
            wrong_far_share=100 * a["wrong_far"] / col[2] if col[2] else float("nan"),
            err_cm_med=hist_quantile(a["hist"], 0.5), err_cm_p90=hist_quantile(a["hist"], 0.9)))
        for r in range(3):
            tot = int(a["T"][r].sum())
            for m in range(3):
                trans.append(dict(
                    model=model, camera=group, region=region, tol=tol, raw_bucket=names[r], model_bucket=names[m],
                    pixels=int(a["T"][r, m]), pct_of_raw_bucket=100 * a["T"][r, m] / tot if tot else float("nan"),
                    still_too_far_pct=(100 * a["still_far"] / a["T"][2, 2]) if (r == 2 and m == 2 and a["T"][2, 2]) else ""))
    for path, rows in ((paths.data("models_summary.csv"), summ), (paths.data("models_transitions.csv"), trans)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    arows = []
    for (m, g), ds in sorted(align.items()):
        sl = np.array([d["slope"] for d in ds])
        rm = np.array([d["ref_rmse_cm"] for d in ds])
        arows.append(dict(model=m, camera=g, images=len(ds), ref_rmse_cm_med=np.median(rm), ref_rmse_cm_p90=np.percentile(rm, 90),
                          slope_med=np.median(sl), frac_nonpositive_slope=float((sl <= 0).mean())))
    with open(paths.data("models_alignment.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(arows[0]))
        w.writeheader()
        w.writerows(arows)
    return summ, trans, arows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="quick test: first N ClearGrasp + N TransCG samples")
    args = ap.parse_args()
    summ, trans, arows = write_outputs(*run(args.limit))
    print(f"{'model':<17}{'camera':<30}{'region':<8}{'tol':<7}{'imgs':>5} {'correct':>8}{'missing':>8}{'wrong':>8}{'far%':>7}{'err med/p90 cm':>16}")
    for r in summ:
        print(f"{r['model']:<17}{r['camera']:<30}{r['region']:<8}{r['tol']:<7}{r['images']:>5} {r['correct']:8.1f}{r['missing']:8.1f}"
              f"{r['wrong']:8.1f}{r['wrong_far_share']:7.1f}{r['err_cm_med']:9.1f}/{r['err_cm_p90']:.1f}")
