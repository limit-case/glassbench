"""Three-bucket breakdown of raw depth vs GT on GT-valid pixels: correct / missing / confidently wrong."""
import sys
import os
import argparse
import csv
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402

TOLS = {"abs1cm": lambda gt: 0.01, "rel5%": lambda gt: 0.05 * gt}
# (raw - gt) in cm histogram, for exact pooled quantiles without keeping every pixel
DIFF_EDGES = np.arange(-500, 1500 + 0.05, 0.05)
FIELDS = ["camera", "id", "region", "tol", "n", "missing", "correct", "wrong", "wrong_far"]


def jobs():
    for cam, split, sid in loader.list_cleargrasp():
        # d415 is test-only; d435 is reported separately for real-test (primary) and real-val
        label = "CG d415" if cam == "d415" else f"CG d435 {split.split('-')[1].split('/')[0]}"
        yield label, "cg", (split, sid)
    for cam in loader.TRANSCG_CAMERAS:
        for v in loader.list_transcg(cam):
            yield f"TransCG {cam}", "transcg", (v, cam)


def count(raw, gt, sel):
    """Per-tolerance bucket counts over pixels in sel (already GT-valid), plus a (raw-gt) cm histogram of wrong ones."""
    raw, gt = raw[sel], gt[sel]
    missing = raw == 0
    out, hists = {}, {}
    for name, tol in TOLS.items():
        err = np.abs(raw - gt)
        present = ~missing
        correct = present & (err <= tol(gt))
        wrong = present & ~correct
        out[name] = dict(n=int(raw.size), missing=int(missing.sum()), correct=int(correct.sum()),
                         wrong=int(wrong.sum()), wrong_far=int((wrong & (raw > gt)).sum()))
        hists[name] = np.histogram(100 * (raw[wrong] - gt[wrong]), DIFF_EDGES)[0].astype(np.int32)
    return out, hists


def hist_quantile(h, q):
    c = np.cumsum(h)
    if c[-1] == 0:
        return float("nan")
    i = np.searchsorted(c, q * c[-1])
    return float(DIFF_EDGES[i] + 0.025)


def fit_bias(raw, gt, valid, mask):
    """Fit raw ~ a*gt + b*x + c*y + d on GT-valid, non-mask, raw>0 pixels; refit once after dropping the
    top 5% |residuals|. x, y are pixel offsets from the image center. Returns ((a, b, c, d), bias_rms_m) or None,
    where bias_rms is the RMS of the fitted model minus GT over the fit pixels (a and d are collinear when the
    GT depth range is narrow, so individually they can be large while the net bias is small)."""
    h, w = raw.shape
    sel = valid & ~mask & (raw > 0)
    yy, xx = np.nonzero(sel)
    if yy.size < 1000:
        return None
    A = np.stack([gt[sel], xx - w / 2, yy - h / 2, np.ones(yy.size)], 1).astype(np.float64)
    y = raw[sel].astype(np.float64)
    with np.errstate(all="ignore"):  # Accelerate BLAS on macOS emits spurious matmul warnings
        p = np.linalg.lstsq(A, y, rcond=None)[0]
        r = np.abs(A @ p - y)
        keep = r <= np.percentile(r, 95)
        p = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        bias_rms = float(np.sqrt(np.mean((A[keep] @ p - A[keep][:, 0]) ** 2)))
    return p, bias_rms


def apply_bias(raw, p):
    """Invert the fitted model on returned pixels: (raw - b*x - c*y - d) / a. Zeros (missing) stay zero."""
    h, w = raw.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a, b, c, d = p
    out = (raw - b * (xx - w / 2) - c * (yy - h / 2) - d) / a
    out[raw == 0] = 0
    return out.astype(np.float32)


def work(job):
    label, kind, args, erode, bias = job
    s = loader.load_cleargrasp(*args) if kind == "cg" else loader.load_transcg(*args)
    params = None
    if bias and kind == "cg":
        fit = fit_bias(s["depth_raw"], s["depth_gt"], s["gt_valid"], s["mask"])
        if fit is not None:
            p, bias_rms = fit
            h, w = s["depth_raw"].shape
            s["depth_raw"] = apply_bias(s["depth_raw"], p)
            params = dict(camera=label, id=s["id"], a=p[0], b_per_px=p[1], c_per_px=p[2], d_cm=100 * p[3],
                          tilt_cm=100 * float(np.hypot(p[1] * w, p[2] * h)), bias_rms_cm=100 * bias_rms)
    mask = s["mask"]
    if erode:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        mask = cv2.erode(mask.astype(np.uint8), k) > 0
    rows, hists = [], {}
    regions = {"mask": mask}
    if kind == "cg":
        regions["nonmask"] = ~s["mask"]  # control region stays the original complement
    for region, m in regions.items():
        out, h = count(s["depth_raw"], s["depth_gt"], m & s["gt_valid"])
        for tol, c in out.items():
            rows.append(dict(camera=label, id=s["id"], region=region, tol=tol, **c))
        if region == "mask":
            hists = h
    return rows, label, hists, params


def pct(a, b):
    return 100 * a / b if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erode", type=int, default=0, help="erode the transparent mask by N pixels")
    ap.add_argument("--bias-correct", action="store_true",
                    help="ClearGrasp only: per-image plane+scale fit of raw vs GT on non-mask pixels, inverted on raw")
    args = ap.parse_args()
    erode, bias = args.erode, args.bias_correct
    suffix = (f"_erode{erode}" if erode else "") + ("_biascorrect" if bias else "")
    out_per_image, out_summary = paths.data(f"three_bucket_per_image{suffix}.csv"), paths.data(f"three_bucket_summary{suffix}.csv")
    rows, diff_hist, params = [], {}, []
    with Pool() as p:
        for rs, label, hists, prm in p.imap(work, [(*j, erode, bias) for j in jobs()], chunksize=8):
            rows += rs
            if prm:
                params.append(prm)
            for tol, h in hists.items():
                diff_hist[label, tol] = diff_hist.get((label, tol), 0) + h
    with open(out_per_image, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)

    summary = []
    keys = sorted({(r["camera"], r["region"], r["tol"]) for r in rows},
                  key=lambda k: (k[0], k[1] != "mask", k[2]))
    for cam, region, tol in keys:
        rs = [r for r in rows if (r["camera"], r["region"], r["tol"]) == (cam, region, tol)]
        tot = {k: sum(r[k] for r in rs) for k in ["n", "missing", "correct", "wrong", "wrong_far"]}
        use = [r for r in rs if r["n"] > 0]
        d = dict(camera=cam, region=region, tol=tol, images=len(rs), images_with_gt=len(use), pixels=tot["n"])
        for b in ["correct", "missing", "wrong"]:
            per = np.array([100 * r[b] / r["n"] for r in use])
            d[f"{b}_pooled"] = pct(tot[b], tot["n"])
            d[f"{b}_med"], d[f"{b}_p10"], d[f"{b}_p90"] = np.percentile(per, [50, 10, 90])
        d["wrong_far_share_pooled"] = pct(tot["wrong_far"], tot["wrong"])
        sh = np.array([100 * r["wrong_far"] / r["wrong"] for r in use if r["wrong"] > 0])
        d["wrong_far_share_med"] = np.median(sh) if sh.size else float("nan")
        if region == "mask":
            h = diff_hist[cam, tol]
            d["wrong_diff_cm_med"], d["wrong_diff_cm_p90"] = hist_quantile(h, 0.5), hist_quantile(h, 0.9)
        summary.append(d)
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for d in summary for k in d)), restval="")
        w.writeheader()
        w.writerows(summary)

    print(f"mask erosion: {erode} px | bias correction (ClearGrasp only): {bias}")
    print("pooled % of GT-valid pixels (per-image median [p10-p90] in brackets)")
    print(f"{'camera':<13}{'region':<8}{'tol':<7}{'imgs':>5}  {'correct':<22}{'missing':<22}{'wrong':<22}{'wrong too far':>14}")
    for d in summary:
        cell = lambda b: f"{d[b + '_pooled']:5.1f} [{d[b + '_med']:4.1f} {d[b + '_p10']:4.1f}-{d[b + '_p90']:4.1f}]"
        print(f"{d['camera']:<13}{d['region']:<8}{d['tol']:<7}{d['images_with_gt']:>5}  {cell('correct'):<22}"
              f"{cell('missing'):<22}{cell('wrong'):<22}{d['wrong_far_share_pooled']:>13.1f}%")
    print("\nconfidently-wrong transparent pixels: raw - gt (cm), pooled")
    for d in summary:
        if d["region"] == "mask":
            print(f"{d['camera']:<13}{d['tol']:<7} median={d['wrong_diff_cm_med']:7.1f}  p90={d['wrong_diff_cm_p90']:7.1f}")
    if params:
        with open(paths.data(f"bias_params{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, list(params[0]))
            w.writeheader()
            w.writerows(params)
        print("\nfitted bias parameters: median [min-max] (tilt = plane gradient across the image, cm)")
        for cam in dict.fromkeys(p["camera"] for p in params):
            ps = [p for p in params if p["camera"] == cam]
            f = lambda k: (np.median([p[k] for p in ps]), min(p[k] for p in ps), max(p[k] for p in ps))
            print(f"{cam:<13} n={len(ps):<4} " + "  ".join(
                f"{k}={f(k)[0]:.4f} [{f(k)[1]:.4f}-{f(k)[2]:.4f}]" for k in ["a", "tilt_cm", "d_cm", "bias_rms_cm"]))


if __name__ == "__main__":
    main()
