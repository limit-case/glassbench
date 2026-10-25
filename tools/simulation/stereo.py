"""Simulated active-stereo depth: OpenCV StereoSGBM with left-right consistency and speckle filtering."""
import json
import os

import cv2
import numpy as np

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "configs", "sgbm.json")
DEFAULTS = dict(texture_min=0, min_disp=8, num_disp=96, block=7, uniqueness=10, speckle_window=100, speckle_range=2, disp12=1,
                prefilter_cap=31, min_depth=0.2, max_depth=2.0)


def load_params(variant=None):
    p = dict(DEFAULTS)
    if os.path.exists(CFG):
        p.update({k: v for k, v in json.load(open(CFG)).items() if not k.startswith("_")})
    if isinstance(p["num_disp"], dict):
        p["num_disp"] = p["num_disp"][variant]
    return p


def texture_std(left, win=9):
    """Local standard deviation of the 8-bit left IR image (contrast / texture measure)."""
    x = left.astype(np.float32)
    m = cv2.boxFilter(x, -1, (win, win), borderType=cv2.BORDER_REPLICATE)
    m2 = cv2.boxFilter(x * x, -1, (win, win), borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def sgbm_depth(left, right, fx, baseline, p=None, variant=None):
    p = p or load_params(variant)
    b = p["block"]
    sg = cv2.StereoSGBM_create(minDisparity=p["min_disp"], numDisparities=p["num_disp"], blockSize=b,
                               P1=8 * b * b, P2=32 * b * b, disp12MaxDiff=p["disp12"],
                               uniquenessRatio=p["uniqueness"], speckleWindowSize=p["speckle_window"],
                               speckleRange=p["speckle_range"], preFilterCap=p["prefilter_cap"],
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    # OpenCV leaves the leftmost (min_disp + num_disp) columns invalid; pad both images so pixels that do have a
    # right-image counterpart can be matched, then crop.
    pad = p["min_disp"] + p["num_disp"]
    lp = cv2.copyMakeBorder(left, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    rp = cv2.copyMakeBorder(right, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    disp = sg.compute(lp, rp)[:, pad:].astype(np.float32) / 16.0
    valid = disp > p["min_disp"]  # SGBM marks invalid as (min_disp - 1)
    depth = np.zeros_like(disp)
    depth[valid] = fx * baseline / disp[valid]
    depth[(depth < p["min_depth"]) | (depth > p["max_depth"])] = 0
    if p.get("texture_min", 0) > 0:
        depth[texture_std(left) < p["texture_min"]] = 0
    return depth
