"""Failure flagging: can the ClearGrasp mask-segmentation network flag the pixels where raw depth is confidently wrong?

Runs datasets/cleargrasp-checkpoints/mask_segmentation (DRN DeepLabV3, RGB only, trained on synthetic data) on the
ClearGrasp real RGB images and compares its predicted transparent mask with (a) the GT mask and (b) the raw
sensor's confidently-wrong pixels.
"""
import csv
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402
from models_eval import MODELS_DIR, bucket_codes, device  # noqa: E402
from three_bucket import TOLS  # noqa: E402

SEG_DIR = os.path.join(MODELS_DIR, "cleargrasp", "pytorch_networks", "masks")
CKPT = os.path.join(loader.DATA, "cleargrasp-checkpoints", "mask_segmentation", "checkpoint_mask.pth")
SIZE = 256  # config: imgHeight = imgWidth = 256, nearest resize, ToTensor() only (no mean/std)


def build():
    sys.path.insert(0, SEG_DIR)
    from modeling import deeplab
    model = deeplab.DeepLab(num_classes=2, backbone="drn", sync_bn=False, freeze_bn=False)
    # The checkpoint pickle only references torch tensor/OrderedDict globals (audited with pickletools).
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval().to(device())


@torch.no_grad()
def predict_mask(model, rgb):
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    x = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device())
    pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def main():
    model = build()
    cg = loader.list_cleargrasp()
    acc, per_img = {}, {}
    for i, (cam, split, sid) in enumerate(cg):
        s = loader.load_cleargrasp(split, sid)
        label = "CG d415" if cam == "d415" else f"CG d435 {split.split('-')[1].split('/')[0]}"
        pm, gm = predict_mask(model, s["rgb"]), s["mask"]
        inter, union = int((pm & gm).sum()), int((pm | gm).sum())
        row = dict(inter=inter, union=union, pred=int(pm.sum()), gt=int(gm.sum()))
        per_img.setdefault((label, "iou"), []).append(inter / union if union else np.nan)
        a = acc.setdefault((label, "mask"), {})
        for k, v in row.items():
            a[k] = a.get(k, 0) + v
        for tol, tol_fn in TOLS.items():
            b = bucket_codes(s["depth_raw"], s["depth_gt"], tol_fn)
            v = s["gt_valid"]
            wrong, correct = v & (b == 2), v & (b == 1)
            c = dict(wrong=wrong.sum(), wrong_flagged=(wrong & pm).sum(),
                     wrong_in_gt=(wrong & gm).sum(), wrong_in_gt_flagged=(wrong & gm & pm).sum(),
                     correct=correct.sum(), correct_flagged=(correct & pm).sum())
            a2 = acc.setdefault((label, tol), {})
            for k, val in c.items():
                a2[k] = a2.get(k, 0) + int(val)
            if c["wrong_in_gt"]:
                per_img.setdefault((label, tol, "recall_in_gt"), []).append(c["wrong_in_gt_flagged"] / c["wrong_in_gt"])
        if i % 50 == 0:
            print(f"{i}/{len(cg)}", flush=True)

    rows = []
    for label in dict.fromkeys(k[0] for k in acc):
        m = acc[label, "mask"]
        ious = np.array(per_img[label, "iou"])
        base = dict(camera=label, images=len(ious), mask_iou_pooled=100 * m["inter"] / m["union"],
                    mask_iou_img_med=100 * np.nanmedian(ious), mask_iou_img_p10=100 * np.nanpercentile(ious, 10),
                    mask_iou_img_p90=100 * np.nanpercentile(ious, 90),
                    pred_coverage_vs_gt=100 * m["pred"] / m["gt"])
        for tol in TOLS:
            c = acc[label, tol]
            r = per_img.get((label, tol, "recall_in_gt"), [np.nan])
            rows.append(dict(base, tol=tol,
                             wrong_px_in_gt_mask_pct=100 * c["wrong_in_gt"] / c["wrong"],
                             flag_recall_in_gt_mask=100 * c["wrong_in_gt_flagged"] / c["wrong_in_gt"],
                             flag_recall_in_gt_mask_img_med=100 * np.median(r),
                             flag_recall_all_wrong=100 * c["wrong_flagged"] / c["wrong"],
                             correct_px_flagged_pct=100 * c["correct_flagged"] / c["correct"],
                             flagged_returned_px_wrong_pct=100 * c["wrong_flagged"] / (c["wrong_flagged"] + c["correct_flagged"])))
    with open(paths.data("flagging_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
