"""Headline numbers, read from the CSVs in data/. Every number in the README and in data/benchmark-stats.json comes from here."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402

CAMERAS = ["CG d415", "CG d435 test", "CG d435 val", "TransCG D435", "TransCG L515"]
RELEASE = "0.1.0"


def rows(name):
    with open(paths.data(name)) as f:
        return list(csv.DictReader(f))


def compute():
    tb = {(r["camera"], r["region"], r["tol"]): r for r in rows("three_bucket_summary.csv")}
    ms = {(r["model"], r["camera"], r["region"], r["tol"]): r for r in rows("models_summary.csv")}
    fl = [r for r in rows("flagging_summary.csv") if r["tol"] == "rel5%"]
    ch = [r for r in rows("calibration_heldout.csv") if r["region"] == "mask" and r["tol"] == "rel5%"]
    groups = []
    for c in CAMERAS:
        r = tb[c, "mask", "rel5%"]
        r1 = tb[c, "mask", "abs1cm"]
        cor, mis, wr = (float(r[k]) for k in ("correct_pooled", "missing_pooled", "wrong_pooled"))
        groups.append(dict(
            camera=c, images=int(r["images_with_gt"]), correct_pct=cor, missing_pct=mis, wrong_pct=wr,
            wrong_among_returned_pct=100 * wr / (cor + wr), wrong_too_far_pct=float(r["wrong_far_share_pooled"]),
            error_cm_median=float(r["wrong_diff_cm_med"]), error_cm_p90=float(r["wrong_diff_cm_p90"]),
            wrong_pct_1cm=float(r1["wrong_pooled"])))
    wrongs = [g["wrong_pct"] for g in groups]
    by = {g["camera"]: g for g in groups}
    tc, d415 = "TransCG D435 (test scenes)", "CG d415"

    def held(pair, which):
        for r in ch:
            if r["pair"] == pair and r["source"] != pair and which in r["source"] and (which != "calibrated" or "uncalibrated" not in r["source"]):
                return float(r["loss_vs_real"])

    pairs = {"d415": "real CG d415 test", "d435": "real CG d435 test", "l515": "real TransCG L515 test scenes"}
    return dict(
        release=RELEASE,
        protocol="Raw depth on the transparent-object mask, GT-valid pixels, 5% tolerance, pooled over pixels.",
        headline=dict(
            real_images=sum(g["images"] for g in groups), evaluation_sets=len(groups),
            wrong_pct_min=min(wrongs), wrong_pct_max=max(wrongs),
            wrong_too_far_pct_min=min(g["wrong_too_far_pct"] for g in groups),
            error_cm_median_min=min(g["error_cm_median"] for g in groups), error_cm_median_max=max(g["error_cm_median"] for g in groups),
            missing_pct_cg_d415=by["CG d415"]["missing_pct"], missing_pct_transcg_l515=by["TransCG L515"]["missing_pct"],
            wrong_pct_transcg_l515=by["TransCG L515"]["wrong_pct"]),
        groups=groups,
        models=dict(
            dfnet_transcg_d435_test_wrong_pct=dict(raw=float(ms["raw", tc, "mask", "rel5%"]["wrong"]), dfnet=float(ms["B_dfnet", tc, "mask", "rel5%"]["wrong"])),
            dfnet_cg_d415_wrong_pct=dict(raw=float(ms["raw", d415, "mask", "rel5%"]["wrong"]), dfnet=float(ms["B_dfnet", d415, "mask", "rel5%"]["wrong"])),
            flag_recall_all_wrong_pct_min=min(float(r["flag_recall_all_wrong"]) for r in fl),
            flag_recall_all_wrong_pct_max=max(float(r["flag_recall_all_wrong"]) for r in fl),
            predicted_mask_area_vs_gt_pct_min=min(float(r["pred_coverage_vs_gt"]) for r in fl),
            predicted_mask_area_vs_gt_pct_max=max(float(r["pred_coverage_vs_gt"]) for r in fl)),
        calibration_heldout_loss_pp={k: dict(uncalibrated=held(p, "uncalibrated"), calibrated=held(p, "calibrated")) for k, p in pairs.items()},
    )
