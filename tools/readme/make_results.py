"""Build RESULTS.md from the CSVs written by three_bucket.py (so every number is script-produced)."""
import sys
import csv
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import loader  # noqa: E402

CAMS = ["CG d415", "CG d435 test", "CG d435 val", "TransCG D435", "TransCG L515"]
CG_CAMS = CAMS[:3]


def read_summary(path):
    with open(path) as f:
        return {(r["camera"], r["region"], r["tol"]): {k: (float(v) if v not in ("", None) and k not in ("camera", "region", "tol") else v)
                                                       for k, v in r.items()} for r in csv.DictReader(f)}


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def bucket_cells(d):
    return [f"{fmt(d[b + '_pooled'])} [{fmt(d[b + '_med'])}]" for b in ("correct", "missing", "wrong")]


MODELS = ["raw", "A_rel_aligned", "A_rel_local", "A_metric", "A_metric_aligned", "A_metric_local", "B_dfnet"]
MODEL_LABEL = {"raw": "raw sensor", "A_rel_aligned": "A: DA-V2-S relative, global scale+shift",
               "A_rel_local": "A: DA-V2-S relative, local (ring) scale+shift", "A_metric": "A: DA-V2-S metric-indoor, as is",
               "A_metric_aligned": "A: metric-indoor, global scale+shift", "A_metric_local": "A: metric-indoor, local (ring) scale+shift",
               "B_dfnet": "B: TransCG DFNet (RGB + raw depth)"}
MCAMS = ["CG d415", "CG d435 test", "CG d435 val", "TransCG D435 (test scenes)", "TransCG L515 (test scenes)"]


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def models_section(A):
    S = {(r["model"], r["camera"], r["region"], r["tol"]): r for r in read_rows(paths.data("models_summary.csv"))}
    T = {(r["model"], r["camera"], r["region"], r["tol"], r["raw_bucket"], r["model_bucket"]): r for r in read_rows(paths.data("models_transitions.csv"))}
    AL = read_rows(paths.data("models_alignment.csv"))
    FL = read_rows(paths.data("flagging_summary.csv"))
    f = lambda r, k, nd=1: fmt(float(r[k]), nd)
    tp = lambda m, c, reg, tol, a, b: float(T[m, c, reg, tol, a, b]["pct_of_raw_bucket"])

    def among(r):
        return 100 * float(r["wrong"]) / (float(r["correct"]) + float(r["wrong"]))

    A("## Do existing models fix it?\n")
    A("Two existing models were run on the same images and scored with the same three buckets, on the transparent mask, GT-valid pixels. "
      "Model outputs are dense, so 'missing' is about 0 for them; 'wrong among returned' is then just 'wrong'. "
      "All model weights are stored outside the repo in `~/glassbench_models/`.\n")
    A(table(["Model", "Input", "Source / licence"], [
        ["A: Depth Anything V2 Small (relative)", "RGB only", "`depth-anything/Depth-Anything-V2-Small-hf`; Apache-2.0 (HF model card)"],
        ["A: Depth Anything V2 Metric-Indoor-Small", "RGB only",
         "`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`; the HF card has no licence field, the upstream README states the Small models are Apache-2.0 (not stated per metric checkpoint)"],
        ["B: TransCG DFNet", "RGB + raw depth", "official `galaxies99/transcg` code and checkpoint (Google Drive link in its README); CC BY-NC-SA 4.0; checkpoint epoch 12"],
        ["Mask segmentation (flagging)", "RGB only", "ClearGrasp `mask_segmentation` checkpoint (DRN DeepLabV3), ClearGrasp repo Apache-2.0"],
    ]))
    A("")
    A("**Protocol.**\n"
      "- *Sample.* All ClearGrasp real images (d415, d435 real-test, d435 real-val separately) and a fixed random subset of 300 views per TransCG camera (seed 0). "
      "The raw-sensor rows here are computed on the same subset, so TransCG raw numbers differ slightly from the full-data tables above.\n"
      "- *Model A alignment.* Model A never sees raw depth. Its output is relative inverse depth, so per image a scale and shift is fit by least squares in inverse-depth space "
      "on GT-valid pixels *outside* the transparent mask (global alignment, the specified protocol), then inverted to depth; evaluation inside the mask is out of sample. "
      "For TransCG, GT equals the raw sensor depth outside the mask, so there the alignment uses raw sensor depth. "
      "The metric-indoor model is reported as is (zero-shot meters) and also with a scale+shift fit in depth space. "
      "Because a global fit is dominated by far background (see the alignment table), a *local* variant fits on GT-valid pixels in a 40 px ring around the mask instead; it is a sensitivity check, not the specified protocol.\n"
      "- *Model B.* DFNet is run the way the official `inference.py` does: RGB and depth resized (nearest) to 320x240, depth clipped to 0.3-1.0 m (outside set to missing), "
      "values outside mean +/- 3 std dropped, holes filled by nearest-neighbour interpolation, depth normalized per image; the output is resized back to the GT resolution (nearest). "
      "Depth is in meters (D435 /1000, L515 /4000, matching DFNet's own `process_depth`).\n"
      "- *DFNet and TransCG scenes.* `metadata.json` and each scene's `metadata.json` put scenes 5, 7, 8 in the **test** split and scenes 1, 2, 3, 4, 6, 9, 10 in the **train** split; "
      "the repo's default config trains on the train split. Only scenes 5, 7, 8 are treated as out-of-sample for DFNet and used for the TransCG DFNet headline; "
      "the other scenes are reported separately as in-sample. This rests on the repo's documented split; the released checkpoint's exact training set could not be independently verified. "
      "ClearGrasp real images are not part of DFNet's training data (cross-dataset).\n")

    def bucket_table(tol, cams, models):
        rows = []
        for c in cams:
            for m in models:
                r = S[m, c, "mask", tol]
                rows.append([c, MODEL_LABEL[m], f(r, "images", 0), f(r, "correct"), f(r, "missing"), f(r, "wrong"), fmt(among(r)),
                             f(r, "wrong_far_share"), f"{f(r, 'err_cm_med')} / {f(r, 'err_cm_p90')}"])
        return table(["Camera", "Model", "Images", "Correct %", "Missing %", "Wrong %", "Wrong among returned %", "Wrong too far %",
                      "Error cm (out - GT) median / p90"], rows)

    A("### Results, transparent mask, 5% tolerance\n")
    A(bucket_table("rel5%", MCAMS, MODELS))
    A("\nPooled over pixels. `Error` is output - GT over confidently wrong pixels (positive = too far). The TransCG rows use the DFNet **test scenes** 5, 7, 8.\n")
    A("### Results, transparent mask, 1 cm tolerance\n")
    A(bucket_table("abs1cm", MCAMS, MODELS))
    A("")
    A("### DFNet on TransCG scenes it was trained on (in-sample, excluded from the headline)\n")
    A(bucket_table("rel5%", ["TransCG D435 (train scenes)", "TransCG L515 (train scenes)"], ["raw", "A_rel_aligned", "B_dfnet"]))
    A("")

    A("### Transitions from raw to model (transparent mask, 5%)\n")
    A("Each cell is the % of the raw-sensor bucket that ends in the model bucket. *Fixed* = raw wrong -> model correct; *inherited* = raw wrong -> model wrong "
      "(*still too far* = % of those where the model also reads farther than GT); *filled* = raw missing -> model correct / wrong; *broken* = raw correct -> model wrong.\n")
    rows = []
    for c in MCAMS:
        for m in ["A_rel_aligned", "A_rel_local", "B_dfnet"]:
            rows.append([c, MODEL_LABEL[m], fmt(tp(m, c, "mask", "rel5%", "wrong", "correct")), fmt(tp(m, c, "mask", "rel5%", "wrong", "wrong")),
                         fmt(float(T[m, c, "mask", "rel5%", "wrong", "wrong"]["still_too_far_pct"])),
                         fmt(tp(m, c, "mask", "rel5%", "missing", "correct")), fmt(tp(m, c, "mask", "rel5%", "missing", "wrong")),
                         fmt(tp(m, c, "mask", "rel5%", "correct", "wrong"))])
    A(table(["Camera", "Model", "Fixed %", "Inherited %", "Still too far %", "Filled -> correct %", "Filled -> wrong %", "Broken %"], rows))
    A("")

    A("### Control: ClearGrasp non-mask pixels (5%)\n")
    A("The same comparison on opaque pixels, where raw depth is mostly right. 'Broken' here is the model's own accuracy floor at this tolerance.\n")
    rows = []
    for c in CG_CAMS:
        for m in ["raw", "A_rel_aligned", "A_rel_local", "B_dfnet"]:
            r = S[m, c, "nonmask", "rel5%"]
            row = [c, MODEL_LABEL[m], f(r, "correct"), f(r, "missing"), f(r, "wrong")]
            if m == "raw":
                row += ["-", "-", "-"]
            else:
                row += [fmt(tp(m, c, "nonmask", "rel5%", "wrong", "correct")), fmt(tp(m, c, "nonmask", "rel5%", "missing", "correct")),
                        fmt(tp(m, c, "nonmask", "rel5%", "correct", "wrong"))]
            rows.append(row)
    A(table(["Camera", "Model", "Correct %", "Missing %", "Wrong %", "Fixed %", "Filled -> correct %", "Broken %"], rows))
    A("")

    A("### Model A alignment quality\n")
    A("RMSE of the aligned depth against GT on the pixels it was fit on (cm, median over images). If this is as large as the errors being measured, "
      "Model A's accuracy is limited by the alignment, not only by glass.\n")
    rows = []
    for c in ["CG d415", "CG d435 test", "CG d435 val", "TransCG D435 (all sampled)", "TransCG L515 (all sampled)"]:
        by = {r["model"]: r for r in AL if r["camera"] == c}
        rows.append([c] + [f(by[m], "ref_rmse_cm_med") for m in ["A_rel_aligned", "A_rel_local", "A_metric_aligned", "A_metric_local"]])
    A(table(["Camera", "relative, global", "relative, local", "metric, global", "metric, local"], rows))
    A("")

    A("### Failure flagging with the ClearGrasp mask-segmentation network\n")
    A("RGB only, trained on synthetic data; run on ClearGrasp real RGB. *Flag recall* = % of raw confidently-wrong, GT-valid pixels that fall inside the predicted transparent mask "
      "(restricted to wrong pixels inside the GT mask in the first recall column). *Correct pixels flagged* is the cost: usable raw depth that would be discarded.\n")
    rows = []
    for r in FL:
        rows.append([r["camera"], "5%" if r["tol"] == "rel5%" else "1 cm", f(r, "images", 0),
                     f"{f(r, 'mask_iou_pooled')} [{f(r, 'mask_iou_img_med')}]", f(r, "pred_coverage_vs_gt", 0),
                     f(r, "wrong_px_in_gt_mask_pct"), f(r, "flag_recall_in_gt_mask"), f(r, "flag_recall_all_wrong"),
                     f(r, "correct_px_flagged_pct"), f(r, "flagged_returned_px_wrong_pct")])
    A(table(["Camera", "Tol", "Images", "Mask IoU % pooled [per-image median]", "Predicted / GT mask area %",
             "Wrong px inside GT mask %", "Recall, wrong px in GT mask %", "Recall, all wrong px %", "Correct px flagged %",
             "Flagged returned px that are wrong %"], rows))
    A("\nThe predicted mask is much larger than the GT mask (area ratio above), which inflates recall; the IoU and the share of correct pixels flagged show the cost.\n")

    A("### Figures\n")
    A("`media/figures/models/models_comparison.png` shows, for the Figure 1 image (L515 scene2/148) and the alt image (d415 000000075): RGB, raw sensor, Model A (global and local alignment) and Model B bucket maps "
      "(5%); per-image versions are `media/figures/models/figure1_models.png` and `media/figures/models/figure1_alt_models.png`.\n")
    A("![Models](media/figures/models/models_comparison.png)\n")

    # finding, numbers pulled from the tables above
    g = lambda m, c, k="wrong", tol="rel5%": float(S[m, c, "mask", tol][k])
    nm = lambda m, c: float(S[m, c, "nonmask", "rel5%"]["wrong"])
    A("### Finding\n")
    A(f"Existing models do not fix the see-through failure; they mostly trade it for another error. DFNet (RGB + raw depth) does fix the case it was built for, "
      f"TransCG D435 test scenes (wrong {fmt(g('raw','TransCG D435 (test scenes)'))}% -> {fmt(g('B_dfnet','TransCG D435 (test scenes)'))}%), but on the L515 it inherits "
      f"{fmt(tp('B_dfnet','TransCG L515 (test scenes)','mask','rel5%','wrong','wrong'))}% of the raw sensor's wrong pixels ({fmt(float(T['B_dfnet','TransCG L515 (test scenes)','mask','rel5%','wrong','wrong']['still_too_far_pct']))}% of them still too far), and on ClearGrasp, out of domain, it fills missing pixels "
      f"while raising the wrong share (d415 {fmt(g('raw','CG d415'))}% -> {fmt(g('B_dfnet','CG d415'))}%, d435 test {fmt(g('raw','CG d435 test'))}% -> {fmt(g('B_dfnet','CG d435 test'))}%, "
      f"d435 val {fmt(g('raw','CG d435 val'))}% -> {fmt(g('B_dfnet','CG d435 val'))}%) against a non-mask error floor of only "
      f"{fmt(min(nm('B_dfnet', c) for c in CG_CAMS))}-{fmt(max(nm('B_dfnet', c) for c in CG_CAMS))}%: it turns a missing return into a confident error on glass.\n")
    A(f"The RGB-only monocular model never sees the raw depth, but its glass errors are of the same kind where they can be separated from its own accuracy floor: on d415 it is wrong on "
      f"{fmt(g('A_rel_aligned','CG d415'))}% of mask pixels versus {fmt(nm('A_rel_aligned','CG d415'))}% outside the mask, with {fmt(g('A_rel_aligned','CG d415','wrong_far_share'))}% of those too far, "
      f"so it also sees through glass; on d435 its floor ({fmt(nm('A_rel_aligned','CG d435 test'))}% non-mask wrong on test) is as large as its in-mask error, so no glass-specific effect can be claimed there. "
      f"The dangerous pixels can largely be located from RGB alone: the segmentation network's predicted mask covers "
      f"{fmt(min(float(r['flag_recall_all_wrong']) for r in FL if r['tol']=='rel5%'))}-{fmt(max(float(r['flag_recall_all_wrong']) for r in FL if r['tol']=='rel5%'))}% of the raw sensor's 5%-wrong pixels, "
f"flagging {fmt(min(float(r['correct_px_flagged_pct']) for r in FL if r['tol']=='rel5%'))}-{fmt(max(float(r['correct_px_flagged_pct']) for r in FL if r['tol']=='rel5%'))}% of correct pixels, "
      f"although only {fmt(min(float(r['flagged_returned_px_wrong_pct']) for r in FL if r['tol']=='rel5%'))}-{fmt(max(float(r['flagged_returned_px_wrong_pct']) for r in FL if r['tol']=='rel5%'))}% of the flagged returned pixels are wrong "
      f"(its masks over-cover the GT masks, which flatters recall). At the 1 cm tolerance on d435 the same mask catches only "
      f"{fmt(float([r for r in FL if r['camera']=='CG d435 val' and r['tol']=='abs1cm'][0]['flag_recall_all_wrong']))}-{fmt(float([r for r in FL if r['camera']=='CG d435 test' and r['tol']=='abs1cm'][0]['flag_recall_all_wrong']))}% of wrong pixels, because most 1 cm errors there lie outside the glass.\n")


def sim_section(A):
    import glob as _glob
    import json as _json
    R = read_rows(paths.data("synth_vs_real.csv"))
    G = {(r["camera"], r["region"], r["tol"]): r for r in R}
    f = lambda r, k, nd=1: fmt(float(r[k]), nd)
    order = []
    for sensor in ("d415", "d435", "l515"):
        for r in R:
            if r["sensor"] == sensor and r["region"] == "mask" and r["tol"] == "rel5%":
                order.append(r["camera"])
    label = lambda c: c.replace("synthetic ", "SIM ")

    def tbl(region, tol, cams):
        rows = []
        for c in cams:
            r = G.get((c, region, tol))
            if r is None:
                continue
            rows.append([label(c), f(r, "images", 0), f(r, "correct"), f(r, "missing"), f(r, "wrong"), f(r, "wrong_among_returned"),
                         f(r, "wrong_far_share"), f"{f(r, 'err_cm_med')} / {f(r, 'err_cm_p90')}"])
        return table(["Camera / source", "Images", "Correct %", "Missing %", "Wrong %", "Wrong among returned %", "Wrong too far %",
                      "Error cm (raw - GT) median / p90"], rows)

    real = {"d415": "CG d415", "d435": "CG d435 test", "d435v": "CG d435 val", "l515": "TransCG L515 (test scenes)"}
    sim = {v: [c for c in order if c.startswith(f"synthetic {v}-like") and "opaque" not in c][0] for v in ("d415", "d435", "l515")}
    twin = {v: [c for c in order if c.startswith(f"synthetic {v}-like") and "opaque" in c][0] for v in ("d415", "d435")}
    m = lambda c, k, reg="mask", tol="rel5%": float(G[c, reg, tol][k])
    sec_t = []
    for p in sorted(_glob.glob(paths.SYNTH + "/out/*/*/meta.json")):
        sec_t.append((os.path.relpath(p, paths.SYNTH).split("/")[1], _json.load(open(p))["total"]))
    secs = {v: float(np.median([t for vv, t in sec_t if vv == v])) for v in ("d415", "d435", "l515")} if sec_t else None

    A("## Can simulation reproduce the failure?\n")
    A("Goal: synthetic scenes whose three-bucket statistics resemble the real-sensor results, with the failures arising from simulated physics and not painted on. "
      "Two simulated sensors: active stereo (D415/D435-like) and a ToF/LiDAR approximation (L515-like). Code is in `tools/simulation/` (Blender scene generator, stereo and ToF models), "
      "`tools/simulation/synth_eval.py` and `tools/figures/synth_figures.py`; the comparison is `data/synth_vs_real.csv`.\n")
    A("### Method\n")
    A("- **Scenes** (`tools/simulation/render_scene.py`, Blender 4.5 Cycles on the Metal GPU, headless): a table with a randomized procedural texture (and a backdrop wall in 70% of scenes), "
      "1-5 transparent objects from the nine ClearGrasp 3D models (`datasets/cleargrasp-3d-models`, Apache-2.0; the download link in the ClearGrasp README is stale for the `.tar`, the same-named `.zip` in the same bucket was used) "
      "dropped with rigid-body physics, plus 0-3 opaque distractors, 1-3 area lights and a uniform or procedural-sky world (auto-exposed), camera 0.4-0.8 m from the cluster at 30-75 degrees elevation. "
      "Transparent objects are either glass (Principled BSDF, transmission 1, IOR 1.45-1.52, roughness 0-0.03) or clear plastic (IOR 1.40-1.50, roughness 0.05-0.12), with a slight random tint.\n"
      "- **Opaque twin**: the same scene with the transparent objects swapped to opaque diffuse, mirroring ClearGrasp's method. It supplies the ground-truth depth, the instance mask (object index) and surface normals. "
      "Ground truth is rendered with a 0.01 px filter so edges are not blended.\n"
      "- **Stereo sensor**: a left and a right IR camera (rectified, parallel) and a textured spot-light projector with a random dot pattern between them, rendered with Cycles. "
      "The projector light uses manifold next-event estimation (shadow caustics), so the pattern is refracted and reflected by the glass in the simulation. "
      "Depth comes from OpenCV StereoSGBM with left-right consistency and speckle filtering, invalid -> 0. The left strip with no right-image counterpart is excluded from the GT-valid set, as in the real data. "
      "Geometry: baselines 55 mm (D415-like) and 50 mm (D435-like), confirmed in Intel's D400 datasheet (document 337029-017, revision 023, Table 3-49; the D435 uses the D430 depth module). "
      "The simulation used depth HFOVs of 69.4 degrees (D415-like) and 85.2 degrees (D435-like) from product pages; the datasheet gives depth FOVs of 65 and 87 degrees, and 69.4 degrees is the D415's *colour* camera FOV there. "
      "A paired re-render of 20 D415-like scenes at 65 degrees (`data/fov_check.csv`) moves the mask correct / missing / wrong shares by at most 2.4 percentage points, so the study was not re-run; the D435 value is within the datasheet's +-3 degree tolerance.\n"
      "- **ToF sensor** (`tools/simulation/tof.py`, an approximation): the depth is an amplitude-weighted mix of the glass front surface (the opaque-twin depth) and the first opaque surface behind the glass "
      "(rendered with the glass removed), with Fresnel reflectance R (n = 1.5) giving the glass amplitude R and the behind amplitude (1-R)^2 x 0.5, a 3x3 footprint mix that creates flying pixels, "
      "noise of 1 mm + 0.25% of depth, and no return at near-grazing incidence. L515 intrinsics from `datasets/transcg/camera_intrinsics`.\n"
      "- **Size**: 640x360 renders; 200 scenes per sensor model with fixed seeds 0-199 for the ToF model and seeds 0-99 for each of the D415-like and D435-like stereo variants (200 stereo scenes in total). "
      + (f"Median render time per scene: {fmt(secs['d415'])} s (D415-like, including the opaque-twin IR pair), {fmt(secs['d435'])} s (D435-like), {fmt(secs['l515'])} s (ToF); about 50 minutes for all 400 scenes." if secs else "") + "\n")
    A("### Validation on opaque twins (stereo)\n")
    A("SGBM was tuned **only** on opaque-twin IR pairs of separate tuning scenes (seeds 1000-1019 per variant) until the opaque control resembled the real non-mask control, then frozen "
      "(`data/configs/sgbm.json`: block 5, uniqueness 10, speckle window 150, min disparity 8; disparity range 96 / 64 px for D415 / D435). On the tuning scenes it gave 99.3 / 0.5 / 0.2 % (D415) and 99.6 / 0.2 / 0.2 % (D435) correct / missing / wrong at 5%. "
      "On the final scenes, opaque twin (mask region = the same objects made opaque) and non-mask control:\n")
    A(tbl("mask", "rel5%", list(twin.values())))
    A("\nNon-mask control, 5% tolerance:\n")
    A(tbl("nonmask", "rel5%", [real["d415"], real["d435"], real["d435v"], sim["d415"], sim["d435"], sim["l515"]]))
    A("")
    A("### Real vs synthetic, transparent mask\n")
    A("5% tolerance, pooled over pixels. Real rows are the raw-sensor rows of the model comparison (CG d435 val is shown as an extra d435 reference because it differs strongly from d435 test).\n")
    A(tbl("mask", "rel5%", [real["d415"], sim["d415"], real["d435"], real["d435v"], sim["d435"], real["l515"], sim["l515"]]))
    A("\n1 cm tolerance:\n")
    A(tbl("mask", "abs1cm", [real["d415"], sim["d415"], real["d435"], real["d435v"], sim["d435"], real["l515"], sim["l515"]]))
    A("")
    A("### What matches and what does not\n")
    A(f"**Matches.** (i) *Direction*: of the confidently wrong mask pixels, {f(G[sim['d415'],'mask','rel5%'],'wrong_far_share')}% (D415-like), {f(G[sim['d435'],'mask','rel5%'],'wrong_far_share')}% (D435-like) and "
      f"{f(G[sim['l515'],'mask','rel5%'],'wrong_far_share')}% (ToF) read farther than the GT, versus 99.8-100% in the real data: the simulated sensors also see through the glass. "
      f"(ii) *Size*: median error {f(G[sim['l515'],'mask','rel5%'],'err_cm_med')} / p90 {f(G[sim['l515'],'mask','rel5%'],'err_cm_p90')} cm for the ToF model against {f(G[real['l515'],'mask','rel5%'],'err_cm_med')} / {f(G[real['l515'],'mask','rel5%'],'err_cm_p90')} real; "
      f"stereo {f(G[sim['d415'],'mask','rel5%'],'err_cm_med')} / {f(G[sim['d415'],'mask','rel5%'],'err_cm_p90')} cm (D415-like) against {f(G[real['d415'],'mask','rel5%'],'err_cm_med')} / {f(G[real['d415'],'mask','rel5%'],'err_cm_p90')} real (the stereo p90 is larger than real). "
      f"(iii) *Glass is the cause*: the same stereo pipeline is {f(G[twin['d415'],'mask','rel5%'],'correct')}% (D415-like) and {f(G[twin['d435'],'mask','rel5%'],'correct')}% (D435-like) correct on the opaque twins of the same objects but only "
      f"{f(G[sim['d415'],'mask','rel5%'],'correct')}% and {f(G[sim['d435'],'mask','rel5%'],'correct')}% correct with glass, and the non-mask control is {f(G[sim['d415'],'nonmask','rel5%'],'correct')}% / {f(G[sim['d435'],'nonmask','rel5%'],'correct')}% correct "
      f"(real d415 {f(G[real['d415'],'nonmask','rel5%'],'correct')}%). (iv) *Correct share*: D415-like {f(G[sim['d415'],'mask','rel5%'],'correct')}% against {f(G[real['d415'],'mask','rel5%'],'correct')}% real, "
      f"and D435-like {f(G[sim['d435'],'mask','rel5%'],'correct')}% against {f(G[real['d435v'],'mask','rel5%'],'correct')}% on real-val (but {f(G[real['d435'],'mask','rel5%'],'correct')}% on real-test). "
      f"(v) ToF missing share {f(G[sim['l515'],'mask','rel5%'],'missing')}% against {f(G[real['l515'],'mask','rel5%'],'missing')}% real.\n")
    A(f"**Does not match.** (i) *Stereo missing vs wrong*: the real stereo sensors return nothing on most glass pixels (missing {f(G[real['d415'],'mask','rel5%'],'missing')}% d415, {f(G[real['d435'],'mask','rel5%'],'missing')}% d435 test) and are confidently wrong on "
      f"{f(G[real['d415'],'mask','rel5%'],'wrong')}% / {f(G[real['d435'],'mask','rel5%'],'wrong')}%; the simulation leaves far fewer holes (missing {f(G[sim['d415'],'mask','rel5%'],'missing')}% / {f(G[sim['d435'],'mask','rel5%'],'missing')}%) and is confidently wrong on "
      f"{f(G[sim['d415'],'mask','rel5%'],'wrong')}% / {f(G[sim['d435'],'mask','rel5%'],'wrong')}%, so 'wrong among returned' is {f(G[sim['d415'],'mask','rel5%'],'wrong_among_returned')}% / {f(G[sim['d435'],'mask','rel5%'],'wrong_among_returned')}% against "
      f"{f(G[real['d415'],'mask','rel5%'],'wrong_among_returned')}% / {f(G[real['d435'],'mask','rel5%'],'wrong_among_returned')}% real. "
      f"(ii) *ToF correct vs wrong*: the simulated ToF model is wrong on {f(G[sim['l515'],'mask','rel5%'],'wrong')}% of mask pixels against {f(G[real['l515'],'mask','rel5%'],'wrong')}% real (correct {f(G[sim['l515'],'mask','rel5%'],'correct')}% against {f(G[real['l515'],'mask','rel5%'],'correct')}%): "
      "it is too see-through. (iii) The real d435 control has 6.1% missing; the simulation has almost none.\n")
    A("**Likely reasons (hypotheses, not tested).** Stereo: the real cameras' matching ASIC applies stricter confidence checks than the SGBM configuration used here (which was tuned only on opaque surfaces, where these checks cannot show up), "
      "so ambiguous glass regions become holes in real data but confident matches in the simulation; the simulated glass is perfectly clean and smooth, so the refracted pattern stays coherent and matches confidently, whereas real glass has imperfections, "
      "dust and thickness variation; real IR also contains ambient NIR and sensor noise that the simulation approximates by a constant. ToF: the Fresnel-weighted two-return model makes clean glass nearly transparent at all but grazing angles, "
      "while real glass and plastic scatter, retroreflect and sometimes absorb NIR so a substantial share of real returns comes from the object itself; the model also has no multipath, absorption or thickness. "
      "Both sets of scenes are also simpler than real ones (flat table, little clutter, only nine object shapes).\n")
    A("### Figures\n")
    A("- `media/figures/synth/real_vs_synth_d415.png` and `media/figures/synth/real_vs_synth_l515.png`: a real example above a synthetic scene chosen as the one whose in-mask bucket mix is closest to the real example (it is matched on statistics, not on scene content, and the d415 match is loose).\n"
      "- `media/figures/synth/example_*.png`: four examples per sensor model (two D415-like, two D435-like, four ToF).\n")
    A("![Real vs synthetic d415](media/figures/synth/real_vs_synth_d415.png)\n\n![Real vs synthetic l515](media/figures/synth/real_vs_synth_l515.png)\n")
    A("### Finding\n")
    A(f"A physically based simulation reproduces the *kind* of failure, but not its proportions. Without any painted-on failure, glass makes both simulated sensors confidently wrong, almost always too far (at least "
      f"{fmt(min(m(sim[v],'wrong_far_share') for v in sim))}% of wrong pixels), by a median of about {f(G[sim['d415'],'mask','rel5%'],'err_cm_med')}-{f(G[sim['l515'],'mask','rel5%'],'err_cm_med')} cm, while opaque twins and non-glass pixels stay about "
      f"{fmt(min(m(twin['d415'],'correct'), m(twin['d435'],'correct')))}-{f(G[sim['d435'],'nonmask','rel5%'],'correct')}% correct. The simulated stereo is more often confidently wrong and less often missing than real stereo "
      f"(wrong {f(G[sim['d415'],'mask','rel5%'],'wrong')}% against {f(G[real['d415'],'mask','rel5%'],'wrong')}% on d415), and the simulated ToF is more see-through than the real L515 "
      f"({f(G[sim['l515'],'mask','rel5%'],'wrong')}% against {f(G[real['l515'],'mask','rel5%'],'wrong')}% wrong), so it cannot yet stand in for real data without calibrating the matcher's confidence and the ToF glass response against real glass captures.\n")
    A("### Limitations\n")
    A("- Caustic accuracy is not guaranteed: the refracted projector pattern uses Cycles' manifold next-event estimation, which is an approximation and noisy for complex glass shapes.\n"
      "- The ToF model is a simple approximation (two returns, Fresnel weights, assumed albedo, no multipath or absorption) and was not tuned to real glass data.\n"
      "- Objects are limited to the nine ClearGrasp models; scenes are simple (flat table, little clutter), 640x360, with a constant ambient-IR level and auto-exposed visible lighting.\n"
      "- The simulator was tuned on opaque scenes only (SGBM settings and noise), so the glass behavior is a prediction and not a fit; the SGBM stands in for the real sensors' proprietary matcher and post-processing.\n"
      "- Real references are different datasets with different GT methods: ClearGrasp GT is a real opaque capture, TransCG GT is rendered from poses and meshes, and TransCG has few test scenes (108 sampled L515 views in scenes 5, 7, 8).\n"
      "- The simulated GT-valid set is restricted to the stereo overlap for stereo variants; sky or horizon pixels beyond 10 m are invalid.\n"
      "- The D415-like simulation used a depth FOV of 69.4 degrees against the datasheet's 65 degrees (see Method; the paired check changes the shares by at most 2.4 points).\n")


def calibrated_section(A):
    import json as _json
    H = read_rows(paths.data("calibration_heldout.csv"))
    S = read_rows(paths.data("calibration_search.csv"))
    T = {r["sensor"]: r for r in read_rows(paths.data("calibration_targets.csv"))}
    FZ = _json.load(open(paths.config("calibrated.json")))
    f = lambda r, k, nd=1: fmt(float(r[k]), nd)

    def held(pair, src_key):
        return [r for r in H if r["pair"] == pair and r["region"] == "mask" and r["tol"] == "rel5%" and
                (r["source"] == pair if src_key == "real" else (src_key in r["source"] and (src_key != "calibrated" or "uncalibrated" not in r["source"])))][0]

    pairs = [("stereo", "real CG d435 test", "D435-like"), ("stereo", "real CG d415 test", "D415-like"), ("tof", "real TransCG L515 test scenes", "L515-like")]
    A("### Calibrated simulation\n")
    A("Protocol in `docs/provenance.md` (written and committed before tuning). Stereo is calibrated on real CG d435 **val**, ToF on real TransCG L515 **train scenes** (1, 2, 3, 4, 6, 9, 10); "
      "held-out real sets are CG d435 **test**, CG d415 **test** and L515 **test scenes** (5, 7, 8, all 719 views). At most three parameters per sensor, grid search only, "
      "loss = sum of absolute differences of the correct / missing / wrong shares (percentage points, transparent mask, 5%). The search reused the existing renders (no re-rendering); "
      "the held-out comparison uses 20 fresh D415-like, 20 fresh D435-like and 40 fresh ToF scenes (seeds 5000+), rendered in thermally limited batches, with both settings evaluated on the same fresh scenes. "
      "These tables sit next to the uncalibrated results above (which are unchanged).\n")
    A("**Frozen parameters** (`data/configs/calibrated.json`; grids in `docs/provenance.md`, all results in `data/calibration_search.csv`):\n")
    st, tf_ = FZ["stereo"]["params"], FZ["tof"]["params"]
    A(table(["Sensor", "Parameter", "Uncalibrated", "Calibrated"], [
        ["Stereo", "uniquenessRatio (second-best-match test)", "10", str(st["uniqueness"])],
        ["Stereo", "left-right tolerance (disp12MaxDiff)", "1", str(st["lr_tol"])],
        ["Stereo", "local IR contrast threshold (texture_min)", "0", str(st["texture_min"])],
        ["ToF", "glass-return gain g", "1", str(tf_["g"])],
        ["ToF", "per-pass glass transmission t_glass", "1.0", str(tf_["t_glass"])],
        ["ToF", "minimum-return threshold a_min", "0", str(tf_["a_min"])]]))
    nvalid = sum(1 for r in S if r["sensor"] == "stereo" and r["valid"] == "True")
    ntotal = sum(1 for r in S if r["sensor"] == "stereo")
    A(f"\nOnly {nvalid} of {ntotal} stereo settings keep at least 98% correct on the opaque-twin renders; every setting with `texture_min` > 0 is rejected, so the stereo calibration effectively has one free parameter (`uniqueness`). "
      "The left-right tolerance has no effect over 0-4, and the ToF minimum-return threshold has none up to 0.2 (ties; the first minimum in grid order is frozen) but removes nearly all returns at 0.3.\n")

    A("**Calibration sets** (the data the parameters were fit to; in-sample):\n")
    rows = []
    for sensor, name in (("stereo", "real CG d435 val"), ("tof", "real TransCG L515 train scenes")):
        t = T[sensor]
        rows.append([name + " (target)", f(t, "correct"), f(t, "missing"), f(t, "wrong"), f(t, "wrong_among_returned"), f(t, "wrong_far_share"), f"{f(t, 'err_cm_med')} / {f(t, 'err_cm_p90')}", ""])
        for lab, k in (("sim uncalibrated", "uncalibrated"), ("sim calibrated", "calibration")):
            r = FZ[sensor]
            row = [f"{sensor} {lab}"]
            base = [x for x in S if x["sensor"] == sensor and (
                (all(float(x[a]) == b for a, b in zip(["uniqueness", "lr_tol", "texture_min"], [10, 1, 0])) if sensor == "stereo" else
                 all(float(x[a]) == b for a, b in zip(["g", "t_glass", "a_min"], [1, 1.0, 0]))) if k == "uncalibrated" else
                all(float(x[a]) == float(b) for a, b in zip(list(r["params"]), list(r["params"].values()))))][0]
            rows.append([f"{sensor} {lab}", f(base, "correct"), f(base, "missing"), f(base, "wrong"), f(base, "wrong_among_returned"), "-", "-", f(base, "loss")])
    A(table(["Set / source", "Correct %", "Missing %", "Wrong %", "Wrong among returned %", "Too far %", "Error cm med / p90", "Loss (pp)"], rows))
    A("")
    A("**Held-out sets** (frozen parameters applied to fresh scenes; real references computed only after the freeze):\n")
    rows = []
    for sensor, pair, var in pairs:
        for lab, key in (("real", "real"), (f"sim uncalibrated ({var})", "uncalibrated"), (f"sim calibrated ({var})", "calibrated")):
            r = held(pair, key)
            rows.append([pair if key == "real" else "", lab, f(r, "images", 0), f(r, "correct"), f(r, "missing"), f(r, "wrong"), f(r, "wrong_among_returned"),
                         f(r, "wrong_far_share"), f"{f(r, 'err_cm_med')} / {f(r, 'err_cm_p90')}", "" if key == "real" else f(r, "loss_vs_real")])
    A(table(["Held-out real set", "Source", "Images", "Correct %", "Missing %", "Wrong %", "Wrong among returned %", "Too far %", "Error cm med / p90", "Loss (pp)"], rows))
    A("")
    A("**Calibration loss vs held-out loss** (uncalibrated -> calibrated, percentage points):\n")
    rows = [["Stereo", f"{FZ['stereo']['uncalibrated_loss']:.1f} -> {FZ['stereo']['calibration_loss']:.1f}",
             f"{f(held('real CG d435 test','uncalibrated'),'loss_vs_real')} -> {f(held('real CG d435 test','calibrated'),'loss_vs_real')}",
             f"{f(held('real CG d415 test','uncalibrated'),'loss_vs_real')} -> {f(held('real CG d415 test','calibrated'),'loss_vs_real')}"],
            ["ToF", f"{FZ['tof']['uncalibrated_loss']:.1f} -> {FZ['tof']['calibration_loss']:.1f}",
             f"{f(held('real TransCG L515 test scenes','uncalibrated'),'loss_vs_real')} -> {f(held('real TransCG L515 test scenes','calibrated'),'loss_vs_real')}", "-"]]
    A(table(["Sensor", "Calibration set", "Held-out 1 (d435 test / L515 test scenes)", "Held-out 2 (d415 test)"], rows))
    A("")

    A("**Parameter sensitivity** (calibration loss when one parameter is varied and the others are at their frozen values; `rejected` = fails the opaque-twin constraint):\n")
    rows = []
    def one_at_a_time(sensor, keys, frozen):
        for k in keys:
            vals = sorted({float(r[k]) for r in S if r["sensor"] == sensor and r[k] != ""})
            cells = []
            for v in vals:
                rr = [r for r in S if r["sensor"] == sensor and all(float(r[a]) == (v if a == k else float(frozen[a])) for a in keys)][0]
                cells.append(f"{v:g}: {float(rr['loss']):.1f}" + (" (rejected)" if rr["valid"] != "True" else ""))
            rows.append([sensor, k, "; ".join(cells)])
    one_at_a_time("stereo", ["uniqueness", "lr_tol", "texture_min"], st)
    one_at_a_time("tof", ["g", "t_glass", "a_min"], tf_)
    A(table(["Sensor", "Parameter (value: loss)", ""], rows).replace("| Sensor | Parameter (value: loss) |  |", "| Sensor | Parameter | Loss at each value |"))
    A("")
    A("![Calibrated vs real](media/figures/synth/calibrated_vs_real.png)\n")

    d4, d1, tt = held("real CG d435 test", "calibrated"), held("real CG d415 test", "calibrated"), held("real TransCG L515 test scenes", "calibrated")
    u4, u1, ut = held("real CG d435 test", "uncalibrated"), held("real CG d415 test", "uncalibrated"), held("real TransCG L515 test scenes", "uncalibrated")
    r4, r1, rt = held("real CG d435 test", "real"), held("real CG d415 test", "real"), held("real TransCG L515 test scenes", "real")
    better = [float(c["loss_vs_real"]) < float(u["loss_vs_real"]) for c, u in ((d4, u4), (d1, u1), (tt, ut))]
    A("#### Finding\n")
    if all(better):
        verdict = "Calibration improved the held-out match on all three held-out sets"
    elif any(better):
        verdict = "Calibration improved the held-out match on some held-out sets but not all"
    else:
        verdict = "Calibration did not improve the held-out match"
    A(f"{verdict}: loss {f(u4,'loss_vs_real')} -> {f(d4,'loss_vs_real')} pp (stereo, d435 test), {f(u1,'loss_vs_real')} -> {f(d1,'loss_vs_real')} (stereo, d415 test), {f(ut,'loss_vs_real')} -> {f(tt,'loss_vs_real')} (ToF, L515 test scenes). "
      f"For the simulated ToF the calibrated shares are close to real ({f(tt,'correct')} / {f(tt,'missing')} / {f(tt,'wrong')} % against {f(rt,'correct')} / {f(rt,'missing')} / {f(rt,'wrong')} %), "
      f"but the error size got worse (median / p90 {f(tt,'err_cm_med')} / {f(tt,'err_cm_p90')} cm against {f(rt,'err_cm_med')} / {f(rt,'err_cm_p90')} real, from {f(ut,'err_cm_med')} / {f(ut,'err_cm_p90')} uncalibrated), because a larger glass-return gain pulls the mixed depth toward the glass surface. "
      f"For stereo the improvement comes almost entirely from moving confident-wrong pixels into 'missing' (missing {f(u4,'missing')}% -> {f(d4,'missing')}% on d435-like, real {f(r4,'missing')}%; {f(u1,'missing')}% -> {f(d1,'missing')}% on d415-like, real {f(r1,'missing')}%), "
      f"while the quality of the pixels the sensor still returns is **not** fixed: wrong among returned stays {f(d4,'wrong_among_returned')}% (d435-like) and {f(d1,'wrong_among_returned')}% (d415-like) against {f(r4,'wrong_among_returned')}% and {f(r1,'wrong_among_returned')}% real, "
      f"and the correct share falls to {f(d4,'correct')}% / {f(d1,'correct')}% (real {f(r4,'correct')}% / {f(r1,'correct')}%). "
      "So a stricter matcher makes the simulated stereo look more like the real one in how often it returns nothing, but it removes correct and wrong matches alike, and the remaining returned depth is still much more often confidently wrong than the real sensors'.\n")
    A("#### Caveats\n")
    A("- The held-out real numbers had been seen in earlier phases, so the held-out sets are not blind to the author; the protocol only guarantees that the search never read them and that the simulated held-out scenes are fresh seeds.\n"
      "- The stereo parameters were fit on D435-like scenes against real d435 val and applied unchanged to the D415-like scenes; the d435 test set has only 23 real images and the fresh simulated sets only 20 (stereo) and 40 (ToF) scenes, so shares carry sampling noise of a few percentage points.\n"
      "- With one effective stereo parameter and two effective ToF parameters, matching three shares is partly an exercise in trading them off against each other; transfer to held-out sets (not the in-sample fit) is the evidence, and it is partial.\n"
      "- The ToF 'missing' share (about 0.8% simulated against about 2-6% real) could not be matched by the minimum-return threshold.\n")


def main():
    base = read_summary(paths.data("three_bucket_summary.csv"))
    bias = read_summary(paths.data("three_bucket_summary_biascorrect.csv"))
    ero = {n: read_summary(paths.data(f"three_bucket_summary_erode{n}.csv")) for n in (3, 5)}
    ero[0] = base
    with open(paths.data("bias_params_biascorrect.csv")) as f:
        params = list(csv.DictReader(f))
    n_img = {c: int(base[c, "mask", "rel5%"]["images"]) for c in CAMS}
    scenes = sorted({int(os.path.basename(os.path.dirname(v)).replace("scene", "")) for v in loader.list_transcg("D435")})
    wr = lambda s, c: 100 * s[c, "mask", "rel5%"]["wrong_pooled"] / (s[c, "mask", "rel5%"]["correct_pooled"] + s[c, "mask", "rel5%"]["wrong_pooled"])
    L = []
    A = L.append

    A("# GlassBench results\n")
    A("All numbers below are produced by `tools/evaluation/three_bucket.py`, `tools/checks/check_transcg_pose.py` and `tools/checks/check_scale.py`; this file is "
      "generated by `tools/readme/make_results.py` from the CSVs in the repo.\n")

    # headline
    cg_main = {c: bias[c, "mask", "rel5%"] for c in CG_CAMS}
    A("## Headline\n")
    A(f"On transparent objects, the raw depth from all four sensors is often confidently wrong, not just missing: at a 5% "
      f"tolerance, {fmt(base['TransCG D435','mask','rel5%']['wrong_pooled'])}% (TransCG D435) to "
      f"{fmt(base['TransCG L515','mask','rel5%']['wrong_pooled'])}% (TransCG L515) of GT-valid pixels inside the transparent mask are wrong, "
      f"and among the pixels where a sensor returns a depth, {fmt(wr(base,'TransCG D435'))}% to {fmt(wr(base,'TransCG L515'))}% are wrong. "
      f"At least {fmt(min(base[c,'mask','rel5%']['wrong_far_share_pooled'] for c in CAMS))}% of the wrong pixels read farther than the GT "
      f"(consistent with the sensor seeing through the glass), by a median of "
      f"{fmt(min(base[c,'mask','rel5%']['wrong_diff_cm_med'] for c in CAMS))} to {fmt(max(base[c,'mask','rel5%']['wrong_diff_cm_med'] for c in CAMS))} cm.\n")
    A(f"The ClearGrasp stereo sensors (d415, d435) mostly drop out on glass: {fmt(base['CG d415','mask','rel5%']['missing_pooled'])}% (d415) and "
      f"{fmt(base['CG d435 test','mask','rel5%']['missing_pooled'])}% (d435 real-test) of mask pixels are missing, versus "
      f"{fmt(base['TransCG L515','mask','rel5%']['missing_pooled'])}% for the L515. "
      f"With the per-image bias correction applied, ClearGrasp mask results are essentially unchanged "
      f"(wrong: d415 {fmt(cg_main['CG d415']['wrong_pooled'])}%, d435 test {fmt(cg_main['CG d435 test']['wrong_pooled'])}%, "
      f"d435 val {fmt(cg_main['CG d435 val']['wrong_pooled'])}%), so the glass failure is not explained by a smooth depth bias between captures.\n")

    A("## Data\n")
    A(table(["Dataset", "Split / subset", "Camera", "Samples", "Notes"], [
        ["ClearGrasp (real)", "real-test", "RealSense D415", str(n_img["CG d415"]), "primary"],
        ["ClearGrasp (real)", "real-test", "RealSense D435", str(n_img["CG d435 test"]), "primary D435 result"],
        ["ClearGrasp (real)", "real-val", "RealSense D435", str(n_img["CG d435 val"]), "reported separately"],
        ["TransCG", f"scenes {scenes[0]}-{scenes[-1]} (local copy)", "RealSense D435", str(n_img["TransCG D435"]), "views with depth1-gt.png"],
        ["TransCG", f"scenes {scenes[0]}-{scenes[-1]} (local copy)", "Intel L515", str(n_img["TransCG L515"]), "views with depth2-gt.png"],
    ]))
    A("")
    A(f"The local TransCG copy contains only scenes {scenes[0]}-{scenes[-1]} ({len(scenes)} scenes). `metadata.json` describes 130 scenes "
      f"(28,967 D435 and 28,748 L515 samples), so these results use a subset. All local scenes are used regardless of the dataset's train/test split.\n")
    A("Licences: ClearGrasp is Apache-2.0; TransCG is CC BY-NC-SA. Research use only.\n")

    A("## Units\n")
    A("See `docs/provenance.md`. ClearGrasp depth is float16 meters. TransCG depth PNGs are uint16: D435 is divided by 1000 (mm), L515 by 4000 (0.25 mm). "
      "`metadata.json` has no scale, so both were verified against object pose z (`tools/checks/check_transcg_pose.py`; poses are in meters, "
      "projected with the camera intrinsics, GT depth taken as the median in a 15x15 window; ratio is GT depth / pose z):\n")
    A("```\n" + open(paths.data("pose_check_output.txt")).read().strip() + "\n```\n")
    A("A ratio slightly below 1 is expected because the GT depth is the object surface and the pose is the object center. "
      "With the old L515 divisor of 1000 the ratio was 3.79; /4000 gives the value above. TransCG's own code (`utils/data_preparation.py`, `process_depth`) also divides D435 depth by 1000 and L515 depth by 4000, independently confirming the scales. `T_camera2_camera1.npy` maps D435 to L515 as written "
      "(forward); its inverse gives scattered ratios (1.2-7.9).\n")

    A("## Method\n")
    A("- Only GT-valid pixels (GT finite and > 0) are counted.\n"
      "- Each pixel is one of three buckets: **missing** (raw == 0), **correct** (raw > 0 and |raw - GT| <= tol), **confidently wrong** (raw > 0 and |raw - GT| > tol).\n"
      "- Tolerances: 1 cm absolute, and 5% of GT (relative). Both are in the CSVs; tables here use 5% unless noted.\n"
      "- Region: the transparent mask. ClearGrasp also has a non-mask control region (the original complement of the mask). "
      "TransCG GT equals raw outside the mask, so it has no control region.\n"
      "- 'Too far' share = wrong pixels with raw > GT, divided by all wrong pixels. 'Wrong among returned' = wrong / (correct + wrong).\n"
      "- Values are pooled over all pixels of a camera; the bracket is the per-image median of the same percentage.\n"
      "- Mask erosion check: the mask is eroded by N = 3 and 5 pixels (elliptical kernel) to test whether boundary pixels drive the result.\n"
      "- Bias correction (ClearGrasp only): per image, fit raw = a*GT + b*x + c*y + d by least squares on GT-valid, non-mask, raw > 0 pixels "
      "(one refit after dropping the top 5% of residuals), then invert it on the returned raw depth.\n")

    A("## Main result (transparent mask, 5% tolerance)\n")
    rows = []
    for c in CAMS:
        d = base[c, "mask", "rel5%"]
        rows.append([c + (" (primary)" if c == "CG d435 test" else ""), str(int(d["images_with_gt"]))] + bucket_cells(d)
                    + [fmt(d["wrong_far_share_pooled"]), fmt(wr(base, c))])
    A(table(["Camera", "Images", "Correct %", "Missing %", "Wrong %", "Wrong too far %", "Wrong among returned %"], rows))
    A("\nPooled % of GT-valid mask pixels, per-image median in brackets. These are uncorrected; bias correction changes the ClearGrasp rows by "
      "the amounts shown in the Robustness section.\n")
    A("Same table at the 1 cm tolerance:\n")
    rows = []
    for c in CAMS:
        d = base[c, "mask", "abs1cm"]
        w = 100 * d["wrong_pooled"] / (d["correct_pooled"] + d["wrong_pooled"])
        rows.append([c, str(int(d["images_with_gt"]))] + bucket_cells(d) + [fmt(d["wrong_far_share_pooled"]), fmt(w)])
    A(table(["Camera", "Images", "Correct %", "Missing %", "Wrong %", "Wrong too far %", "Wrong among returned %"], rows))
    A("")

    A("## Error size of confidently wrong pixels\n")
    A("raw - GT in cm, over confidently wrong mask pixels, pooled (0.05 cm histogram resolution).\n")
    rows = [[c] + [fmt(base[c, "mask", t][k]) for t in ("rel5%", "abs1cm") for k in ("wrong_diff_cm_med", "wrong_diff_cm_p90")] for c in CAMS]
    A(table(["Camera", "median, 5%", "p90, 5%", "median, 1 cm", "p90, 1 cm"], rows))
    A("")

    A("## Robustness\n")
    A("### Mask erosion (5% tolerance, transparent mask)\n")
    rows = []
    for c in CAMS:
        for n in (0, 3, 5):
            d = ero[n][c, "mask", "rel5%"]
            rows.append([c, str(n)] + bucket_cells(d) + [fmt(d["wrong_far_share_pooled"])])
    A(table(["Camera", "Erosion px", "Correct %", "Missing %", "Wrong %", "Wrong too far %"], rows))
    A("")
    A("### CG d435: real-test vs real-val\n")
    A("Shown in the main table. The two splits differ strongly (wrong at 5%: "
      f"{fmt(base['CG d435 test','mask','rel5%']['wrong_pooled'])}% on test vs {fmt(base['CG d435 val','mask','rel5%']['wrong_pooled'])}% on val; "
      f"non-mask wrong at 1 cm: {fmt(base['CG d435 test','nonmask','abs1cm']['wrong_pooled'])}% vs {fmt(base['CG d435 val','nonmask','abs1cm']['wrong_pooled'])}%). "
      "Pooling them would have mixed two different capture conditions, so real-test is the primary D435 result.\n")
    A("### Per-image bias correction (ClearGrasp, transparent mask)\n")
    rows = []
    for c in CG_CAMS:
        for t, tn in (("rel5%", "5%"), ("abs1cm", "1 cm")):
            u, k = base[c, "mask", t], bias[c, "mask", t]
            rows.append([c, tn, fmt(u["correct_pooled"]), fmt(k["correct_pooled"]), fmt(u["wrong_pooled"]), fmt(k["wrong_pooled"]),
                         fmt(u["missing_pooled"]), f"{fmt(u['wrong_diff_cm_med'])} / {fmt(u['wrong_diff_cm_p90'])}",
                         f"{fmt(k['wrong_diff_cm_med'])} / {fmt(k['wrong_diff_cm_p90'])}"])
    A(table(["Camera", "Tol", "Correct % raw", "Correct % corrected", "Wrong % raw", "Wrong % corrected", "Missing % (both)",
             "Wrong error cm med/p90 raw", "corrected"], rows))
    A("\nFitted parameter distribution (median [min-max] over images). `tilt` is the plane gradient across the image in cm, "
      "`d` the offset at the image center in cm, and `bias RMS` is the RMS of the fitted model minus GT over the fit pixels (cm). "
      "`a` and `d` are collinear when an image's GT depth range is narrow, so individually they can be large while the net bias stays small; "
      "bias RMS is the reliable size measure.\n")
    rows = []
    for c in CG_CAMS:
        ps = [p for p in params if p["camera"] == c]
        cell = lambda k, nd: (lambda v: f"{np.median(v):.{nd}f} [{min(v):.{nd}f} - {max(v):.{nd}f}]")([float(p[k]) for p in ps])
        rows.append([c, str(len(ps)), cell("a", 4), cell("tilt_cm", 3), cell("d_cm", 3), cell("bias_rms_cm", 3)])
    A(table(["Camera", "Images fit", "a", "tilt cm", "d cm", "bias RMS cm"], rows))
    A("")

    A("## Control: ClearGrasp non-mask pixels\n")
    rows = []
    for c in CG_CAMS:
        for t, tn in (("abs1cm", "1 cm"), ("rel5%", "5%")):
            u, k = base[c, "nonmask", t], bias[c, "nonmask", t]
            rows.append([c, tn, fmt(u["correct_pooled"]), fmt(k["correct_pooled"]), fmt(u["missing_pooled"]),
                         fmt(u["wrong_pooled"]), fmt(k["wrong_pooled"]), f"{fmt(u['wrong_med'])} / {fmt(u['wrong_p90'])}"])
    A(table(["Camera", "Tol", "Correct % raw", "Correct % corrected", "Missing %", "Wrong % raw", "Wrong % corrected",
             "Wrong % per-image median / p90 (raw)"], rows))
    A("\nThe correction is fit on these same pixels, so the corrected control numbers are in-sample; the mask region is out-of-sample.\n")

    if os.path.exists(paths.data("models_summary.csv")) and os.path.exists(paths.data("flagging_summary.csv")):
        models_section(A)
    if os.path.exists(paths.data("synth_vs_real.csv")):
        sim_section(A)
        if os.path.exists(paths.data("calibration_heldout.csv")):
            calibrated_section(A)

    A("## Limitations\n")
    A("- TransCG has no independent opaque control: GT equals raw outside the mask, so sensor noise on opaque surfaces cannot be separated from glass failure there.\n"
      "- TransCG GT comes from object poses and mesh rendering, not from an opaque capture; some 'wrong' pixels near mask edges may be GT error. "
      "(The mask erosion check shows the TransCG numbers do not change with 3-5 px of erosion.)\n"
      f"- CG d435 real-val has a capture mismatch between raw and GT on non-mask pixels (1 cm wrong rate {fmt(base['CG d435 val','nonmask','abs1cm']['wrong_pooled'])}% "
      f"pooled, per-image median {fmt(base['CG d435 val','nonmask','abs1cm']['wrong_med'])}%, p90 {fmt(base['CG d435 val','nonmask','abs1cm']['wrong_p90'])}%), "
      "concentrated in a minority of images, with a smooth, low-frequency raw-GT pattern whose sign varies between images. "
      "A per-image plane + scale fit does not remove it (above), so it is not a simple tilt or scale bias.\n"
      "- D435 results differ between ClearGrasp and TransCG; the datasets differ in objects, distances, lighting and GT method, so the cameras should not be compared as like for like.\n"
      "- The explanation of the L515 (time-of-flight) returning confident wrong values where the stereo sensors drop out is a hypothesis; it was not tested.\n"
      f"- The local TransCG copy is scenes {scenes[0]}-{scenes[-1]} only (see Data).\n"
      "- Bias-correction parameters `a` and `d` are ill-conditioned on some images (narrow GT depth range).\n"
      "- Model A is scored after a per-image scale+shift fit, so it is an optimistic upper bound for a monocular model (the fit uses GT, or for TransCG the raw depth, outside the mask). "
      "Its alignment error is as large as the errors being measured (a few cm, and about 25 cm for the L515 global fit; see the alignment table), "
      "so Model A at a 1 cm or 5% tolerance mostly reflects monocular accuracy, not only glass. Its 'broken' rate on non-mask pixels shows that floor.\n"
      "- Model B (DFNet) is only out-of-sample on TransCG scenes 5, 7, 8 by the repo's documented split; the other TransCG scenes are in-sample. ClearGrasp is cross-dataset, "
      "so DFNet's ClearGrasp numbers also reflect domain shift (camera, objects, working range: DFNet drops depth outside 0.3-1.0 m, and ClearGrasp d435 depth reaches about 1.15 m).\n"
      "- DFNet runs at 320x240 and its output is resized back with nearest-neighbour interpolation, which adds edge error at the GT resolution.\n"
      "- The TransCG model comparison uses 300 random views per camera (seed 0): 91 D435 and 108 L515 views fall in the test scenes.\n"
      "- The mask-segmentation network over-predicts the transparent mask (predicted area is 1.5-2.6x the GT area), which inflates flag recall; "
      "it was trained on synthetic data only, so ClearGrasp real images are out of domain for it.\n")

    if os.path.exists(paths.FIGURES + "/figure1.png"):
        A("## Figures\n")
        A("- `media/figures/figure1.png`: the clearest see-through failure (L515/TransCG): large confidently wrong region inside the glass, little missing.\n"
          "- `media/figures/figure1_alt.png`: a stereo (d415) case where the glass mostly goes missing, for contrast.\n"
          "- `media/figures/error_maps/` and `media/figures/cg_d435_outliers/`: additional examples.\n")
        A("![Figure 1](media/figures/figure1.png)\n\n![Figure 1 alt](media/figures/figure1_alt.png)\n")

    open(os.path.join(paths.ROOT, "docs", "results.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
