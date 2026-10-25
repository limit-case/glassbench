"""Field descriptions for every published file. The single source for data/fields.json and docs/schema.md;
make_data_docs.py fails if a CSV header and this dictionary disagree."""

PCT = "%"
BUCKET = ("Share of GT-valid pixels in the region, in percent.",)

_SUMMARY = [
    ("camera", "Evaluation set: `CG d415`, `CG d435 test`, `CG d435 val` (ClearGrasp real), `TransCG D435`, `TransCG L515`."),
    ("region", "`mask` = inside the transparent-object mask; `nonmask` = ClearGrasp control region (TransCG has none, see measurement.md)."),
    ("tol", "Tolerance: `abs1cm` = 1 cm absolute; `rel5%` = 5% of ground-truth depth."),
    ("images", "Images in the set."),
    ("images_with_gt", "Images with at least one GT-valid pixel in the region."),
    ("pixels", "GT-valid pixels counted, summed over images."),
    ("correct_pooled", "Percent of pixels with a return within tolerance, pooled over all pixels."),
    ("correct_med", "Same, per-image median."),
    ("correct_p10", "Same, per-image 10th percentile."),
    ("correct_p90", "Same, per-image 90th percentile."),
    ("missing_pooled", "Percent of pixels with no return (raw depth = 0), pooled."),
    ("missing_med", "Same, per-image median."),
    ("missing_p10", "Same, per-image 10th percentile."),
    ("missing_p90", "Same, per-image 90th percentile."),
    ("wrong_pooled", "Percent of pixels with a return outside tolerance (confidently wrong), pooled."),
    ("wrong_med", "Same, per-image median."),
    ("wrong_p10", "Same, per-image 10th percentile."),
    ("wrong_p90", "Same, per-image 90th percentile."),
    ("wrong_far_share_pooled", "Percent of wrong pixels whose raw depth is farther than ground truth, pooled."),
    ("wrong_far_share_med", "Same, per-image median."),
    ("wrong_diff_cm_med", "Median of raw - GT in cm over wrong pixels, pooled (0.05 cm histogram resolution)."),
    ("wrong_diff_cm_p90", "90th percentile of raw - GT in cm over wrong pixels, pooled."),
]

_PER_IMAGE = [
    ("camera", "Evaluation set."),
    ("id", "Image identifier within the source dataset (split/camera/index for ClearGrasp, scene/view for TransCG)."),
    ("region", "`mask` or `nonmask`."),
    ("tol", "`abs1cm` or `rel5%`."),
    ("n", "GT-valid pixels in the region."),
    ("missing", "Pixels with no return."),
    ("correct", "Pixels with a return within tolerance."),
    ("wrong", "Pixels with a return outside tolerance."),
    ("wrong_far", "Wrong pixels with raw depth farther than ground truth."),
]

_SIM_BUCKETS = [
    ("correct", "Percent correct, pooled over pixels."),
    ("missing", "Percent missing, pooled."),
    ("wrong", "Percent confidently wrong, pooled."),
    ("wrong_among_returned", "wrong / (correct + wrong), in percent."),
    ("wrong_far_share", "Percent of wrong pixels that read farther than ground truth."),
    ("err_cm_med", "Median of output - GT in cm over wrong pixels."),
    ("err_cm_p90", "90th percentile of output - GT in cm over wrong pixels."),
]

CSV = {
    "three_bucket_summary.csv": ("Main result: three-bucket summary of raw depth per evaluation set, region and tolerance.", _SUMMARY),
    "three_bucket_summary_erode3.csv": ("Same as `three_bucket_summary.csv` with the transparent mask eroded by 3 px.", _SUMMARY),
    "three_bucket_summary_erode5.csv": ("Same as `three_bucket_summary.csv` with the transparent mask eroded by 5 px.", _SUMMARY),
    "three_bucket_summary_biascorrect.csv": ("Same as `three_bucket_summary.csv` after the per-image plane + scale bias correction (ClearGrasp only; TransCG rows are uncorrected).", _SUMMARY),
    "three_bucket_per_image.csv": ("Three-bucket pixel counts per image, region and tolerance.", _PER_IMAGE),
    "three_bucket_per_image_erode3.csv": ("Per-image counts with the mask eroded by 3 px.", _PER_IMAGE),
    "three_bucket_per_image_erode5.csv": ("Per-image counts with the mask eroded by 5 px.", _PER_IMAGE),
    "three_bucket_per_image_biascorrect.csv": ("Per-image counts after bias correction.", _PER_IMAGE),
    "bias_params_biascorrect.csv": ("Fitted bias model per ClearGrasp image: raw = a*GT + b*x + c*y + d.", [
        ("camera", "Evaluation set."), ("id", "Image identifier."),
        ("a", "Depth scale factor (dimensionless)."),
        ("b_per_px", "Horizontal gradient, metres per pixel from the image centre."),
        ("c_per_px", "Vertical gradient, metres per pixel from the image centre."),
        ("d_cm", "Offset at the image centre, cm."),
        ("tilt_cm", "Plane gradient across the image, cm."),
        ("bias_rms_cm", "RMS of fitted model minus GT over the fit pixels, cm. The reliable size of the bias."),
    ]),
    "object_proxy.csv": ("Per-object consequence proxy: how far the median returned depth of each transparent object is from ground truth. Analysis only; not a grasping experiment.", [
        ("group", "Evaluation set."),
        ("objects", "Connected components of the GT transparent mask of at least 200 pixels."),
        ("objects_with_return", "Objects with enough returned GT-valid pixels to take a median (at least 20; see object_proxy.py)."),
        ("no_return_pct", "Percent of objects without enough returned pixels: a visible failure."),
        ("mostly_missing_pct", "Percent of objects with fewer than half of their GT-valid pixels returned."),
        ("off_gt2cm_pct", "Percent of objects with a return whose median raw - GT error exceeds 2 cm in magnitude."),
        ("off_gt5cm_pct", "Same for 5 cm."),
        ("off_gt5cm_too_far_pct", "Percent of the objects off by more than 5 cm whose error reads too far."),
        ("median_abs_err_cm", "Median over objects with a return of |median raw - GT|, cm."),
        ("median_signed_err_cm", "Median over objects with a return of the signed median raw - GT, cm (positive = too far)."),
    ]),
    "models_summary.csv": ("Raw sensor and model outputs scored with the same three buckets.", [
        ("model", "`raw`; `A_rel_aligned`, `A_rel_local` (Depth Anything V2 Small relative, global / ring alignment); `A_metric` (metric-indoor, as is); `A_metric_aligned`, `A_metric_local`; `B_dfnet` (TransCG DFNet)."),
        ("camera", "Evaluation set; TransCG rows are split into `(test scenes)`, `(train scenes)` and `(all sampled)`."),
        ("region", "`mask` or `nonmask`."), ("tol", "`abs1cm` or `rel5%`."),
        ("images", "Images scored."), ("pixels", "GT-valid pixels counted."),
        ("correct", "Percent correct, pooled."), ("missing", "Percent missing, pooled (about 0 for dense models)."),
        ("wrong", "Percent confidently wrong, pooled."),
        ("correct_img_med", "Per-image median percent correct."), ("missing_img_med", "Per-image median percent missing."),
        ("wrong_img_med", "Per-image median percent wrong."),
        ("wrong_far_share", "Percent of wrong pixels reading farther than ground truth."),
        ("err_cm_med", "Median of output - GT in cm over wrong pixels."), ("err_cm_p90", "90th percentile of the same."),
    ]),
    "models_transitions.csv": ("Where each raw-sensor bucket ends up after a model.", [
        ("model", "Model, as in `models_summary.csv`."), ("camera", "Evaluation set."), ("region", "`mask` or `nonmask`."),
        ("tol", "`abs1cm` or `rel5%`."),
        ("raw_bucket", "Bucket of the pixel in the raw sensor: `correct`, `missing`, `wrong`."),
        ("model_bucket", "Bucket of the same pixel in the model output."),
        ("pixels", "Pixels moving from `raw_bucket` to `model_bucket`."),
        ("pct_of_raw_bucket", "`pixels` as a percent of all pixels in `raw_bucket`."),
        ("still_too_far_pct", "For wrong -> wrong: percent of those pixels where the model also reads farther than ground truth."),
    ]),
    "models_alignment.csv": ("Quality of the per-image scale+shift fit used for Depth Anything V2.", [
        ("model", "Aligned variant: `A_rel_aligned`, `A_rel_local`, `A_metric_aligned`, `A_metric_local`."),
        ("camera", "Evaluation set."), ("images", "Images fitted."),
        ("ref_rmse_cm_med", "Median over images of the RMSE of aligned depth against GT on the fit pixels, cm."),
        ("ref_rmse_cm_p90", "90th percentile of the same."),
        ("slope_med", "Median fitted slope."),
        ("frac_nonpositive_slope", "Fraction of images whose fitted slope was zero or negative."),
    ]),
    "flagging_summary.csv": ("Can the ClearGrasp mask-segmentation network flag the confidently wrong pixels?", [
        ("camera", "Evaluation set."), ("images", "Images scored."),
        ("mask_iou_pooled", "Mask intersection-over-union against the GT mask, pooled, percent."),
        ("mask_iou_img_med", "Per-image median IoU, percent."), ("mask_iou_img_p10", "Per-image 10th percentile."),
        ("mask_iou_img_p90", "Per-image 90th percentile."),
        ("pred_coverage_vs_gt", "Predicted mask area as a percent of GT mask area."),
        ("tol", "`abs1cm` or `rel5%`."),
        ("wrong_px_in_gt_mask_pct", "Percent of raw confidently wrong pixels that lie inside the GT mask."),
        ("flag_recall_in_gt_mask", "Percent of wrong pixels inside the GT mask that fall in the predicted mask."),
        ("flag_recall_in_gt_mask_img_med", "Per-image median of the same."),
        ("flag_recall_all_wrong", "Percent of all raw confidently wrong pixels that fall in the predicted mask."),
        ("correct_px_flagged_pct", "Percent of correct raw pixels that the predicted mask would discard."),
        ("flagged_returned_px_wrong_pct", "Percent of flagged pixels that returned a depth and are wrong."),
    ]),
    "synth_vs_real.csv": ("Simulated sensors (uncalibrated settings) against real results.", [
        ("sensor", "`d415`, `d435` (active stereo) or `l515` (ToF approximation)."),
        ("source", "`real`, `synthetic` or `synthetic_opaque_twin`."),
        ("camera", "Evaluation set or synthetic set with its scene count."),
        ("region", "`mask` or `nonmask`."), ("tol", "`abs1cm` or `rel5%`."),
        ("images", "Images or scenes."), ("pixels", "GT-valid pixels counted."),
    ] + _SIM_BUCKETS + [
        ("correct_img_med", "Per-image median percent correct."), ("missing_img_med", "Per-image median percent missing."),
        ("wrong_img_med", "Per-image median percent wrong."),
    ]),
    "calibration_targets.csv": ("Real targets of the calibration sets, computed before any search.", [
        ("sensor", "`stereo` or `tof`."), ("set", "Real calibration set."), ("images", "Images."), ("pixels", "GT-valid pixels."),
    ] + _SIM_BUCKETS),
    "calibration_search.csv": ("Every grid-search setting and its calibration loss.", [
        ("sensor", "`stereo` or `tof`."),
        ("uniqueness", "Stereo: SGBM uniqueness ratio."), ("lr_tol", "Stereo: left-right check tolerance (disp12MaxDiff)."),
        ("texture_min", "Stereo: minimum local IR contrast (9x9 standard deviation)."),
        ("g", "ToF: glass-return gain."), ("t_glass", "ToF: per-pass glass transmission."),
        ("a_min", "ToF: minimum return amplitude."),
        ("correct", "Percent correct on the simulated calibration scenes."), ("missing", "Percent missing."), ("wrong", "Percent wrong."),
        ("wrong_among_returned", "wrong / (correct + wrong), percent."),
        ("loss", "Calibration loss in percentage points: |correct - real| + |missing - real| + |wrong - real|."),
        ("opaque_correct", "Stereo only: percent correct on opaque-twin renders (must be at least 98)."),
        ("valid", "True when the setting satisfies the opaque-twin constraint."),
    ]),
    "calibration_heldout.csv": ("Held-out evaluation after the calibration parameters were frozen.", [
        ("sensor", "`stereo` or `tof`."), ("pair", "Real held-out set the row is compared with."),
        ("source", "`real ...`, `sim uncalibrated (...)` or `sim calibrated (...)`."),
        ("region", "`mask` or `nonmask`."), ("tol", "`abs1cm` or `rel5%`."),
        ("images", "Images or scenes."), ("pixels", "GT-valid pixels counted."),
    ] + _SIM_BUCKETS + [
        ("loss_vs_real", "Simulated rows only: sum of absolute differences of correct, missing and wrong against the real row, percentage points."),
    ]),
    "fov_check.csv": ("Sensitivity of D415-like results to the horizontal field of view.", [
        ("setting", "`uncalibrated` or `calibrated` simulator settings."), ("hfov_deg", "Horizontal field of view in degrees."),
        ("missing", "Percent missing."), ("correct", "Percent correct."), ("wrong", "Percent wrong."),
        ("wrong_among_returned", "wrong / (correct + wrong), percent."), ("pixels", "GT-valid pixels counted."),
    ]),
}

TEXT = {
    "pose_check_output.txt": "Console output of `tools/checks/check_transcg_pose.py`: ratio of ground-truth depth to projected object-pose depth, used to verify the TransCG depth units.",
}

CONFIGS = {
    "configs/sgbm.json": ("Frozen OpenCV SGBM settings of the stereo simulator.", [
        ("min_disp", "Minimum disparity, pixels."), ("block", "Matching block size."), ("uniqueness", "Uniqueness ratio before calibration."),
        ("speckle_window", "Speckle filter window."), ("speckle_range", "Speckle filter range."), ("disp12", "Left-right check tolerance before calibration."),
        ("prefilter_cap", "Prefilter cap."), ("min_depth", "Shortest depth kept, metres."), ("max_depth", "Longest depth kept, metres."),
        ("num_disp", "Disparity range in pixels per variant (`d415`, `d435`)."), ("_note", "How the settings were frozen."),
    ]),
    "configs/calibrated.json": ("Calibration parameters frozen before the held-out evaluation.", [
        ("_note", "How the parameters were chosen."),
        ("stereo.params", "`uniqueness`, `lr_tol`, `texture_min` (see `calibration_search.csv`)."),
        ("stereo.calibration_loss", "Loss on the calibration set, percentage points."),
        ("stereo.uncalibrated_loss", "Loss of the original settings on the same set."),
        ("tof.params", "`g`, `t_glass`, `a_min`."),
        ("tof.calibration_loss", "Loss on the calibration set, percentage points."),
        ("tof.uncalibrated_loss", "Loss of the original settings on the same set."),
    ]),
}

SAMPLE_FILES = [
    ("rgb.png", "8-bit RGB. Visible-light render with the transparent objects as glass or clear plastic."),
    ("ir_left.png, ir_right.png", "8-bit grey. Stereo scenes only: rectified IR pair lit by a dot-pattern projector, refracted by the glass."),
    ("gt_depth_mm.png", "uint16, millimetres. z-depth from the opaque twin (transparent objects swapped for opaque diffuse). 0 = no surface within 10 m."),
    ("gt_valid.png", "8-bit. 255 where the ground truth counts as valid; for stereo scenes the left strip with no right-image counterpart is excluded."),
    ("mask.png", "8-bit. 255 on transparent objects."),
    ("depth_behind_mm.png", "uint16, millimetres. ToF scenes only: depth with the transparent objects removed."),
    ("raw_uncalibrated_mm.png", "uint16, millimetres. Simulated raw sensor depth with the original settings. 0 = no return."),
    ("raw_calibrated_mm.png", "uint16, millimetres. Same with the calibrated settings in `data/configs/calibrated.json`."),
    ("meta.json", "Scene description, below."),
]

SAMPLE_META = [
    ("seed", "integer", "Scene seed."),
    ("variant", "string", "`d415`, `d435` or `l515`."),
    ("intrinsics", "object", "`fx`, `fy`, `cx`, `cy` in pixels; `width`, `height` in pixels."),
    ("baseline", "number", "Stereo baseline in metres; null for the ToF variant."),
    ("cam_matrix_world", "4x4 array", "Camera-to-world pose."),
    ("camera_dist", "number", "Camera distance from the scene centre, metres."),
    ("camera_elev_deg", "number", "Camera elevation, degrees."),
    ("exposure", "number", "Exposure of the visible-light render."),
    ("texture", "string", "Table texture."),
    ("wall", "boolean", "Whether a backdrop wall is present."),
    ("world", "string", "World lighting."),
    ("n_lights", "integer", "Number of lights."),
    ("objects", "array", "Transparent objects: `model` (ClearGrasp 3D model file) and `plastic` (clear plastic instead of glass)."),
    ("n_opaque", "integer", "Opaque distractor objects."),
    ("total", "number", "Render time, seconds."),
]
