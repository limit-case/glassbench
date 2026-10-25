# Measurement

How every published number is computed. The code is in [`../tools/`](../tools/); the outputs are in [`../data/`](../data/), described field by field in [`schema.md`](schema.md).

## The three buckets

Every pixel inside a region is in exactly one bucket. Only pixels where the ground truth is finite and greater than zero (GT-valid) are counted.

| Bucket | Rule |
|---|---|
| Missing | raw depth = 0. The sensor returned nothing. A visible failure. |
| Correct | raw depth > 0 and \|raw - GT\| is within tolerance. |
| Confidently wrong | raw depth > 0 and \|raw - GT\| is outside tolerance. A silent failure. |

Two derived quantities are published beside the three shares.

- **Wrong among returned** = wrong / (correct + wrong). The chance that a depth the sensor chose to report is wrong.
- **Wrong too far** = wrong pixels with raw > GT, divided by all wrong pixels. A share near 100% means the sensor reads the surface behind the glass.

## Tolerances

| Name in the CSVs | Tolerance |
|---|---|
| `abs1cm` | 1 cm absolute |
| `rel5%` | 5% of the ground-truth depth at that pixel |

Tables and the README use `rel5%` unless they say otherwise. Both tolerances are published for every row.

## Regions

| Region | Definition |
|---|---|
| `mask` | The transparent-object mask supplied with the dataset. |
| `nonmask` | ClearGrasp only: the complement of the mask, a control where raw depth should mostly be right. TransCG ground truth equals raw depth outside the mask, so TransCG has no control region. |

## Pooling and spread

A pooled share sums pixel counts over every image of an evaluation set and divides once. The per-image median, 10th and 90th percentiles (`_med`, `_p10`, `_p90`) are taken over the per-image shares. Pooled numbers are the headline; per-image numbers show whether a few images dominate.

## Error sizes

For confidently wrong pixels the error is raw - GT in cm, positive when the sensor reads too far. Pooled medians and 90th percentiles come from a histogram with 0.05 cm bins, so they are exact to within that bin and no pixel array is kept. This is why published errors end in .x25 or .x75 before rounding.

## Evaluation sets

| Set | Source | Camera | Note |
|---|---|---|---|
| `CG d415` | ClearGrasp real-test | RealSense D415 | |
| `CG d435 test` | ClearGrasp real-test | RealSense D435 | The primary ClearGrasp D435 result. |
| `CG d435 val` | ClearGrasp real-val | RealSense D435 | Reported separately: its capture conditions differ strongly from test, so pooling would mix them. |
| `TransCG D435` | TransCG, views with `depth1-gt.png` | RealSense D435 | |
| `TransCG L515` | TransCG, views with `depth2-gt.png` | Intel L515 (ToF) | |

The TransCG copy used here has scenes 1-10 of the 130 in the released metadata. All local scenes are used regardless of the dataset's train/test split.

## Robustness checks

| Output | What changes |
|---|---|
| `three_bucket_*_erode3.csv`, `_erode5.csv` | The mask is eroded by 3 or 5 px (elliptical kernel) to test whether boundary pixels drive the result. |
| `three_bucket_*_biascorrect.csv` | ClearGrasp only. Per image, fit raw = a*GT + b*x + c*y + d by least squares on GT-valid, non-mask pixels with raw > 0, refit once after dropping the 5% largest residuals, then invert the fit on the returned raw depth. Tests whether a smooth bias between captures explains the failure. The fitted parameters are in `bias_params_biascorrect.csv`; the bias RMS is the reliable measure of their size, because `a` and `d` are collinear on images with a narrow depth range. |

## Models

Two published models are scored with the same three buckets, on the same images, pooled the same way.

| Model | Input | How it is run |
|---|---|---|
| A: Depth Anything V2 Small, relative | RGB only | Output is relative inverse depth. Per image, a scale and shift is fit by least squares in inverse-depth space on GT-valid pixels outside the mask, then inverted to depth. For TransCG, where GT equals raw depth outside the mask, the fit uses raw depth. A second, local variant fits on a 40 px ring around the mask; it is a sensitivity check. |
| A: Depth Anything V2 Metric-Indoor Small | RGB only | Reported as is (zero-shot metres) and with the same global and local scale+shift in depth space. |
| B: TransCG DFNet | RGB + raw depth | The official inference procedure: inputs resized (nearest) to 320x240, depth clipped to 0.3-1.0 m, values beyond mean +/- 3 std dropped, holes filled by nearest neighbour, depth normalised per image; output resized back (nearest). |

Dense models return a value everywhere, so their missing share is about 0 and wrong-among-returned equals wrong. Model A is scored after a fit that uses ground truth (or raw depth for TransCG) outside the mask, so it is an optimistic bound for a monocular model; `models_alignment.csv` gives the fit error, which is as large as the errors being measured on some sets. DFNet is out of sample on TransCG scenes 5, 7 and 8 by the repository's documented split; the other scenes are reported separately as in-sample. ClearGrasp images are not part of DFNet's training data. The TransCG model rows use a fixed random subset of 300 views per camera (seed 0).

`models_transitions.csv` follows each raw bucket to the model bucket: *fixed* is raw wrong to model correct, *inherited* is raw wrong to model wrong, *filled* is raw missing to correct or wrong, *broken* is raw correct to model wrong.

## Flagging

`flagging_summary.csv` runs the ClearGrasp mask-segmentation network (RGB only, trained on synthetic data) on the real ClearGrasp RGB images. **Flag recall** is the percent of raw confidently wrong pixels that fall inside the predicted transparent mask. Its cost is the percent of correct raw pixels the predicted mask would discard. The predicted mask is larger than the true mask (`pred_coverage_vs_gt`), which raises recall.

## Per-object proxy

`object_proxy.csv` takes each connected component of the GT mask of at least 200 px and the median raw - GT error over its returned pixels (an object returns depth if at least 20 of its pixels have a raw value). The 2 cm and 5 cm thresholds are illustrative parallel-jaw and suction placement tolerances. It is not a grasping experiment. Touching objects merge into one component.

## Simulation

The simulated sensors are scored with the same buckets on the same region and tolerances (`synth_vs_real.csv`). Ground truth comes from an opaque twin of each scene: the same scene with every transparent object swapped for an opaque diffuse one.

- **Stereo (D415-like, D435-like).** Left and right IR cameras and a dot projector are rendered in Blender Cycles; the projector pattern is refracted by the glass. Depth comes from OpenCV StereoSGBM with left-right consistency and speckle filtering. Baselines are 55 mm and 50 mm.
- **ToF (L515-like).** An approximation: per pixel, an amplitude-weighted mix of the glass front surface (Fresnel reflectance, n = 1.5) and the first opaque surface behind it (transmitted twice, albedo 0.5), a 3x3 footprint mix, noise of 1 mm + 0.25% of depth, and no return at near-grazing incidence. It has no multipath and no glass absorption.

Caustics use Cycles' manifold next-event estimation, which is an approximation and noisy for complex glass. The matcher stands in for the real sensors' proprietary one.

## Calibration loss

Each simulated sensor has at most three parameters, searched on a grid, one global setting per sensor.

| Sensor | Parameters and grid |
|---|---|
| Stereo | `uniqueness` {10, 20, 30, 40, 50}; `lr_tol` {0, 1, 2, 4}; `texture_min` {0, 3, 6, 10, 15, 20, 30} |
| ToF | `g` {1, 2, 4, 8, 16, 32}; `t_glass` {1.0, 0.8, 0.6, 0.4, 0.2}; `a_min` {0, 0.05, 0.1, 0.15, 0.2, 0.3} |

The loss is, in percentage points,

    loss = |correct_sim - correct_real| + |missing_sim - missing_real| + |wrong_sim - wrong_real|

on the transparent mask, GT-valid pixels, 5% tolerance, pooled over pixels. A stereo setting is rejected if it is less than 98% correct on the opaque-twin renders. The protocol, split and caveats are in [`provenance.md`](provenance.md).

## The sample table

`python examples/load_sample.py` applies the same rule to the nine-scene open sample, pooled per sensor, for the uncalibrated and calibrated simulator settings. It prints the shares for the 5% tolerance by default; `--tol abs1cm` gives the 1 cm table. It covers nine simulated scenes and is for checking that the files load, not a benchmark result.
