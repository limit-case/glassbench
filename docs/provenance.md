# Provenance

Where the data comes from, how its units were verified, how the synthetic scenes are rendered, and how the simulator was calibrated.

## Source datasets

No dataset file is redistributed in this repository. The scripts read files you download yourself into `datasets/` (git-ignored).

| Dataset | What is used | Licence | Source |
|---|---|---|---|
| ClearGrasp | Real test and validation images (RealSense D415 and D435), the 3D models of nine transparent objects, the mask-segmentation checkpoint | Apache-2.0 | [github.com/Shreeyak/cleargrasp](https://github.com/Shreeyak/cleargrasp), Sajjan et al., ICRA 2020 |
| TransCG | Real RGB-D views from a RealSense D435 and an Intel L515, with refined ground-truth depth, masks, poses and camera intrinsics | CC BY-NC-SA 4.0 | [graspnet.net/transcg](https://graspnet.net/transcg), Fang et al., RA-L 2022 |
| Depth Anything V2 Small | Model A weights, downloaded from Hugging Face on first use | Apache-2.0 per the model card | [depth-anything](https://github.com/DepthAnything/Depth-Anything-V2), Yang et al., NeurIPS 2024 |
| TransCG DFNet | Model B checkpoint, distributed through the TransCG repository | CC BY-NC-SA 4.0 | [github.com/galaxies99/transcg](https://github.com/galaxies99/transcg) |

Download links, expected folder names and the model folder are listed in [`access.md`](access.md).

### Ground truth

ClearGrasp captures each transparent scene twice, once with the transparent objects and once with spray-painted opaque copies in the same poses. The opaque capture gives the ground-truth depth. Opaque ground truth contains NaNs; those pixels are not GT-valid. TransCG ground truth is rendered from object poses and meshes, not captured, so some wrong pixels near mask edges may be ground-truth error. The erosion check in [`measurement.md`](measurement.md) shows the TransCG numbers do not move with 3-5 px of erosion.

## Depth units

Verified on 2026-10-22.

| Source | Storage | Unit |
|---|---|---|
| ClearGrasp D415 / D435 | EXR float16 | metres. Opaque GT has NaNs, so `gt_valid` is false there. |
| TransCG D435 | uint16 PNG | divide by 1000 (millimetres) |
| TransCG L515 | uint16 PNG | **divide by 4000 (0.25 mm steps)** |

TransCG's `metadata.json` carries no depth scale, so both scales were checked against object poses. `tools/checks/check_transcg_pose.py` projects each object pose into the image with the camera intrinsics and compares the median GT depth in a 15x15 window with the pose z (20 views, 61 poses). The ratio is slightly below 1 because GT is the object surface and the pose is the object centre.

| Camera | Divisor | GT depth / pose z (median) |
|---|---|---|
| D435 | 1000 | 0.943 |
| L515 | 1000 | 3.79 |
| L515 | **4000** | **0.949** |

The L515 is the finding: with the divisor 1000 that works for the D435, L515 depth is 4 times too large. The pose-ratio check gives 0.949 with 4000. TransCG's own `process_depth` in `utils/data_preparation.py` independently uses 1000 for the D435 and 4000 for the L515. The pose file `T_camera2_camera1.npy` maps D435 to L515 as written; its inverse gives scattered ratios of 1.2-7.9. The full console output is [`../data/pose_check_output.txt`](../data/pose_check_output.txt).

## Rendering pipeline

The synthetic scenes are our own renders. They use the nine ClearGrasp 3D models (Apache-2.0).

| Step | What happens | Code |
|---|---|---|
| Scene | A table with a randomised procedural texture (and a backdrop wall in 70% of scenes), 1-5 transparent objects dropped with rigid-body physics, 0-3 opaque distractors, 1-3 area lights, uniform or sky world. Camera 0.4-0.8 m from the cluster at 30-75 degrees elevation. | `tools/simulation/render_scene.py` |
| Materials | Glass (transmission 1, IOR 1.45-1.52, roughness 0-0.03) or clear plastic (IOR 1.40-1.50, roughness 0.05-0.12) with a slight random tint. | same |
| Opaque twin | The same scene with transparent objects swapped for opaque diffuse. Supplies ground-truth depth, the object mask and normals. | same |
| Stereo | Left and right IR cameras and a random dot projector rendered in Cycles on a Metal GPU, then OpenCV StereoSGBM. Settings frozen on opaque-twin tuning scenes only (seeds 1000-1019 per variant). | `tools/simulation/stereo.py`, `data/configs/sgbm.json` |
| ToF | Two-return Fresnel model on saved depth layers. | `tools/simulation/tof.py` |
| Sample | Writes the open sample from the first three scenes per sensor with a transparent mask covering at least 1% of the image. | `tools/simulation/make_samples.py` |

Renders are 640x360. The study used 100 scenes per stereo variant (seeds 0-99) and 200 ToF scenes (seeds 0-199). Rendering needs Blender 4.5.

## Calibration protocol

Written before any tuning. The goal was to move the simulated sensors toward the real results under a strict held-out protocol, using existing renders and as little new rendering as possible.

| Sensor | Calibration set (real) | Scenes searched | Held-out (real) | Held-out simulated |
|---|---|---|---|---|
| Stereo | real CG d435 val (173 images) | the 100 D435-like renders, seeds 0-99 | real CG d435 test (23), real CG d415 test (90) | 20 fresh D435-like and 20 fresh D415-like scenes, seeds 5000-5019 |
| ToF | real TransCG L515 train scenes 1, 2, 3, 4, 6, 9, 10 | the first 100 of the 200 L515-like renders | real TransCG L515 test scenes 5, 7, 8 | 40 fresh L515-like scenes, seeds 5000-5039 |

Procedure:

1. Compute the real targets of the calibration sets only (`tools/calibration/calib_targets.py` writes `data/calibration_targets.csv`).
2. Run the grid searches (`tools/calibration/calibrate.py` writes `data/calibration_search.csv`), choose the parameters with the lowest calibration loss subject to the opaque-twin constraint, and freeze them in `data/configs/calibrated.json`.
3. Only then render the fresh evaluation scenes, in batches of ten with three-minute pauses (`tools/simulation/render_fresh.sh`), and compute the held-out real and simulated numbers (`tools/calibration/calibration_eval.py` writes `data/calibration_heldout.csv`).
4. Report real, uncalibrated simulated and calibrated simulated on the same fresh scenes, with calibration and held-out loss. If calibration does not improve the held-out match, say so.

Caveats stated in advance:

- The held-out real numbers were already seen in earlier phases of the work, so the held-out sets are not blind. What the protocol enforces is mechanical: the search reads only calibration-set targets, the held-out evaluation is a separate script run after the parameters were frozen, and the held-out simulated scenes are fresh seeds that the search never touched.
- The calibration real data and the simulated scenes differ in objects, layout and resolution. A better match on the calibration set can come from compensating for those differences and not from better physics. Transfer to the held-out sets is what tests that.
- The parameters are physically interpretable (matcher confidence, ToF glass response) but they are fitted knobs, not measurements.

## Known limits

- TransCG has no independent opaque control and its ground truth is rendered from poses and meshes.
- The TransCG copy used is scenes 1-10 only.
- ClearGrasp real-val has a capture mismatch between raw and GT on non-mask pixels, concentrated in a minority of images; a per-image plane + scale fit does not remove it.
- D435 results differ between ClearGrasp and TransCG; objects, distances, lighting and ground-truth method differ, so the cameras are not compared like for like.
- The explanation that a ToF sensor returns confident wrong values where stereo sensors drop out is a hypothesis; it was not tested.
- The simulated ToF sensor is an approximation, objects are limited to nine ClearGrasp models, scenes are simple, and held-out simulation sets are small.
