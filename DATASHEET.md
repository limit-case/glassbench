# Datasheet — GlassBench

Following Gebru et al., *Datasheets for Datasets* (2018). It covers two things: the benchmark, which is a set of measurements made on public datasets, and the open synthetic sample, which is data we generated.

## Motivation

**Why was it created?** Depth cameras fail on glass in two ways. They return nothing, which a robot can see, or they return a plausible but wrong value, which it cannot. Published work on transparent-object depth mostly measures completion error. We wanted the share of pixels in each failure mode, per camera, on real captures, and a way to test whether a simulator reproduces those shares.

**Who created and funded it?** Limit Case, from its own funds.

## Composition

**What does the benchmark contain?** Result files, not images. For five evaluation sets from two public datasets (ClearGrasp real images from RealSense D415 and D435, TransCG real images from RealSense D435 and Intel L515) it publishes, per set, the share of ground-truth-valid pixels inside the transparent-object mask that are correct, missing or confidently wrong, at two tolerances, with error sizes and the share that read too far. It also publishes the same numbers for two published models, for a mask network used as a failure flag, for a per-object proxy, and for simulated sensors before and after calibration. Every file and field is in [`docs/schema.md`](docs/schema.md).

**What does the sample contain?** Nine synthetic scenes rendered by us: three each for a D415-like and a D435-like stereo sensor and an L515-like time-of-flight approximation, 640x360 pixels. Each has an RGB render, ground-truth depth, a mask, and simulated raw depth with two simulator settings.

**Sampling.** The real evaluation sets are every image in the downloaded ClearGrasp test and validation splits and every view with ground truth in the local TransCG copy, which is scenes 1-10 of 130. They are not a random sample of any population. The model comparison on TransCG uses a fixed random subset of 300 views per camera. The sample scenes are the first three per sensor, by seed, with a transparent-object mask covering at least 1% of the image, so they are not representative of the full-scale generator.

**Is anything missing?** Yes. TransCG has no opaque control region. A cell is empty where a measure does not apply to a row; it never means zero.

**Does it contain confidential or offensive content?** No. The sample is renders of tabletop objects. The real images are not redistributed.

**Does it relate to people?** No. The sample has no people. The source datasets are tabletop scenes.

## Collection

The real images were captured by the ClearGrasp and TransCG authors. We did not capture any. ClearGrasp ground truth comes from an opaque twin captured in the same pose; TransCG ground truth is rendered from object poses and meshes. Details and the depth-unit verification are in [`docs/provenance.md`](docs/provenance.md).

The sample was rendered by us with Blender Cycles from the ClearGrasp 3D models. Ground truth comes from an opaque twin of each scene.

## Preprocessing

Depth is converted to metres (ClearGrasp float16 metres; TransCG D435 uint16 / 1000; TransCG L515 uint16 / 4000). A pixel is GT-valid when ground truth is finite and greater than zero. No other cleaning, filtering or normalisation is applied to the real captures before scoring. The rules are in [`docs/measurement.md`](docs/measurement.md).

## Uses

- **Measuring** how often a camera or a depth model fails silently on glass, with a rule other people can rerun.
- **Testing a simulator**: whether its failure shares and failure direction match real captures of the same camera.
- **Developing flagging methods** that decide which returned pixels not to trust.

## Limitations

- **The simulated sensors are approximations.** Stereo depth comes from OpenCV semi-global matching, not the real cameras' matcher; ToF depth comes from a two-return model without multipath or glass absorption. They reproduce the direction of the failure but not the proportions without calibration. After calibration the ToF held-out shares move close to the real ones; the stereo improves only partly.
- **Held-out sets are not blind.** The real held-out numbers had been seen in earlier phases; the protocol enforces the split mechanically, not blindness.
- **TransCG ground truth is rendered**, so some wrong pixels near mask edges may be ground-truth error, and there is no opaque control.
- **Only TransCG scenes 1-10 are used**, not all 130.
- **ClearGrasp real-val has a raw-to-ground-truth mismatch** on non-mask pixels in a minority of images, which is why it is reported apart from real-test.
- **Model A is scored after a fit that uses ground truth** outside the mask, so it is an optimistic bound for a monocular model. DFNet is out of sample only on TransCG scenes 5, 7 and 8, and its ClearGrasp numbers include domain shift.
- **The per-object proxy is not a grasping experiment.** Its 2 cm and 5 cm thresholds are illustrative.
- **Nine objects, simple scenes.** The generator uses the nine ClearGrasp models on a table with little clutter.
- **Sample statistics are not benchmark results.** Three scenes per sensor is too few to generalise from.

## Distribution

The code is Apache-2.0. The sample, the documentation and the result files are CC BY 4.0. ClearGrasp is Apache-2.0. TransCG is CC BY-NC-SA 4.0, research use only: it is not redistributed here, and the figures and result rows derived from TransCG follow its terms. Details in [`docs/access.md`](docs/access.md).

Custom synthetic batches for a specific camera and object set are by request, through the Custom batch request form in this repository's issues.

## Maintenance

Maintained by Limit Case. Errors in the data, the code or the documentation can be reported as issues on [limit-case/glassbench](https://github.com/limit-case/glassbench/issues) using the bug report form. Corrections are recorded in [`CHANGELOG.md`](CHANGELOG.md).
