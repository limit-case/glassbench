# GlassBench synthetic sample

Nine scenes: three each for a D415-like and a D435-like active-stereo sensor (`d415/`, `d435/`) and an L515-like time-of-flight approximation (`l515/`), 640x360 pixels. Our own Blender (Cycles) renders of tabletop scenes with transparent objects, with simulated depth-sensor outputs. Made by `tools/simulation/render_scene.py` and `tools/simulation/make_samples.py`; the scenes are the first three per sensor, by seed, whose transparent-object mask covers at least 1% of the image. Full-scale generation (400 scenes) is reproducible from the repository; only this sample is distributed.

**Licence: CC BY 4.0.** Attribution: "GlassBench synthetic sample, Limit Case, 2026". The scenes use the ClearGrasp 3D models (Apache-2.0, <https://github.com/Shreeyak/cleargrasp>).

Every file and field is described in [`../../docs/schema.md`](../../docs/schema.md). Load the sample with `python examples/load_sample.py`.

## Caveats

These are simulations. The stereo depth comes from OpenCV semi-global matching, not the real cameras' matcher, and the ToF depth from a simple two-return approximation; caustics through glass use an approximate light-transport method. The simulated sensors reproduce the direction of the real failure (confident wrong depth that reads too far) but not its exact proportions; see [`../../docs/results.md`](../../docs/results.md). The sample is for inspection and method development, not a replacement for real captures.
