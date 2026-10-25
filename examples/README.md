# Examples

The open synthetic sample and a short walkthrough of loading it.

## The sample

[`sample/`](sample/) holds nine scenes rendered by us: three each for a D415-like and a D435-like active-stereo sensor and an L515-like time-of-flight sensor, 640x360 pixels. Each scene has the visible render, the ground-truth depth from an opaque twin, the transparent-object mask, and simulated raw depth with the original and the calibrated settings. Every file is described in [`../docs/schema.md`](../docs/schema.md).

Licence: CC BY 4.0. The scenes use the ClearGrasp 3D models (Apache-2.0).

## Load it

```bash
pip install -r tools/requirements.txt
python examples/load_sample.py
```

```
sensor  scenes        setting  correct %  missing %  wrong %  too far %
d415         3   uncalibrated       33.9        8.9     57.2      100.0
d415         3     calibrated       25.6       38.1     36.4      100.0
d435         3   uncalibrated       37.4        2.9     59.7       99.9
d435         3     calibrated       30.5       21.6     47.9      100.0
l515         3   uncalibrated       13.1        1.5     85.4      100.0
l515         3     calibrated       28.1        1.5     70.4      100.0
```

This is the three-bucket rule from [`../docs/measurement.md`](../docs/measurement.md) on nine simulated scenes: GT-valid pixels inside the transparent mask, 5% tolerance, pooled per sensor. With only three scenes per sensor the shares are noisy, so read them as a check that the files load and the rule runs, not as a result. The benchmark results are in [`../data/`](../data/).

## Use it in your own code

`load_scene` returns one scene as arrays in metres:

```python
import sys
sys.path.insert(0, "examples")
from load_sample import load_scene, buckets

s = load_scene("examples/sample/l515/00000")
region = s["mask"] & s["gt_valid"]
n, missing, correct, wrong, too_far = buckets(s["raw_uncalibrated"], s["gt"], region, lambda gt: 0.05 * gt)
print(f"{100 * wrong / n:.1f}% of glass pixels are confidently wrong, {100 * missing / n:.1f}% are missing")
```

Depth PNGs are uint16 millimetres; 0 means no surface (ground truth) or no return (raw). `meta.json` has the intrinsics, baseline and objects.
