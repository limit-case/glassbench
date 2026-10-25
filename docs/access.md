# Access

## What is free

Everything in this repository, under the licences below. No sign-up and no form.

- The code in [`tools/`](../tools/) for every stage: loaders, unit checks, three-bucket evaluation, model evaluation, flagging, simulation and calibration.
- The result files in [`data/`](../data/): every CSV, the benchmark statistics, the field descriptions, the JSON Schemas and the simulator configs.
- The documentation in [`docs/`](.) and the [datasheet](../DATASHEET.md).
- The open synthetic sample in [`examples/sample/`](../examples/sample/): nine scenes, three for each of a D415-like, a D435-like and an L515-like simulated sensor.

To load the sample and print its three-bucket table:

```bash
pip install -r tools/requirements.txt
python examples/load_sample.py
```

### Reproducing the real-data results

The real captures are not in this repository. Download them from their owners into `datasets/` (git-ignored), then follow the commands in [`../tools/README.md`](../tools/README.md).

| Dataset | Download | Folder |
|---|---|---|
| ClearGrasp test and validation images | [cleargrasp-dataset-test-val.tar](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-test-val.tar) | `datasets/cleargrasp-dataset-test-val/` |
| ClearGrasp checkpoints | [cleargrasp-checkpoints.zip](https://storage.googleapis.com/cleargrasp/cleargrasp-checkpoints.zip) | `datasets/cleargrasp-checkpoints/` |
| ClearGrasp 3D models | [cleargrasp-3d-models.zip](https://storage.googleapis.com/cleargrasp/cleargrasp-3d-models.zip) (the `.tar` link in the ClearGrasp README no longer works; this `.zip` does) | `datasets/cleargrasp-3d-models/` |
| TransCG | [graspnet.net/transcg](https://graspnet.net/transcg) (needs `metadata.json`, `camera_intrinsics/` and `scene*/`) | `datasets/transcg/` |

For the model comparison, clone [galaxies99/transcg](https://github.com/galaxies99/transcg) and [Shreeyak/cleargrasp](https://github.com/Shreeyak/cleargrasp) and put the DFNet checkpoint in `~/glassbench_models/` (`transcg/`, `dfnet.pth`, `cleargrasp/`). Depth Anything V2 weights download from Hugging Face on first use. Weights are never stored in this repository.

## What is available on request

**Custom synthetic batches.** The open sample comes from a larger generator. We can run that generator for a specific camera and object set, calibrated against real captures of that camera, and deliver the rendered scenes with ground truth and simulated raw depth.

To ask, [open a Custom batch request](https://github.com/limit-case/glassbench/issues/new?template=custom-batch-request.yml). The form asks for:

1. Your organisation.
2. The robot and task.
3. The camera model.
4. The objects and materials that fail.
5. The approximate volume.
6. Your timeline.

The request is a public GitHub issue, so describe the need at a level you are comfortable making public. We reply on the issue. Terms for a batch are confirmed in that reply before any work starts.

## Licences

| What | Licence |
|---|---|
| Code in `tools/` and `examples/load_sample.py` | Apache-2.0 ([`LICENSE.txt`](../LICENSE.txt)) |
| Synthetic sample in `examples/sample/` | CC BY 4.0. Attribution: "GlassBench synthetic sample, Limit Case, 2026". The scenes use the ClearGrasp 3D models (Apache-2.0). |
| Documentation, result files in `data/`, and figures made from our own renders | CC BY 4.0 |
| ClearGrasp data, models and code | Apache-2.0 |
| TransCG data, code and DFNet checkpoint | CC BY-NC-SA 4.0. **Not redistributed here.** Research use only, with attribution to Fang et al., RA-L 2022. |
| Figures that show TransCG frames or maps derived from them | Follow the TransCG terms: CC BY-NC-SA 4.0, research use only. In this repository: `docs/media/figures/figure1.png`, `docs/media/figures/error_maps/TransCG_*`, `docs/media/figures/models/*` and `docs/media/figures/synth/real_vs_synth_l515.png`. |
| Figures that show ClearGrasp frames | Apache-2.0 |

The result files in `data/` are aggregate statistics computed from the source datasets. Rows for TransCG are derived from TransCG; if your use of them would not be allowed under the TransCG terms, do not use those rows.

Cloning this repository does not grant any right to the datasets above. Read their licences at the sources.
