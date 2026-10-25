# Tools

Every script that produces a number or a figure in this repository, grouped by stage. Run all commands from the repository root. Paths are resolved by [`paths.py`](paths.py), so the working directory does not matter.

## Two commands

```bash
pip install -r tools/requirements.txt
python examples/load_sample.py
```

The first installs numpy and OpenCV. The second loads the nine-scene open sample and prints its three-bucket table.

## Reproduce the real-data tables

Needs the full requirements and the downloads listed in [`../docs/access.md`](../docs/access.md).

```bash
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements-full.txt
.venv/bin/python tools/evaluation/three_bucket.py        # data/three_bucket_summary.csv, the main table
```

| Stage | Folder | Command | Writes |
|---|---|---|---|
| Loaders | `loaders/` | imported by every stage | |
| Unit checks | `checks/` | `python tools/checks/check_transcg_pose.py`, `python tools/checks/check_scale.py` | `data/pose_check_output.txt` (by redirecting the first command) |
| Three-bucket evaluation | `evaluation/` | `python tools/evaluation/three_bucket.py`, then again with `--erode 3`, `--erode 5` and `--bias-correct` | `data/three_bucket_*.csv`, `data/bias_params_biascorrect.csv` |
| Per-object proxy | `evaluation/` | `python tools/evaluation/object_proxy.py` | `data/object_proxy.csv` |
| Model evaluation | `evaluation/` | `python tools/evaluation/models_eval.py` | `data/models_*.csv` |
| Flagging | `flagging/` | `python tools/flagging/flag_eval.py` | `data/flagging_summary.csv` |
| Simulation | `simulation/` | `blender -b -P tools/simulation/render_scene.py -- --variant d415 --seeds 0 1 2 --out synth/out [--ir-twin]`, then `python tools/simulation/synth_eval.py` | `data/synth_vs_real.csv` |
| Calibration | `calibration/` | `calib_targets.py`, `calibrate.py`, `tools/simulation/render_fresh.sh`, `calibration_eval.py` | `data/calibration_*.csv` |
| Figures | `figures/` | `figure1.py`, `error_maps.py`, `models_figure.py`, `synth_figures.py`, `calibration_figure.py` | `docs/media/figures/` |
| Documents | `readme/` | `make_results.py`, `make_data_docs.py`, `make_readme.py` | `docs/results.md`, `docs/schema.md`, `data/benchmark-stats.json`, `data/fields.json`, `README.md` |

`python tools/validate_release.py` checks every CSV header against `data/fields.json` and every JSON file against `data/schema/`.

## Notes

- Raw datasets go in `datasets/`, model weights in `~/glassbench_models/`, full-scale renders in `synth/`. All three are git-ignored.
- The evaluation and document stages take minutes on a laptop. Model evaluation needs PyTorch (tested on Apple M4 with MPS) and rendering needs Blender 4.5 with a GPU.
- `make_readme.py` reads only `data/*.csv`, so the README cannot drift from the results.
