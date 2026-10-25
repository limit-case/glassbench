"""Repository paths shared by every script, and the sys.path setup that lets the stage folders import each other.

Scripts begin with the same three lines so they run from any working directory:

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import paths
"""
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
DATA = os.path.join(ROOT, "data")                  # tracked machine-readable outputs
CONFIGS = os.path.join(DATA, "configs")            # simulator configs
RAW = os.path.join(ROOT, "datasets")               # downloaded ClearGrasp / TransCG files (git-ignored)
SYNTH = os.path.join(ROOT, "synth")                # full-scale renders (git-ignored)
SAMPLE = os.path.join(ROOT, "examples", "sample")  # the open synthetic sample
MEDIA = os.path.join(ROOT, "docs", "media")
FIGURES = os.path.join(MEDIA, "figures")

for _stage in sorted(os.listdir(TOOLS)):
    _d = os.path.join(TOOLS, _stage)
    if os.path.isdir(_d) and not _stage.startswith(("_", ".")) and _d not in sys.path:
        sys.path.insert(0, _d)


def data(name):
    return os.path.join(DATA, name)


def config(name):
    return os.path.join(CONFIGS, name)
