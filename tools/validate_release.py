"""Check the release files against their descriptions and JSON Schemas.

    python tools/validate_release.py

Checks that every CSV header matches data/fields.json, that data/*.json and every sample meta.json validate
against data/schema/, and that every schema is itself valid JSON Schema. Needs jsonschema (tools/requirements-full.txt).
"""
import csv
import glob
import json
import os
import sys

from jsonschema import Draft202012Validator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def main():
    errors = []
    schemas = {os.path.basename(p): load(os.path.relpath(p, ROOT)) for p in glob.glob(os.path.join(ROOT, "data", "schema", "*.json"))}
    for name, schema in schemas.items():
        Draft202012Validator.check_schema(schema)

    fields = load("data/fields.json")["files"]
    for path, spec in fields.items():
        if not path.endswith(".csv"):
            continue
        with open(os.path.join(ROOT, path)) as f:
            header = next(csv.reader(f))
        if header != [x["name"] for x in spec["fields"]]:
            errors.append(f"{path}: header does not match data/fields.json")

    pairs = [("data/benchmark-stats.json", "benchmark-stats"), ("data/configs/calibrated.json", "calibrated-config"), ("data/configs/sgbm.json", "sgbm-config")]
    pairs += [(os.path.relpath(p, ROOT), "sample-meta") for p in sorted(glob.glob(os.path.join(ROOT, "examples", "sample", "*", "*", "meta.json")))]
    for path, key in pairs:
        for e in Draft202012Validator(schemas[f"{key}.schema.json"]).iter_errors(load(path)):
            errors.append(f"{path}: {e.message}")

    for e in errors:
        print("FAIL", e)
    print(f"{len(fields)} described files, {len(pairs)} JSON files, {len(schemas)} schemas: {'FAILED' if errors else 'all valid'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
