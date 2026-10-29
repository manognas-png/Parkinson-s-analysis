"""Build the healthy-stride templates for the demo: the average control stride for each
foot (heel strike to heel strike of that foot, 100 points), for the 16- and 6-sensor
versions. Each stride is divided by its own single-support force, so no body weight is
needed. Saved to demo/template.json for demo/engine.js.

Run from the project root:  python demo/build_template.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.demo_metrics import STRIDE_POINTS, VERSIONS, template_channels, template_strides  # noqa: E402
from src.load import DATA_DIR, FILENAME_PATTERN, load_trial, parse_filename  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "template.json"


def main():
    files = sorted(p for p in DATA_DIR.glob("*Co*_01.txt") if FILENAME_PATTERN.match(p.name))
    strides = {version: {"L": [], "R": []} for version in VERSIONS}
    for path in files:
        trial = load_trial(path)
        for version in VERSIONS:
            for foot in ["L", "R"]:
                strides[version][foot].append(template_strides(trial, version, foot))

    template = {
        "description": "Average control stride per foot (heel strike to heel strike of that foot), each stride "
                       "divided by its single-support force; all control _01 trials",
        "n_points": STRIDE_POINTS,
        "subjects": [parse_filename(p)["subject_id"] for p in files],
    }
    for version in VERSIONS:
        template[version] = {"channels": template_channels(version)}
        for foot in ["L", "R"]:
            stacked = np.concatenate(strides[version][foot])
            template[version][foot] = {"n_strides": int(len(stacked)),
                                       "values": np.round(stacked.mean(axis=0), 10).tolist()}
            print(f"{version}-sensor template, {foot} foot: {len(stacked)} strides from {len(files)} controls")
    OUTPUT.write_text(json.dumps(template))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
