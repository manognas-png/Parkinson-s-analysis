"""Build the healthy-stride templates for the demo: the average control stride
(left heel strike to left heel strike, 100 points, body-weight units) for the
16-sensor and 6-sensor versions. Saved to demo/template.json for demo/engine.js.

Run from the project root:  python demo/build_template.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.demo_metrics import STRIDE_POINTS, VERSIONS, template_channels, template_strides  # noqa: E402
from src.load import DATA_DIR, FILENAME_PATTERN, load_demographics, load_trial, parse_filename  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "template.json"


def main():
    weights = load_demographics().set_index("subject_id")["weight_kg"]
    files = sorted(p for p in DATA_DIR.glob("*Co*_01.txt") if FILENAME_PATTERN.match(p.name))
    strides = {version: [] for version in VERSIONS}
    used = []
    for path in files:
        subject = parse_filename(path)["subject_id"]
        weight = weights.get(subject, np.nan)
        if not np.isfinite(weight):
            continue  # template needs body-weight units
        trial = load_trial(path)
        for version in VERSIONS:
            strides[version].append(template_strides(trial, version, weight))
        used.append(subject)

    template = {
        "description": "Average control stride (left heel strike to left heel strike), "
                       "force / body weight, built from all control _01 trials with a recorded weight",
        "n_points": STRIDE_POINTS,
        "subjects": used,
    }
    for version in VERSIONS:
        stacked = np.concatenate(strides[version])
        template[f"template_{version}"] = {
            "channels": template_channels(version),
            "n_strides": int(len(stacked)),
            "values": np.round(stacked.mean(axis=0), 10).tolist(),
        }
        print(f"{version}-sensor template: {len(stacked)} strides from {len(used)} controls")
    OUTPUT.write_text(json.dumps(template))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
