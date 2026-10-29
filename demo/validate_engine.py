"""Check that demo/engine.js reproduces the Python pipeline (src/demo_metrics.py).

Runs both on 10 trials (5 controls, 5 PD, all three studies), matches strides by
version, foot and heel-strike time, and reports the largest Python - JavaScript
difference for every metric. One extra run without body weight tests the newton path
(template deviation needs no body weight and must match there too).

Needs Node.js on the PATH:  module load node.js/22.14.0
Run from the project root:  python demo/validate_engine.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.demo_metrics import METRICS, trial_metrics  # noqa: E402
from src.load import DATA_DIR, load_demographics, load_trial  # noqa: E402

DEMO = Path(__file__).resolve().parent
TOLERANCE = 1e-9  # "equal within rounding" for float64 arithmetic
TRIALS = {  # 5 controls and 5 PD across Ga, Ju, Si (all with a recorded weight)
    "GaCo01": "Control", "GaCo02": "Control", "JuCo01": "Control", "JuCo02": "Control", "SiCo01": "Control",
    "GaPt05": "PD", "GaPt24": "PD", "JuPt01": "PD", "SiPt04": "PD", "SiPt05": "PD",
}
NO_WEIGHT_TRIAL = "GaCo01"


def run_javascript(node, path, weight):
    command = [node, str(DEMO / "run_engine.js"), str(path), "none" if weight is None else str(weight),
               str(DEMO / "template.json")]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return pd.DataFrame(json.loads(output)["strides"]).astype({m: float for m in METRICS})


def compare(python_rows, js_rows, label):
    """Match strides on (version, foot, heel-strike time); return per-metric max differences."""
    key = ["version", "foot", "heel_strike"]
    py = pd.DataFrame(python_rows).assign(heel_strike=lambda d: d["heel_strike"].round(6))
    js = js_rows.assign(heel_strike=lambda d: d["heel_strike"].round(6))
    merged = py.merge(js, on=key, how="outer", suffixes=("_py", "_js"), indicator=True)
    rows = []
    for version in ["16", "6"]:
        part = merged[merged["version"] == version]
        both = part[part["_merge"] == "both"]
        row = {"trial": label, "version": version, "strides_python": int((part["_merge"] != "right_only").sum()),
               "strides_js": int((part["_merge"] != "left_only").sum()), "strides_matched": len(both)}
        for metric in METRICS:
            a, b = both[f"{metric}_py"].to_numpy(), both[f"{metric}_js"].to_numpy()
            same_missing = np.isnan(a) == np.isnan(b)
            row[metric] = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
            if not same_missing.all():
                row[metric] = np.inf  # one side has a value, the other does not
        rows.append(row)
    return rows


def main():
    node = shutil.which("node")
    if node is None:
        sys.exit("ERROR: Node.js not found. Run:  module load node.js/22.14.0")
    print(subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip(), "|", node)
    template_json = json.loads((DEMO / "template.json").read_text())
    templates = {v: {f: np.array(template_json[v][f]["values"]) for f in ["L", "R"]} for v in ["16", "6"]}
    weights = load_demographics().set_index("subject_id")["weight_kg"]

    rows = []
    for subject in TRIALS:
        path = DATA_DIR / f"{subject}_01.txt"
        trial = load_trial(path)
        weight = float(weights[subject])
        rows += compare(trial_metrics(trial, weight, templates), run_javascript(node, path, weight), subject)
    trial = load_trial(DATA_DIR / f"{NO_WEIGHT_TRIAL}_01.txt")
    rows += compare(trial_metrics(trial, None, templates),
                    run_javascript(node, DATA_DIR / f"{NO_WEIGHT_TRIAL}_01.txt", None),
                    f"{NO_WEIGHT_TRIAL} (newtons)")

    table = pd.DataFrame(rows)
    table.to_csv(DEMO / "validation_report.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n== Per trial: strides found and max |Python - JavaScript| per metric ==")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1e}"))

    summary = table.groupby("version")[METRICS].max().T
    summary.columns = [f"max difference, {v}-sensor" for v in summary.columns]
    counts_match = (table["strides_python"] == table["strides_js"]).all() and \
                   (table["strides_matched"] == table["strides_python"]).all()
    print("\n== Summary over all 11 runs: max |Python - JavaScript| ==")
    print(summary.to_string(float_format=lambda v: f"{v:.1e}"))
    passed = counts_match and (summary.to_numpy() <= TOLERANCE).all()
    print(f"\nstride sets identical: {counts_match}; all differences <= {TOLERANCE}: "
          f"{(summary.to_numpy() <= TOLERANCE).all()}")
    print("VALIDATION PASSED" if passed else "VALIDATION FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
