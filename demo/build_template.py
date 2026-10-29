"""Build the healthy-stride templates and the insole weight calibration for the demo.

1. Insole weight estimate per subject and version: mean of left + right total force over
   complete steady strides (src/demo_metrics.weight_estimate_raw).
2. Accuracy against recorded weight (estimated / recorded), controls and PD separately.
3. If the median ratio is off by more than BIAS_THRESHOLD, calibrate: factor = median of
   recorded / estimated. Accuracy of the calibrated estimate is measured with 5-fold
   cross-validation over subjects (factor learned on the training folds only). The factor
   used by the demo is learned from all subjects with a weight except the demo-pair subjects.
4. Templates: average control stride per foot, divided by the calibrated estimate.

Saved to demo/template.json (templates + calibration + accuracy) and
demo/weight_estimate_report.csv. Part of slurm/demo_rebuild.sbatch.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.demo_metrics import GRAVITY, STRIDE_POINTS, VERSIONS, template_channels, template_strides, weight_estimate_raw  # noqa: E402,E501
from src.load import DATA_DIR, FILENAME_PATTERN, load_demographics, load_trial, parse_filename  # noqa: E402

DEMO = Path(__file__).resolve().parent
OUTPUT = DEMO / "template.json"
BIAS_THRESHOLD = 0.03  # calibrate if the median estimated/recorded ratio is more than 3 % away from 1
N_FOLDS = 5
SEED = 0


def demo_subjects():
    """Subjects shown in the demo (kept out of the calibration), if demo_data.json exists."""
    path = DEMO / "demo_data.json"
    if not path.exists():
        return set()
    pairs = json.loads(path.read_text())["pairs"]
    return {p["control"] for p in pairs} | {p["pd"] for p in pairs}


def spread(values):
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {"median": q[2], "q25": q[1], "q75": q[3], "p05": q[0], "p95": q[4]}


def main():
    weights = load_demographics().set_index("subject_id")["weight_kg"]
    files = sorted(p for p in DATA_DIR.glob("*_01.txt") if FILENAME_PATTERN.match(p.name))
    trials = {parse_filename(p)["subject_id"]: (parse_filename(p)["group"], load_trial(p)) for p in files}

    # 1-2. Raw estimates and their accuracy
    rows = []
    for subject, (group, trial) in trials.items():
        row = {"subject_id": subject, "group": group, "recorded_n": weights.get(subject, np.nan) * GRAVITY}
        for version in VERSIONS:
            row[f"raw_{version}"] = weight_estimate_raw(trial, version)
        rows.append(row)
    table = pd.DataFrame(rows)
    known = table.dropna(subset=["recorded_n", "raw_16", "raw_6"]).reset_index(drop=True)
    print(f"{len(known)} subjects with a recorded weight and steady strides")

    report = []
    for version in VERSIONS:
        known[f"ratio_raw_{version}"] = known[f"raw_{version}"] / known["recorded_n"]
        for group in ["Control", "PD", "All"]:
            part = known if group == "All" else known[known["group"] == group]
            report.append({"version": version, "estimate": "raw", "group": group, "n": len(part),
                           **spread(part[f"ratio_raw_{version}"])})

    # 3. Calibration decision, cross-validated accuracy, demo factor
    calibration, accuracy = {}, {}
    for version in VERSIONS:
        bias = abs(known[f"ratio_raw_{version}"].median() - 1)
        apply = bias > BIAS_THRESHOLD
        out_of_fold = np.full(len(known), np.nan)
        for train, test in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(known):
            factor = np.median(known["recorded_n"].iloc[train] / known[f"raw_{version}"].iloc[train]) if apply else 1.0
            out_of_fold[test] = known[f"raw_{version}"].iloc[test] * factor / known["recorded_n"].iloc[test]
        known[f"ratio_cv_{version}"] = out_of_fold
        for group in ["Control", "PD", "All"]:
            part = known if group == "All" else known[known["group"] == group]
            report.append({"version": version, "estimate": "calibrated (cross-validated)" if apply else "raw",
                           "group": group, "n": len(part), **spread(part[f"ratio_cv_{version}"])})
        training = known[~known["subject_id"].isin(demo_subjects())]
        calibration[version] = float(np.median(training["recorded_n"] / training[f"raw_{version}"])) if apply else 1.0
        accuracy[version] = {
            "calibrated": bool(apply),
            "median_abs_error_pct": float(100 * np.median(np.abs(out_of_fold - 1))),
            "p90_abs_error_pct": float(100 * np.percentile(np.abs(out_of_fold - 1), 90)),
            "raw_median_ratio": float(known[f"ratio_raw_{version}"].median()),
            "n_subjects": int(len(known)),
            "n_training_subjects_for_factor": int(len(training)),
        }
        print(f"{version} sensors: raw median ratio {accuracy[version]['raw_median_ratio']:.3f} -> "
              f"{'calibrate' if apply else 'no calibration'}; factor {calibration[version]:.4f} "
              f"(from {len(training)} subjects, demo subjects excluded); cross-validated median |error| "
              f"{accuracy[version]['median_abs_error_pct']:.1f} %, 90th percentile "
              f"{accuracy[version]['p90_abs_error_pct']:.1f} %")

    report = pd.DataFrame(report)
    report.to_csv(DEMO / "weight_estimate_report.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n== Estimated / recorded body weight (median, IQR, 5-95 %) ==")
    print(report.round(3).to_string(index=False))

    # 4. Templates with the calibrated estimator
    template = {
        "description": "Average control stride per foot (heel strike to heel strike of that foot), divided by the "
                       "calibrated insole weight estimate (mean left + right total force over steady strides)",
        "n_points": STRIDE_POINTS,
        "weight_estimate": {"calibration": calibration, "accuracy": accuracy, "bias_threshold": BIAS_THRESHOLD},
    }
    controls = [s for s, (g, _) in trials.items() if g == "Control"]
    template["subjects"] = controls
    for version in VERSIONS:
        template[version] = {"channels": template_channels(version)}
        for foot in ["L", "R"]:
            stacked = np.concatenate([template_strides(trials[s][1], version, foot, calibration) for s in controls])
            template[version][foot] = {"n_strides": int(len(stacked)),
                                       "values": np.round(stacked.mean(axis=0), 10).tolist()}
            print(f"{version}-sensor template, {foot} foot: {len(stacked)} strides from {len(controls)} controls")
    OUTPUT.write_text(json.dumps(template))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
