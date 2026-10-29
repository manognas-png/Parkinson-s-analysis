"""Prepare the data embedded in demo/gait_demo.html (written to demo/demo_data.json):

- group medians (controls, PD) of the per-stride demo metrics, 16- and 6-sensor versions
- group stance curves (heel load and centre of pressure, 0-100 % of stance), mean and SD
- 3 control-PD pairs from the Ga study, similar in age and walking speed, with PD patients
  near the PD median of early-stance heel load and controls within the middle 50 % of all
  controls (both typical of their group, not extreme)
- a 10-second steady-walking window of each of those 6 recordings, downsampled to 50 Hz

Run as a SLURM job (about 2 min of computing):  sbatch --account=bioinfmath540f26_class slurm/demo_data.sbatch
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.demo_metrics import VERSIONS, stance_curves, trial_metrics  # noqa: E402
from src.gait import steady_window  # noqa: E402
from src.load import DATA_DIR, FILENAME_PATTERN, load_demographics, load_trial, parse_filename  # noqa: E402

DEMO = Path(__file__).resolve().parent
N_JOBS = 8  # matches --cpus-per-task in slurm/demo_data.sbatch
WINDOW_S = 10
DOWNSAMPLE = 2  # 100 Hz -> 50 Hz
N_PAIRS = 3
MAX_AGE_DIFFERENCE = 5  # years
MAX_SPEED_DIFFERENCE = 0.15  # m/s
CARD_METRICS = ["heel_load_early", "cop_05_15", "cadence", "template_deviation"]
TRIAL_COLUMNS = ["time", *[f"{f}{i}" for f in "LR" for i in range(1, 9)], "L_total", "R_total"]


def load_templates():
    data = json.loads((DEMO / "template.json").read_text())
    return {v: {f: np.array(data[v][f]["values"]) for f in "LR"} for v in VERSIONS}


def subject_summary(path, weight_kg, templates):
    """Per-subject medians of the card metrics and mean stance curves, per version."""
    trial = load_trial(path)
    weight = weight_kg if np.isfinite(weight_kg) else None
    rows = pd.DataFrame(trial_metrics(trial, weight, templates))
    summary = {"subject_id": parse_filename(path)["subject_id"]}
    for version in VERSIONS:
        part = rows[rows["version"] == version] if len(rows) else rows
        for metric in CARD_METRICS:
            values = part[metric].dropna() if len(part) else pd.Series(dtype=float)
            if metric == "heel_load_early" and weight is None:
                values = pd.Series(dtype=float)  # newtons are not comparable across people
            summary[f"{metric}_{version}"] = float(values.median()) if len(values) else np.nan
    curves = stance_curves(trial, weight) if weight else None
    return summary, curves


def choose_pairs(table, pd_median_heel):
    """Ga PD patients closest to the PD median heel load, each matched to a different Ga control
    within MAX_AGE_DIFFERENCE years and MAX_SPEED_DIFFERENCE m/s (smallest combined distance).
    Controls must be typical too: heel load within the interquartile range of all controls."""
    ga = table[(table["study"] == "Ga") & table["speed_01"].notna() & table["heel_load_early_16"].notna()]
    patients = ga[ga["group"] == "PD"].assign(distance_to_median=lambda d: (d["heel_load_early_16"]
                                                                             - pd_median_heel).abs())
    patients = patients.sort_values("distance_to_median")
    all_controls = table.loc[table["group"] == "Control", "heel_load_early_16"]
    low, high = all_controls.quantile([0.25, 0.75])
    controls = ga[(ga["group"] == "Control") & ga["heel_load_early_16"].between(low, high)]
    print(f"typical controls: heel load {low:.3f}-{high:.3f} (all-controls IQR); "
          f"{len(controls)} Ga controls qualify")
    pairs, used = [], set()
    for _, patient in patients.iterrows():
        candidates = controls[~controls["subject_id"].isin(used)]
        age_gap = (candidates["age"] - patient["age"]).abs()
        speed_gap = (candidates["speed_01"] - patient["speed_01"]).abs()
        ok = (age_gap <= MAX_AGE_DIFFERENCE) & (speed_gap <= MAX_SPEED_DIFFERENCE)
        if not ok.any():
            continue
        score = (age_gap / MAX_AGE_DIFFERENCE + speed_gap / MAX_SPEED_DIFFERENCE)[ok]
        control = candidates.loc[score.idxmin()]
        used.add(control["subject_id"])
        pairs.append({"control": control["subject_id"], "pd": patient["subject_id"],
                      "control_age": control["age"], "pd_age": patient["age"],
                      "control_speed": control["speed_01"], "pd_speed": patient["speed_01"],
                      "control_heel": control["heel_load_early_16"], "pd_heel": patient["heel_load_early_16"],
                      "pd_hoehn_yahr": patient["hoehn_yahr"]})
        if len(pairs) == N_PAIRS:
            break
    return pd.DataFrame(pairs)


def demo_window(subject, weight_kg):
    """10 s of steady walking (from a left heel strike), downsampled to 50 Hz, rounded to 0.1 N."""
    trial = load_trial(DATA_DIR / f"{subject}_01.txt")
    left = trial[[f"L{i}" for i in range(1, 9)]].sum(axis=1).to_numpy()
    start, end, _ = steady_window(left, trial["time"].to_numpy(), WINDOW_S)
    window = trial[(trial["time"] >= start) & (trial["time"] < end)].iloc[::DOWNSAMPLE]
    columns = {c: np.round(window[c].to_numpy(), 1).tolist() for c in TRIAL_COLUMNS if c != "time"}
    columns["time"] = np.round(window["time"].to_numpy() - window["time"].iloc[0], 4).tolist()
    return {"subject": subject, "weight_kg": weight_kg, "start_s": round(float(start), 2), "trial": columns}


def main():
    start = time.time()
    demographics = load_demographics()
    weights = demographics.set_index("subject_id")["weight_kg"]
    templates = load_templates()
    files = sorted(p for p in DATA_DIR.glob("*_01.txt") if FILENAME_PATTERN.match(p.name))
    results = Parallel(n_jobs=N_JOBS)(
        delayed(subject_summary)(p, weights.get(parse_filename(p)["subject_id"], np.nan), templates) for p in files)
    meta = pd.DataFrame([parse_filename(p) for p in files])[["subject_id", "study", "group"]]
    table = meta.merge(pd.DataFrame([s for s, _ in results]), on="subject_id").merge(
        demographics[["subject_id", "age", "speed_01", "weight_kg", "hoehn_yahr"]], on="subject_id", how="left")
    table.to_csv(DEMO / "demo_subject_metrics.csv", index=False)
    print(f"{len(table)} subjects summarized in {time.time() - start:.0f} s")

    # Group medians for the cards
    medians = {}
    for version in VERSIONS:
        medians[version] = {group: {m: float(table.loc[table["group"] == group, f"{m}_{version}"].median())
                                    for m in CARD_METRICS} for group in ["Control", "PD"]}
    print("\n== Group medians of subject medians ==")
    print(pd.DataFrame({(v, g): medians[v][g] for v in medians for g in medians[v]}).round(4).to_string())

    # Group stance curves for the group panel (subjects with a recorded weight)
    curves = {}
    for version in VERSIONS:
        curves[version] = {}
        for group in ["Control", "PD"]:
            heel = [c[version]["heel"] for (s, c), g in zip(results, meta["group"]) if c and g == group
                    and c[version]["heel"] is not None]
            cop = [c[version]["cop"] for (s, c), g in zip(results, meta["group"]) if c and g == group
                   and c[version]["cop"] is not None]
            curves[version][group] = {
                "heel_mean": np.round(np.mean(heel, axis=0), 4).tolist(),
                "heel_sd": np.round(np.std(heel, axis=0), 4).tolist(),
                "cop_mean": np.round(np.mean(cop, axis=0), 4).tolist(),
                "cop_sd": np.round(np.std(cop, axis=0), 4).tolist(),
                "n_subjects": len(heel),
            }

    # Pairs
    pd_median_heel = table.loc[table["group"] == "PD", "heel_load_early_16"].median()
    pairs = choose_pairs(table, pd_median_heel)
    print(f"\n== Chosen pairs (PD median early-stance heel load, all PD: {pd_median_heel:.3f} body weight; "
          f"control median {table.loc[table['group'] == 'Control', 'heel_load_early_16'].median():.3f}) ==")
    print(pairs.round(3).to_string(index=False))
    if len(pairs) < N_PAIRS:
        print(f"WARNING: only {len(pairs)} pairs met the age/speed limits")

    windows = {}
    for subject in [*pairs["control"], *pairs["pd"]]:
        windows[subject] = demo_window(subject, float(weights[subject]))
    data = {"medians": medians, "curves": curves, "pairs": pairs.to_dict(orient="records"), "windows": windows,
            "pd_median_heel": float(pd_median_heel)}
    (DEMO / "demo_data.json").write_text(json.dumps(data))
    print(f"\nsaved {DEMO / 'demo_data.json'} ({(DEMO / 'demo_data.json').stat().st_size / 1e3:.0f} kB); "
          f"total {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
