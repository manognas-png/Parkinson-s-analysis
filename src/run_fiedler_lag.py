"""Step 6b: per-foot, per-stride lagged cross-correlation networks.

For each _01 trial and each foot: one 8-sensor network per steady stride, with
edge weight = max |cross-correlation| within +/- 0.3 s and the lag at that max.
Per subject: median Fiedler value over strides, its CV over strides, and the
median heel (1-3) -> forefoot (6-8) lag; left and right feet averaged.

Run as a SLURM job (not on the login node):  sbatch slurm/fiedler_lag.sbatch
Options: --max-lag SECONDS (default 0.3), --suffix TEXT (added to output file names)
"""

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.features_math import algebraic_connectivity, lag_network
from src.gait import coefficient_of_variation, steady_stride_bounds
from src.load import DATA_DIR, FILENAME_PATTERN, SAMPLING_RATE_HZ, load_demographics, load_trial, parse_filename
from src.plot_explore import FIGURE_DIR, GROUP_COLORS, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save

N_JOBS = 4
MAX_LAG_S = 0.3  # default search window for the lag, +/- seconds
FEET = {"L": "L_total", "R": "R_total"}
HEEL = [0, 1, 2]  # sensors 1-3
FOREFOOT = [5, 6, 7]  # sensors 6-8
FEATURES = ["fiedler_median", "fiedler_cv", "heel_forefoot_lag_s"]
EXPECTED_SUBJECTS = 165
DIAGNOSTICS = ["strides_with_inactive_sensor", "heel_forefoot_negative_peak_fraction",
               "heel_forefoot_at_boundary_fraction"]


def fail(message):
    sys.exit(f"ERROR: {message}")


def check_inputs():
    """Stop before any work if the raw data or demographics are missing."""
    if not DATA_DIR.is_dir():
        fail(f"data folder not found: {DATA_DIR}")
    if not (DATA_DIR / "demographics.txt").exists():
        fail(f"missing {DATA_DIR / 'demographics.txt'}")
    files = sorted(p for p in DATA_DIR.glob("*_01.txt") if FILENAME_PATTERN.match(p.name))
    if len(files) != EXPECTED_SUBJECTS:
        fail(f"expected {EXPECTED_SUBJECTS} _01 trial files in {DATA_DIR}, found {len(files)}")
    demographics = load_demographics()
    if "speed_01" not in demographics.columns:
        fail("demographics has no speed_01 column")
    return files, demographics


def foot_features(trial, foot, max_lag):
    """Per-stride networks for one foot -> subject-level numbers + median lag matrix.

    `max_lag` is in samples. Lags that land exactly on +/- max_lag are counted:
    there the true peak may lie outside the window (the lag is censored).
    """
    sensors = [f"{foot}{i}" for i in range(1, 9)]
    time_s = trial["time"].to_numpy()
    signals = trial[sensors].to_numpy().T

    fiedler, heel_forefoot_lag, lag_matrices = [], [], []
    n_inactive, n_negative, n_boundary, n_pairs = 0, 0, 0, 0
    for start, end in steady_stride_bounds(trial[FEET[foot]], trial["time"]):
        first, last = np.searchsorted(time_s, [start, end])
        A, lag, sign = lag_network(signals[:, first:last], max_lag)
        lag_s = lag / SAMPLING_RATE_HZ

        fiedler.append(algebraic_connectivity(A))
        heel_forefoot_lag.append(np.nanmean(lag_s[np.ix_(HEEL, FOREFOOT)]))
        lag_matrices.append(lag_s)
        n_inactive += int((A.sum(axis=1) == 0).any())
        n_negative += int((sign[np.ix_(HEEL, FOREFOOT)] < 0).sum())
        n_boundary += int((np.abs(lag[np.ix_(HEEL, FOREFOOT)]) >= max_lag).sum())
        n_pairs += len(HEEL) * len(FOREFOOT)

    if not fiedler:
        return {"n_strides": 0}, None
    return {
        "n_strides": len(fiedler),
        "fiedler_median": np.median(fiedler),
        "fiedler_cv": coefficient_of_variation(fiedler),
        "heel_forefoot_lag_s": np.nanmedian(heel_forefoot_lag),
        "strides_with_inactive_sensor": n_inactive,
        "heel_forefoot_negative_peak_fraction": n_negative / n_pairs,
        "heel_forefoot_at_boundary_fraction": n_boundary / n_pairs,
    }, np.nanmedian(lag_matrices, axis=0)


def subject_features(path, max_lag):
    """Both feet for one trial; features averaged over feet (NaN-aware)."""
    info = parse_filename(path)
    trial = load_trial(path)
    record = {"subject_id": info["subject_id"], "study": info["study"], "group": info["group"]}
    per_foot = {}
    left_lag_matrix = None
    for foot in FEET:
        values, lag_matrix = foot_features(trial, foot, max_lag)
        per_foot[foot] = values
        record[f"n_strides_{foot}"] = values["n_strides"]
        if foot == "L":
            left_lag_matrix = lag_matrix
    for name in [*FEATURES, *DIAGNOSTICS]:
        values = [per_foot[f].get(name, np.nan) for f in FEET]
        record[name] = np.nanmean(values) if not np.all(np.isnan(values)) else np.nan
    return record, left_lag_matrix


def compare_raw_and_adjusted(table):
    """Raw vs speed-adjusted (Huber, controls only) PD vs control comparison."""
    from src.plot_svd import compare_groups
    from src.speed_adjust import speed_adjusted_residuals

    rows = []
    for feature in FEATURES:
        table[f"{feature}_adjusted"], coefficients = speed_adjusted_residuals(table, feature)
        print(f"  speed coefficient in controls, {feature:20s} {coefficients[1]:+.4f} per m/s")
        raw = compare_groups(table.dropna(subset=[feature]), feature)
        adjusted = compare_groups(table.dropna(subset=[f"{feature}_adjusted"]), f"{feature}_adjusted")
        for study in raw.index:
            rows.append({"feature": feature, "study": study,
                         "raw_p_pd_higher": raw.loc[study, "p_pd_higher"], "raw_p_value": raw.loc[study, "p_value"],
                         "adjusted_p_pd_higher": adjusted.loc[study, "p_pd_higher"],
                         "adjusted_p_value": adjusted.loc[study, "p_value"]})
    return pd.DataFrame(rows)


def plot_lag_matrices(lag_matrices, max_lag_s, filename):
    """Figure 12: left-foot 8 x 8 lag matrix (ms), averaged over controls vs PD."""
    means = {group: np.nanmean(np.stack(m), axis=0) * 1000 for group, m in lag_matrices.items()}
    limit = max(np.nanmax(np.abs(m)) for m in means.values())
    labels = ["L1 heel", "L2", "L3", "L4", "L5", "L6", "L7", "L8 toe"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (group, matrix) in zip(axes, means.items()):
        image = ax.imshow(matrix, cmap="BrBG", vmin=-limit, vmax=limit)
        for i in range(8):
            for j in range(8):
                if i != j:
                    ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(matrix[i, j]) > 0.6 * limit else TEXT_COLOR)
        ax.set_xticks(range(8))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(8))
        ax.set_yticklabels(labels)
        ax.set_xlabel("sensor j")
        ax.set_ylabel("sensor i")
        ax.grid(False)
        ax.set_title(f"{group} (n = {len(lag_matrices[group])})", color=GROUP_COLORS[group], loc="left")
    fig.colorbar(image, ax=axes, label="lag (ms): > 0 means j follows i", shrink=0.85)
    fig.suptitle(f"Left foot: lag at the strongest cross-correlation (search window ±{max_lag_s} s), "
                 "median over strides, mean over subjects", x=0.06, ha="left")
    save(fig, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lag", type=float, default=MAX_LAG_S, help="lag search window, +/- seconds")
    parser.add_argument("--suffix", default="", help="added to output file names, e.g. _lag0.5")
    args = parser.parse_args()
    max_lag = int(round(args.max_lag * SAMPLING_RATE_HZ))
    suffix = args.suffix

    start = time.time()
    print(f"lag search window: +/- {args.max_lag} s ({max_lag} samples)")
    files, demographics = check_inputs()

    results = Parallel(n_jobs=N_JOBS)(delayed(subject_features)(path, max_lag) for path in files)
    table = pd.DataFrame([record for record, _ in results])
    table = table.merge(demographics[["subject_id", "speed_01"]], on="subject_id", how="left")
    print(f"{len(table)} subjects processed in {time.time() - start:.0f} s")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.drop(columns="speed_01").to_csv(METRICS_DIR / f"fiedler_lag_features{suffix}.csv", index=False)
    print(f"saved {(METRICS_DIR / f'fiedler_lag_features{suffix}.csv').relative_to(RESULTS_DIR.parent)}")

    print("\nSteady strides per foot (median [min-max]): "
          f"L {table['n_strides_L'].median():.0f} [{table['n_strides_L'].min()}-{table['n_strides_L'].max()}], "
          f"R {table['n_strides_R'].median():.0f} [{table['n_strides_R'].min()}-{table['n_strides_R'].max()}]")
    print(f"Subjects without any steady stride: {list(table.loc[table['fiedler_median'].isna(), 'subject_id'])}")
    print("Median per group:")
    print(table.groupby("group")[[*FEATURES, *DIAGNOSTICS]].median().round(4).T.to_string())
    at_edge = table["heel_forefoot_at_boundary_fraction"]
    print(f"Heel->forefoot lags on the window edge (+/- {args.max_lag} s): median {at_edge.median():.1%} of pairs "
          f"per subject; subjects with > 25% on the edge: {(at_edge > 0.25).sum()} of {at_edge.notna().sum()}")

    print()
    comparison = compare_raw_and_adjusted(table)
    comparison.to_csv(METRICS_DIR / f"fiedler_lag_comparison{suffix}.csv", index=False)
    view = comparison.assign(
        raw=lambda d: d["raw_p_pd_higher"].map("{:.2f}".format) + " (" + d["raw_p_value"].map("{:.3f}".format) + ")",
        adjusted=lambda d: d["adjusted_p_pd_higher"].map("{:.2f}".format) + " ("
        + d["adjusted_p_value"].map("{:.3f}".format) + ")",
    ).pivot(index="feature", columns="study", values=["raw", "adjusted"])
    view = view.reindex(FEATURES).swaplevel(axis=1)[[(s, k) for s in ["All", "Ga", "Ju", "Si"]
                                                     for k in ["raw", "adjusted"]]]
    pd.set_option("display.width", 220)
    print("\nP(PD > Ctrl) (Mann-Whitney p): raw vs adjusted for speed + study (Huber, controls only)")
    print(view.to_string())

    lag_matrices = {"Control": [], "PD": []}
    for (record, matrix) in results:
        if matrix is not None:
            lag_matrices[record["group"]].append(matrix)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_lag_matrices(lag_matrices, args.max_lag, f"12_lag_networks{suffix}.png")
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
