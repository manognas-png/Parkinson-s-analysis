"""Job B: supervised sensor selection (greedy forward selection by CV AUC, nested
inside subject-wise CV), 100 random 6-sensor sets, bootstrap confidence intervals,
and the pre-stated success criterion.

Features per sensor subset: mode-1 energy, effective modes, heel-forefoot coupling,
computed exactly from each subject's 16 x 16 covariance / correlation matrices.

Run as a SLURM job (not on the login node):
    sbatch slurm/time_one_fold.sbatch           # timing only: one fold -> projected runtime
    sbatch slurm/supervised_selection.sbatch    # full run
    --exclude-dead-sensors                      # sensor-5 check: drop trials with a near-dead sensor
"""

import argparse
import sys
import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import bootstrap_auc, fit_model, repeated_group_cv, subject_scores, summarize
from src.features_math import (
    SENSOR_COLUMNS,
    center_rows,
    preprocess,
    qr_pivot_sensors,
    sensor_matrix,
    sensor_modes,
    subset_spectral_features,
)
from src.load import DATA_DIR, FILENAME_PATTERN, load_demographics, load_trial, parse_filename
from src.plot_explore import FIGURE_DIR, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save
from src.plot_svd import DEAD_SENSOR_STD_N
from src.run_ml import make_models
from src.run_sensor_selection import ANATOMICAL

N_JOBS = 16  # matches --cpus-per-task in slurm/supervised_selection.sbatch
N_SPLITS = 5
N_REPEATS = 10
SEED = 0
SIZES = list(range(2, 9))
N_RANDOM = 100
RANDOM_SIZE = 6
N_BOOT = 2000
INNER_C = 1.0  # fixed C inside the greedy search (tuning C there would multiply the cost)
EXPECTED_SUBJECTS = 165
GAIT = ["energy_mode1", "effective_modes", "heel_forefoot"]

# Stated before any results were seen:
SUCCESS_CRITERION = (
    "A 6-sensor set succeeds if (1) its mean CV AUC (gait features only, logistic regression) "
    "is >= 90% of the 16-sensor AUC from the same pipeline, AND (2) it is higher than the median "
    "AUC of 100 random 6-sensor sets."
)


def fail(message):
    sys.exit(f"ERROR: {message}")


def check_inputs():
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


def subject_matrices(path):
    """Centered Gram, correlation and standardized Gram matrices for one trial,
    plus each sensor's standard deviation and mean force (N)."""
    info = parse_filename(path)
    X = sensor_matrix(load_trial(path))
    Xc = center_rows(X)
    Z = preprocess(X, standardize=True)
    standardized_gram = Z @ Z.T
    return (info, Xc @ Xc.T, standardized_gram / X.shape[1], standardized_gram,
            X.std(axis=1), X.mean(axis=1))


def compare_sensor5(info, sensor_std, sensor_mean):
    """Is sensor 5 genuinely different in PD, among trials where every sensor works?"""
    print("Sensor 5 (medial midfoot, under the arch), trials without a near-dead sensor:")
    is_pd = (info["group"] == "PD").to_numpy()
    for name in ["L5", "R5"]:
        k = SENSOR_COLUMNS.index(name)
        for label, values in [("mean force (N)", sensor_mean[:, k]), ("std (N)", sensor_std[:, k])]:
            u, p = mannwhitneyu(values[is_pd], values[~is_pd])
            print(f"  {name} {label:15s} control median {np.median(values[~is_pd]):7.1f} | "
                  f"PD median {np.median(values[is_pd]):7.1f} | P(PD > Ctrl) {u / (is_pd.sum() * (~is_pd).sum()):.2f} "
                  f"(Mann-Whitney p {p:.3f})")


def spectral_table(cov, corr, names):
    """Features for one sensor subset, one row per subject; impossible columns dropped."""
    table = pd.DataFrame([subset_spectral_features(c, r, names) for c, r in zip(cov, corr)])
    return table.dropna(axis=1, how="all")


def inner_auc(cov, corr, y, groups, names, seed):
    """Mean inner-CV AUC (subject-wise) of a fixed-C logistic model on one subset."""
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("model", LogisticRegression(C=INNER_C, class_weight="balanced", max_iter=5000))])
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_score(model, spectral_table(cov, corr, names), y, groups=groups,
                           cv=cv, scoring="roc_auc").mean()


def greedy_path(cov, corr, y, groups, seed):
    """Best pair by inner-CV AUC, then add one sensor at a time up to max(SIZES).

    Starts from pairs: a single sensor gives no usable features (its energy
    fraction is always 1 and coupling needs two sensors).
    """
    pair_scores = {pair: inner_auc(cov, corr, y, groups, list(pair), seed)
                   for pair in combinations(SENSOR_COLUMNS, 2)}
    selected = list(max(pair_scores, key=pair_scores.get))
    path = {2: list(selected)}
    while len(selected) < max(SIZES):
        remaining = [s for s in SENSOR_COLUMNS if s not in selected]
        scores = {s: inner_auc(cov, corr, y, groups, selected + [s], seed) for s in remaining}
        selected.append(max(scores, key=scores.get))
        path[len(selected)] = list(selected)
    return path


def outer_fold(cov, corr, y, groups, speed, train, test, seed):
    """Select sensors on the training subjects only, then score the test subjects.

    Features are per-subject quantities (no cross-subject fitting), so computing
    them for all subjects at once leaks nothing; the selection and the model
    (imputer, scaler, tuned C) only ever see `train`.
    """
    path = greedy_path(cov[train], corr[train], y[train], groups[train], seed)
    scores = {}
    for size, names in path.items():
        features = spectral_table(cov, corr, names)
        for set_name, X in {"gait": features, "gait+speed": features.assign(speed_01=speed)}.items():
            model = make_models(seed)["logistic"]
            fitted = fit_model(clone(model), X.iloc[train], y[train], groups[train])
            scores[(size, set_name)] = fitted.predict_proba(X.iloc[test])[:, 1]
    return path, scores


def outer_splits(y, groups):
    """The same splits evaluate.repeated_group_cv uses (seed + repeat), for paired comparisons."""
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            yield repeat, fold, train, test


def evaluate_fixed(cov, corr, y, groups, speed, study, names, set_name, n_jobs):
    """Repeated subject-wise CV for a fixed sensor set (step 7 logistic model)."""
    X = spectral_table(cov, corr, names)
    if set_name == "gait+speed":
        X = X.assign(speed_01=speed)
    predictions, _ = repeated_group_cv(make_models(SEED)["logistic"], X, y, groups, n_splits=N_SPLITS,
                                       n_repeats=N_REPEATS, seed=SEED, n_jobs=n_jobs,
                                       row_info=pd.DataFrame({"study": study}))
    return predictions


def time_one_fold(cov, corr, y, groups, speed, study):
    """Time the expensive pieces once and project the full run on N_JOBS cores."""
    _, _, train, test = next(outer_splits(y, groups))
    t = time.time()
    outer_fold(cov, corr, y, groups, speed, train, test, SEED)
    fold_seconds = time.time() - t

    t = time.time()
    evaluate_fixed(cov, corr, y, groups, speed, study, SENSOR_COLUMNS[:6], "gait", n_jobs=1)
    fixed_seconds = time.time() - t  # 50 outer fits, one core

    t = time.time()
    table = pd.DataFrame({"y": y, "a": np.random.default_rng(0).random(len(y))})
    bootstrap_auc(table, n_boot=200)
    boot_seconds_per_column = (time.time() - t) * N_BOOT / 200

    n_folds = N_REPEATS * N_SPLITS
    n_fixed = 2 * (1 + len(SIZES) + len(ANATOMICAL)) + N_RANDOM
    n_boot_columns = 2 * (3 + len(SIZES))
    greedy_total = fold_seconds * n_folds / N_JOBS
    fixed_total = fixed_seconds * n_fixed / N_JOBS
    boot_total = boot_seconds_per_column * n_boot_columns
    print(f"one greedy outer fold: {fold_seconds:.1f} s (1 core)")
    print(f"one fixed-set evaluation ({n_folds} outer fits): {fixed_seconds:.1f} s (1 core)")
    print(f"bootstrap: {boot_seconds_per_column:.1f} s per score column")
    print(f"projected on {N_JOBS} cores: greedy {greedy_total / 60:.1f} min ({n_folds} folds), "
          f"fixed sets {fixed_total / 60:.1f} min ({n_fixed} evaluations), bootstrap {boot_total / 60:.1f} min")
    print(f"PROJECTED TOTAL: {(greedy_total + fixed_total + boot_total) / 60:.1f} min "
          f"(+ about 2 min for loading data)")


def plot_results(auc_table, random_aucs, frequency, qr_sets, reference, filename="14_supervised_selection.png"):
    """Figure 14: AUC vs number of sensors; how often each sensor was selected."""
    fig, (ax_auc, ax_freq) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    gait = auc_table[auc_table["feature_set"] == "gait"]
    greedy = gait[gait["method"] == "greedy (nested)"].sort_values("n_sensors")
    qr_rows = gait[gait["method"] == "qr"].sort_values("n_sensors")
    # Everything here is the mean AUC over CV repeats (± SD over repeats), the same
    # scale as the random-set box; bootstrap CIs of the pooled AUC are in the CSV.
    ax_auc.errorbar(greedy["n_sensors"], greedy["auc_mean"], yerr=greedy["auc_sd_repeats"], color="#6250d6",
                    marker="o", linewidth=2, capsize=3, label="greedy forward selection (nested CV)")
    ax_auc.errorbar(qr_rows["n_sensors"], qr_rows["auc_mean"], yerr=qr_rows["auc_sd_repeats"], color="#1baf7a",
                    marker="s", linewidth=2, capsize=3, label="QR pivoting (fixed sets)")
    ax_auc.boxplot(random_aucs, positions=[RANDOM_SIZE], widths=0.4, showfliers=True,
                   boxprops={"color": TEXT_COLOR}, medianprops={"color": TEXT_COLOR})
    ax_auc.plot([], [], color=TEXT_COLOR, label=f"{N_RANDOM} random {RANDOM_SIZE}-sensor sets")
    ax_auc.axhline(reference, color=TEXT_COLOR, linewidth=1, label=f"all 16 sensors ({reference:.3f})")
    ax_auc.axhline(0.9 * reference, color=TEXT_COLOR, linewidth=1, linestyle="--", label="90% of 16-sensor AUC")
    ax_auc.axhline(0.5, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax_auc.set_xticks(SIZES)
    ax_auc.set_xticklabels(SIZES)
    ax_auc.set_xlabel("Number of sensors")
    ax_auc.set_ylabel("CV AUC, gait features only (logistic): mean ± SD over repeats")
    ax_auc.set_title("AUC vs number of sensors", loc="left")
    ax_auc.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    image = ax_freq.imshow(frequency.T.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for i, size in enumerate(frequency.index):
        for sensor in qr_sets.get(size, []):
            j = SENSOR_COLUMNS.index(sensor)
            ax_freq.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="#1baf7a", linewidth=2))
    ax_freq.set_xticks(range(len(frequency.index)))
    ax_freq.set_xticklabels(frequency.index)
    ax_freq.set_yticks(range(16))
    ax_freq.set_yticklabels(SENSOR_COLUMNS)
    ax_freq.set_xlabel("Number of sensors")
    ax_freq.grid(False)
    ax_freq.set_title("Fraction of outer folds selecting each sensor (green box = QR choice)", loc="left")
    fig.colorbar(image, ax=ax_freq, label="selection frequency")
    fig.suptitle("Supervised vs QR sensor selection", x=0.06, ha="left")
    fig.tight_layout()
    save(fig, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-one-fold", action="store_true", help="only time one fold and project the runtime")
    parser.add_argument("--exclude-dead-sensors", action="store_true",
                        help=f"drop trials where any sensor has std < {DEAD_SENSOR_STD_N} N (sensor-5 check)")
    args = parser.parse_args()

    start = time.time()
    print("SUCCESS CRITERION (stated in advance):\n  " + SUCCESS_CRITERION + "\n")
    files, demographics = check_inputs()
    loaded = Parallel(n_jobs=N_JOBS)(delayed(subject_matrices)(p) for p in files)
    info = pd.DataFrame([row[0] for row in loaded]).merge(demographics[["subject_id", "speed_01"]],
                                                          on="subject_id", how="left")
    cov = np.stack([row[1] for row in loaded])
    corr = np.stack([row[2] for row in loaded])
    y = (info["group"] == "PD").astype(int).to_numpy()
    groups = info["subject_id"].to_numpy()
    speed = info["speed_01"].to_numpy()
    study = info["study"].to_numpy()
    print(f"loaded {len(info)} subjects in {time.time() - start:.0f} s")

    suffix = ""
    if args.exclude_dead_sensors:
        sensor_std = np.stack([row[4] for row in loaded])
        sensor_mean = np.stack([row[5] for row in loaded])
        dead = sensor_std.min(axis=1) < DEAD_SENSOR_STD_N
        dropped = info.loc[dead, ["subject_id", "group"]].assign(
            sensor=[SENSOR_COLUMNS[k] for k in sensor_std[dead].argmin(axis=1)])
        print(f"\nExcluding {dead.sum()} trials with a near-dead sensor (std < {DEAD_SENSOR_STD_N} N):")
        print(dropped.to_string(index=False))
        keep = ~dead
        loaded = [row for row, k in zip(loaded, keep) if k]
        info = info[keep].reset_index(drop=True)
        cov, corr = cov[keep], corr[keep]
        y, groups, speed, study = y[keep], groups[keep], speed[keep], study[keep]
        compare_sensor5(info, sensor_std[keep], sensor_mean[keep])
        print(f"{len(info)} subjects remain ({y.sum()} PD, {len(y) - y.sum()} control)\n")
        suffix = "_no_dead_sensors"

    if args.time_one_fold:
        time_one_fold(cov, corr, y, groups, speed, study)
        return

    # QR sets for every size (same basis as Job A: pooled standardized control data)
    pooled = sum(row[3] for row, g in zip(loaded, info["group"]) if g == "Control")
    modes_all = sensor_modes(pooled)
    qr_sets = {k: [SENSOR_COLUMNS[i] for i in qr_pivot_sensors(modes_all[:, :k], k)] for k in SIZES}

    # 1. Greedy forward selection, nested in the outer folds
    splits = list(outer_splits(y, groups))
    fold_results = Parallel(n_jobs=N_JOBS)(
        delayed(outer_fold)(cov, corr, y, groups, speed, train, test, SEED + repeat)
        for repeat, fold, train, test in splits)
    print(f"greedy selection done ({time.time() - start:.0f} s)")

    base = pd.DataFrame({"row": np.arange(len(y)), "subject": groups, "y": y, "study": study})
    greedy_predictions = {}
    selection_rows = []
    for (repeat, fold, _, test), (path, scores) in zip(splits, fold_results):
        for size, names in path.items():
            selection_rows.append({"repeat": repeat, "fold": fold, "n_sensors": size, "sensors": ",".join(names)})
        for (size, set_name), score in scores.items():
            part = base.iloc[test].assign(repeat=repeat, fold=fold, score=score)
            greedy_predictions.setdefault((size, set_name), []).append(part)
    greedy_predictions = {k: pd.concat(v, ignore_index=True) for k, v in greedy_predictions.items()}
    selections = pd.DataFrame(selection_rows)

    # 2. Fixed sets: all 16, QR per size, anatomical, 100 random 6-sensor sets
    fixed_sets = {"all16": SENSOR_COLUMNS, **{f"qr{k}": names for k, names in qr_sets.items()},
                  **{f"anatomical{k}": names for k, names in ANATOMICAL.items()}}
    fixed_predictions = {}
    for name, names in fixed_sets.items():
        for set_name in ["gait", "gait+speed"]:
            fixed_predictions[(name, set_name)] = evaluate_fixed(cov, corr, y, groups, speed, study, names,
                                                                 set_name, N_JOBS)
    rng = np.random.default_rng(SEED)
    random_sets = [sorted(rng.choice(SENSOR_COLUMNS, size=RANDOM_SIZE, replace=False)) for _ in range(N_RANDOM)]
    random_aucs = np.array([
        summarize(evaluate_fixed(cov, corr, y, groups, speed, study, names, "gait", N_JOBS))["auc"].mean()
        for names in random_sets])
    print(f"fixed and random sets done ({time.time() - start:.0f} s)")

    # 3. AUC table with bootstrap CIs (paired: same resampled subjects for every column)
    columns = {}
    for (size, set_name), predictions in greedy_predictions.items():
        columns[f"greedy{size}|{set_name}"] = predictions
    for (name, set_name), predictions in fixed_predictions.items():
        columns[f"{name}|{set_name}"] = predictions
    score_table = pd.DataFrame({"y": subject_scores(next(iter(columns.values())))["y"]})
    for column, predictions in columns.items():
        score_table[column] = subject_scores(predictions)["score"]
    boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)

    rows = []
    for column, predictions in columns.items():
        name, set_name = column.split("|")
        per_repeat = summarize(predictions)["auc"]
        reference = f"all16|{set_name}"
        ratio = boot[column] / boot[reference]
        above_chance = (boot[column] - 0.5) / (boot[reference] - 0.5)
        method = ("greedy (nested)" if name.startswith("greedy") else
                  "qr" if name.startswith("qr") else "anatomical" if name.startswith("anatomical") else "all16")
        rows.append({
            "method": method, "variant": name, "feature_set": set_name,
            "n_sensors": 16 if name == "all16" else int("".join(ch for ch in name if ch.isdigit())),
            "sensors": ",".join(fixed_sets[name]) if name in fixed_sets else "varies by fold",
            "auc_mean": per_repeat.mean(), "auc_sd_repeats": per_repeat.std(ddof=1),
            "auc_pooled": boot[column].mean(), "ci_low": boot[column].quantile(0.025),
            "ci_high": boot[column].quantile(0.975),
            "ratio_to_all16": ratio.mean(), "ratio_ci_low": ratio.quantile(0.025), "ratio_ci_high": ratio.quantile(0.975),
            "above_chance_kept": above_chance.mean(),
        })
    auc_table = pd.DataFrame(rows).sort_values(["feature_set", "method", "n_sensors"])
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    auc_table.to_csv(METRICS_DIR / f"supervised_selection{suffix}.csv", index=False)
    pd.DataFrame({"sensors": [",".join(s) for s in random_sets], "auc_mean": random_aucs}) \
        .to_csv(METRICS_DIR / f"random6_aucs{suffix}.csv", index=False)

    # Selection frequency and overlap with QR
    frequency = pd.DataFrame(0.0, index=SIZES, columns=SENSOR_COLUMNS)
    jaccard = {}
    for size, part in selections.groupby("n_sensors"):
        chosen = [set(s.split(",")) for s in part["sensors"]]
        for sensors in chosen:
            for sensor in sensors:
                frequency.loc[size, sensor] += 1 / len(chosen)
        qr_set = set(qr_sets[size])
        jaccard[size] = np.mean([len(s & qr_set) / len(s | qr_set) for s in chosen])
    frequency.to_csv(METRICS_DIR / f"selection_frequency{suffix}.csv")

    pd.set_option("display.width", 220)
    print("\n== AUC (mean over repeats; pooled bootstrap 95% CI; ratio to all 16 with CI) ==")
    print(auc_table.drop(columns=["sensors"]).round(3).to_string(index=False))
    print("\n== Greedy selection vs QR ==")
    for size in SIZES:
        top = frequency.loc[size].sort_values(ascending=False).head(size)
        print(f"  {size} sensors | QR: {qr_sets[size]} | greedy, most often chosen: "
              f"{[f'{s} ({v:.0%})' for s, v in top.items()]} | mean Jaccard overlap with QR: {jaccard[size]:.2f}")

    # Success criterion
    gait = auc_table[auc_table["feature_set"] == "gait"].set_index("variant")
    reference_auc = gait.loc["all16", "auc_mean"]
    random_median = float(np.median(random_aucs))
    print(f"\n== SUCCESS CRITERION ==\n  {SUCCESS_CRITERION}")
    print(f"  16-sensor AUC {reference_auc:.3f}; 90% threshold {0.9 * reference_auc:.3f}; "
          f"median of {N_RANDOM} random 6-sensor sets {random_median:.3f} "
          f"(5-95%: {np.percentile(random_aucs, 5):.3f}-{np.percentile(random_aucs, 95):.3f})")
    for variant in [f"qr{RANDOM_SIZE}", f"anatomical{RANDOM_SIZE}", f"greedy{RANDOM_SIZE}"]:
        row = gait.loc[variant]
        keeps = row["auc_mean"] >= 0.9 * reference_auc
        beats = row["auc_mean"] > random_median
        print(f"  {variant:12s} AUC {row['auc_mean']:.3f} (95% CI {row['ci_low']:.3f}-{row['ci_high']:.3f}); "
              f"ratio {row['ratio_to_all16']:.2f} (CI {row['ratio_ci_low']:.2f}-{row['ratio_ci_high']:.2f}) "
              f"-> keeps 90%: {'yes' if keeps else 'NO'}; beats random median: {'yes' if beats else 'NO'} "
              f"=> {'PASS' if keeps and beats else 'FAIL'}")
    print("  (greedy6 = the selection procedure, evaluated without leakage; its chosen sets differ by fold)")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_results(auc_table, random_aucs, frequency, qr_sets, reference_auc,
                 filename=f"14_supervised_selection{suffix}.png")
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
