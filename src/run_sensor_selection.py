"""Minimum sensor set: QR-pivoted sensor selection, reconstruction error,
features and classifiers on reduced sensor sets, placement tolerance, hand-off table.

Run as a SLURM job (not on the login node):  sbatch slurm/sensor_selection.sbatch
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluate import repeated_group_cv, summarize
from src.features_math import (
    SENSOR_COLUMNS,
    SENSOR_POSITIONS,
    center_rows,
    dmd_features,
    downsample,
    effective_n_modes,
    energy_fractions,
    heel_forefoot_coupling,
    preprocess,
    qr_pivot_sensors,
    reconstruction_matrix,
    relative_reconstruction_error,
    sensor_matrix,
    sensor_modes,
    svd,
)
from src.gait import steady_window
from src.load import DATA_DIR, FILENAME_PATTERN, load_demographics, load_trial, parse_filename
from src.plot_dmd import DOWNSAMPLE_FACTOR, DT, N_DELAYS, RANK, WINDOW_SECONDS
from src.plot_explore import FIGURE_DIR, GROUP_COLORS, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save
from src.run_ml import make_models

N_JOBS = 4  # matches --cpus-per-task in slurm/sensor_selection.sbatch
N_SPLITS = 5
N_REPEATS = 10
SEED = 0
SIZES = [3, 4, 5, 6]  # number of sensors = number of modes r
ML_SIZES = [4, 6]
N_RANDOM = 100
EXPECTED_SUBJECTS = 165

# Anatomical sets: heel (1), lateral midfoot (4; sensor 5 is the one most often
# nearly dead), medial forefoot (7, under the big-toe metatarsal head)
ANATOMICAL = {4: ["L1", "L7", "R1", "R7"], 6: ["L1", "L4", "L7", "R1", "R4", "R7"]}

GAIT_FEATURES = ["energy_mode1", "effective_modes", "dmd_freq", "heel_forefoot"]
METHOD_COLORS = {"qr": "#1baf7a", "random": "#a3a29b", "anatomical": "#eda100"}


def fail(message):
    sys.exit(f"ERROR: {message}")


def check_inputs():
    """Stop before any work if raw data or demographics are missing."""
    if not DATA_DIR.is_dir():
        fail(f"data folder not found: {DATA_DIR}")
    if not (DATA_DIR / "demographics.txt").exists():
        fail(f"missing {DATA_DIR / 'demographics.txt'}")
    files = sorted(p for p in DATA_DIR.glob("*_01.txt") if FILENAME_PATTERN.match(p.name))
    if len(files) != EXPECTED_SUBJECTS:
        fail(f"expected {EXPECTED_SUBJECTS} _01 trial files in {DATA_DIR}, found {len(files)}")
    demographics = load_demographics()
    if not {"speed_01", "age", "sex"} <= set(demographics.columns):
        fail("demographics is missing speed_01 / age / sex")
    return files, demographics


# ---------- Part 1: per-subject Gram matrices and peak forces ----------


def subject_stats(path):
    """Gram matrix of the standardized 16 x T data, and each sensor's peak force (N)."""
    info = parse_filename(path)
    X = sensor_matrix(load_trial(path))
    Z = preprocess(X, standardize=True)
    return info["subject_id"], info["study"], info["group"], Z @ Z.T, X.max(axis=1)


# ---------- Part 2: reconstruction from a subset of sensors ----------


def names_to_index(names):
    return [SENSOR_COLUMNS.index(n) for n in names]


def index_to_names(index):
    return [SENSOR_COLUMNS[i] for i in index]


def errors_for_set(sensors, modes_all, grams):
    """Relative reconstruction error per subject, using r = len(sensors) modes."""
    M = reconstruction_matrix(modes_all[:, :len(sensors)], list(sensors))
    return np.array([relative_reconstruction_error(G, M) for G in grams])


def group_medians(errors, groups):
    return {g: float(np.median(errors[groups == g])) for g in ["Control", "PD"]}


def nearest_neighbor(sensor, exclude):
    """Closest sensor on the same foot (x/y from format.txt), not in `exclude`.

    Ties are broken by the lower sensor number.
    """
    x0, y0 = SENSOR_POSITIONS[sensor]
    candidates = [s for s in SENSOR_COLUMNS if s[0] == sensor[0] and s not in exclude]
    return min(candidates, key=lambda s: (np.hypot(SENSOR_POSITIONS[s][0] - x0,
                                                   SENSOR_POSITIONS[s][1] - y0), int(s[1:])))


# ---------- Part 3: gait features from a subset of sensors ----------


def proxy_foot_force(trial, names):
    """Sum of the selected sensors on the left foot (right foot if none on the left).

    A reduced insole has no 'total force' column; this stands in for it when
    finding heel strikes and steady walking.
    """
    left = [n for n in names if n.startswith("L")]
    chosen = left if left else [n for n in names if n.startswith("R")]
    return trial[chosen].sum(axis=1).to_numpy()


def subset_features(trial, names):
    """Gait features computable from the sensors `names` only."""
    X = trial[names].to_numpy().T
    _, s, _ = svd(center_rows(X))
    features = {
        "energy_mode1": energy_fractions(s)[0],
        "effective_modes": effective_n_modes(s),
        "heel_forefoot": heel_forefoot_coupling(X, names),
        "dmd_freq": np.nan,
    }
    start, end, _ = steady_window(proxy_foot_force(trial, names), trial["time"].to_numpy(), WINDOW_SECONDS)
    if start is not None:
        in_window = (trial["time"] >= start).to_numpy() & (trial["time"] < end).to_numpy()
        W = center_rows(downsample(X[:, in_window], DOWNSAMPLE_FACTOR))
        if W.shape[1] > 2 * N_DELAYS:
            features["dmd_freq"] = dmd_features(W, DT, N_DELAYS, RANK)["dmd_dominant_freq"]
    return features


def subject_variant_features(path, variants):
    """Features for every sensor variant for one subject (trial loaded once)."""
    info = parse_filename(path)
    trial = load_trial(path)
    return [{"subject_id": info["subject_id"], "variant": name, **subset_features(trial, names)}
            for name, names in variants.items()]


# ---------- Part 4: classifiers ----------


def classify(features, demographics, variants, model_names):
    """Repeated subject-wise CV AUC for each variant x feature set x model."""
    table = features.merge(demographics, on="subject_id", how="left")
    y_all = (table["group"] == "PD").astype(int)
    rows = []
    for variant in variants:
        part = table[table["variant"] == variant].reset_index(drop=True)
        gait = [c for c in GAIT_FEATURES if part[c].notna().any()]  # drop impossible features
        y = y_all[table["variant"] == variant].to_numpy()
        for set_name, columns in {"gait": gait, "gait+speed": [*gait, "speed_01"]}.items():
            for model_name, model in make_models(SEED).items():
                if model_name not in model_names:
                    continue
                predictions, _ = repeated_group_cv(model, part[columns], y, part["subject_id"].to_numpy(),
                                                   n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED,
                                                   n_jobs=N_JOBS, row_info=part[["study"]])
                auc = summarize(predictions)["auc"]
                rows.append({"variant": variant, "feature_set": set_name, "model": model_name,
                             "features": ",".join(columns), "auc_mean": auc.mean(), "auc_std": auc.std(ddof=1)})
                print(f"  {variant:28s} {set_name:10s} {model_name:14s} AUC {auc.mean():.3f} ± {auc.std(ddof=1):.3f}")
    return pd.DataFrame(rows)


# ---------- Part 5: figure ----------


def plot_selection(selection, random_summary, sets_to_show):
    """Figure 13: reconstruction error vs number of sensors; chosen sensors on the foot."""
    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(1, 3, k + 1) for k in range(3)]
    for ax, group in zip(axes[:2], ["Control", "PD"]):
        rand = random_summary[random_summary["group"] == group]
        ax.fill_between(rand["n_sensors"], rand["p05"], rand["p95"], color=METHOD_COLORS["random"],
                        alpha=0.35, linewidth=0, label="random sets (5–95%)")
        ax.plot(rand["n_sensors"], rand["median"], color=METHOD_COLORS["random"], linewidth=2,
                label="random sets (median)")
        qr_rows = selection[(selection["method"] == "qr") & (selection["group"] == group)]
        ax.plot(qr_rows["n_sensors"], qr_rows["error_median"], color=METHOD_COLORS["qr"], linewidth=2,
                marker="o", markersize=7, label="QR pivoting")
        anat = selection[(selection["method"] == "anatomical") & (selection["group"] == group)]
        ax.scatter(anat["n_sensors"], anat["error_median"], color=METHOD_COLORS["anatomical"], s=80,
                   marker="D", zorder=3, label="anatomical")
        oracle = selection[(selection["method"] == "all16_r_modes") & (selection["group"] == group)]
        ax.plot(oracle["n_sensors"], oracle["error_median"], color=TEXT_COLOR, linewidth=1,
                linestyle="--", label="all 16 sensors, r modes (floor)")
        ax.set_xticks(SIZES)
        ax.set_xlabel("Number of sensors (= modes r)")
        ax.set_title(f"{group}: reconstruction error", color=GROUP_COLORS[group], loc="left")
    axes[0].set_ylabel("Relative error ‖X − X̂‖ / ‖X‖ (median over subjects, log scale)")
    axes[1].sharey(axes[0])
    axes[0].set_yscale("log")  # random sets range from ~0.5 to ~50
    axes[0].legend(frameon=False, fontsize=8)

    ax = axes[2]
    xy = np.array([SENSOR_POSITIONS[n] for n in SENSOR_COLUMNS])
    ax.scatter(xy[:, 0], xy[:, 1], s=60, facecolors="none", edgecolors=TEXT_COLOR, linewidths=1)
    # Larger set drawn as big faint discs, smaller set as solid dots on top
    for (label, names), size, alpha in zip(sorted(sets_to_show.items(), key=lambda kv: -len(kv[1])),
                                           [420, 140], [0.3, 1.0]):
        pts = np.array([SENSOR_POSITIONS[n] for n in names])
        ax.scatter(pts[:, 0], pts[:, 1], s=size, color=METHOD_COLORS["qr"], alpha=alpha, label=label,
                   linewidths=0, zorder=2)
    for name, (x, y) in zip(SENSOR_COLUMNS, xy):
        ax.annotate(name, (x, y), xytext=(0, -20), textcoords="offset points", ha="center",
                    fontsize=8, color=TEXT_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1100, 1100)
    ax.axis("off")
    ax.legend(frameon=False, loc="lower center", fontsize=8, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("QR-selected sensors (toe at top)", loc="left")
    fig.suptitle("Minimum sensor set: reconstructing all 16 sensors from a few", x=0.06, ha="left")
    fig.tight_layout()
    save(fig, "13_sensor_selection.png")


# ---------- main ----------


def main():
    start = time.time()
    files, demographics = check_inputs()
    demographics = demographics[["subject_id", "speed_01", "age", "sex"]]

    # Part 1: Gram matrices -> pooled control modes
    stats = Parallel(n_jobs=N_JOBS)(delayed(subject_stats)(p) for p in files)
    subjects = np.array([s[0] for s in stats])
    groups = np.array([s[2] for s in stats])
    grams = [s[3] for s in stats]
    peaks = pd.DataFrame([s[4] for s in stats], columns=SENSOR_COLUMNS)
    pooled = sum(G for G, g in zip(grams, groups) if g == "Control")
    modes_all = sensor_modes(pooled)
    energy = np.linalg.eigvalsh(pooled)[::-1]
    print(f"Part 1 done ({time.time() - start:.0f} s). Pooled control energy in modes 1-6: "
          f"{np.round(np.cumsum(energy / energy.sum())[:6], 3)}")

    # Part 2: QR selection, reconstruction, random and anatomical baselines
    selection_rows, qr_sets = [], {}
    for r in SIZES:
        modes = modes_all[:, :r]
        chosen = qr_pivot_sensors(modes, r)
        qr_sets[r] = index_to_names(chosen)
        condition = np.linalg.cond(modes[chosen])
        oracle = np.array([relative_reconstruction_error(G, modes @ modes.T) for G in grams])
        for method, sensors, cond in [("qr", chosen, condition),
                                      *([("anatomical", names_to_index(ANATOMICAL[r]),
                                          np.linalg.cond(modes[names_to_index(ANATOMICAL[r])]))]
                                        if r in ANATOMICAL else [])]:
            errors = errors_for_set(sensors, modes_all, grams)
            for group, median in group_medians(errors, groups).items():
                selection_rows.append({"method": method, "n_sensors": r, "group": group,
                                       "sensors": ",".join(index_to_names(sensors)),
                                       "condition_number": cond, "error_median": median,
                                       "error_p90": float(np.percentile(errors[groups == group], 90))})
        for group, median in group_medians(oracle, groups).items():
            selection_rows.append({"method": "all16_r_modes", "n_sensors": r, "group": group,
                                   "sensors": "all", "condition_number": 1.0, "error_median": median,
                                   "error_p90": float(np.percentile(oracle[groups == group], 90))})

    rng = np.random.default_rng(SEED)
    random_rows = []
    for r in SIZES:
        for draw in range(N_RANDOM):
            sensors = np.sort(rng.choice(16, size=r, replace=False))
            errors = errors_for_set(sensors, modes_all, grams)
            random_rows += [{"n_sensors": r, "draw": draw, "group": g, "error_median": m}
                            for g, m in group_medians(errors, groups).items()]
    random_draws = pd.DataFrame(random_rows)
    random_summary = random_draws.groupby(["n_sensors", "group"])["error_median"].agg(
        median="median", p05=lambda v: v.quantile(0.05), p95=lambda v: v.quantile(0.95)).reset_index()
    selection = pd.DataFrame(selection_rows)
    qr_rank = []
    for _, row in selection[selection["method"].isin(["qr", "anatomical"])].iterrows():
        draws = random_draws[(random_draws["n_sensors"] == row["n_sensors"]) & (random_draws["group"] == row["group"])]
        qr_rank.append((draws["error_median"] < row["error_median"]).mean())
    selection.loc[selection["method"].isin(["qr", "anatomical"]), "fraction_of_random_sets_better"] = qr_rank

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    selection.to_csv(METRICS_DIR / "sensor_selection.csv", index=False)
    random_summary.to_csv(METRICS_DIR / "sensor_selection_random.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n== QR-pivoted sensor sets (selection order) ==")
    for r, names in qr_sets.items():
        print(f"  r = {r}: {names}")
    print("\n== Reconstruction error (median over subjects) ==")
    print(selection.pivot_table(index=["method", "n_sensors"], columns="group",
                                values="error_median").round(3).to_string())
    print("\n== Random sets (median and 5-95% over draws of the group-median error) ==")
    print(random_summary.round(3).to_string(index=False))
    print("\n== Fraction of random sets that beat QR / anatomical, and condition numbers ==")
    print(selection[selection["method"].isin(["qr", "anatomical"])]
          [["method", "n_sensors", "group", "sensors", "condition_number", "fraction_of_random_sets_better"]]
          .round(3).to_string(index=False))

    # Placement tolerance: swap each chosen sensor for its nearest neighbor
    tolerance_rows, variants = [], {"all16": SENSOR_COLUMNS}
    for r in ML_SIZES:
        variants[f"qr{r}"] = qr_sets[r]
        variants[f"anatomical{r}"] = ANATOMICAL[r]
        base_errors = group_medians(errors_for_set(names_to_index(qr_sets[r]), modes_all, grams), groups)
        for sensor in qr_sets[r]:
            neighbor = nearest_neighbor(sensor, exclude=qr_sets[r])
            swapped = [neighbor if s == sensor else s for s in qr_sets[r]]
            name = f"qr{r}_{sensor}->{neighbor}"
            variants[name] = swapped
            errors = group_medians(errors_for_set(names_to_index(swapped), modes_all, grams), groups)
            tolerance_rows.append({"base": f"qr{r}", "variant": name, "moved": sensor, "to": neighbor,
                                   "sensors": ",".join(swapped),
                                   **{f"error_{g}": errors[g] for g in errors},
                                   **{f"error_change_{g}": errors[g] - base_errors[g] for g in errors}})
    tolerance = pd.DataFrame(tolerance_rows)
    print(f"\nPart 2 done ({time.time() - start:.0f} s)")

    # Part 3: features for every variant
    results = Parallel(n_jobs=N_JOBS)(delayed(subject_variant_features)(p, variants) for p in files)
    features = pd.DataFrame([row for rows in results for row in rows])
    meta = pd.DataFrame({"subject_id": subjects, "study": [s[1] for s in stats], "group": groups})
    features = features.merge(meta, on="subject_id")
    features.to_csv(METRICS_DIR / "sensor_subset_features.csv", index=False)
    print(f"Part 3 done ({time.time() - start:.0f} s); missing values per variant:")
    print(features.groupby("variant")[GAIT_FEATURES].apply(lambda d: d.isna().sum()).to_string())

    # Part 4: classifiers. Main variants: both models; placement swaps: logistic only
    main_variants = [v for v in variants if "->" not in v]
    swap_variants = [v for v in variants if "->" in v]
    print("\nClassifiers on main variants:")
    ml = classify(features, demographics, main_variants, ["logistic", "random_forest"])
    print("Classifiers on placement swaps (logistic only):")
    ml = pd.concat([ml, classify(features, demographics, swap_variants, ["logistic"])], ignore_index=True)
    reference = ml[ml["variant"] == "all16"].set_index(["feature_set", "model"])["auc_mean"]
    ml["auc_all16"] = [reference.loc[(f, m)] for f, m in zip(ml["feature_set"], ml["model"])]
    ml["auc_lost"] = ml["auc_all16"] - ml["auc_mean"]
    ml.to_csv(METRICS_DIR / "sensor_subset_ml.csv", index=False)

    print("\n== AUC by sensor variant (mean over repeats) and AUC lost vs all 16 ==")
    print(ml[ml["variant"].isin(main_variants)]
          .pivot_table(index="variant", columns=["feature_set", "model"], values=["auc_mean", "auc_lost"])
          .round(3).to_string())

    swap_auc = ml[(ml["variant"].isin(swap_variants)) & (ml["model"] == "logistic")]
    for feature_set in ["gait", "gait+speed"]:
        base = ml[(ml["model"] == "logistic") & (ml["feature_set"] == feature_set)].set_index("variant")["auc_mean"]
        column = f"auc_change_{feature_set}"
        tolerance[column] = [swap_auc[(swap_auc["variant"] == v) & (swap_auc["feature_set"] == feature_set)]
                             ["auc_mean"].iloc[0] - base.loc[b] for v, b in zip(tolerance["variant"], tolerance["base"])]
    tolerance.to_csv(METRICS_DIR / "sensor_tolerance.csv", index=False)
    print("\n== Placement tolerance (nearest-neighbor swap): change in error and logistic AUC ==")
    print(tolerance.drop(columns=["sensors"]).round(3).to_string(index=False))

    # Part 5: hand-off table
    best_model = ml[(ml["variant"] == "all16") & (ml["feature_set"] == "gait+speed")] \
        .sort_values("auc_mean").iloc[-1]["model"]
    handoff_rows = []
    for variant in main_variants:
        names = variants[variant]
        size = len(names)
        errors = (group_medians(errors_for_set(names_to_index(names), modes_all, grams), groups)
                  if size < 16 else {"Control": 0.0, "PD": 0.0})
        aucs = ml[(ml["variant"] == variant) & (ml["model"] == best_model)].set_index("feature_set")
        for name in names:
            x, y = SENSOR_POSITIONS[name]
            handoff_rows.append({
                "variant": variant, "n_sensors": size, "sensor": name, "foot": "left" if name[0] == "L" else "right",
                "x": x, "y": y,
                "peak_force_median_N": peaks[name].median(), "peak_force_p95_N": peaks[name].quantile(0.95),
                "recon_error_control": errors["Control"], "recon_error_pd": errors["PD"],
                "model": best_model,
                "auc_gait": aucs.loc["gait", "auc_mean"], "auc_gait_speed": aucs.loc["gait+speed", "auc_mean"],
                "auc_lost_gait": aucs.loc["gait", "auc_lost"], "auc_lost_gait_speed": aucs.loc["gait+speed", "auc_lost"],
            })
    handoff = pd.DataFrame(handoff_rows)
    handoff.to_csv(METRICS_DIR / "sensor_handoff.csv", index=False)
    print(f"\nsaved {(METRICS_DIR / 'sensor_handoff.csv').relative_to(RESULTS_DIR.parent)} "
          f"(AUC columns use the {best_model} model)")
    print(handoff[handoff["variant"] != "all16"].round(3).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_selection(selection, random_summary, {"QR 4": qr_sets[4], "QR 6": qr_sets[6]})
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
