"""EXPLORATORY (post hoc): heel + arch + medial forefoot layout (sensors 1, 5, 7 per foot)
vs the anatomical set (1, 4, 7) and all 16 sensors.

This layout was chosen after seeing the sensor-5 results, so it is NOT part of the
pre-stated success-criterion table. Same procedure as that table: gait features only,
logistic regression, subject-wise CV (5 folds x 10 repeats), paired bootstrap CIs,
comparison with the same 100 random 6-sensor sets (their AUCs are read from Job B's
outputs, which used this exact pipeline and these exact splits).

Run as a SLURM job (not on the login node):  sbatch slurm/exploratory_layout.sbatch
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluate import bootstrap_auc, subject_scores, summarize
from src.plot_explore import FIGURE_DIR, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save
from src.plot_svd import DEAD_SENSOR_STD_N
from src.run_sensor_selection import ANATOMICAL
from src.run_supervised_selection import (
    N_BOOT,
    N_JOBS,
    N_RANDOM,
    SEED,
    check_inputs,
    evaluate_fixed,
    fail,
    subject_matrices,
)
from src.features_math import SENSOR_COLUMNS

LAYOUTS = {
    "all16": SENSOR_COLUMNS,
    "anatomical6 (1,4,7)": ANATOMICAL[6],
    "arch6 (1,5,7)": ["L1", "L5", "L7", "R1", "R5", "R7"],
}
LAYOUT_COLORS = {"all16": TEXT_COLOR, "anatomical6 (1,4,7)": "#eda100", "arch6 (1,5,7)": "#6250d6"}
DATASETS = {
    "all trials": "random6_aucs.csv",
    "without near-dead-sensor trials": "random6_aucs_no_dead_sensors.csv",
}


def load_random_aucs():
    """AUCs of the 100 random 6-sensor sets from Job B (stop if missing)."""
    random_aucs = {}
    for dataset, filename in DATASETS.items():
        path = METRICS_DIR / filename
        if not path.exists():
            fail(f"missing {path} (run Job B and the sensor-5 check first)")
        table = pd.read_csv(path)
        if len(table) != N_RANDOM:
            fail(f"{path} has {len(table)} rows, expected {N_RANDOM}")
        random_aucs[dataset] = table["auc_mean"].to_numpy()
    return random_aucs


def evaluate_dataset(cov, corr, y, groups, speed, study, dataset, random_aucs):
    """AUC, paired bootstrap CIs, ratio to all 16 and rank among random sets, for each layout."""
    predictions = {name: evaluate_fixed(cov, corr, y, groups, speed, study, names, "gait", N_JOBS)
                   for name, names in LAYOUTS.items()}
    score_table = pd.DataFrame({"y": subject_scores(predictions["all16"])["y"]})
    for name, p in predictions.items():
        score_table[name] = subject_scores(p)["score"]
    boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)

    rows = []
    for name in LAYOUTS:
        per_repeat = summarize(predictions[name])["auc"]
        ratio = boot[name] / boot["all16"]
        rows.append({
            "dataset": dataset, "layout": name, "sensors": ",".join(LAYOUTS[name]), "n_subjects": len(y),
            "auc_mean": per_repeat.mean(), "auc_sd_repeats": per_repeat.std(ddof=1),
            "auc_pooled": boot[name].mean(), "ci_low": boot[name].quantile(0.025),
            "ci_high": boot[name].quantile(0.975),
            "ratio_to_all16": ratio.mean(), "ratio_ci_low": ratio.quantile(0.025),
            "ratio_ci_high": ratio.quantile(0.975),
            "fraction_of_random_sets_beaten": float((random_aucs < per_repeat.mean()).mean()),
        })
    difference = boot["arch6 (1,5,7)"] - boot["anatomical6 (1,4,7)"]
    contrast = {"dataset": dataset, "difference_arch6_minus_anatomical6": difference.mean(),
                "ci_low": difference.quantile(0.025), "ci_high": difference.quantile(0.975),
                "fraction_of_bootstraps_arch6_better": float((difference > 0).mean())}
    return rows, contrast


def plot_layouts(table, random_aucs):
    """Figure 15: AUC (mean ± SD over repeats) of each layout vs the random 6-sensor sets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        ax.boxplot(random_aucs[dataset], positions=[0], widths=0.5,
                   boxprops={"color": TEXT_COLOR}, medianprops={"color": TEXT_COLOR})
        part = table[table["dataset"] == dataset].reset_index(drop=True)
        for i, row in part.iterrows():
            ax.errorbar(i + 1, row["auc_mean"], yerr=row["auc_sd_repeats"], fmt="o", markersize=9,
                        capsize=4, color=LAYOUT_COLORS[row["layout"]], linewidth=2)
            ax.annotate(f"{row['auc_mean']:.3f}\nratio {row['ratio_to_all16']:.2f}\n"
                        f"beats {row['fraction_of_random_sets_beaten']:.0%}",
                        (i + 1, row["auc_mean"]), xytext=(12, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT_COLOR)
        ax.axhline(0.5, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(part) + 1))
        ax.set_xticklabels([f"{N_RANDOM} random\n6-sensor sets", *[n.replace(" ", "\n", 1) for n in part["layout"]]])
        ax.set_xlim(-0.6, len(part) + 0.9)
        ax.set_title(f"{dataset} (n = {part['n_subjects'].iloc[0]})", loc="left")
    axes[0].set_ylabel("CV AUC, gait features only (logistic): mean ± SD over repeats")
    fig.suptitle("Exploratory (post hoc): heel + arch + medial forefoot (1, 5, 7) vs anatomical (1, 4, 7) vs all 16",
                 x=0.06, ha="left")
    fig.tight_layout()
    save(fig, "15_layout_exploratory.png")


def main():
    start = time.time()
    print("EXPLORATORY (post hoc) analysis: the arch6 layout was chosen after seeing the sensor-5 results.\n"
          "It is not part of the pre-stated success-criterion table.\n")
    files, demographics = check_inputs()
    random_aucs = load_random_aucs()

    loaded = Parallel(n_jobs=N_JOBS)(delayed(subject_matrices)(p) for p in files)
    info = pd.DataFrame([row[0] for row in loaded]).merge(demographics[["subject_id", "speed_01"]],
                                                          on="subject_id", how="left")
    cov = np.stack([row[1] for row in loaded])
    corr = np.stack([row[2] for row in loaded])
    sensor_std = np.stack([row[4] for row in loaded])
    y = (info["group"] == "PD").astype(int).to_numpy()
    groups = info["subject_id"].to_numpy()
    speed = info["speed_01"].to_numpy()
    study = info["study"].to_numpy()
    working = sensor_std.min(axis=1) >= DEAD_SENSOR_STD_N
    print(f"{len(info)} subjects; {(~working).sum()} trials with a near-dead sensor "
          f"(std < {DEAD_SENSOR_STD_N} N): {list(info.loc[~working, 'subject_id'])}\n")

    rows, contrasts = [], []
    for dataset, keep in [("all trials", np.ones(len(y), dtype=bool)),
                          ("without near-dead-sensor trials", working)]:
        dataset_rows, contrast = evaluate_dataset(cov[keep], corr[keep], y[keep], groups[keep], speed[keep],
                                                  study[keep], dataset, random_aucs[dataset])
        rows += dataset_rows
        contrasts.append(contrast)
        print(f"{dataset} done ({time.time() - start:.0f} s)")

    table = pd.DataFrame(rows)
    contrast_table = pd.DataFrame(contrasts)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "layout_comparison_exploratory.csv", index=False)
    contrast_table.to_csv(METRICS_DIR / "layout_contrast_exploratory.csv", index=False)
    print(f"saved {(METRICS_DIR / 'layout_comparison_exploratory.csv').relative_to(RESULTS_DIR.parent)} "
          f"and layout_contrast_exploratory.csv")

    pd.set_option("display.width", 220)
    print("\n== Exploratory (post hoc) layout comparison: gait features only, logistic, 10 x 5-fold CV ==")
    print(table.drop(columns=["sensors"]).round(3).to_string(index=False))
    for dataset in DATASETS:
        r = random_aucs[dataset]
        print(f"  random 6-sensor sets ({dataset}): median {np.median(r):.3f}, 5-95% "
              f"{np.percentile(r, 5):.3f}-{np.percentile(r, 95):.3f}")
    print("\n== Paired bootstrap: arch6 (1,5,7) minus anatomical6 (1,4,7), pooled AUC ==")
    print(contrast_table.round(3).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_layouts(table, random_aucs)
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
