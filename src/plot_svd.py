"""SVD of the 16 foot sensors: energy plots, spatial modes, per-subject features.

Everything is done twice: sensors centered only, and centered + scaled to unit
variance ("standardized").

Run from the project root:  python -m src.plot_svd
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.features_math import (
    SENSOR_COLUMNS,
    energy_fractions,
    fix_signs,
    n_components_for_energy,
    preprocess,
    sensor_matrix,
    svd,
    svd_features,
)
from src.load import DATA_DIR, build_summary, load_trial
from src.plot_explore import (
    CONTROL_FILE,
    GRID_COLOR,
    GROUP_COLORS,
    METRICS_DIR,
    PD_FILE,
    RESULTS_DIR,
    TEXT_COLOR,
    FIGURE_DIR,
    save,
)

ENERGY_TARGET = 0.9
N_MODES_TO_PLOT = 3

# name -> standardize flag
VARIANTS = {"centered": False, "standardized": True}
VARIANT_LABELS = {"centered": "centered", "standardized": "centered + unit variance"}

# Sensors with less spread than this (N) are treated as nearly dead
DEAD_SENSOR_STD_N = 5


def decompose(filename, standardize):
    """SVD of one trial's preprocessed 16 x T sensor matrix."""
    X = preprocess(sensor_matrix(load_trial(DATA_DIR / filename)), standardize)
    U, s, _ = svd(X)
    return U, s


def plot_energy(results, variant):
    """Singular values (log scale) and cumulative energy fraction."""
    fig, (ax_scree, ax_cum) = plt.subplots(1, 2, figsize=(11, 4))
    modes = np.arange(1, len(SENSOR_COLUMNS) + 1)

    for group, (filename, U, s) in results.items():
        color = GROUP_COLORS[group]
        label = f"{group} ({filename[:6]})"
        ax_scree.plot(modes, s, color=color, linewidth=2, marker="o", markersize=5, label=label)

        cumulative = np.cumsum(energy_fractions(s))
        k = n_components_for_energy(s, ENERGY_TARGET)
        ax_cum.plot(modes, cumulative, color=color, linewidth=2, marker="o", markersize=5,
                    label=f"{label}: {k} modes for {ENERGY_TARGET:.0%}")
        ax_cum.plot(k, cumulative[k - 1], marker="o", markersize=11, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.5)

    ax_scree.set_yscale("log")
    ax_scree.set_xlabel("Mode k")
    ax_scree.set_ylabel("Singular value σₖ (log scale)")
    ax_scree.set_title("Scree plot", loc="left")
    ax_scree.legend(frameon=False)

    ax_cum.axhline(ENERGY_TARGET, color=TEXT_COLOR, linewidth=1, linestyle="--")
    ax_cum.set_ylim(0.2, 1.02)
    ax_cum.set_xlabel("Number of modes k")
    ax_cum.set_ylabel("Cumulative energy fraction")
    ax_cum.set_title("Energy captured by the first k modes", loc="left")
    ax_cum.legend(frameon=False, loc="lower right")

    for ax in (ax_scree, ax_cum):
        ax.set_xticks(modes)
    fig.suptitle(f"SVD energy, sensors {VARIANT_LABELS[variant]}", x=0.06, ha="left")
    fig.tight_layout()
    save(fig, f"05_svd_energy_{variant}.png")


def plot_modes(results, variant):
    """First left singular vectors as bar charts over the 16 sensors."""
    # Bars 0-7 = left foot, gap, bars 9-16 = right foot
    positions = np.r_[np.arange(8), np.arange(9, 17)]
    tick_labels = ["L1\nheel", "L2", "L3", "L4", "L5", "L6", "L7", "L8\ntoe",
                   "R1\nheel", "R2", "R3", "R4", "R5", "R6", "R7", "R8\ntoe"]

    fig, axes = plt.subplots(N_MODES_TO_PLOT, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for col, (group, (filename, U, s)) in enumerate(results.items()):
        fractions = energy_fractions(s)
        for row in range(N_MODES_TO_PLOT):
            ax = axes[row, col]
            ax.bar(positions, U[:, row], width=0.8, color=GROUP_COLORS[group])
            ax.axhline(0, color=TEXT_COLOR, linewidth=0.8)
            ax.axvline(8, color=GRID_COLOR, linewidth=1)
            ax.set_title(f"{group} ({filename[:6]}), mode {row + 1}: {fractions[row]:.0%} of energy",
                         color=GROUP_COLORS[group], loc="left")
            ax.grid(axis="x", visible=False)
        axes[0, col].text(3.5, 0.72, "Left foot", ha="center", color=TEXT_COLOR)
        axes[0, col].text(12.5, 0.72, "Right foot", ha="center", color=TEXT_COLOR)

    for ax in axes[:, 0]:
        ax.set_ylabel("Weight in uₖ")
    for ax in axes[-1, :]:
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
    axes[0, 0].set_ylim(-0.7, 0.8)
    fig.suptitle(f"First three spatial modes, sensors {VARIANT_LABELS[variant]}", x=0.06, ha="left")
    save(fig, f"06_svd_modes_{variant}.png")


def svd_feature_table(summary):
    """SVD features (both variants) for every _01 trial, plus a dead-sensor flag."""
    rows = []
    for _, row in summary[summary["trial"] == 1].iterrows():
        trial = load_trial(DATA_DIR / row["filename"])
        record = {"subject_id": row["subject_id"], "study": row["study"], "group": row["group"],
                  "min_sensor_std": sensor_matrix(trial).std(axis=1).min()}
        for variant, standardize in VARIANTS.items():
            features = svd_features(trial, ENERGY_TARGET, standardize)
            record[f"n_components_90_{variant}"] = features["n_components_90"]
            record[f"energy_mode1_{variant}"] = features["energy_mode1"]
        rows.append(record)
    table = pd.DataFrame(rows)
    table["dead_sensor"] = table["min_sensor_std"] < DEAD_SENSOR_STD_N
    return table


def compare_groups(table, column):
    """Per-study control vs PD medians, rank-based effect size and Mann-Whitney p-value.

    p_pd_higher = probability that a random PD subject has a higher value than a
    random control (0.5 = no difference; ties count half).
    """
    rows = []
    for study, subset in [*table.groupby("study"), ("All", table)]:
        control = subset.loc[subset["group"] == "Control", column]
        pd_values = subset.loc[subset["group"] == "PD", column]
        u, p = mannwhitneyu(pd_values, control, alternative="two-sided")
        rows.append({
            "study": study,
            "control_median": control.median(),
            "pd_median": pd_values.median(),
            "p_pd_higher": u / (len(pd_values) * len(control)),
            "p_value": p,
        })
    return pd.DataFrame(rows).set_index("study")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for variant, standardize in VARIANTS.items():
        U_control, s_control = decompose(CONTROL_FILE, standardize)
        U_pd, s_pd = decompose(PD_FILE, standardize)

        # SVD signs are arbitrary: fix the control's, then point the PD modes the same way
        U_control = fix_signs(U_control)
        U_pd = fix_signs(U_pd, reference=U_control)
        results = {"Control": (CONTROL_FILE, U_control, s_control), "PD": (PD_FILE, U_pd, s_pd)}

        plot_energy(results, variant)
        plot_modes(results, variant)

    table = svd_feature_table(build_summary())
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "svd_features.csv", index=False)
    print(f"saved {(METRICS_DIR / 'svd_features.csv').relative_to(RESULTS_DIR.parent)}")

    dead = table[table["dead_sensor"]]
    print(f"\nTrials with a nearly dead sensor (std < {DEAD_SENSOR_STD_N} N): "
          f"{len(dead)} ({dead['group'].value_counts().to_dict()})")

    pd.set_option("display.width", 120)
    for subset_name, subset in [("all trials", table), ("without dead-sensor trials", table[~table["dead_sensor"]])]:
        for feature in ["energy_mode1", "n_components_90"]:
            for variant in VARIANTS:
                column = f"{feature}_{variant}"
                print(f"\n== {column} ({subset_name}) ==")
                print(compare_groups(subset, column).round(3).to_string())


if __name__ == "__main__":
    main()
