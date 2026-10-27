"""Sensor correlation networks: Fiedler split on a foot layout, per-subject features.

Run from the project root:  python -m src.plot_network
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features_math import (
    IS_HEEL,
    IS_LEFT,
    SENSOR_COLUMNS,
    SENSOR_POSITIONS,
    correlation_network,
    fiedler,
    network_features,
    sensor_matrix,
    von_neumann_entropy,
)
from src.load import DATA_DIR, build_summary, load_trial
from src.plot_explore import (
    CONTROL_FILE,
    GROUP_COLORS,
    METRICS_DIR,
    PD_FILE,
    RESULTS_DIR,
    TEXT_COLOR,
    FIGURE_DIR,
    save,
)
from src.plot_svd import compare_groups

FEATURES = ["fiedler_value", "von_neumann_entropy", "mean_edge_weight",
            "within_foot_corr", "left_right_corr", "heel_forefoot_corr",
            "fiedler_value_thresholded", "entropy_thresholded"]

# Node colors for the two sides of the Fiedler split (not group colors)
FIEDLER_COLORS = {"+": "#eda100", "-": "#6250d6"}


def plot_fiedler_networks(examples):
    """Figure 7: correlation network on the insole layout, nodes colored by Fiedler sign."""
    xy = np.array([SENSOR_POSITIONS[name] for name in SENSOR_COLUMNS])
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    for ax, (group, filename) in zip(axes, examples.items()):
        A = correlation_network(sensor_matrix(load_trial(DATA_DIR / filename)))
        value, vector = fiedler(A)

        # Edges, weakest first so strong ones are drawn on top
        i_upper, j_upper = np.triu_indices(len(A), k=1)
        for k in np.argsort(A[i_upper, j_upper]):
            i, j = i_upper[k], j_upper[k]
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=TEXT_COLOR,
                    linewidth=3 * A[i, j] ** 2, alpha=0.15 + 0.6 * A[i, j] ** 2, zorder=1)

        colors = [FIEDLER_COLORS["+"] if v >= 0 else FIEDLER_COLORS["-"] for v in vector]
        ax.scatter(xy[:, 0], xy[:, 1], s=80 + 1500 * np.abs(vector) ** 2, c=colors,
                   edgecolors="#fcfcfb", linewidths=2, zorder=2)
        for name, (x, y) in zip(SENSOR_COLUMNS, xy):
            ax.annotate(name, (x, y), xytext=(0, -22), textcoords="offset points",
                        ha="center", color=TEXT_COLOR, fontsize=8)

        ax.set_title(f"{group} ({filename[:6]})\nFiedler value {value:.2f}, "
                     f"entropy {von_neumann_entropy(A):.3f}",
                     color=GROUP_COLORS[group], loc="left")
        ax.text(-500, 1050, "Left foot", ha="center", color=TEXT_COLOR)
        ax.text(500, 1050, "Right foot", ha="center", color=TEXT_COLOR)
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1100, 1150)
        ax.set_aspect("equal")
        ax.axis("off")

    for sign, color in FIEDLER_COLORS.items():
        axes[1].scatter([], [], s=80, c=color, label=f"Fiedler vector {sign}")
    axes[1].plot([], [], color=TEXT_COLOR, linewidth=2.5, label="|correlation| (thicker = stronger)")
    fig.legend(loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Sensor correlation networks: toe at top, heel at bottom", x=0.06, ha="left")
    save(fig, "07_network_fiedler.png")


def split_type(vector):
    """Name the partition made by the Fiedler vector's sign (+ side = heel side)."""
    positive = vector >= 0
    if np.array_equal(positive, IS_HEEL):
        return "heel (1-3) vs forefoot (4-8)"
    if np.array_equal(positive, IS_LEFT) or np.array_equal(positive, ~IS_LEFT):
        return "left vs right"
    return "other"


def network_feature_table(summary):
    """Network features and Fiedler split for every _01 trial, with walking speed."""
    rows = []
    for _, row in summary[summary["trial"] == 1].iterrows():
        trial = load_trial(DATA_DIR / row["filename"])
        _, vector = fiedler(correlation_network(sensor_matrix(trial)))
        rows.append({
            "subject_id": row["subject_id"], "study": row["study"], "group": row["group"],
            "speed_01": row["speed_01"], **network_features(trial),
            "fiedler_split": split_type(vector),
            **{f"heel_side_{name}": v >= 0 for name, v in zip(SENSOR_COLUMNS, vector)},
        })
    return pd.DataFrame(rows)


def describe_splits(table):
    """How often each split type occurs, and how often each sensor lands on the heel side."""
    print("\n== Fiedler split type (subjects) ==")
    print(pd.crosstab([table["study"], table["group"]], table["fiedler_split"]))
    heel_side = table.groupby("group")[[f"heel_side_{name}" for name in SENSOR_COLUMNS]].mean()
    heel_side.columns = SENSOR_COLUMNS
    print("\n== Fraction of subjects with each sensor on the heel (+) side ==")
    print(heel_side.round(2).to_string())


def speed_correlations(table):
    """Spearman rho (p) of each feature with speed_01: all subjects, and per group."""
    rows = []
    for feature in FEATURES:
        row = {"feature": feature}
        for name, subset in [("All", table), *table.groupby("group")]:
            pair = subset[[feature, "speed_01"]].dropna()
            rho, p = spearmanr(pair[feature], pair["speed_01"])
            row[name] = f"{rho:+.2f} ({p:.3f})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_fiedler_networks({"Control": CONTROL_FILE, "PD": PD_FILE})

    table = network_feature_table(build_summary())
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    feature_columns = ["subject_id", "study", "group", *FEATURES, "fiedler_split"]
    table[feature_columns].to_csv(METRICS_DIR / "network_features.csv", index=False)
    print(f"saved {(METRICS_DIR / 'network_features.csv').relative_to(RESULTS_DIR.parent)}")

    describe_splits(table)

    pd.set_option("display.width", 120)
    for feature in FEATURES:
        print(f"\n== {feature} ==")
        print(compare_groups(table, feature).round(3).to_string())

    print("\n== Spearman correlation with speed_01: rho (p) ==")
    print(speed_correlations(table).to_string())


if __name__ == "__main__":
    main()
