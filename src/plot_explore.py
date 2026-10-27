"""Exploratory plots of the raw gait signals. Saves PNGs to results/figures/.

Run from the project root:  python -m src.plot_explore
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no display window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.gait import (
    coefficient_of_variation,
    detect_heel_strikes,
    outlier_mask,
    stride_times,
)
from src.load import DATA_DIR, build_summary, load_trial

FIGURE_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

# Example subjects: same study (Ga), same sex, similar age, normal walk (_01)
CONTROL_FILE = "GaCo01_01.txt"  # male, 66
PD_FILE = "GaPt24_01.txt"  # male, 68, Hoehn & Yahr 2.5

# 10-second window, starting after the subject is up to speed
WINDOW_START_S = 20
WINDOW_LENGTH_S = 10

# Strides needed for a trustworthy CV
MIN_STRIDES_FOR_CV = 20

# Colors (validated categorical palette); group colors are reused in every figure
GROUP_COLORS = {"Control": "#2a78d6", "PD": "#eb6834"}
FOOT_COLORS = {"L_total": "#1baf7a", "R_total": "#6250d6"}
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.size": 9,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def window(trial):
    """Rows of `trial` inside the plotting window, with time reset to start at 0."""
    in_window = trial["time"].between(WINDOW_START_S, WINDOW_START_S + WINDOW_LENGTH_S)
    out = trial[in_window].copy()
    out["time"] -= WINDOW_START_S
    return out


def save(fig, name):
    path = FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path.relative_to(FIGURE_DIR.parent.parent)}")


def plot_total_force(trials):
    """Figure 1: left and right total force, control vs PD, shared y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, (group, (filename, trial)) in zip(axes, trials.items()):
        w = window(trial)
        for column, label in [("L_total", "Left foot"), ("R_total", "Right foot")]:
            ax.plot(w["time"], w[column], color=FOOT_COLORS[column], linewidth=1.5, label=label)
        ax.set_title(f"{group} ({filename[:6]})", color=GROUP_COLORS[group], loc="left")
        ax.set_xlabel(f"Time (s, from t = {WINDOW_START_S} s)")
    axes[0].set_ylabel("Total vertical force (N)")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Total force under each foot, 10-second stretch", x=0.06, ha="left")
    save(fig, "01_total_force.png")


def plot_sensor_heatmap(trials):
    """Figure 2: left-foot sensors (heel at bottom, toe at top) over time."""
    sensors = [f"L{i}" for i in range(1, 9)]  # L1 = heel ... L8 = toe
    windows = {group: window(trial) for group, (_, trial) in trials.items()}
    vmax = max(w[sensors].to_numpy().max() for w in windows.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, (group, w) in zip(axes, windows.items()):
        image = ax.imshow(
            w[sensors].to_numpy().T,
            aspect="auto",
            origin="lower",  # row 0 (L1, heel) at the bottom
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            extent=[w["time"].iloc[0], w["time"].iloc[-1], -0.5, 7.5],
            interpolation="nearest",
        )
        ax.grid(False)
        ax.set_title(f"{group} ({trials[group][0][:6]})", color=GROUP_COLORS[group], loc="left")
        ax.set_xlabel(f"Time (s, from t = {WINDOW_START_S} s)")
    axes[0].set_yticks(range(8))
    axes[0].set_yticklabels(["L1 heel", "L2", "L3", "L4", "L5", "L6", "L7", "L8 toe"])
    fig.colorbar(image, ax=axes, label="Force (N)", pad=0.02)
    fig.suptitle("Left-foot sensor forces, heel to toe", x=0.06, ha="left")
    save(fig, "02_left_sensor_heatmap.png")


def plot_stride_times(trials):
    """Figure 3: stride time vs stride number; removed outliers shown as hollow markers."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for group, (filename, trial) in trials.items():
        strides = stride_times(detect_heel_strikes(trial["L_total"], trial["time"]))
        keep = outlier_mask(strides)
        numbers = np.arange(1, len(strides) + 1)
        color = GROUP_COLORS[group]
        cv = coefficient_of_variation(strides[keep])

        ax.plot(numbers[keep], strides[keep], color=color, linewidth=1.5, marker="o",
                markersize=3, label=f"{group} ({filename[:6]}), CV = {cv:.1f}%")
        ax.plot(numbers[~keep], strides[~keep], linestyle="none", marker="o", markersize=6,
                markerfacecolor="none", markeredgecolor=color)
    ax.plot([], [], linestyle="none", marker="o", markerfacecolor="none",
            markeredgecolor=TEXT_COLOR, label="Removed (turn / pause, >30% from median)")
    ax.set_xlabel("Stride number")
    ax.set_ylabel("Stride time (s)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Stride time over the whole trial (left heel strike to left heel strike)", loc="left")
    save(fig, "03_stride_times.png")


def stride_cv_table(summary):
    """Stride-time CV for every _01 trial."""
    rows = []
    for _, row in summary[summary["trial"] == 1].iterrows():
        trial = load_trial(DATA_DIR / row["filename"])
        strides = stride_times(detect_heel_strikes(trial["L_total"], trial["time"]))
        kept = strides[outlier_mask(strides)]
        rows.append({
            "subject_id": row["subject_id"],
            "study": row["study"],
            "group": row["group"],
            "n_strides": len(kept),
            "n_removed": len(strides) - len(kept),
            "stride_cv": coefficient_of_variation(kept),
        })
    return pd.DataFrame(rows)


def plot_cv_histogram(cv_table):
    """Figure 4: distribution of stride-time CV, control vs PD, on a shared x-axis."""
    usable = cv_table[cv_table["n_strides"] >= MIN_STRIDES_FOR_CV]
    bins = np.arange(0, np.ceil(usable["stride_cv"].max()) + 0.5, 0.5)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)
    for ax, group in zip(axes, ["Control", "PD"]):
        values = usable.loc[usable["group"] == group, "stride_cv"]
        ax.hist(values, bins=bins, color=GROUP_COLORS[group], edgecolor="#fcfcfb", linewidth=1)
        ax.axvline(values.median(), color=TEXT_COLOR, linewidth=1, linestyle="--")
        ax.text(values.median(), ax.get_ylim()[1] * 0.92, f"  median {values.median():.2f}%",
                color=TEXT_COLOR, va="top")
        ax.set_title(f"{group} (n = {len(values)})", color=GROUP_COLORS[group], loc="left")
        ax.set_ylabel("Subjects")
    axes[1].set_xlabel("Stride-time coefficient of variation (%)")
    fig.suptitle("Stride-time variability across all subjects (normal walk, _01)", x=0.06, ha="left")
    save(fig, "04_stride_cv_histogram.png")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    trials = {
        "Control": (CONTROL_FILE, load_trial(DATA_DIR / CONTROL_FILE)),
        "PD": (PD_FILE, load_trial(DATA_DIR / PD_FILE)),
    }
    plot_total_force(trials)
    plot_sensor_heatmap(trials)
    plot_stride_times(trials)

    cv_table = stride_cv_table(build_summary())
    plot_cv_histogram(cv_table)

    # Short text summary for checking
    too_few = cv_table[cv_table["n_strides"] < MIN_STRIDES_FOR_CV]
    print(f"\n_01 trials: {len(cv_table)}, excluded (< {MIN_STRIDES_FOR_CV} strides): "
          f"{list(too_few['subject_id'])}")
    print(cv_table.groupby("group")["stride_cv"].describe().round(2))
    print(cv_table.groupby(["study", "group"])["stride_cv"].median().round(2).unstack())


if __name__ == "__main__":
    main()
