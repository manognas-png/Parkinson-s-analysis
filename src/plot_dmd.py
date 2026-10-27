"""Dynamic Mode Decomposition of steady walking: eigenvalue / spectrum plots and
per-subject DMD features.

Run from the project root:  python -m src.plot_dmd
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features_math import (
    center_rows,
    delay_embed,
    dmd,
    dmd_features,
    downsample,
    sensor_matrix,
)
from src.gait import steady_window
from src.load import DATA_DIR, SAMPLING_RATE_HZ, build_summary, load_trial
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

WINDOW_SECONDS = 10
DOWNSAMPLE_FACTOR = 2  # 100 Hz -> 50 Hz
DT = DOWNSAMPLE_FACTOR / SAMPLING_RATE_HZ
N_DELAYS = 50  # 1 s of history, about one stride
RANK = 20


def steady_matrix(trial):
    """Centered 16 x T sensor matrix of the steady window at 50 Hz, plus window info."""
    start, end, stride_freq = steady_window(trial["L_total"], trial["time"], WINDOW_SECONDS)
    if start is None:
        return None, np.nan, np.nan
    window = trial[(trial["time"] >= start) & (trial["time"] < end)]
    X = center_rows(downsample(sensor_matrix(window), DOWNSAMPLE_FACTOR))
    return X, end - start, stride_freq


def plot_eigenvalues(results):
    """Figure 8: DMD eigenvalues on the unit circle (full view and zoom near z = 1)."""
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 5))
    circle = np.exp(1j * np.linspace(0, 2 * np.pi, 400))

    for ax in (ax_full, ax_zoom):
        ax.plot(circle.real, circle.imag, color=TEXT_COLOR, linewidth=1)
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.5)
        ax.axvline(0, color=TEXT_COLOR, linewidth=0.5)
        ax.set_aspect("equal")

    for group, (filename, result, _) in results.items():
        lam = result["eigenvalues"]
        size = 20 + 300 * np.abs(result["amplitudes"]) / np.abs(result["amplitudes"]).max()
        for ax in (ax_full, ax_zoom):
            ax.scatter(lam.real, lam.imag, s=size, facecolors="none", edgecolors=GROUP_COLORS[group],
                       linewidths=1.5, label=f"{group} ({filename[:6]})")

    ax_full.set_xlim(-1.15, 1.15)
    ax_full.set_ylim(-1.15, 1.15)
    ax_full.set_title("All eigenvalues (marker size = mode amplitude)", loc="left")
    ax_full.legend(frameon=False, loc="lower left")
    ax_zoom.set_xlim(0.5, 1.05)
    ax_zoom.set_ylim(-0.02, 0.8)
    ax_zoom.set_title("Zoom near z = 1 (angle = 2π · f · Δt)", loc="left")
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("Re(λ)")
    ax_full.set_ylabel("Im(λ)")
    fig.suptitle("DMD eigenvalues: on the circle = sustained, inside = damped", x=0.06, ha="left")
    save(fig, "08_dmd_eigenvalues.png")


def plot_spectrum(results):
    """Figure 9: mode frequency vs amplitude, with multiples of the stride frequency marked."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, (group, (filename, result, stride_freq)) in zip(axes, results.items()):
        positive = result["frequencies"] > 0
        f = result["frequencies"][positive]
        a = np.abs(result["amplitudes"][positive])
        for k in range(1, 8):
            ax.axvline(k * stride_freq, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        ax.vlines(f, 0, a, color=GROUP_COLORS[group], linewidth=2)
        ax.scatter(f, a, color=GROUP_COLORS[group], s=30, zorder=3)
        ax.set_title(f"{group} ({filename[:6]}): stride frequency {stride_freq:.2f} Hz "
                     f"(dotted lines = 1×, 2×, 3× …)", color=GROUP_COLORS[group], loc="left")
        ax.set_ylabel("Mode amplitude |b|")
        ax.set_ylim(bottom=0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim(0, 8)
    fig.suptitle("DMD spectrum of steady walking (positive-frequency modes)", x=0.06, ha="left")
    save(fig, "09_dmd_spectrum.png")


def dmd_feature_table(summary):
    """DMD features for every _01 trial (NaN when there is no steady window)."""
    rows = []
    for _, row in summary[summary["trial"] == 1].iterrows():
        X, window_s, stride_freq = steady_matrix(load_trial(DATA_DIR / row["filename"]))
        record = {"subject_id": row["subject_id"], "study": row["study"], "group": row["group"],
                  "dmd_window_s": window_s, "steady_stride_freq": stride_freq}
        if X is not None and X.shape[1] > 2 * N_DELAYS:
            record.update(dmd_features(X, DT, N_DELAYS, RANK))
        rows.append(record)
    return pd.DataFrame(rows)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for group, filename in {"Control": CONTROL_FILE, "PD": PD_FILE}.items():
        X, _, stride_freq = steady_matrix(load_trial(DATA_DIR / filename))
        results[group] = (filename, dmd(delay_embed(X, N_DELAYS), RANK, DT), stride_freq)
    plot_eigenvalues(results)
    plot_spectrum(results)

    table = dmd_feature_table(build_summary())
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "dmd_features.csv", index=False)
    print(f"saved {(METRICS_DIR / 'dmd_features.csv').relative_to(RESULTS_DIR.parent)}")

    usable = table.dropna(subset=["dmd_dominant_freq"])
    print(f"\nTrials with DMD features: {len(usable)} of {len(table)}; "
          f"window length median {usable['dmd_window_s'].median():.1f} s, "
          f"min {usable['dmd_window_s'].min():.1f} s")
    ratio = usable["dmd_dominant_freq"] / usable["steady_stride_freq"]
    print("Dominant frequency / stride frequency:")
    print(ratio.round(1).value_counts().sort_index().to_string())
    rho, p = spearmanr(usable["dmd_dominant_freq"], usable["steady_stride_freq"])
    print(f"Spearman(dominant freq, stride freq) = {rho:.3f} (p = {p:.1e})")
    print(usable.groupby("group")[[c for c in usable.columns if c.startswith("dmd_")]].median().round(4).T)


if __name__ == "__main__":
    main()
