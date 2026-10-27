"""Figure 10: speed-adjusted residuals of the key features, control vs PD, per study.

Run from the project root (after speed_adjust's inputs exist):
    python -m src.plot_key_features
"""

import matplotlib.pyplot as plt
import numpy as np

from src.plot_explore import GROUP_COLORS, TEXT_COLOR, save
from src.plot_svd import compare_groups
from src.speed_adjust import load_feature_table, speed_adjusted_residuals

KEY_FEATURES = {
    "heel_forefoot_corr": "Heel–forefoot coupling (|r|)",
    "swing_cv": "Swing-time CV (%)",
    "energy_mode1_centered": "SVD mode-1 energy (centered)",
    "dmd_reconstruction_error": "DMD reconstruction error",
    "dmd_dominant_freq": "DMD dominant frequency (Hz)",
}
STUDIES = ["Ga", "Ju", "Si"]
GROUPS = ["Control", "PD"]
BOX_OFFSET = 0.2  # Control left of the study position, PD right


def main():
    table = load_feature_table()
    rng = np.random.default_rng(0)  # fixed jitter so the figure is reproducible

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    legend_ax = axes.flat[len(KEY_FEATURES)]  # spare panel holds the legend
    for ax, (feature, label) in zip(axes.flat, KEY_FEATURES.items()):
        table["residual"], _ = speed_adjusted_residuals(table, feature)
        data = table.dropna(subset=["residual"])
        stats = compare_groups(data, "residual")

        for s, study in enumerate(STUDIES):
            for g, group in enumerate(GROUPS):
                values = data.loc[(data["study"] == study) & (data["group"] == group), "residual"]
                x = s + (g - 0.5) * 2 * BOX_OFFSET
                color = GROUP_COLORS[group]
                ax.boxplot(values, positions=[x], widths=0.3, patch_artist=True, showfliers=False,
                           boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
                           whiskerprops={"color": color}, capprops={"color": color},
                           medianprops={"color": TEXT_COLOR, "linewidth": 1.5})
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(values)), values, s=10,
                           color=color, alpha=0.7, linewidths=0, zorder=3)

            p_higher, p_value = stats.loc[study, ["p_pd_higher", "p_value"]]
            ax.text(s, 1.0, f"P = {p_higher:.2f}\np = {p_value:.3f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=TEXT_COLOR)

        pooled = stats.loc["All"]
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(STUDIES)))
        ax.set_xticklabels(STUDIES)
        ax.set_xlim(-0.6, len(STUDIES) - 0.4)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.25 * (ymax - ymin))  # room for the P / p labels
        ax.set_title(f"{label}\npooled P(PD > Ctrl) = {pooled['p_pd_higher']:.2f}, "
                     f"p = {pooled['p_value']:.3f}", loc="left")
        ax.set_ylabel("Residual after speed + study")
        ax.grid(axis="x", visible=False)

    for group in GROUPS:
        legend_ax.scatter([], [], s=60, color=GROUP_COLORS[group], label=group)
    legend_ax.legend(loc="center", frameon=False, fontsize=11,
                     title="P = P(PD > Ctrl) of residuals\np = Mann–Whitney", title_fontsize=9)
    legend_ax.axis("off")
    fig.suptitle("Key features after adjusting for walking speed and study "
                 "(0 = what a control of that speed and study would show)", x=0.06, ha="left")
    fig.tight_layout()
    save(fig, "10_key_features.png")


if __name__ == "__main__":
    main()
