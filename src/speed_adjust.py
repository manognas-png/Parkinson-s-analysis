"""Do the gait features separate PD from controls beyond walking speed and study?

For each feature: fit feature ~ speed_01 + study on controls only (robust Huber
regression), compute residuals for everyone, and compare PD vs control residuals.

Run from the project root (after plot_explore, plot_svd, plot_network):
    python -m src.speed_adjust
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from src.load import load_demographics
from src.plot_explore import METRICS_DIR, RESULTS_DIR
from src.plot_svd import compare_groups

FEATURES = [
    "energy_mode1_centered",
    "energy_mode1_standardized",
    "effective_n_modes_centered",
    "effective_n_modes_standardized",
    "mean_edge_weight",
    "heel_forefoot_corr",
    "fiedler_value_thresholded",
    "entropy_thresholded",
    "swing_cv",
    "dmd_dominant_freq",
    "dmd_dominant_growth_rate",
    "dmd_harmonic_amplitude_fraction",
    "dmd_reconstruction_error",
]


def load_feature_table():
    """All per-subject features in one table, with walking speed."""
    svd_table = pd.read_csv(METRICS_DIR / "svd_features.csv")
    network = pd.read_csv(METRICS_DIR / "network_features.csv")
    gait = pd.read_csv(METRICS_DIR / "steady_gait_cv.csv")
    gait.loc[~gait["enough_strides"], "swing_cv"] = np.nan  # < 20 steady strides
    dmd_table = pd.read_csv(METRICS_DIR / "dmd_features.csv")

    table = (
        svd_table
        .merge(network.drop(columns=["study", "group"]), on="subject_id")
        .merge(gait[["subject_id", "swing_cv"]], on="subject_id")
        .merge(dmd_table.drop(columns=["study", "group"]), on="subject_id")
        .merge(load_demographics()[["subject_id", "speed_01"]], on="subject_id")
    )
    return table


def design_matrix(table):
    """Columns: intercept, speed, one indicator per study except Ga (the baseline)."""
    return np.column_stack([
        np.ones(len(table)),
        table["speed_01"],
        table["study"] == "Ju",
        table["study"] == "Si",
    ]).astype(float)


def speed_adjusted_residuals(table, feature):
    """Residuals of `feature` after removing what speed + study predict in controls.

    Uses Huber regression: squared loss for small residuals, absolute loss for
    large ones, so a few extreme controls can't drag the fit.

    Returns the residuals (NaN where the feature or speed is missing) and the
    fitted coefficients [intercept, speed, Ju, Si].
    """
    X = design_matrix(table)
    y = table[feature].to_numpy(dtype=float)
    usable = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    fit_rows = usable & (table["group"] == "Control").to_numpy()

    # alpha=0: no ridge penalty; the intercept column is handled by fit_intercept
    huber = HuberRegressor(alpha=0, max_iter=1000).fit(X[fit_rows, 1:], y[fit_rows])
    coefficients = np.r_[huber.intercept_, huber.coef_]
    residuals = np.full(len(y), np.nan)
    residuals[usable] = y[usable] - X[usable] @ coefficients
    return residuals, coefficients


def format_result(row):
    return f"{row['p_pd_higher']:.2f} ({row['p_value']:.3f})"


def main():
    table = load_feature_table()

    growing = table[table["dmd_max_abs_eigenvalue"] > 1]
    print(f"Subjects with a growing DMD mode (max |lambda| > 1): {len(growing)} of "
          f"{table['dmd_max_abs_eigenvalue'].notna().sum()} ({growing['group'].value_counts().to_dict()}); "
          f"largest {table['dmd_max_abs_eigenvalue'].max():.4f}\n")

    comparison_rows = []
    print("Speed coefficient in controls (feature change per +1 m/s):")
    for feature in FEATURES:
        residuals, coefficients = speed_adjusted_residuals(table, feature)
        table[f"{feature}_adjusted"] = residuals
        print(f"  {feature:32s} {coefficients[1]:+.4f}")

        raw = compare_groups(table.dropna(subset=[feature]), feature)
        adjusted = compare_groups(table.dropna(subset=[f"{feature}_adjusted"]), f"{feature}_adjusted")
        for study in raw.index:
            comparison_rows.append({
                "feature": feature,
                "study": study,
                "raw_p_pd_higher": raw.loc[study, "p_pd_higher"],
                "raw_p_value": raw.loc[study, "p_value"],
                "adjusted_p_pd_higher": adjusted.loc[study, "p_pd_higher"],
                "adjusted_p_value": adjusted.loc[study, "p_value"],
            })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(METRICS_DIR / "speed_adjusted_comparison.csv", index=False)
    print(f"\nsaved {(METRICS_DIR / 'speed_adjusted_comparison.csv').relative_to(RESULTS_DIR.parent)}")

    # Side-by-side view: P(PD > Ctrl) (Mann-Whitney p), raw vs speed-adjusted
    view = comparison.assign(
        raw=lambda d: d.apply(lambda r: format_result({"p_pd_higher": r["raw_p_pd_higher"],
                                                       "p_value": r["raw_p_value"]}), axis=1),
        adjusted=lambda d: d.apply(lambda r: format_result({"p_pd_higher": r["adjusted_p_pd_higher"],
                                                            "p_value": r["adjusted_p_value"]}), axis=1),
    ).pivot(index="feature", columns="study", values=["raw", "adjusted"])
    view = view.reindex(FEATURES).swaplevel(axis=1)[[(s, k) for s in ["All", "Ga", "Ju", "Si"]
                                                     for k in ["raw", "adjusted"]]]
    pd.set_option("display.width", 220)
    print("\nP(PD > Ctrl) (Mann-Whitney p): raw vs adjusted for speed + study")
    print(view.to_string())

    patients = table[table["group"] == "PD"]
    controls = table[table["group"] == "Control"]
    print(f"\nspeed_01 range: controls {controls['speed_01'].min():.2f}-{controls['speed_01'].max():.2f}, "
          f"PD {patients['speed_01'].min():.2f}-{patients['speed_01'].max():.2f}; "
          f"PD below slowest control: {(patients['speed_01'] < controls['speed_01'].min()).sum()}")


if __name__ == "__main__":
    main()
