"""Compare the Ga / Ju / Si studies on severity, age, speed and recording length,
and check whether mode-1 energy tracks PD severity.

Run from the project root (after src.plot_svd):  python -m src.compare_studies
"""

import pandas as pd
from scipy.stats import spearmanr

from src.load import build_summary
from src.plot_explore import METRICS_DIR

DESCRIPTORS = ["hoehn_yahr", "updrs", "updrs_motor", "age", "speed_01", "duration_s"]
SEVERITY = ["hoehn_yahr", "updrs", "updrs_motor"]
MODE1_FEATURES = ["energy_mode1_centered", "energy_mode1_standardized"]


def median_iqr(values):
    values = values.dropna()
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return f"{median:.2f} [{q1:.2f}-{q3:.2f}] n={len(values)}"


def main():
    summary = build_summary()
    trials = summary[summary["trial"] == 1]
    svd_table = pd.read_csv(METRICS_DIR / "svd_features.csv")
    data = trials.merge(svd_table[["subject_id", *MODE1_FEATURES]], on="subject_id")

    pd.set_option("display.width", 160)
    for group in ["PD", "Control"]:
        subset = data[data["group"] == group]
        table = subset.groupby("study")[DESCRIPTORS].agg(median_iqr).T
        print(f"\n== {group}: median [IQR] ==")
        print(table.to_string())

    print("\n== Hoehn & Yahr stage counts (PD) ==")
    print(pd.crosstab(data.loc[data["group"] == "PD", "study"], data.loc[data["group"] == "PD", "hoehn_yahr"]))

    print("\n== Spearman correlation, mode-1 energy vs severity (PD only): rho (p) ==")
    patients = data[data["group"] == "PD"]
    rows = []
    for study, subset in [*patients.groupby("study"), ("All", patients)]:
        row = {"study": study}
        for feature in MODE1_FEATURES:
            for score in SEVERITY:
                pair = subset[[feature, score]].dropna()
                rho, p = spearmanr(pair[feature], pair[score])
                row[f"{feature.replace('energy_mode1_', '')}~{score}"] = f"{rho:+.2f} ({p:.2f})"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("study").T.to_string())


if __name__ == "__main__":
    main()
