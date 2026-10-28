"""Step 7: classical ML (PD vs control) with subject-wise evaluation.

Compares four feature sets x two models with repeated grouped cross-validation,
leave-one-study-out testing and held-out permutation importance.

Run as a SLURM job (not on the login node):  sbatch slurm/run_ml.sbatch
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import METRICS, leave_one_group_out, repeated_group_cv, summarize
from src.load import DATA_DIR, load_demographics
from src.plot_explore import FIGURE_DIR, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save

N_JOBS = 8
N_SPLITS = 5
N_REPEATS = 10
SEED = 0
PERMUTATION_REPEATS = 20

GAIT_FEATURES = [
    "energy_mode1_centered",
    "effective_n_modes_centered",
    "dmd_dominant_freq",
    "heel_forefoot_corr",
    "swing_cv",
    "dmd_reconstruction_error",
]
FEATURE_SETS = {
    "1 speed": ["speed_01"],
    "2 speed+age+sex": ["speed_01", "age", "is_female"],
    "3 gait": GAIT_FEATURES,
    "4 gait+speed+age+sex": [*GAIT_FEATURES, "speed_01", "age", "is_female"],
}
MODEL_COLORS = {"logistic": "#1baf7a", "random_forest": "#6250d6"}
STUDY_MARKERS = {"Ga": "o", "Ju": "s", "Si": "^"}

REQUIRED_FILES = {
    METRICS_DIR / "svd_features.csv": ["subject_id", "energy_mode1_centered", "effective_n_modes_centered"],
    METRICS_DIR / "network_features.csv": ["subject_id", "heel_forefoot_corr"],
    METRICS_DIR / "steady_gait_cv.csv": ["subject_id", "swing_cv", "enough_strides"],
    METRICS_DIR / "dmd_features.csv": ["subject_id", "dmd_dominant_freq", "dmd_reconstruction_error"],
    DATA_DIR / "demographics.txt": [],
}


def fail(message):
    sys.exit(f"ERROR: {message}")


def check_input_files():
    """Stop before any work if a feature file or column is missing."""
    missing = [str(p) for p in REQUIRED_FILES if not Path(p).exists()]
    if missing:
        fail("missing input files (run the earlier steps first):\n  " + "\n  ".join(missing))
    for path, columns in REQUIRED_FILES.items():
        if columns:
            header = pd.read_csv(path, nrows=0).columns
            absent = [c for c in columns if c not in header]
            if absent:
                fail(f"{path} is missing columns {absent}")


def load_dataset():
    """One row per subject: all feature-set columns, label, study."""
    from src.speed_adjust import load_feature_table  # merges the feature CSVs + speed

    table = load_feature_table()
    demographics = load_demographics()[["subject_id", "age", "sex"]]
    table = table.merge(demographics, on="subject_id", how="left")
    table["is_female"] = (table["sex"] == "female").astype(float).where(table["sex"].notna())
    table["label"] = (table["group"] == "PD").astype(int)

    needed = sorted({c for columns in FEATURE_SETS.values() for c in columns})
    absent = [c for c in needed if c not in table.columns]
    if absent:
        fail(f"merged table is missing feature columns {absent}")
    if table["subject_id"].duplicated().any():
        fail("more than one row per subject in the merged table")
    if table["label"].nunique() != 2 or len(table) < 100:
        fail(f"unexpected data: {len(table)} subjects, labels {table['label'].unique()}")
    return table


def make_models(seed):
    """Both models as a pipeline inside a grid search with subject-wise inner CV."""
    inner_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    logistic = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=5000)),  # L2 by default
    ])
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # trees need no scaling
        ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         random_state=seed, n_jobs=1)),
    ])
    return {
        "logistic": GridSearchCV(logistic, {"model__C": np.logspace(-3, 2, 11)},
                                 scoring="roc_auc", cv=inner_cv),
        "random_forest": GridSearchCV(forest, {"model__max_depth": [2, 3, 5, None]},
                                      scoring="roc_auc", cv=inner_cv),
    }


def mean_std_rows(per_repeat, feature_set, model, evaluation, subset_column=None):
    """Long-format rows (metric, mean, std over repeats) for the results CSV."""
    rows = []
    groups = per_repeat.groupby(subset_column) if subset_column else [("All", per_repeat)]
    for subset, part in groups:
        for metric in METRICS:
            rows.append({"feature_set": feature_set, "model": model, "evaluation": evaluation,
                         "subset": subset, "metric": metric, "mean": part[metric].mean(),
                         "std": part[metric].std(ddof=1), "n": len(part)})
    return rows


def plot_auc(cv_auc, loso):
    """Figure 11: repeated-CV AUC (left) and leave-one-study-out AUC (right)."""
    fig, (ax_cv, ax_loso) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    sets = list(FEATURE_SETS)
    offsets = {"logistic": -0.15, "random_forest": 0.15}
    rng = np.random.default_rng(0)

    for model, color in MODEL_COLORS.items():
        for i, feature_set in enumerate(sets):
            values = cv_auc[(cv_auc["feature_set"] == feature_set) & (cv_auc["model"] == model)]["auc"]
            x = i + offsets[model]
            ax_cv.scatter(x + rng.uniform(-0.04, 0.04, len(values)), values, s=12, color=color, alpha=0.5)
            ax_cv.errorbar(x, values.mean(), yerr=values.std(ddof=1), fmt="o", color=color,
                           markersize=8, capsize=4, linewidth=2)
            subset = loso[(loso["feature_set"] == feature_set) & (loso["model"] == model)]
            for _, row in subset.iterrows():
                ax_loso.scatter(x, row["auc"], s=60, color=color, marker=STUDY_MARKERS[row["held_out"]])
        ax_cv.scatter([], [], s=60, color=color, label=model.replace("_", " "))

    for study, marker in STUDY_MARKERS.items():
        ax_loso.scatter([], [], s=60, color=TEXT_COLOR, marker=marker, label=f"test on {study}")
    for ax, title in [(ax_cv, f"Repeated subject-wise CV ({N_REPEATS}×{N_SPLITS}-fold): mean ± SD"),
                      (ax_loso, "Leave-one-study-out")]:
        ax.axhline(0.5, color=TEXT_COLOR, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s.replace(" ", "\n", 1) for s in sets])
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="lower right")
    ax_cv.set_ylabel("AUC (PD vs control)")
    ax_cv.set_ylim(0.3, 1.0)
    fig.suptitle("Classifying PD vs control: AUC by feature set and model (0.5 = chance)", x=0.06, ha="left")
    save(fig, "11_ml_auc.png")


def main():
    start = time.time()
    check_input_files()
    table = load_dataset()
    y = table["label"].to_numpy()
    groups = table["subject_id"].to_numpy()
    row_info = table[["study"]]
    print(f"{len(table)} subjects ({y.sum()} PD, {len(y) - y.sum()} control); "
          f"missing values per column:\n{table[sorted(FEATURE_SETS['4 gait+speed+age+sex'])].isna().sum().to_string()}\n")

    result_rows, cv_auc_rows, loso_rows, importances = [], [], [], {}
    for feature_set, columns in FEATURE_SETS.items():
        X = table[columns]
        for model_name, model in make_models(SEED).items():
            with_importance = PERMUTATION_REPEATS if feature_set.startswith("4") else 0
            predictions, importance = repeated_group_cv(
                model, X, y, groups, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED,
                n_jobs=N_JOBS, permutation_repeats=with_importance, row_info=row_info)
            per_repeat = summarize(predictions)
            per_study = summarize(predictions, by="study")
            result_rows += mean_std_rows(per_repeat, feature_set, model_name, "repeated_cv")
            result_rows += mean_std_rows(per_study, feature_set, model_name, "repeated_cv_by_study", "study")
            cv_auc_rows += [{"feature_set": feature_set, "model": model_name, **r}
                            for r in per_repeat[["repeat", "auc"]].to_dict("records")]
            if importance is not None:
                importances[model_name] = importance

            loso = summarize(leave_one_group_out(model, X, y, groups, table["study"], row_info),
                             keys=(), by="held_out")
            loso_rows += [{"feature_set": feature_set, "model": model_name, **r} for r in loso.to_dict("records")]
            for _, r in loso.iterrows():
                result_rows += [{"feature_set": feature_set, "model": model_name, "evaluation": "leave_one_study_out",
                                 "subset": r["held_out"], "metric": m, "mean": r[m], "std": np.nan, "n": 1}
                                for m in METRICS]
            print(f"done: {feature_set:22s} {model_name:14s} CV AUC {per_repeat['auc'].mean():.3f} "
                  f"± {per_repeat['auc'].std(ddof=1):.3f}  ({time.time() - start:.0f} s)")

    results = pd.DataFrame(result_rows)
    cv_auc = pd.DataFrame(cv_auc_rows)
    loso = pd.DataFrame(loso_rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "ml_results.csv", index=False)
    print(f"\nsaved {(METRICS_DIR / 'ml_results.csv').relative_to(RESULTS_DIR.parent)}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_auc(cv_auc, loso)

    pd.set_option("display.width", 200)
    cv = results[results["evaluation"] == "repeated_cv"]
    table_view = cv.assign(value=lambda d: d["mean"].map("{:.3f}".format) + " ± " + d["std"].map("{:.3f}".format))
    print("\n== Repeated subject-wise CV (mean ± SD over repeats) ==")
    print(table_view.pivot(index=["feature_set", "model"], columns="metric", values="value")[METRICS].to_string())

    by_study = results[results["evaluation"] == "repeated_cv_by_study"]
    by_study = by_study[by_study["metric"] == "auc"]
    print("\n== AUC per study from pooled CV predictions (mean ± SD over repeats) ==")
    print(by_study.assign(value=lambda d: d["mean"].map("{:.3f}".format) + " ± " + d["std"].map("{:.3f}".format))
          .pivot(index=["feature_set", "model"], columns="subset", values="value").to_string())

    print("\n== Leave-one-study-out (tested study in columns) ==")
    print(loso.pivot(index=["feature_set", "model"], columns="held_out", values="auc").round(3).to_string())

    # Do gait features add to speed + age + sex? Same splits per repeat -> paired differences
    print("\n== Set 4 minus set 2, CV AUC per repeat (paired, same splits) ==")
    for model_name in MODEL_COLORS:
        auc = cv_auc[cv_auc["model"] == model_name].pivot(index="repeat", columns="feature_set", values="auc")
        difference = auc["4 gait+speed+age+sex"] - auc["2 speed+age+sex"]
        print(f"  {model_name:14s} mean {difference.mean():+.3f} ± {difference.std(ddof=1):.3f}; "
              f"set 4 better in {(difference > 0).sum()}/{len(difference)} repeats")

    set4 = cv[(cv["feature_set"] == "4 gait+speed+age+sex") & (cv["metric"] == "auc")]
    best = set4.loc[set4["mean"].idxmax(), "model"]
    importance = importances[best].drop(columns=["repeat", "fold"])
    summary = pd.DataFrame({"mean_auc_drop": importance.mean(), "sd": importance.std(ddof=1)}) \
        .sort_values("mean_auc_drop", ascending=False)
    summary.to_csv(METRICS_DIR / "ml_permutation_importance.csv")
    print(f"\n== Permutation importance, feature set 4, best model = {best} "
          f"(AUC drop on held-out folds, mean ± SD over {len(importance)} folds) ==")
    print(summary.round(4).to_string())
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
