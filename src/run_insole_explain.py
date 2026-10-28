"""Step 10c: explaining the insole-only PD classifier.

1. Per-study paired bootstrap CIs: insole (body-weight-normalized and raw) minus
   measured speed, on all subjects and on the age-matched subsets.
2. Permutation importance on held-out folds for the six insole features (plus the
   speed and TUG estimates permuted together, since they are correlated), and the
   standardized logistic coefficients.

Age is not included: adding it changed the insole AUC by <= 0.003 (Step 10b).

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/insole_explain.sbatch
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import bootstrap_auc, fit_model, subject_scores
from src.plot_explore import FIGURE_DIR, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save
from src.run_insole_regression import N_BOOT, N_JOBS, N_REPEATS, N_SPLITS, SEED, check_inputs, insole_estimates
from src.run_insole_robustness import INSOLE, age_matched_subjects, load_insole_data
from src.run_ml import make_models

N_PERMUTATIONS = 20  # shuffles per feature per held-out fold
FEATURE_LABELS = {
    "est_speed": "estimated speed",
    "est_log_tug": "estimated TUG (log)",
    "template_dev": "template deviation",
    "energy_mode1": "mode-1 energy",
    "effective_modes": "effective modes",
    "heel_forefoot": "heel–forefoot coupling",
}
PERMUTATION_GROUPS = {**{f: [f] for f in INSOLE}, "est. speed + est. TUG together": ["est_speed", "est_log_tug"]}
VERSIONS = {"insole (BW-normalized)": "bw", "insole (raw)": "raw"}
STUDIES = ["Ga", "Ju", "Si"]


def explain_fold(X, F, stride_subject, targets, info, train, test, seed, include_speed):
    """Fit the insole classifier (and measured speed) on `train`; on `test`: scores,
    AUC drop per permuted feature group, and standardized coefficients."""
    is_control = (info["group"] == "Control").to_numpy()
    table = info.join(insole_estimates(X, F, stride_subject, targets, is_control, train, test, seed))
    y = (info["group"] == "PD").astype(int).to_numpy()
    groups = info["subject_id"].to_numpy()

    scores = {}
    if include_speed:
        speed_model = fit_model(clone(make_models(seed)["logistic"]), table.loc[train, ["speed_01"]],
                                y[train], groups[train])
        scores["measured speed"] = speed_model.predict_proba(table.loc[test, ["speed_01"]])[:, 1]

    model = fit_model(clone(make_models(seed)["logistic"]), table.loc[train, INSOLE], y[train], groups[train])
    X_test = table.loc[test, INSOLE].reset_index(drop=True)
    scores["insole"] = model.predict_proba(X_test)[:, 1]
    baseline = roc_auc_score(y[test], scores["insole"])

    rng = np.random.default_rng(seed)
    importance = {}
    for name, members in PERMUTATION_GROUPS.items():
        drops = []
        for _ in range(N_PERMUTATIONS):
            shuffled = X_test.copy()
            shuffled[members] = X_test[members].to_numpy()[rng.permutation(len(X_test))]
            drops.append(baseline - roc_auc_score(y[test], model.predict_proba(shuffled)[:, 1]))
        importance[name] = float(np.mean(drops))
    coefficients = dict(zip(INSOLE, model.best_estimator_.named_steps["model"].coef_[0]))
    return scores, importance, coefficients


def paired_differences(score_table, subsets, comparisons):
    """Bootstrap AUCs and insole - measured speed differences within each subset of subjects."""
    rows = []
    for subset_name, subjects in subsets.items():
        part = score_table.loc[score_table.index.isin(subjects)]
        boot = bootstrap_auc(part, n_boot=N_BOOT, seed=SEED)
        for insole, reference in comparisons:
            d = boot[insole] - boot[reference]
            rows.append({"subset": subset_name, "insole_version": insole,
                         "n_pd": int((part["y"] == 1).sum()), "n_control": int((part["y"] == 0).sum()),
                         "auc_insole": boot[insole].mean(), "auc_measured_speed": boot[reference].mean(),
                         "difference": d.mean(), "ci_low": d.quantile(0.025), "ci_high": d.quantile(0.975),
                         "fraction_positive": (d > 0).mean()})
    return pd.DataFrame(rows)


def plot_explain(differences, importance):
    """Figure 21: per-study differences (forest plot) and permutation importance."""
    fig, (ax_diff, ax_imp) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    colors = {"insole (BW-normalized)": "#6250d6", "insole (raw)": "#1baf7a"}
    labels = differences["subset"].unique().tolist()
    for k, (version, color) in enumerate(colors.items()):
        part = differences[differences["insole_version"] == version].set_index("subset").reindex(labels)
        positions = np.arange(len(labels)) + (k - 0.5) * 0.3
        ax_diff.errorbar(part["difference"], positions,
                         xerr=[part["difference"] - part["ci_low"], part["ci_high"] - part["difference"]],
                         fmt="o", color=color, capsize=3, linewidth=2, markersize=7, label=version)
    ax_diff.axvline(0, color=TEXT_COLOR, linewidth=1)
    ax_diff.set_yticks(range(len(labels)))
    ax_diff.set_yticklabels(labels)
    ax_diff.invert_yaxis()
    ax_diff.set_xlabel("AUC difference: insole − measured speed (95% bootstrap CI)")
    ax_diff.set_title("Does the insole beat measured speed, per study?", loc="left")
    ax_diff.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax_diff.grid(axis="y", visible=False)

    order = importance[importance["version"] == "insole (BW-normalized)"] \
        .sort_values("mean")["feature"].tolist()
    for k, (version, color) in enumerate(colors.items()):
        part = importance[importance["version"] == version].set_index("feature").reindex(order)
        positions = np.arange(len(order)) + (k - 0.5) * 0.35
        ax_imp.barh(positions, part["mean"], height=0.33, xerr=part["sd"], color=color, alpha=0.85,
                    error_kw={"ecolor": TEXT_COLOR, "elinewidth": 1}, label=version)
    ax_imp.axvline(0, color=TEXT_COLOR, linewidth=1)
    ax_imp.set_yticks(range(len(order)))
    ax_imp.set_yticklabels([FEATURE_LABELS.get(f, f) for f in order])
    ax_imp.set_xlabel("AUC drop when the feature is shuffled (held-out folds, mean ± SD)")
    ax_imp.set_title("Permutation importance of the insole features", loc="left")
    ax_imp.legend(frameon=False, loc="lower right")
    ax_imp.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "21_insole_explain.png")


def main():
    start = time.time()
    print("Age is not included: adding it changed the insole AUC by <= 0.003 in Step 10b.\n")
    files, demographics = check_inputs()
    info, X_raw, X_bw, F_raw, F_bw, stride_subject, targets = load_insole_data(files, demographics)
    y = (info["group"] == "PD").astype(int).to_numpy()
    subject_ids = info["subject_id"].to_numpy()
    print(f"{len(X_raw)} strides from {info['cadence'].notna().sum()} subjects; loaded in {time.time() - start:.0f} s")

    splits = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, subject_ids)):
            splits.append((repeat, fold, train, test))

    predictions, importance_rows, coefficient_rows = {}, [], []
    data = {"bw": (X_bw, F_bw), "raw": (X_raw, F_raw)}
    for version, key in VERSIONS.items():
        X, F = data[key]
        include_speed = key == "bw"  # measured speed does not depend on the strides; fit it once
        results = Parallel(n_jobs=N_JOBS)(
            delayed(explain_fold)(X, F, stride_subject, targets, info, train, test, SEED + 100 * r + f, include_speed)
            for r, f, train, test in splits)
        names = {"insole": version, **({"measured speed": "measured speed"} if include_speed else {})}
        for short, name in names.items():
            predictions[name] = pd.concat([
                pd.DataFrame({"repeat": r, "fold": f, "subject": subject_ids[test], "y": y[test], "score": s[short]})
                for (r, f, _, test), (s, _, _) in zip(splits, results)], ignore_index=True)
        for (r, f, _, _), (_, importance, coefficients) in zip(splits, results):
            importance_rows += [{"version": version, "repeat": r, "fold": f, "feature": k, "auc_drop": v}
                                for k, v in importance.items()]
            coefficient_rows.append({"version": version, "repeat": r, "fold": f, **coefficients})
        print(f"{version} done ({time.time() - start:.0f} s)")

    # 1. Per-study paired CIs (subject scores averaged over repeats)
    score_table = pd.DataFrame({"y": subject_scores(predictions["measured speed"])["y"],
                                **{n: subject_scores(p)["score"] for n, p in predictions.items()}})
    pairs = age_matched_subjects(info)
    matched = set(pairs["pd"]) | set(pairs["control"])
    study_of = info.set_index("subject_id")["study"]
    subsets = {"all studies": set(subject_ids),
               **{f"{s}": set(study_of.index[study_of == s]) for s in STUDIES},
               "all studies, age-matched": matched,
               **{f"{s}, age-matched": {i for i in matched if study_of[i] == s} for s in STUDIES}}
    differences = paired_differences(score_table, subsets,
                                     [(v, "measured speed") for v in VERSIONS])
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    differences.to_csv(METRICS_DIR / "insole_per_study_differences.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n== 1. Insole - measured speed: pooled AUC per subset, paired bootstrap 95% CI ==")
    print(differences.round(3).to_string(index=False))

    # 2. Permutation importance and coefficients
    importance = pd.DataFrame(importance_rows)
    summary = importance.groupby(["version", "feature"])["auc_drop"].agg(
        mean="mean", sd=lambda v: v.std(ddof=1), fraction_positive=lambda v: (v > 0).mean()).reset_index()
    summary.to_csv(METRICS_DIR / "insole_permutation_importance.csv", index=False)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficient_summary = coefficients.groupby("version")[INSOLE].agg(["mean", "std"]).T
    coefficient_summary.to_csv(METRICS_DIR / "insole_coefficients.csv")
    print(f"\n== 2. Permutation importance on held-out folds (AUC drop; {N_PERMUTATIONS} shuffles x "
          f"{len(splits)} folds) ==")
    print(summary.sort_values(["version", "mean"], ascending=[True, False]).round(4).to_string(index=False))
    print("\n== Standardized logistic coefficients (positive = more PD-like), mean and SD over folds ==")
    print(coefficient_summary.round(3).to_string())
    print("\n== Context for correlated features: Spearman correlations of the 6-sensor features with "
          "measured speed and TUG (all subjects) ==")
    guide = info[["speed_01", "tug_s", "energy_mode1", "effective_modes", "heel_forefoot"]].corr("spearman")
    print(guide.round(2).to_string())

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_explain(differences, summary)
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
