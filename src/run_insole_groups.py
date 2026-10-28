"""Step 10d: what the insole classifier relies on, by feature group, and whether the
insole speed estimate carries information beyond measured speed.

1. Group permutation importance on held-out folds: speed estimates (speed + TUG),
   template deviation, all three gait features, and everything (sanity check).
2. Residual test: in each fold, regress estimated speed on measured speed using the
   training-fold CONTROLS only; on held-out subjects, residual = estimated - predicted.
   Does the residual (i.e. the speed-adjusted insole estimate) separate PD from controls?

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/insole_groups.sbatch
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import bootstrap_auc, fit_model, subject_scores, summarize
from src.plot_explore import FIGURE_DIR, GROUP_COLORS, METRICS_DIR, TEXT_COLOR, save
from src.plot_svd import compare_groups
from src.run_insole_regression import GAIT6, N_BOOT, N_JOBS, N_REPEATS, N_SPLITS, SEED, check_inputs, insole_estimates
from src.run_insole_robustness import INSOLE, load_insole_data
from src.run_ml import make_models

N_PERMUTATIONS = 20
GROUPS = {
    "speed estimates (speed + TUG)": ["est_speed", "est_log_tug"],
    "template deviation": ["template_dev"],
    "gait features (mode-1 energy, effective modes, heel-forefoot)": GAIT6,
    "everything (sanity check)": INSOLE,
}
VERSIONS = {"insole (BW-normalized)": "bw", "insole (raw)": "raw"}
STUDIES = ["Ga", "Ju", "Si"]


def group_fold(X, F, stride_subject, targets, info, train, test, seed):
    """Held-out group permutation importance and the speed-adjusted residual of estimated speed."""
    is_control = (info["group"] == "Control").to_numpy()
    table = info.join(insole_estimates(X, F, stride_subject, targets, is_control, train, test, seed))
    y = (info["group"] == "PD").astype(int).to_numpy()
    groups = info["subject_id"].to_numpy()

    # 1. Group permutation importance
    model = fit_model(clone(make_models(seed)["logistic"]), table.loc[train, INSOLE], y[train], groups[train])
    X_test = table.loc[test, INSOLE].reset_index(drop=True)
    baseline = roc_auc_score(y[test], model.predict_proba(X_test)[:, 1])
    rng = np.random.default_rng(seed)
    importance = {}
    for name, members in GROUPS.items():
        drops = []
        for _ in range(N_PERMUTATIONS):
            shuffled = X_test.copy()
            shuffled[members] = X_test[members].to_numpy()[rng.permutation(len(X_test))]
            drops.append(baseline - roc_auc_score(y[test], model.predict_proba(shuffled)[:, 1]))
        importance[name] = float(np.mean(drops))

    # 2. Residual of estimated speed after measured speed, fitted on training controls only
    controls = table.loc[train[is_control[train]], ["speed_01", "est_speed"]].dropna()
    line = LinearRegression().fit(controls[["speed_01"]], controls["est_speed"])
    held_out = table.loc[test]
    residual = (held_out["est_speed"] - (line.intercept_ + line.coef_[0] * held_out["speed_01"])).to_numpy()
    return baseline, importance, residual, (line.intercept_, line.coef_[0])


def plot_groups(importance, residuals):
    """Figure 22: group importance; speed-adjusted residual of estimated speed by study and group."""
    fig, (ax_imp, ax_res) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = {"insole (BW-normalized)": "#6250d6", "insole (raw)": "#1baf7a"}
    order = list(GROUPS)[::-1]
    for k, (version, color) in enumerate(colors.items()):
        part = importance[importance["version"] == version].set_index("group").reindex(order)
        positions = np.arange(len(order)) + (k - 0.5) * 0.35
        ax_imp.barh(positions, part["mean"], height=0.33, xerr=part["sd"], color=color, alpha=0.85,
                    error_kw={"ecolor": TEXT_COLOR, "elinewidth": 1}, label=version)
    ax_imp.axvline(0, color=TEXT_COLOR, linewidth=1)
    ax_imp.set_yticks(range(len(order)))
    ax_imp.set_yticklabels([g.replace(" (", "\n(") for g in order], fontsize=8)
    ax_imp.set_xlabel("AUC drop when the group is shuffled together (held-out folds, mean ± SD)")
    ax_imp.set_title("Group permutation importance", loc="left")
    ax_imp.legend(frameon=False, loc="upper right")
    ax_imp.grid(axis="y", visible=False)

    part = residuals[residuals["version"] == "insole (BW-normalized)"]
    rng = np.random.default_rng(0)
    for s, study in enumerate(STUDIES):
        for g, group in enumerate(["Control", "PD"]):
            values = part.loc[(part["study"] == study) & (part["group"] == group), "residual"].dropna()
            x = s + (g - 0.5) * 0.4
            color = GROUP_COLORS[group]
            ax_res.boxplot(values, positions=[x], widths=0.3, patch_artist=True, showfliers=False,
                           boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
                           whiskerprops={"color": color}, capprops={"color": color},
                           medianprops={"color": TEXT_COLOR})
            ax_res.scatter(x + rng.uniform(-0.07, 0.07, len(values)), values, s=10, color=color,
                           alpha=0.7, linewidths=0, zorder=3)
    for group in ["Control", "PD"]:
        ax_res.scatter([], [], s=40, color=GROUP_COLORS[group], label=group)
    ax_res.axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle="--")
    ax_res.set_xticks(range(len(STUDIES)))
    ax_res.set_xticklabels(STUDIES)
    ax_res.set_ylabel("estimated − expected speed (m/s)")
    ax_res.set_title("Insole speed estimate beyond measured speed (BW-normalized)", loc="left")
    ax_res.legend(frameon=False)
    ax_res.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "22_insole_groups.png")


def main():
    start = time.time()
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

    data = {"bw": (X_bw, F_bw), "raw": (X_raw, F_raw)}
    importance_rows, residual_rows, line_rows, baselines = [], [], [], {}
    for version, key in VERSIONS.items():
        X, F = data[key]
        results = Parallel(n_jobs=N_JOBS)(
            delayed(group_fold)(X, F, stride_subject, targets, info, train, test, SEED + 100 * r + f)
            for r, f, train, test in splits)
        baselines[version] = np.mean([b for b, _, _, _ in results])
        for (r, f, _, test), (_, importance, residual, line) in zip(splits, results):
            importance_rows += [{"version": version, "repeat": r, "fold": f, "group": g, "auc_drop": v}
                                for g, v in importance.items()]
            residual_rows += [{"version": version, "repeat": r, "fold": f, "subject": s, "residual": v}
                              for s, v in zip(subject_ids[test], residual)]
            line_rows.append({"version": version, "intercept": line[0], "slope": line[1]})
        print(f"{version} done ({time.time() - start:.0f} s)")

    meta = info[["subject_id", "study", "group"]].rename(columns={"subject_id": "subject"})
    importance = pd.DataFrame(importance_rows).groupby(["version", "group"])["auc_drop"].agg(
        mean="mean", sd=lambda v: v.std(ddof=1), fraction_positive=lambda v: (v > 0).mean()).reset_index()
    residual_long = pd.DataFrame(residual_rows).merge(meta, on="subject")
    lines = pd.DataFrame(line_rows).groupby("version")[["intercept", "slope"]].agg(["mean", "std"])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(METRICS_DIR / "insole_group_importance.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n== 1. Group permutation importance (AUC drop on held-out folds; "
          f"{N_PERMUTATIONS} shuffles x {len(splits)} folds) ==")
    for version, auc in baselines.items():
        print(f"  {version}: mean held-out fold AUC before shuffling {auc:.3f}")
    print(importance.sort_values(["version", "mean"], ascending=[True, False]).round(4).to_string(index=False))

    # 2. Residual test
    print("\n== 2. Estimated speed beyond measured speed (line fitted on training-fold controls) ==")
    print("  fitted line: estimated = intercept + slope x measured speed (mean, SD over folds)")
    print(lines.round(3).to_string())
    rows, subject_level = [], []
    for version in VERSIONS:
        part = residual_long[(residual_long["version"] == version)].dropna(subset=["residual"])
        predictions = part.assign(y=(part["group"] == "PD").astype(int), score=part["residual"])
        per_repeat = summarize(predictions)["auc"]
        per_study = summarize(predictions, by="study").groupby("study")["auc"].agg(["mean", "std"])
        subjects = subject_scores(predictions).join(meta.set_index("subject"))
        subjects = subjects.rename(columns={"score": "residual"})
        subject_level.append(subjects.assign(version=version).reset_index())
        tests = compare_groups(subjects.reset_index(), "residual")
        for study in ["All", *STUDIES]:
            subset = subjects if study == "All" else subjects[subjects["study"] == study]
            boot = bootstrap_auc(subset[["y", "residual"]], n_boot=N_BOOT, seed=SEED)["residual"]
            rows.append({"version": version, "study": study,
                         "auc_mean_over_repeats": per_repeat.mean() if study == "All" else per_study.loc[study, "mean"],
                         "auc_pooled": boot.mean(), "ci_low": boot.quantile(0.025), "ci_high": boot.quantile(0.975),
                         "p_pd_higher": tests.loc[study, "p_pd_higher"], "mann_whitney_p": tests.loc[study, "p_value"],
                         "median_residual_control": subset.loc[subset["y"] == 0, "residual"].median(),
                         "median_residual_pd": subset.loc[subset["y"] == 1, "residual"].median()})
    residual_table = pd.DataFrame(rows)
    residual_table.to_csv(METRICS_DIR / "insole_speed_residual.csv", index=False)
    residuals = pd.concat(subject_level, ignore_index=True)
    residuals.to_csv(METRICS_DIR / "insole_speed_residual_subjects.csv", index=False)
    print("\n  AUC of the residual for PD vs control (> 0.5: PD's insole speed estimate is HIGHER than "
          "their measured speed predicts; < 0.5: LOWER)")
    print(residual_table.round(3).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_groups(importance, residuals)
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
