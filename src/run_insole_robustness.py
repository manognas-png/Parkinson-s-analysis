"""Step 10b: robustness of the insole-only PD classifier (Step 10 Part B).

(a) age as a covariate, plus an age-only baseline
(b) strides WITHOUT body-weight normalization (truly insole-only: no recorded weight
    needed), plus a body-weight-only baseline
(c) per-study evaluation on age-matched PD/control subsets (1:1, caliper)

Question: does the insole-only AUC still exceed measured speed after these?

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/insole_robustness.sbatch
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import bootstrap_auc, subject_scores, summarize
from src.load import load_demographics, parse_filename
from src.plot_explore import FIGURE_DIR, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save
from src.run_insole_regression import (
    GAIT6,
    N_BOOT,
    N_JOBS,
    N_REPEATS,
    N_SPLITS,
    SEED,
    check_inputs,
    classifier_fold,
    subject_data,
)

AGE_CALIPER_YEARS = 5
INSOLE = ["est_speed", "est_log_tug", "template_dev", *GAIT6]

# Feature sets run on body-weight-normalized strides (as in Step 10)
NORMALIZED_SETS = {
    "measured speed": ["speed_01"],
    "age": ["age"],
    "body weight": ["weight_kg"],
    "measured speed + age": ["speed_01", "age"],
    "insole (BW-normalized)": INSOLE,
    "insole (BW-normalized) + age": [*INSOLE, "age"],
}
# Feature sets run on raw strides (newtons; no recorded body weight used)
RAW_SETS = {
    "insole (raw)": INSOLE,
    "insole (raw) + age": [*INSOLE, "age"],
}
# (insole set, reference) pairs for paired AUC differences
COMPARISONS = [
    ("insole (BW-normalized)", "measured speed"),
    ("insole (BW-normalized) + age", "measured speed + age"),
    ("insole (raw)", "measured speed"),
    ("insole (raw) + age", "measured speed + age"),
    ("insole (raw)", "insole (BW-normalized)"),
]


def load_insole_data(files, demographics):
    """Raw and body-weight-normalized strides plus per-subject info, for all trials.

    Returns info (one row per subject), X_raw, X_bw (strides x 6 x 100), F_raw, F_bw
    (flattened strides + log duration, the ridge inputs), stride_subject and targets
    (speed, log TUG).
    """
    weights = demographics.set_index("subject_id")["weight_kg"]
    data = Parallel(n_jobs=N_JOBS)(
        delayed(subject_data)(p, weights.get(parse_filename(p)["subject_id"], np.nan), normalize=False)
        for p in files)
    meta = pd.DataFrame([parse_filename(p) for p in files])[["subject_id", "study", "group"]]
    info = pd.DataFrame([{k: v for k, v in d.items() if k not in ("strides", "log_duration")} for d in data])
    info = meta.merge(info, on="subject_id").merge(
        demographics[["subject_id", "speed_01", "tug_s", "age", "weight_kg"]], on="subject_id", how="left")
    X_raw = np.concatenate([d["strides"] for d in data]).astype(np.float32)
    D = np.concatenate([d["log_duration"] for d in data])
    stride_subject = np.concatenate([[i] * len(d["strides"]) for i, d in enumerate(data)]).astype(int)
    X_bw = (X_raw / info["body_weight_n"].to_numpy()[stride_subject, None, None]).astype(np.float32)
    flat = lambda X: np.hstack([X.reshape(len(X), -1), D[:, None]]).astype(np.float32)
    targets = np.column_stack([info["speed_01"], np.log(info["tug_s"])])
    return info, X_raw, X_bw, flat(X_raw), flat(X_bw), stride_subject, targets


def age_matched_subjects(info, caliper=AGE_CALIPER_YEARS):
    """1:1 PD-control pairs within each study, minimizing total age difference
    (Hungarian algorithm), keeping only pairs within `caliper` years."""
    pairs = []
    for study, part in info.groupby("study"):
        pd_rows = part[part["group"] == "PD"]
        controls = part[part["group"] == "Control"]
        cost = np.abs(pd_rows["age"].to_numpy()[:, None] - controls["age"].to_numpy()[None, :])
        rows, cols = linear_sum_assignment(np.where(cost <= caliper, cost, 1e6))
        for r, c in zip(rows, cols):
            if cost[r, c] <= caliper:
                pairs.append({"study": study, "pd": pd_rows["subject_id"].iloc[r],
                              "control": controls["subject_id"].iloc[c], "age_difference": cost[r, c]})
    return pd.DataFrame(pairs)


def describe_groups(info, matched_ids):
    """Age and body weight by study and group, before and after matching."""
    rows = []
    for label, part in [("all subjects", info), ("age-matched", info[info["subject_id"].isin(matched_ids)])]:
        for (study, group), g in part.groupby(["study", "group"]):
            rows.append({"subset": label, "study": study, "group": group, "n": len(g),
                         "age_median": g["age"].median(), "weight_median": g["weight_kg"].median()})
    return pd.DataFrame(rows)


def auc_summary(predictions, subjects=None):
    """AUC per repeat (optionally on a subset of subjects): overall and per study."""
    p = predictions if subjects is None else predictions[predictions["subject"].isin(subjects)]
    per_repeat = summarize(p)["auc"]
    per_study = summarize(p, by="study").groupby("study")["auc"].agg(["mean", "std"])
    return per_repeat, per_study


def plot_robustness(table):
    """Figure 20: AUC by feature set, all subjects vs age-matched subsets."""
    fig, ax = plt.subplots(figsize=(13, 5))
    names = table["feature_set"].tolist()
    for i, row in table.reset_index(drop=True).iterrows():
        color = "#6250d6" if row["feature_set"].startswith("insole") else TEXT_COLOR
        ax.errorbar(i - 0.12, row["auc_all"], yerr=row["auc_all_sd"], fmt="o", color=color, markersize=8,
                    capsize=3, linewidth=2, label="all subjects" if i == 0 else None)
        ax.errorbar(i + 0.12, row["auc_matched"], yerr=row["auc_matched_sd"], fmt="D", color=color, markersize=7,
                    capsize=3, linewidth=2, markerfacecolor="none", label="age-matched subset" if i == 0 else None)
    ax.axhline(0.5, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("AUC, PD vs control (mean ± SD over repeats)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Insole-only classifier robustness: age, body-weight normalization, age matching", loc="left")
    fig.tight_layout()
    save(fig, "20_insole_robustness.png")


def main():
    start = time.time()
    files, demographics = check_inputs()
    info, X_raw, X_bw, F_raw, F_bw, stride_subject, targets = load_insole_data(files, demographics)
    y = (info["group"] == "PD").astype(int).to_numpy()
    print(f"{len(X_raw)} strides from {info['cadence'].notna().sum()} subjects; loaded in {time.time() - start:.0f} s")

    pairs = age_matched_subjects(info)
    matched_ids = set(pairs["pd"]) | set(pairs["control"])
    groups_table = describe_groups(info, matched_ids)
    print(f"\n== Age matching: 1:1 within study, caliper {AGE_CALIPER_YEARS} years ==")
    print(pairs.groupby("study").agg(pairs=("pd", "size"), mean_age_difference=("age_difference", "mean"))
          .round(2).to_string())
    print(groups_table.pivot_table(index=["subset", "study"], columns="group",
                                   values=["n", "age_median", "weight_median"]).round(1).to_string())

    splits = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, info["subject_id"])):
            splits.append((repeat, fold, train, test))

    runs = {"normalized": (X_bw, F_bw, NORMALIZED_SETS), "raw": (X_raw, F_raw, RAW_SETS)}
    predictions = {}
    for run, (X, F, feature_sets) in runs.items():
        results = Parallel(n_jobs=N_JOBS)(
            delayed(classifier_fold)(X, F, stride_subject, targets, info, train, test, SEED + 100 * r + f,
                                     feature_sets)
            for r, f, train, test in splits)
        for name in feature_sets:
            predictions[name] = pd.concat([
                pd.DataFrame({"repeat": r, "fold": f, "subject": info["subject_id"].to_numpy()[test],
                              "y": y[test], "study": info["study"].to_numpy()[test], "score": scores[name]})
                for (r, f, _, test), (scores, _) in zip(splits, results)], ignore_index=True)
        print(f"{run} run done ({time.time() - start:.0f} s)")

    # AUC: all subjects and age-matched subsets, overall and per study
    rows = []
    for name, p in predictions.items():
        all_repeat, all_study = auc_summary(p)
        matched_repeat, matched_study = auc_summary(p, matched_ids)
        rows.append({"feature_set": name, "auc_all": all_repeat.mean(), "auc_all_sd": all_repeat.std(ddof=1),
                     **{f"all_{s}": all_study.loc[s, "mean"] for s in ["Ga", "Ju", "Si"]},
                     "auc_matched": matched_repeat.mean(), "auc_matched_sd": matched_repeat.std(ddof=1),
                     **{f"matched_{s}": matched_study.loc[s, "mean"] for s in ["Ga", "Ju", "Si"]}})
    table = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "insole_robustness.csv", index=False)
    pairs.to_csv(METRICS_DIR / "age_matched_pairs.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== AUC (mean over repeats): all subjects | age-matched subsets ==")
    print(table.round(3).to_string(index=False))

    # Paired bootstrap differences (same resampled subjects), all subjects and matched
    print("\n== Paired bootstrap differences in pooled AUC (95% CI) ==")
    difference_rows = []
    for subset_name, subset in [("all subjects", None), ("age-matched", matched_ids)]:
        scores = {}
        for name, p in predictions.items():
            q = p if subset is None else p[p["subject"].isin(subset)]
            scores[name] = subject_scores(q)
        score_table = pd.DataFrame({"y": next(iter(scores.values()))["y"], **{n: s["score"] for n, s in scores.items()}})
        boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)
        for a, b in COMPARISONS:
            d = boot[a] - boot[b]
            difference_rows.append({"subset": subset_name, "comparison": f"{a} - {b}", "difference": d.mean(),
                                    "ci_low": d.quantile(0.025), "ci_high": d.quantile(0.975),
                                    "fraction_positive": (d > 0).mean()})
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(METRICS_DIR / "insole_robustness_differences.csv", index=False)
    print(differences.round(3).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_robustness(table)
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
