"""Step 11: stance-indexed AP centre-of-pressure features (AP-COP10).

Baseline and features follow Sifat et al. 2026, arXiv:2609.22956 (reported AUC 0.894,
95% CI 0.843-0.937, with elastic-net stability selection and 20 x 5 CV stratified by
diagnosis and study). Here: our usual logistic model and 10 x 5 subject-wise splits.

1. AP-COP10 from all 16 sensors (steady-stride stances): CV AUC vs their ~0.89.
2. Speed: each feature raw vs speed-adjusted (Huber on training-fold controls,
   speed + study); COP10 + measured speed vs speed alone.
3. EXPLORATORY: fewer sensors: 6-sensor set (1, 4, 7), heel + forefoot (1, 7), and a
   simulated position strip (total force + AP position per foot).
4. EXPLORATORY: COP10 (16- and 6-sensor) added to the insole-only model: AUC gain and
   group permutation importance.

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/cop.sbatch
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.cop import COP_FEATURES, STANCE_POINTS, WINDOWS, trial_cop_features
from src.evaluate import bootstrap_auc, fit_model, subject_scores, summarize
from src.load import load_trial, parse_filename
from src.plot_explore import FIGURE_DIR, GROUP_COLORS, METRICS_DIR, TEXT_COLOR, save
from src.plot_svd import compare_groups
from src.run_insole_regression import GAIT6, N_BOOT, N_JOBS, N_REPEATS, N_SPLITS, SEED, check_inputs, insole_estimates
from src.run_insole_robustness import INSOLE, load_insole_data
from src.run_ml import make_models

CITATION = "Sifat et al. 2026, arXiv:2609.22956"
PAPER_AUC = (0.894, 0.843, 0.937)
N_PERMUTATIONS = 20
FEET = ["L", "R"]
ALL8 = {f: [f"{f}{i}" for i in range(1, 9)] for f in FEET}
TOTALS = {"L": "L_total", "R": "R_total"}
SENSOR_SETS = {  # key: (label, sensors per foot, total-force channel used for stances or None = sum of sensors)
    "c16": ("16 sensors", ALL8, TOTALS),
    "c6": ("6 sensors (1, 4, 7)", {f: [f"{f}1", f"{f}4", f"{f}7"] for f in FEET}, None),
    "c2": ("heel + forefoot (1, 7)", {f: [f"{f}1", f"{f}7"] for f in FEET}, None),
    "cs": ("position strip (total force + AP position)", ALL8, TOTALS),  # identical to c16 by construction
}


def columns(key):
    return [f"{key}_{f}" for f in COP_FEATURES]


MODELS = {
    "measured speed": ["speed_01"],
    "COP10, 16 sensors": columns("c16"),
    "COP10 (16) + measured speed": [*columns("c16"), "speed_01"],
    "COP10, 6 sensors": columns("c6"),
    "COP10, heel + forefoot": columns("c2"),
    "COP10, position strip": columns("cs"),
    "insole-only": INSOLE,
    "insole-only + COP10 (16)": [*INSOLE, *columns("c16")],
    "insole-only + COP10 (6)": [*INSOLE, *columns("c6")],
}
EXPLORATORY = {"COP10, 6 sensors", "COP10, heel + forefoot", "COP10, position strip", "insole-only",
               "insole-only + COP10 (16)", "insole-only + COP10 (6)"}
IMPORTANCE_GROUPS = {
    "speed estimates": ["est_speed", "est_log_tug"],
    "template deviation": ["template_dev"],
    "gait features": GAIT6,
}
COMBINED = {"insole-only + COP10 (16)": "c16", "insole-only + COP10 (6)": "c6"}
COMPARISONS = [  # (model, reference, label)
    ("COP10, 16 sensors", "measured speed", "COP10 (16) vs speed alone"),
    ("COP10 (16) + measured speed", "measured speed", "COP10 (16) + speed vs speed alone"),
    ("COP10, 6 sensors", "COP10, 16 sensors", "6 sensors vs 16 [exploratory]"),
    ("COP10, heel + forefoot", "COP10, 16 sensors", "heel + forefoot vs 16 [exploratory]"),
    ("COP10, position strip", "COP10, 16 sensors", "position strip vs 16 [exploratory]"),
    ("insole-only + COP10 (16)", "insole-only", "insole + COP10 (16) vs insole [exploratory]"),
    ("insole-only + COP10 (6)", "insole-only", "insole + COP10 (6) vs insole [exploratory]"),
]


def subject_cop(path):
    """AP-COP10 for every sensor set, plus mean stance traces (left/right averaged)."""
    trial = load_trial(path)
    record, traces = {"subject_id": parse_filename(path)["subject_id"]}, {}
    for key, (_, sensors, totals) in SENSOR_SETS.items():
        features, mean_traces = trial_cop_features(trial, sensors, totals)
        record.update({f"{key}_{name}": value for name, value in features.items()})
        traces[key] = np.nanmean([mean_traces["L"], mean_traces["R"]], axis=0)
    return record, traces


def cop_fold(X, F, stride_subject, targets, table, train, test, seed):
    """All models' test scores, speed-adjusted COP residuals, and combined-model importance."""
    is_control = (table["group"] == "Control").to_numpy()
    t = table.join(insole_estimates(X, F, stride_subject, targets, is_control, train, test, seed))
    y = (t["group"] == "PD").astype(int).to_numpy()
    groups = t["subject_id"].to_numpy()

    scores, fitted = {}, {}
    for name, cols in MODELS.items():
        fitted[name] = fit_model(clone(make_models(seed)["logistic"]), t.loc[train, cols], y[train], groups[train])
        scores[name] = fitted[name].predict_proba(t.loc[test, cols])[:, 1]

    # Speed-adjusted COP features: Huber (speed + study) fitted on training controls only
    design = lambda rows: np.column_stack([t.loc[rows, "speed_01"], t.loc[rows, "study"] == "Ju",
                                           t.loc[rows, "study"] == "Si"]).astype(float)
    residuals = {}
    train_controls = train[is_control[train]]
    for feature in columns("c16"):
        D, v = design(train_controls), t.loc[train_controls, feature].to_numpy()
        ok = np.isfinite(D).all(axis=1) & np.isfinite(v)
        huber = HuberRegressor(alpha=0, max_iter=1000).fit(D[ok], v[ok])
        D_test = design(test)
        residual = np.full(len(test), np.nan)
        usable = np.isfinite(D_test).all(axis=1)
        residual[usable] = t.loc[test, feature].to_numpy()[usable] - huber.predict(D_test[usable])
        residuals[feature] = residual

    # Group permutation importance for the combined models (held-out fold)
    rng = np.random.default_rng(seed)
    importance = []
    for name, key in COMBINED.items():
        cols = MODELS[name]
        X_test = t.loc[test, cols].reset_index(drop=True)
        baseline = roc_auc_score(y[test], fitted[name].predict_proba(X_test)[:, 1])
        for group, members in {**IMPORTANCE_GROUPS, "COP10": columns(key)}.items():
            drops = []
            for _ in range(N_PERMUTATIONS):
                shuffled = X_test.copy()
                shuffled[members] = X_test[members].to_numpy()[rng.permutation(len(X_test))]
                drops.append(baseline - roc_auc_score(y[test], fitted[name].predict_proba(shuffled)[:, 1]))
            importance.append({"model": name, "group": group, "auc_drop": float(np.mean(drops))})
    return scores, residuals, importance


def plot_trajectories(traces, info):
    """Figure 23: mean AP-COP over stance (left/right averaged), controls vs PD, per sensor set."""
    keys = ["c16", "c6", "c2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    phase = np.linspace(0, 100, STANCE_POINTS)
    for ax, key in zip(axes, keys):
        for a, b in WINDOWS:
            ax.axvspan(a, b, color=TEXT_COLOR, alpha=0.07, linewidth=0)
        for group in ["Control", "PD"]:
            curves = np.stack([traces[i][key] for i in np.flatnonzero((info["group"] == group).to_numpy())])
            mean, sd = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
            ax.plot(phase, mean, color=GROUP_COLORS[group], linewidth=2, label=group)
            ax.fill_between(phase, mean - sd, mean + sd, color=GROUP_COLORS[group], alpha=0.15, linewidth=0)
        label = SENSOR_SETS[key][0] + ("" if key == "c16" else " [exploratory]")
        ax.set_title(label, loc="left")
        ax.set_xlabel("% of stance (shaded: the five AP-COP10 windows)")
    axes[0].set_ylabel("AP-COP (0 = heel, 1 = toe)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"AP centre of pressure through stance (features after {CITATION})", x=0.06, ha="left")
    fig.tight_layout()
    save(fig, "23_cop_trajectories.png")


def plot_models(auc_table):
    """Figure 24: AUC by model (mean ± SD over repeats; band = pooled 95% CI)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, row in auc_table.reset_index(drop=True).iterrows():
        color = "#a3a29b" if row["exploratory"] else "#6250d6"
        ax.vlines(i, row["ci_low"], row["ci_high"], color=color, alpha=0.35, linewidth=9)
        ax.errorbar(i, row["auc_mean"], yerr=row["auc_sd"], fmt="o", color=color if row["exploratory"] else "#6250d6",
                    markersize=8, capsize=3, linewidth=2)
    ax.axhline(PAPER_AUC[0], color=TEXT_COLOR, linewidth=1, linestyle="--")
    ax.text(len(auc_table) - 0.5, PAPER_AUC[0] + 0.005, f"reported AP-COP10 AUC {PAPER_AUC[0]}", ha="right",
            fontsize=8, color=TEXT_COLOR)
    ax.axhline(0.5, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(auc_table)))
    ax.set_xticklabels([n.replace(", ", ",\n").replace(" + ", "\n+ ") + ("\n[exploratory]" if e else "")
                        for n, e in zip(auc_table["model"], auc_table["exploratory"])], fontsize=8)
    ax.set_ylabel("AUC, PD vs control")
    ax.set_title("AP-COP10 models: mean ± SD over repeats; band = pooled 95% bootstrap CI (grey = exploratory)",
                 loc="left")
    fig.tight_layout()
    save(fig, "24_cop_models.png")


def main():
    start = time.time()
    print(f"AP-COP10 features after {CITATION}; reported AUC {PAPER_AUC[0]} "
          f"(95% CI {PAPER_AUC[1]}-{PAPER_AUC[2]}).\n")
    files, demographics = check_inputs()
    info, X_raw, X_bw, F_raw, F_bw, stride_subject, targets = load_insole_data(files, demographics)
    cop = Parallel(n_jobs=N_JOBS)(delayed(subject_cop)(p) for p in files)
    cop_table = pd.DataFrame([record for record, _ in cop])
    traces = [t for _, t in cop]
    table = info.merge(cop_table, on="subject_id", how="left")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table[["subject_id", "study", "group", *[c for k in SENSOR_SETS for c in columns(k)]]] \
        .to_csv(METRICS_DIR / "cop_features.csv", index=False)
    same = np.allclose(table[columns("c16")].to_numpy(), table[columns("cs")].to_numpy(), equal_nan=True)
    print(f"loaded in {time.time() - start:.0f} s; subjects with 16-sensor COP10: "
          f"{table[columns('c16')].notna().all(axis=1).sum()} of {len(table)}; "
          f"position strip identical to 16 sensors: {same}")

    y = (table["group"] == "PD").astype(int).to_numpy()
    subject_ids = table["subject_id"].to_numpy()
    splits = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, subject_ids)):
            splits.append((repeat, fold, train, test))
    results = Parallel(n_jobs=N_JOBS)(
        delayed(cop_fold)(X_bw, F_bw, stride_subject, targets, table, train, test, SEED + 100 * r + f)
        for r, f, train, test in splits)
    print(f"cross-validation done ({time.time() - start:.0f} s)")

    meta = table[["subject_id", "study", "group"]].rename(columns={"subject_id": "subject"})
    long = lambda values_of: pd.concat([
        pd.DataFrame({"repeat": r, "fold": f, "subject": subject_ids[test], "y": y[test],
                      "study": table["study"].to_numpy()[test], "score": values_of(res)})
        for (r, f, _, test), res in zip(splits, results)], ignore_index=True)

    # Model AUCs, pooled CIs, paired differences
    predictions = {name: long(lambda res, n=name: res[0][n]) for name in MODELS}
    score_table = pd.DataFrame({"y": subject_scores(predictions["measured speed"])["y"],
                                **{n: subject_scores(p)["score"] for n, p in predictions.items()}})
    boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)
    rows = []
    for name, p in predictions.items():
        per_repeat = summarize(p)["auc"]
        per_study = summarize(p, by="study").groupby("study")["auc"].mean()
        rows.append({"model": name, "exploratory": name in EXPLORATORY, "auc_mean": per_repeat.mean(),
                     "auc_sd": per_repeat.std(ddof=1), "auc_pooled": boot[name].mean(),
                     "ci_low": boot[name].quantile(0.025), "ci_high": boot[name].quantile(0.975),
                     **{f"auc_{s}": per_study[s] for s in ["Ga", "Ju", "Si"]}})
    auc_table = pd.DataFrame(rows)
    differences = []
    for model, reference, label in COMPARISONS:
        d, ratio = boot[model] - boot[reference], boot[model] / boot[reference]
        differences.append({"comparison": label, "difference": d.mean(), "ci_low": d.quantile(0.025),
                            "ci_high": d.quantile(0.975), "auc_ratio": ratio.mean(),
                            "ratio_ci_low": ratio.quantile(0.025), "ratio_ci_high": ratio.quantile(0.975),
                            "exploratory": "exploratory" in label})
    differences = pd.DataFrame(differences)
    auc_table[~auc_table["exploratory"]].to_csv(METRICS_DIR / "cop_auc.csv", index=False)
    auc_table[auc_table["exploratory"]].to_csv(METRICS_DIR / "cop_auc_exploratory.csv", index=False)
    differences.to_csv(METRICS_DIR / "cop_differences.csv", index=False)

    pd.set_option("display.width", 230)
    print(f"\n== Model AUCs (logistic, 10 x 5 subject-wise CV); pooled = repeat-averaged scores, comparable to "
          f"the paper's {PAPER_AUC[0]} ==")
    print(auc_table.round(3).to_string(index=False))
    print("\n== Paired bootstrap differences in pooled AUC (95% CI); ratio = AUC kept ==")
    print(differences.round(3).to_string(index=False))

    # Each 16-sensor COP feature: raw vs speed-adjusted
    residual_table = pd.concat([
        pd.DataFrame({"repeat": r, "subject": subject_ids[test], **{f: res[1][f] for f in columns("c16")}})
        for (r, _, _, test), res in zip(splits, results)], ignore_index=True)
    adjusted = residual_table.groupby("subject")[columns("c16")].mean().join(meta.set_index("subject")).reset_index()
    feature_rows = []
    for feature in columns("c16"):
        raw_ok = table[[feature, "group"]].dropna()
        adj_ok = adjusted[[feature, "group"]].dropna()
        raw_tests = compare_groups(table.dropna(subset=[feature]), feature)
        adj_tests = compare_groups(adjusted.dropna(subset=[feature]), feature)
        feature_rows.append({
            "feature": feature.removeprefix("c16_"),
            "raw_auc": roc_auc_score(raw_ok["group"] == "PD", raw_ok[feature]),
            "raw_p": raw_tests.loc["All", "p_value"],
            "adjusted_auc": roc_auc_score(adj_ok["group"] == "PD", adj_ok[feature]),
            "adjusted_p": adj_tests.loc["All", "p_value"],
            **{f"adjusted_auc_{s}": adj_tests.loc[s, "p_pd_higher"] for s in ["Ga", "Ju", "Si"]},
        })
    feature_speed = pd.DataFrame(feature_rows)
    feature_speed.to_csv(METRICS_DIR / "cop_feature_speed.csv", index=False)
    print("\n== Each 16-sensor COP feature: P(PD > Ctrl) raw vs speed-adjusted (Huber on training-fold "
          "controls, speed + study) ==")
    print(feature_speed.round(3).to_string(index=False))

    # Combined-model group importance [exploratory]
    importance = pd.DataFrame([row for res in results for row in res[2]]).groupby(["model", "group"])["auc_drop"] \
        .agg(mean="mean", sd=lambda v: v.std(ddof=1), fraction_positive=lambda v: (v > 0).mean()).reset_index()
    importance.to_csv(METRICS_DIR / "cop_combined_importance_exploratory.csv", index=False)
    print("\n== [exploratory] Group permutation importance in the combined models (AUC drop, held-out folds) ==")
    print(importance.sort_values(["model", "mean"], ascending=[True, False]).round(4).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_trajectories(traces, table)
    plot_models(auc_table)
    print(f"\nFeatures after {CITATION}. Steps 3-4 (fewer sensors, combined models) are exploratory.")
    print(f"total time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
