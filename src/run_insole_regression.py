"""Step 10: insole-only estimates of walking speed and Timed Up-and-Go (TUG) time,
and an insole-only PD classifier built on them.

TUG is used ONLY as a fall-risk PROXY: this dataset records no falls.

Part A (regression, main target = continuous value): for speed and log(TUG),
  - CNN on strides (6 sensors x 100 points + stride duration), one network, two outputs
  - baseline: linear regression on cadence
  - baseline: ridge regression on the same stride inputs as the CNN
Part B (classifier, logistic, PD vs control), insole-only feature sets vs measured speed:
  - measured speed alone (reference; not insole-only)
  - estimated speed
  - estimated speed + estimated TUG + template deviation + 6-sensor gait features
  Estimates are cross-fitted inside each training fold (no leakage), using ridge.

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/insole_regression.sbatch
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import bootstrap_auc, fit_model, subject_scores, summarize
from src.features_math import center_rows, correlation_matrix, sensor_matrix, subset_spectral_features
from src.gait import resample_strides, steady_stride_bounds
from src.load import DATA_DIR, FILENAME_PATTERN, load_demographics, load_trial, parse_filename
from src.plot_explore import FIGURE_DIR, GROUP_COLORS, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save
from src.run_ml import make_models
from src.run_sensor_selection import ANATOMICAL

N_JOBS = 16  # matches --cpus-per-task in slurm/insole_regression.sbatch
DEVICE = "cpu"
N_SPLITS = 5
N_REPEATS = 10
SEED = 0
N_POINTS = 100
SENSORS = ANATOMICAL[6]  # L1, L4, L7, R1, R4, R7
GRAVITY = 9.81
VALIDATION_FRACTION = 0.15
MAX_EPOCHS = 500
PATIENCE = 30
RIDGE_ALPHAS = np.logspace(-1, 4, 6)
N_BOOT = 2000
TUG_PROXY_THRESHOLD = 13.5  # seconds; commonly used fall-risk cut-off for TUG
EXPECTED_SUBJECTS = 165
GAIT6 = ["energy_mode1", "effective_modes", "heel_forefoot"]
TARGETS = ["speed", "tug"]  # tug is modeled as log(TUG) and reported in seconds
MODELS = ["cadence_linear", "ridge_strides", "cnn_strides"]
MODEL_LABELS = {"cadence_linear": "linear on cadence", "ridge_strides": "ridge on stride inputs",
                "cnn_strides": "CNN on strides"}
FEATURE_SETS = {
    "measured speed": ["speed_01"],
    "estimated speed": ["est_speed"],
    "est. speed + est. TUG + template + 6-sensor gait": ["est_speed", "est_log_tug", "template_dev", *GAIT6],
}
PROXY_NOTE = "TUG is used only as a fall-risk PROXY; no falls were recorded in this dataset."


def fail(message):
    sys.exit(f"ERROR: {message}")


def check_inputs():
    try:
        import torch  # noqa: F401
    except ImportError:
        fail("PyTorch is not installed in .venv: source .venv/bin/activate && pip install -r requirements.txt")
    if not (DATA_DIR / "demographics.txt").exists():
        fail(f"missing {DATA_DIR / 'demographics.txt'}")
    files = sorted(p for p in DATA_DIR.glob("*_01.txt") if FILENAME_PATTERN.match(p.name))
    if len(files) != EXPECTED_SUBJECTS:
        fail(f"expected {EXPECTED_SUBJECTS} _01 trial files in {DATA_DIR}, found {len(files)}")
    demographics = load_demographics()
    needed = {"speed_01", "tug_s", "weight_kg"}
    if not needed <= set(demographics.columns):
        fail(f"demographics is missing {sorted(needed - set(demographics.columns))}")
    return files, demographics


# ---------- data ----------


def subject_data(path, weight_kg):
    """Insole-only data for one trial: 6-sensor strides / body weight, stride durations,
    cadence, and the 6-sensor gait features (from that subject's own signals only)."""
    info = parse_filename(path)
    trial = load_trial(path)
    time_s = trial["time"].to_numpy()
    proxy = trial[[s for s in SENSORS if s.startswith("L")]].sum(axis=1).to_numpy()
    bounds = steady_stride_bounds(proxy, time_s)
    body_weight_n = weight_kg * GRAVITY if np.isfinite(weight_kg) else (trial["L_total"] + trial["R_total"]).mean()
    strides = resample_strides(trial[SENSORS].to_numpy().T, time_s, bounds, N_POINTS) / body_weight_n
    durations = np.array([end - start for start, end in bounds])

    X16 = sensor_matrix(trial)
    Xc = center_rows(X16)
    gait = subset_spectral_features(Xc @ Xc.T, correlation_matrix(X16), SENSORS)
    return {
        "subject_id": info["subject_id"],
        "strides": strides.astype(np.float32),
        "log_duration": np.log(durations).astype(np.float32),
        "cadence": 120 / np.median(durations) if len(durations) else np.nan,  # steps per minute
        **gait,
    }


def ridge_search():
    """Ridge with alpha tuned by subject-grouped inner CV (strides of a subject stay together)."""
    return GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
                        {"ridge__alpha": RIDGE_ALPHAS}, cv=GroupKFold(n_splits=3),
                        scoring="neg_mean_squared_error")


def subject_medians(values, stride_subject, subjects):
    """Median of per-stride values for each subject position in `subjects` (NaN if no strides)."""
    return np.array([np.median(values[stride_subject == s]) if (stride_subject == s).any() else np.nan
                     for s in subjects])


def ridge_fit_predict(F, stride_subject, target, train_subjects, test_subjects):
    """Fit ridge on strides of training subjects with a known target; per-subject median prediction."""
    rows = np.isin(stride_subject, train_subjects) & np.isfinite(target[stride_subject])
    model = ridge_search().fit(F[rows], target[stride_subject[rows]], groups=stride_subject[rows])
    test_rows = np.isin(stride_subject, test_subjects)
    predictions = np.full(len(F), np.nan)
    predictions[test_rows] = model.predict(F[test_rows])
    return subject_medians(predictions, stride_subject, test_subjects)


# ---------- Part A: regression fold ----------


def regression_fold(X, D, F, stride_subject, targets, cadence, train, test, seed):
    """All three models for speed and log(TUG): train on `train` subjects, predict `test` subjects."""
    import torch

    from src.models import predict_strides, train_stride_regressor

    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    results = {}

    # CNN: one network, two standardized outputs (NaN targets masked out of the loss)
    mean, sd = np.nanmean(targets[train], axis=0), np.nanstd(targets[train], axis=0)
    standardized = (targets - mean) / sd
    val_subjects = rng.choice(train, max(1, int(round(VALIDATION_FRACTION * len(train)))), replace=False)
    fit_rows = np.isin(stride_subject, np.setdiff1d(train, val_subjects))
    val_rows = np.isin(stride_subject, val_subjects)
    model, best_epoch, _ = train_stride_regressor(
        X[fit_rows], D[fit_rows], standardized[stride_subject[fit_rows]],
        X[val_rows], D[val_rows], standardized[stride_subject[val_rows]], seed,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, device=DEVICE)
    test_rows = np.isin(stride_subject, test)
    stride_predictions = np.full((len(X), 2), np.nan)
    stride_predictions[test_rows] = predict_strides(model, X[test_rows], D[test_rows]) * sd + mean
    for k, target in enumerate(TARGETS):
        results[("cnn_strides", target)] = subject_medians(stride_predictions[:, k], stride_subject, test)

    # Ridge on the same stride inputs, one model per target
    for k, target in enumerate(TARGETS):
        results[("ridge_strides", target)] = ridge_fit_predict(F, stride_subject, targets[:, k], train, test)

    # Linear on cadence
    for k, target in enumerate(TARGETS):
        ok = np.isfinite(cadence[train]) & np.isfinite(targets[train, k])
        linear = LinearRegression().fit(cadence[train][ok, None], targets[train, k][ok])
        prediction = np.full(len(test), np.nan)
        has_cadence = np.isfinite(cadence[test])
        prediction[has_cadence] = linear.predict(cadence[test][has_cadence, None])
        results[("cadence_linear", target)] = prediction
    return results, best_epoch


# ---------- Part B: stacked insole-only classifier fold ----------


def template_scores(X, stride_subject, train_controls, subjects):
    """Median squared distance of each subject's strides to the mean control stride."""
    template = X[np.isin(stride_subject, train_controls)].mean(axis=0)
    distance = ((X - template) ** 2).mean(axis=(1, 2))
    return subject_medians(distance, stride_subject, subjects)


def insole_estimates(X, F, stride_subject, targets, is_control, train, test, seed):
    """Estimated speed, log(TUG) and template deviation for train and test subjects.

    Training subjects get cross-fitted values (inner folds that exclude them);
    test subjects get values from models fitted on all training subjects.
    """
    estimates = pd.DataFrame(index=np.r_[train, test], columns=["est_speed", "est_log_tug", "template_dev"],
                             dtype=float)
    inner = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for inner_train, inner_test in inner.split(train, groups=train):
        fit_subjects, score_subjects = train[inner_train], train[inner_test]
        estimates.loc[score_subjects, "est_speed"] = ridge_fit_predict(F, stride_subject, targets[:, 0],
                                                                       fit_subjects, score_subjects)
        estimates.loc[score_subjects, "est_log_tug"] = ridge_fit_predict(F, stride_subject, targets[:, 1],
                                                                         fit_subjects, score_subjects)
        estimates.loc[score_subjects, "template_dev"] = template_scores(
            X, stride_subject, fit_subjects[is_control[fit_subjects]], score_subjects)
    estimates.loc[test, "est_speed"] = ridge_fit_predict(F, stride_subject, targets[:, 0], train, test)
    estimates.loc[test, "est_log_tug"] = ridge_fit_predict(F, stride_subject, targets[:, 1], train, test)
    estimates.loc[test, "template_dev"] = template_scores(X, stride_subject, train[is_control[train]], test)
    return estimates


def classifier_fold(X, F, stride_subject, targets, info, train, test, seed):
    """Logistic PD classifier for each feature set; returns test-subject scores."""
    is_control = (info["group"] == "Control").to_numpy()
    estimates = insole_estimates(X, F, stride_subject, targets, is_control, train, test, seed)
    table = info.join(estimates)
    y = (info["group"] == "PD").astype(int).to_numpy()
    groups = info["subject_id"].to_numpy()
    scores = {}
    for name, columns in FEATURE_SETS.items():
        model = make_models(seed)["logistic"]
        fitted = fit_model(clone(model), table.loc[train, columns], y[train], groups[train])
        scores[name] = fitted.predict_proba(table.loc[test, columns])[:, 1]
    return scores, estimates.loc[test]


# ---------- reporting ----------


def regression_metrics(predictions, info):
    """Per repeat: MAE, RMSE, R2, Spearman (all subjects and per study), in original units."""
    measured = {"speed": info["speed_01"].to_numpy(), "tug": info["tug_s"].to_numpy()}
    rows = []
    for (repeat, model, target), part in predictions.groupby(["repeat", "model", "target"]):
        truth = measured[target][part["subject"]]
        estimate = part["prediction"].to_numpy()
        studies = info["study"].to_numpy()[part["subject"]]
        for study in ["All", "Ga", "Ju", "Si"]:
            ok = np.isfinite(truth) & np.isfinite(estimate) & ((studies == study) | (study == "All"))
            rows.append({"repeat": repeat, "model": model, "target": target, "study": study, "n": ok.sum(),
                         "mae": mean_absolute_error(truth[ok], estimate[ok]),
                         "rmse": float(np.sqrt(np.mean((truth[ok] - estimate[ok]) ** 2))),
                         "r2": r2_score(truth[ok], estimate[ok]),
                         "spearman": spearmanr(truth[ok], estimate[ok])[0]})
    return pd.DataFrame(rows)


def plot_regression(mean_predictions, info):
    """Figure 18: subject-level estimates (mean over repeats) vs measured values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    measured = {"speed": info["speed_01"], "tug": info["tug_s"]}
    for row, target in enumerate(TARGETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            estimate = mean_predictions[(model, target)]
            for group in ["Control", "PD"]:
                g = (info["group"] == group).to_numpy()
                ax.scatter(measured[target][g], estimate[g], s=16, color=GROUP_COLORS[group], alpha=0.75,
                           linewidths=0, label=group)
            ok = np.isfinite(measured[target]) & np.isfinite(estimate)
            low, high = np.nanmin(measured[target]), np.nanmax(measured[target])
            ax.plot([low, high], [low, high], color=TEXT_COLOR, linewidth=1, linestyle="--")
            if target == "tug":
                ax.axvline(TUG_PROXY_THRESHOLD, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
                ax.axhline(TUG_PROXY_THRESHOLD, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
                ax.set_xscale("log")
                ax.set_yscale("log")
            r2 = r2_score(measured[target][ok], estimate[ok])
            rho = spearmanr(measured[target][ok], estimate[ok])[0]
            ax.set_title(f"{MODEL_LABELS[model]}: R² {r2:.2f}, Spearman {rho:.2f}", loc="left")
            ax.set_xlabel("measured " + ("speed (m/s)" if target == "speed" else "TUG (s)"))
            ax.set_ylabel("insole estimate")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Insole-only estimates of walking speed (top) and TUG time (bottom; fall-risk proxy only, "
                 f"dotted = {TUG_PROXY_THRESHOLD} s); held-out subjects, mean over repeats", x=0.04, ha="left")
    fig.tight_layout()
    save(fig, "18_insole_regression.png")


def plot_classifier(auc_table):
    """Figure 19: AUC of the insole-only classifiers vs measured speed."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    markers = {"Ga": "o", "Ju": "s", "Si": "^"}
    for i, row in auc_table.reset_index(drop=True).iterrows():
        ax.errorbar(i, row["auc_mean"], yerr=row["auc_sd_repeats"], fmt="o", markersize=9, capsize=4,
                    color="#6250d6" if i else TEXT_COLOR, linewidth=2)
        ax.vlines(i, row["ci_low"], row["ci_high"], color="#6250d6" if i else TEXT_COLOR, alpha=0.3, linewidth=8)
        for study, marker in markers.items():
            ax.scatter(i + 0.18, row[f"auc_{study}"], marker=marker, s=40, color=TEXT_COLOR, alpha=0.6)
    for study, marker in markers.items():
        ax.scatter([], [], marker=marker, s=40, color=TEXT_COLOR, alpha=0.6, label=f"AUC within {study}")
    ax.axhline(0.5, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(auc_table)))
    ax.set_xticklabels([n.replace(" + ", " +\n") for n in auc_table["feature_set"]])
    ax.set_ylabel("AUC, PD vs control (mean ± SD; band = 95% CI)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Insole-only PD classifier vs measured speed (logistic, 10 × 5-fold subject-wise CV)", loc="left")
    fig.tight_layout()
    save(fig, "19_insole_classifier.png")


def main():
    start = time.time()
    print(PROXY_NOTE + "\n")
    files, demographics = check_inputs()
    weights = demographics.set_index("subject_id")["weight_kg"]

    data = Parallel(n_jobs=N_JOBS)(
        delayed(subject_data)(p, weights.get(parse_filename(p)["subject_id"], np.nan)) for p in files)
    meta = pd.DataFrame([parse_filename(p) for p in files])[["subject_id", "study", "group"]]
    info = pd.DataFrame([{k: v for k, v in d.items() if k not in ("strides", "log_duration")} for d in data])
    info = meta.merge(info, on="subject_id").merge(demographics[["subject_id", "speed_01", "tug_s"]],
                                                   on="subject_id", how="left")
    X = np.concatenate([d["strides"] for d in data])
    D = np.concatenate([d["log_duration"] for d in data])
    stride_subject = np.concatenate([[i] * len(d["strides"]) for i, d in enumerate(data)]).astype(int)
    F = np.hstack([X.reshape(len(X), -1), D[:, None]]).astype(np.float32)  # ridge inputs = CNN inputs
    targets = np.column_stack([info["speed_01"], np.log(info["tug_s"])])
    cadence = info["cadence"].to_numpy()
    y = (info["group"] == "PD").astype(int).to_numpy()
    tug = info["tug_s"]
    print(f"{len(X)} strides from {(info['cadence'].notna()).sum()} subjects; "
          f"speed known for {info['speed_01'].notna().sum()}, TUG known for {tug.notna().sum()} "
          f"({(tug > TUG_PROXY_THRESHOLD).sum()} above {TUG_PROXY_THRESHOLD} s, all "
          f"{info.loc[tug > TUG_PROXY_THRESHOLD, 'group'].unique().tolist()}); loaded in {time.time() - start:.0f} s\n")

    splits = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, info["subject_id"])):
            splits.append((repeat, fold, train, test))

    # Part A
    regression = Parallel(n_jobs=N_JOBS)(
        delayed(regression_fold)(X, D, F, stride_subject, targets, cadence, train, test, SEED + 100 * r + f)
        for r, f, train, test in splits)
    epochs = [e for _, e in regression]
    print(f"Part A done ({time.time() - start:.0f} s); CNN best epoch median {np.median(epochs):.0f} "
          f"[{min(epochs)}-{max(epochs)}] of max {MAX_EPOCHS}")
    rows = []
    for (repeat, fold, _, test), (results, _) in zip(splits, regression):
        for (model, target), values in results.items():
            values = np.exp(values) if target == "tug" else values
            rows += [{"repeat": repeat, "fold": fold, "subject": s, "model": model, "target": target,
                      "prediction": v} for s, v in zip(test, values)]
    predictions = pd.DataFrame(rows)
    metrics = regression_metrics(predictions, info)
    summary = metrics.groupby(["target", "model", "study"])[["n", "mae", "rmse", "r2", "spearman"]] \
        .agg(["mean", "std"]).round(3)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(METRICS_DIR / "insole_regression.csv")
    mean_predictions = {key: part.groupby("subject")["prediction"].mean().reindex(range(len(info))).to_numpy()
                        for key, part in predictions.groupby(["model", "target"])}
    saved = info[["subject_id", "study", "group", "speed_01", "tug_s", "cadence"]].copy()
    for (model, target), values in mean_predictions.items():
        saved[f"est_{target}_{model}"] = values
    saved.to_csv(METRICS_DIR / "insole_regression_predictions.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== Part A: regression on held-out subjects (mean ± SD over repeats; TUG in seconds) ==")
    view = metrics[metrics["study"] == "All"].groupby(["target", "model"])[["mae", "rmse", "r2", "spearman"]]
    print((view.mean().round(3).astype(str) + " ± " + view.std().round(3).astype(str)).to_string())
    print("\nR² by study (mean over repeats):")
    print(metrics.pivot_table(index=["target", "model"], columns="study", values="r2").round(3).to_string())

    # TUG > 13.5 s flag from the regression output (secondary, proxy only)
    print(f"\n== Secondary: flagging TUG > {TUG_PROXY_THRESHOLD} s (fall-risk proxy) with the regression output ==")
    known = tug.notna().to_numpy()
    flag = (tug > TUG_PROXY_THRESHOLD).to_numpy()
    for model in MODELS:
        aucs = []
        for repeat, part in predictions[(predictions["model"] == model) & (predictions["target"] == "tug")] \
                .groupby("repeat"):
            subjects = part["subject"].to_numpy()
            ok = known[subjects] & part["prediction"].notna().to_numpy()
            aucs.append(roc_auc_score(flag[subjects][ok], part["prediction"].to_numpy()[ok]))
        print(f"  estimated TUG, {MODEL_LABELS[model]:24s} AUC {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}")
    ok = known & info["speed_01"].notna().to_numpy()
    print(f"  measured speed (reference, slower = higher risk) AUC "
          f"{roc_auc_score(flag[ok], -info['speed_01'].to_numpy()[ok]):.3f}")
    print(f"  ({flag.sum()} subjects above {TUG_PROXY_THRESHOLD} s, all PD; {PROXY_NOTE})")

    # Part B
    classification = Parallel(n_jobs=N_JOBS)(
        delayed(classifier_fold)(X, F, stride_subject, targets, info, train, test, SEED + 100 * r + f)
        for r, f, train, test in splits)
    print(f"\nPart B done ({time.time() - start:.0f} s)")
    columns, auc_rows = {}, []
    for name in FEATURE_SETS:
        parts = [pd.DataFrame({"repeat": r, "fold": f, "subject": info["subject_id"].to_numpy()[test],
                               "y": y[test], "study": info["study"].to_numpy()[test], "score": scores[name]})
                 for (r, f, _, test), (scores, _) in zip(splits, classification)]
        p = pd.concat(parts, ignore_index=True)
        per_repeat = summarize(p)["auc"]
        per_study = summarize(p, by="study").groupby("study")["auc"].mean()
        columns[name] = subject_scores(p)
        auc_rows.append({"feature_set": name, "insole_only": name != "measured speed",
                         "auc_mean": per_repeat.mean(), "auc_sd_repeats": per_repeat.std(ddof=1),
                         **{f"auc_{s}": per_study[s] for s in ["Ga", "Ju", "Si"]}})
    score_table = pd.DataFrame({"y": columns["measured speed"]["y"], **{n: c["score"] for n, c in columns.items()}})
    boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)
    auc_table = pd.DataFrame(auc_rows)
    auc_table["ci_low"] = [boot[n].quantile(0.025) for n in FEATURE_SETS]
    auc_table["ci_high"] = [boot[n].quantile(0.975) for n in FEATURE_SETS]
    auc_table.to_csv(METRICS_DIR / "insole_classifier.csv", index=False)
    print("\n== Part B: PD vs control classifier (logistic; estimates cross-fitted with ridge) ==")
    print(auc_table.round(3).to_string(index=False))
    names = list(FEATURE_SETS)
    for a, b in [(names[1], names[0]), (names[2], names[0]), (names[2], names[1])]:
        d = boot[a] - boot[b]
        print(f"  [{a}] - [{b}]: {d.mean():+.3f} (95% CI {d.quantile(0.025):+.3f} to {d.quantile(0.975):+.3f})")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_regression(mean_predictions, info)
    plot_classifier(auc_table)
    print(f"\n{PROXY_NOTE}\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
