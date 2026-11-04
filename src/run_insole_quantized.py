"""Step 10e: does the insole-only classifier survive float32 / int16 storage?

Same 10 x 5 subject-wise splits and training as Step 10 Part B (body-weight-normalized
6-sensor strides). In each fold the DEPLOYED model is extracted:
  - ridge speed and ridge TUG (scaler folded into the weights: 601 weights + 1 intercept each)
  - template (mean training-control stride, 6 x 100)
  - logistic regression (imputation medians; scaler folded into 6 weights + 1 intercept)
and the held-out subjects are scored with its parameters stored as float64 (reference),
float32 (parameters and arithmetic) and int16 (each stored array symmetric-quantized with
one scale; arithmetic in float64, i.e. storage precision only).

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/insole_quantized.sbatch
"""

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import bootstrap_auc, fit_model, subject_scores, summarize
from src.plot_explore import METRICS_DIR
from src.run_insole_regression import (
    N_BOOT, N_JOBS, N_REPEATS, N_SPLITS, SEED, check_inputs, insole_estimates, ridge_search, subject_medians,
)
from src.run_insole_robustness import INSOLE, load_insole_data
from src.run_ml import make_models

PRECISIONS = ["float64", "float32", "int16"]


def quantize(array, precision):
    """Stored copy of a parameter array at the given precision (returned as float for scoring)."""
    array = np.asarray(array, dtype=np.float64)
    if precision == "float64":
        return array
    if precision == "float32":
        return array.astype(np.float32)
    scale = np.max(np.abs(array)) / 32767 if np.any(array) else 1.0
    return (np.clip(np.round(array / scale), -32767, 32767).astype(np.int16) * scale).astype(np.float64)


def folded_linear(scaler, coef, intercept):
    """w' x + b' equal to coef . (x - mean) / scale + intercept."""
    w = np.ravel(coef) / scaler.scale_
    return w, float(np.ravel(intercept)[0] - np.sum(w * scaler.mean_))


def deployed_model(X, F, stride_subject, targets, is_control, train, classifier):
    """Parameters a device would store, refitted exactly as Step 10 does for test subjects."""
    params = {}
    for k, name in enumerate(["speed", "tug"]):
        rows = np.isin(stride_subject, train) & np.isfinite(targets[:, k][stride_subject])
        search = ridge_search().fit(F[rows], targets[:, k][stride_subject[rows]], groups=stride_subject[rows])
        pipe = search.best_estimator_
        params[f"{name}_w"], params[f"{name}_b"] = folded_linear(pipe["scale"], pipe["ridge"].coef_,
                                                                 pipe["ridge"].intercept_)
    params["template"] = X[np.isin(stride_subject, train[is_control[train]])].mean(axis=0)
    pipe = classifier.best_estimator_
    params["medians"] = pipe["impute"].statistics_
    params["logit_w"], params["logit_b"] = folded_linear(pipe["scale"], pipe["model"].coef_, pipe["model"].intercept_)
    return params


def score_subjects(params, precision, X, F, stride_subject, gait, test):
    """Classifier score (logit) for held-out subjects with parameters stored at `precision`."""
    dtype = np.float32 if precision == "float32" else np.float64
    p = {k: np.asarray(quantize(v, precision), dtype=dtype) for k, v in params.items()}
    Fp, Xp = F.astype(dtype), X.astype(dtype)
    rows = np.isin(stride_subject, test)
    speed = np.full(len(F), np.nan)
    tug = np.full(len(F), np.nan)
    deviation = np.full(len(F), np.nan)
    speed[rows] = Fp[rows] @ p["speed_w"] + p["speed_b"]
    tug[rows] = Fp[rows] @ p["tug_w"] + p["tug_b"]
    deviation[rows] = ((Xp[rows] - p["template"]) ** 2).mean(axis=(1, 2))
    features = np.column_stack([
        subject_medians(speed, stride_subject, test), subject_medians(tug, stride_subject, test),
        subject_medians(deviation, stride_subject, test), gait.loc[test].to_numpy(),
    ]).astype(dtype)
    features = np.where(np.isnan(features), p["medians"], features)  # imputation, as in the pipeline
    return (features @ p["logit_w"] + p["logit_b"]).astype(np.float64), features[:, 0].astype(np.float64)


def quantized_fold(X, F, stride_subject, targets, info, train, test, seed):
    is_control = (info["group"] == "Control").to_numpy()
    y = (info["group"] == "PD").astype(int).to_numpy()
    groups = info["subject_id"].to_numpy()
    estimates = insole_estimates(X, F, stride_subject, targets, is_control, train, test, seed)
    table = info.join(estimates)
    classifier = fit_model(clone(make_models(seed)["logistic"]), table.loc[train, INSOLE], y[train], groups[train])
    params = deployed_model(X, F, stride_subject, targets, is_control, train, classifier)
    gait = table[INSOLE[3:]]
    out = {}
    for precision in PRECISIONS:
        out[precision], est_speed = score_subjects(params, precision, X, F, stride_subject, gait, test)
        if precision == "float64":  # rebuilt deployment must reproduce Step 10's test-time path
            reference = classifier.decision_function(table.loc[test, INSOLE])
            out["check_score"] = float(np.max(np.abs(out["float64"] - reference)))
            out["check_speed"] = float(np.nanmax(np.abs(est_speed - estimates.loc[test, "est_speed"].to_numpy())))
    out["n_parameters"] = int(sum(np.size(v) for v in params.values()))
    return out


def main():
    start = time.time()
    files, demographics = check_inputs()
    info, X_raw, X_bw, F_raw, F_bw, stride_subject, targets = load_insole_data(files, demographics)
    y = (info["group"] == "PD").astype(int).to_numpy()
    subject_ids = info["subject_id"].to_numpy()
    splits = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, subject_ids)):
            splits.append((repeat, fold, train, test))
    results = Parallel(n_jobs=N_JOBS)(
        delayed(quantized_fold)(X_bw, F_bw, stride_subject, targets, info, train, test, SEED + 100 * r + f)
        for r, f, train, test in splits)
    print(f"cross-validation done ({time.time() - start:.0f} s)")

    checks = pd.DataFrame([{k: r[k] for k in ["check_score", "check_speed", "n_parameters"]} for r in results])
    print(f"stored parameters per fold: {checks['n_parameters'].iloc[0]} (folded); rebuilt float64 model vs Step 10 "
          f"path: max |score diff| {checks['check_score'].max():.1e}, max |speed diff| {checks['check_speed'].max():.1e}")

    predictions = {p: pd.concat([
        pd.DataFrame({"repeat": r, "fold": f, "subject": subject_ids[test], "y": y[test],
                      "study": info["study"].to_numpy()[test], "score": res[p]})
        for (r, f, _, test), res in zip(splits, results)], ignore_index=True) for p in PRECISIONS}
    score_table = pd.DataFrame({"y": subject_scores(predictions["float64"])["y"],
                                **{p: subject_scores(v)["score"] for p, v in predictions.items()}})
    boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)
    rows = []
    for p in PRECISIONS:
        per_repeat = summarize(predictions[p])["auc"]
        d = boot[p] - boot["float64"]
        max_score_change = np.max(np.abs(predictions[p]["score"].to_numpy() - predictions["float64"]["score"].to_numpy()))
        rows.append({"precision": p, "bytes_per_number": {"float64": 8, "float32": 4, "int16": 2}[p],
                     "model_size_bytes": checks["n_parameters"].iloc[0] * {"float64": 8, "float32": 4, "int16": 2}[p],
                     "auc_mean": per_repeat.mean(), "auc_sd": per_repeat.std(ddof=1), "auc_pooled": boot[p].mean(),
                     "ci_low": boot[p].quantile(0.025), "ci_high": boot[p].quantile(0.975),
                     "auc_difference_vs_float64": d.mean(), "difference_ci_low": d.quantile(0.025),
                     "difference_ci_high": d.quantile(0.975), "max_abs_logit_change": max_score_change})
    table = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "insole_quantized.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n== Insole-only classifier: AUC with parameters stored as float64 / float32 / int16 ==")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
