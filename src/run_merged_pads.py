"""Step 12b (EXPLORATORY): merged-sensor classifier with physically realistic pad models.

Per foot, two large pads measure b = A x with x = (heel, midfoot, forefoot) = sensors 1, 4, 7:
  (a) adjacent pads, midfoot force split 50/50: A = [[1, 0.5, 0], [0, 0.5, 1]]
  (b) two non-overlapping pads:                 A = [[1, 1, 0],   [0, 0,   1]]
  (c) stacked layers (midfoot counted twice):   A = [[1, 1, 0],   [0, 1,   1]]
Strides are detected from S1 + S2 of the left foot (not from the real sensors), and all
features come from the two pads per foot. Compared with the real 6 sensors.

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/merged_pads.sbatch
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import bootstrap_auc, fit_model, subject_scores, summarize
from src.features_math import center_rows
from src.gait import resample_strides, steady_stride_bounds
from src.load import load_trial
from src.plot_explore import FIGURE_DIR, METRICS_DIR, TEXT_COLOR, save
from src.run_insole_regression import N_BOOT, N_JOBS, N_POINTS, N_REPEATS, N_SPLITS, SEED, SENSORS, check_inputs, insole_estimates
from src.run_insole_robustness import INSOLE, load_insole_data
from src.run_merged_sensors import gait_features
from src.run_ml import make_models

PAD_MODELS = {
    "(a) adjacent pads, midfoot split 50/50": np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]),
    "(b) non-overlapping pads": np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "(c) stacked layers": np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
}
REAL = "real 6 sensors"


def pad_matrix(A):
    """4 x 6 map from (L1, L4, L7, R1, R4, R7) to (L S1, L S2, R S1, R S2)."""
    return np.block([[A, np.zeros((2, 3))], [np.zeros((2, 3)), A]])


def subject_pads(path, body_weight_n):
    """Per pad model: strides detected from left S1 + S2, resampled pad signals / body weight,
    log stride durations and the whole-trial pad covariance."""
    trial = load_trial(path)
    time_s = trial["time"].to_numpy()
    signals = trial[SENSORS].to_numpy().T
    out = {}
    for name, A in PAD_MODELS.items():
        pads = pad_matrix(A) @ signals
        bounds = steady_stride_bounds(pads[0] + pads[1], time_s)
        strides = resample_strides(pads, time_s, bounds, N_POINTS) / body_weight_n
        durations = np.array([end - start for start, end in bounds])
        centered = center_rows(pads)
        out[name] = {"strides": strides.astype(np.float32), "log_duration": np.log(durations).astype(np.float32),
                     "cov": centered @ centered.T}
    return out


def stack(per_subject):
    """Concatenate subjects' strides -> X, F (ridge inputs), stride_subject."""
    X = np.concatenate([d["strides"] for d in per_subject])
    D = np.concatenate([d["log_duration"] for d in per_subject])
    stride_subject = np.concatenate([[i] * len(d["strides"]) for i, d in enumerate(per_subject)]).astype(int)
    F = np.hstack([X.reshape(len(X), -1), D[:, None]]).astype(np.float32)
    return X.astype(np.float32), F, stride_subject


def pads_fold(variants, targets, info, train, test, seed):
    """Insole-only classifier for each variant (estimates cross-fitted inside the fold)."""
    is_control = (info["group"] == "Control").to_numpy()
    y = (info["group"] == "PD").astype(int).to_numpy()
    groups = info["subject_id"].to_numpy()
    scores = {}
    for name, (X, F, stride_subject, gait) in variants.items():
        table = info[["subject_id"]].join(gait)
        table = table.join(insole_estimates(X, F, stride_subject, targets, is_control, train, test, seed))
        model = fit_model(clone(make_models(seed)["logistic"]), table.loc[train, INSOLE], y[train], groups[train])
        scores[name] = model.predict_proba(table.loc[test, INSOLE])[:, 1]
    return scores


def plot_pads(auc_table):
    """Figure 26: AUC of each pad model vs the real 6 sensors."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, row in auc_table.reset_index(drop=True).iterrows():
        color = TEXT_COLOR if row["variant"] == REAL else "#6250d6"
        ax.vlines(i, row["ci_low"], row["ci_high"], color=color, alpha=0.3, linewidth=10)
        ax.errorbar(i, row["auc_mean"], yerr=row["auc_sd"], fmt="o", color=color, capsize=3, markersize=8)
        ax.annotate(f"{row['auc_mean']:.3f}", (i, row["auc_mean"]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT_COLOR)
    ax.axhline(0.5, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(auc_table)))
    ax.set_xticklabels([v.replace(", ", ",\n") for v in auc_table["variant"]], fontsize=8)
    ax.set_ylabel("AUC, insole-only classifier (mean ± SD; band = pooled 95% CI)")
    ax.set_title("Exploratory (post hoc): two pads per foot, realistic pad models, strides from S1 + S2",
                 loc="left")
    fig.tight_layout()
    save(fig, "26_merged_pads_exploratory.png")


def main():
    start = time.time()
    print("EXPLORATORY analysis: realistic pad models, strides detected from S1 + S2.\n")
    for name, A in PAD_MODELS.items():
        print(f"  {name}: A = {A.tolist()}, rank {np.linalg.matrix_rank(A)}, S1 + S2 = {A.sum(axis=0).tolist()} "
              f"x (heel, midfoot, forefoot)")
    files, demographics = check_inputs()
    info, X_raw, X_bw, F_raw, F_bw, stride_subject, targets = load_insole_data(files, demographics)
    extracted = Parallel(n_jobs=N_JOBS)(
        delayed(subject_pads)(p, bw) for p, bw in zip(files, info["body_weight_n"]))

    real_gait = info[["energy_mode1", "effective_modes", "heel_forefoot"]]
    variants = {REAL: (X_bw, F_bw, stride_subject, real_gait)}
    real_counts = np.bincount(stride_subject, minlength=len(info))
    count_rows = [{"variant": REAL, "strides_total": len(X_bw), "strides_median_per_subject": np.median(real_counts),
                   "subjects_without_strides": int((real_counts == 0).sum()),
                   "subjects_with_same_count_as_real": len(info)}]
    for name in PAD_MODELS:
        per_subject = [e[name] for e in extracted]
        X, F, ss = stack(per_subject)
        gait = gait_features(np.stack([d["cov"] for d in per_subject]), np.eye(4))
        variants[name] = (X, F, ss, gait)
        counts = np.bincount(ss, minlength=len(info))
        count_rows.append({"variant": name, "strides_total": len(X), "strides_median_per_subject": np.median(counts),
                           "subjects_without_strides": int((counts == 0).sum()),
                           "subjects_with_same_count_as_real": int((counts == real_counts).sum())})
    counts_table = pd.DataFrame(count_rows)
    pd.set_option("display.width", 220)
    print(f"\nloaded in {time.time() - start:.0f} s")
    print("\n== Steady strides per variant (strides detected from S1 + S2 for the pad models) ==")
    print(counts_table.to_string(index=False))

    y = (info["group"] == "PD").astype(int).to_numpy()
    subject_ids = info["subject_id"].to_numpy()
    splits = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, subject_ids)):
            splits.append((repeat, fold, train, test))
    results = Parallel(n_jobs=N_JOBS)(
        delayed(pads_fold)(variants, targets, info, train, test, SEED + 100 * r + f) for r, f, train, test in splits)
    print(f"\ncross-validation done ({time.time() - start:.0f} s)")

    predictions = {v: pd.concat([
        pd.DataFrame({"repeat": r, "fold": f, "subject": subject_ids[test], "y": y[test],
                      "study": info["study"].to_numpy()[test], "score": res[v]})
        for (r, f, _, test), res in zip(splits, results)], ignore_index=True) for v in variants}
    score_table = pd.DataFrame({"y": subject_scores(predictions[REAL])["y"],
                                **{v: subject_scores(p)["score"] for v, p in predictions.items()}})
    boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)
    rows = []
    for v, p in predictions.items():
        per_repeat = summarize(p)["auc"]
        per_study = summarize(p, by="study").groupby("study")["auc"].mean()
        d, ratio = boot[v] - boot[REAL], boot[v] / boot[REAL]
        rows.append({"variant": v, "auc_mean": per_repeat.mean(), "auc_sd": per_repeat.std(ddof=1),
                     "auc_pooled": boot[v].mean(), "ci_low": boot[v].quantile(0.025), "ci_high": boot[v].quantile(0.975),
                     **{f"auc_{s}": per_study[s] for s in ["Ga", "Ju", "Si"]},
                     "difference_vs_real": d.mean(), "difference_ci_low": d.quantile(0.025),
                     "difference_ci_high": d.quantile(0.975), "auc_kept": ratio.mean()})
    auc_table = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    auc_table.to_csv(METRICS_DIR / "merged_pads_exploratory.csv", index=False)
    counts_table.to_csv(METRICS_DIR / "merged_pads_strides_exploratory.csv", index=False)
    print("\n== Insole-only classifier: pad models vs real 6 sensors (pooled AUC, paired bootstrap 95% CI) ==")
    print(auc_table.round(3).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_pads(auc_table)
    print(f"\nEXPLORATORY. total time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
