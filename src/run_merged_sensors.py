"""Step 12 (EXPLORATORY): merged "large sensors" with linear recovery, A x = b.

Per foot, two large sensors S1 = sensor 1 + sensor 4 (heel + midfoot) and
S2 = sensor 4 + sensor 7 (midfoot + forefoot): b = A x with A = [[1, 1, 0], [0, 1, 1]]
and x = (heel, midfoot, forefoot). A has rank 2; its null space n = (1, -1, 1)/sqrt(3)
is invisible to the merged sensors.

1. Sparsity: fraction of each stance phase with 0 / 1 / 2 / 3 of the points loaded (> 20 N).
2. Recovery x_hat = W b, per foot, three ways: (a) pseudo-inverse; (b) SVD prior with 2
   modes learned from training-fold controls; (c) ridge from (S1, S2) trained on
   training-fold subjects. Error vs the real sensors 1, 4, 7 (controls / PD, heel).
3. Does the PD heel signal survive (early-stance heel loading, real vs recovered)?
4. Insole-only classifier with (i) the 4 merged signals, (ii) the recovered 6 sensors,
   vs the real 6 sensors.

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/merged_sensors.sbatch
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import null_space
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.cop import stance_bounds
from src.evaluate import bootstrap_auc, fit_model, subject_scores, summarize
from src.features_math import center_rows
from src.gait import CONTACT_THRESHOLD_N
from src.load import load_trial
from src.plot_explore import FIGURE_DIR, GROUP_COLORS, METRICS_DIR, TEXT_COLOR, save
from src.plot_svd import compare_groups
from src.run_insole_regression import N_BOOT, N_JOBS, N_REPEATS, N_SPLITS, SEED, SENSORS, check_inputs, insole_estimates
from src.run_insole_robustness import INSOLE, load_insole_data
from src.run_ml import make_models

A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
A_PINV = np.linalg.pinv(A)
FEET = {"L": [0, 1, 2], "R": [3, 4, 5]}  # positions in SENSORS = L1, L4, L7, R1, R4, R7
POINTS = 101  # stance resampled to 0..100 %
PHASES = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
EARLY_HEEL = (5, 15)  # % of stance, the first AP-COP window
METHODS = ["pseudo-inverse", "SVD prior", "ridge"]
VARIANTS = ["real 6 sensors", "merged 4 signals", *[f"recovered: {m}" for m in METHODS]]


# ---------- data ----------


def subject_extras(path, body_weight_n):
    """6-sensor covariance (whole trial), stance traces per foot in body-weight units,
    and the fraction of each stance phase with 0/1/2/3 points loaded."""
    trial = load_trial(path)
    time_s = trial["time"].to_numpy()
    signals = trial[SENSORS].to_numpy().T
    Xc = center_rows(signals)
    traces, counts = {}, np.zeros((len(PHASES), 4))
    for foot, index in FEET.items():
        x = signals[index]
        foot_traces = []
        for first, last in stance_bounds(x.sum(axis=0), time_s):
            grid = np.linspace(first, last - 1, POINTS)
            resampled = np.stack([np.interp(grid, np.arange(first, last), ch[first:last]) for ch in x])
            loaded = (resampled > CONTACT_THRESHOLD_N).sum(axis=0)
            for p, (a, b) in enumerate(PHASES):
                counts[p] += np.bincount(loaded[a:b], minlength=4)[:4]
            foot_traces.append(resampled / body_weight_n)
        traces[foot] = np.array(foot_traces).reshape(len(foot_traces), 3, POINTS)
    fractions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return Xc @ Xc.T, traces, fractions


# ---------- recovery ----------


def svd_prior_matrix(traces):
    """W = U2 (A U2)^-1 from the top 2 uncentred modes of (heel, midfoot, forefoot)."""
    samples = np.concatenate([t.transpose(0, 2, 1).reshape(-1, 3) for t in traces if len(t)])
    eigenvalues, U = np.linalg.eigh(samples.T @ samples)
    U2 = U[:, np.argsort(eigenvalues)[::-1][:2]]
    return U2 @ np.linalg.inv(A @ U2), np.linalg.cond(A @ U2)


def ridge_matrix(traces):
    """Least-squares W (3 x 2), no intercept (zero load in, zero load out)."""
    samples = np.concatenate([t.transpose(0, 2, 1).reshape(-1, 3) for t in traces if len(t)])
    model = Ridge(alpha=1.0, fit_intercept=False).fit(samples @ A.T, samples)
    return model.coef_


def recover(traces, W):
    """x_hat = W A x for stance traces (n, 3, points)."""
    return np.einsum("ij,jk,nkt->nit", W, A, traces)


def early_heel(traces):
    """Median over stances of the mean heel load in 5-15 % of stance."""
    if len(traces) == 0:
        return np.nan
    a, b = EARLY_HEEL
    return float(np.median(traces[:, 0, a:b + 1].mean(axis=1)))


def relative_error(real, recovered, channel=None):
    real = real if channel is None else real[:, channel]
    recovered = recovered if channel is None else recovered[:, channel]
    return float(np.linalg.norm(recovered - real) / np.linalg.norm(real))


# ---------- features for a sensor variant ----------


def variant_transform(variant, W):
    """Matrix M mapping the real 6 sensors to the variant's channels (per foot blocks)."""
    if variant == "real 6 sensors":
        return np.eye(6)
    if variant == "merged 4 signals":
        return np.block([[A, np.zeros((2, 3))], [np.zeros((2, 3)), A]])
    return np.block([[W["L"] @ A, np.zeros((3, 3))], [np.zeros((3, 3)), W["R"] @ A]])


def gait_features(covariances, M):
    """Mode-1 energy, effective modes and heel-forefoot coupling from transformed covariances.

    Coupling: 6 channels -> mean |r| of heel with midfoot and forefoot, per foot;
    4 merged channels -> mean |r| between S1 and S2, per foot (the closest analogue).
    """
    rows = []
    for cov in covariances:
        c = M @ cov @ M.T
        energies = np.clip(np.linalg.eigvalsh(c)[::-1], 0, None)
        fractions = energies / energies.sum()
        sd = np.sqrt(np.clip(np.diag(c), 1e-12, None))
        corr = c / np.outer(sd, sd)
        pairs = [(0, 1), (0, 2), (3, 4), (3, 5)] if len(c) == 6 else [(0, 1), (2, 3)]
        rows.append({"energy_mode1": fractions[0], "effective_modes": 1 / np.sum(fractions ** 2),
                     "heel_forefoot": float(np.mean([abs(corr[i, j]) for i, j in pairs]))})
    return pd.DataFrame(rows)


def variant_strides(X, M, D):
    """Strides (n, channels, 100) and ridge inputs F for a variant."""
    Xv = np.einsum("ij,njt->nit", M, X).astype(np.float32)
    return Xv, np.hstack([Xv.reshape(len(Xv), -1), D[:, None]]).astype(np.float32)


# ---------- one fold ----------


def merged_fold(X, D, stride_subject, targets, info, covariances, traces, train, test, seed):
    is_control = (info["group"] == "Control").to_numpy()
    y = (info["group"] == "PD").astype(int).to_numpy()
    groups = info["subject_id"].to_numpy()

    W, condition = {"pseudo-inverse": {"L": A_PINV, "R": A_PINV}}, {}
    W["SVD prior"], W["ridge"] = {}, {}
    for foot in FEET:
        W["SVD prior"][foot], condition[foot] = svd_prior_matrix([traces[i][foot] for i in train if is_control[i]])
        W["ridge"][foot] = ridge_matrix([traces[i][foot] for i in train])

    # Recovery error and early-stance heel load on held-out subjects
    recovery = []
    for i in test:
        real = np.concatenate([traces[i][f] for f in FEET])
        if len(real) == 0:
            continue
        row = {"subject": groups[i],
               "heel_real": np.nanmean([early_heel(traces[i][f]) for f in FEET])}
        for method in METHODS:
            rec = {f: recover(traces[i][f], W[method][f]) for f in FEET}
            joined = np.concatenate([rec[f] for f in FEET])
            row[f"error_{method}"] = relative_error(real, joined)
            row[f"heel_error_{method}"] = relative_error(real, joined, channel=0)
            row[f"heel_{method}"] = np.nanmean([early_heel(rec[f]) for f in FEET])
        recovery.append(row)

    # Classifier per variant
    scores = {}
    for variant in VARIANTS:
        method = variant.removeprefix("recovered: ")
        M = variant_transform(variant, W.get(method))
        Xv, Fv = variant_strides(X, M, D)
        table = info[["subject_id", "speed_01"]].join(gait_features(covariances, M))
        table = table.join(insole_estimates(Xv, Fv, stride_subject, targets, is_control, train, test, seed))
        model = fit_model(clone(make_models(seed)["logistic"]), table.loc[train, INSOLE], y[train], groups[train])
        scores[variant] = model.predict_proba(table.loc[test, INSOLE])[:, 1]
    learned = {m: {f: W[m][f] for f in FEET} for m in ["SVD prior", "ridge"]}
    return recovery, scores, learned, condition


# ---------- figure ----------


def plot_merged(sparsity, heel, auc_table):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.2, 1.1, 1]})
    ax = axes[0]
    labels = [f"{a}-{min(b, 100)}%" for a, b in PHASES]
    colors = ["#e4e3df", "#1baf7a", "#eda100", "#6250d6"]
    for g, group in enumerate(["Control", "PD"]):
        part = sparsity[sparsity["group"] == group].set_index("phase").reindex(labels)
        bottom = np.zeros(len(labels))
        x = np.arange(len(labels)) + (g - 0.5) * 0.4
        for k in range(4):
            values = part[f"loaded_{k}"].to_numpy()
            ax.bar(x, values, width=0.38, bottom=bottom, color=colors[k], edgecolor="#fcfcfb",
                   label=f"{k} loaded" if g == 0 else None)
            bottom += values
        ax.text(x[0], 1.02, group[0], ha="center", fontsize=8, color=GROUP_COLORS[group])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("stance phase (left bar C = control, right bar P = PD)")
    ax.set_ylabel("fraction of stance samples")
    ax.set_title("How many of heel / midfoot / forefoot carry > 20 N", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    columns = ["heel_real", *[f"heel_{m}" for m in METHODS]]
    names = ["real heel", *METHODS]
    rng = np.random.default_rng(0)
    for c, column in enumerate(columns):
        for g, group in enumerate(["Control", "PD"]):
            values = heel.loc[heel["group"] == group, column].dropna()
            x = c + (g - 0.5) * 0.36
            color = GROUP_COLORS[group]
            ax.boxplot(values, positions=[x], widths=0.3, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
                       whiskerprops={"color": color}, capprops={"color": color}, medianprops={"color": TEXT_COLOR})
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(values)), values, s=7, color=color, alpha=0.6, linewidths=0)
    for group in ["Control", "PD"]:
        ax.scatter([], [], color=GROUP_COLORS[group], s=30, label=group)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("early-stance heel load (body weight)")
    ax.set_title("Does the PD heel signal survive recovery?", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for i, row in auc_table.reset_index(drop=True).iterrows():
        ax.vlines(i, row["ci_low"], row["ci_high"], color="#6250d6", alpha=0.3, linewidth=9)
        ax.errorbar(i, row["auc_mean"], yerr=row["auc_sd"], fmt="o", color="#6250d6", capsize=3, markersize=7)
    ax.axhline(0.5, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(auc_table)))
    ax.set_xticklabels([v.replace(": ", ":\n").replace(" 6", "\n6").replace(" 4", "\n4") for v in auc_table["variant"]],
                       fontsize=8)
    ax.set_ylabel("AUC, insole-only classifier")
    ax.set_title("Classifier: real vs merged vs recovered", loc="left")
    fig.suptitle("Exploratory (post hoc): merged large sensors S1 = 1 + 4, S2 = 4 + 7 with linear recovery",
                 x=0.04, ha="left")
    fig.tight_layout()
    save(fig, "25_merged_sensors_exploratory.png")


# ---------- main ----------


def main():
    start = time.time()
    print("EXPLORATORY analysis: merged large sensors with linear recovery.\n")
    rank = np.linalg.matrix_rank(A)
    print(f"A = {A.tolist()}  rank {rank}; null space {np.round(null_space(A).ravel(), 4).tolist()} "
          f"(= (1, -1, 1)/sqrt(3) up to sign)")
    print(f"pseudo-inverse A+ =\n{np.round(A_PINV, 4)}\n")

    files, demographics = check_inputs()
    info, X_raw, X_bw, F_raw, F_bw, stride_subject, targets = load_insole_data(files, demographics)
    D = F_bw[:, -1]
    extras = Parallel(n_jobs=N_JOBS)(delayed(subject_extras)(p, bw) for p, bw in zip(files, info["body_weight_n"]))
    covariances = np.stack([e[0] for e in extras])
    traces = [e[1] for e in extras]
    print(f"loaded in {time.time() - start:.0f} s; stances per subject (both feet) median "
          f"{np.median([sum(len(t[f]) for f in FEET) for t in traces]):.0f}")

    # 1. Sparsity
    rows = []
    for (subject, group), (_, _, fractions) in zip(info[["subject_id", "group"]].itertuples(index=False), extras):
        for p, (a, b) in enumerate(PHASES):
            rows.append({"subject": subject, "group": group, "phase": f"{a}-{min(b, 100)}%",
                         **{f"loaded_{k}": fractions[p, k] for k in range(4)}})
    per_subject = pd.DataFrame(rows)
    sparsity = per_subject.groupby(["group", "phase"])[[f"loaded_{k}" for k in range(4)]].mean().reset_index()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    sparsity.to_csv(METRICS_DIR / "merged_sparsity_exploratory.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n== 1. Sparsity: mean fraction of stance samples with k of (heel, midfoot, forefoot) > 20 N ==")
    print(sparsity.round(3).to_string(index=False))

    # 2-4. Cross-validated recovery and classifiers
    y = (info["group"] == "PD").astype(int).to_numpy()
    subject_ids = info["subject_id"].to_numpy()
    splits = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, subject_ids)):
            splits.append((repeat, fold, train, test))
    results = Parallel(n_jobs=N_JOBS)(
        delayed(merged_fold)(X_bw, D, stride_subject, targets, info, covariances, traces, train, test,
                             SEED + 100 * r + f)
        for r, f, train, test in splits)
    print(f"\ncross-validation done ({time.time() - start:.0f} s)")

    meta = info[["subject_id", "study", "group"]].rename(columns={"subject_id": "subject"})
    print("\n== Learned recovery matrices W (x_hat = W b), mean over folds; pseudo-inverse for reference ==")
    for method in ["SVD prior", "ridge"]:
        for foot in FEET:
            print(f"  {method}, {foot}:\n{np.round(np.mean([r[2][method][foot] for r in results], axis=0), 3)}")
    conditions = np.array([[r[3][f] for f in FEET] for r in results])
    print(f"  condition number of A U2 (SVD prior): median {np.median(conditions):.2f}, max {conditions.max():.2f}")

    recovery = pd.DataFrame([row for r in results for row in r[0]])
    recovery = recovery.groupby("subject").mean().join(meta.set_index("subject")).reset_index()
    recovery.to_csv(METRICS_DIR / "merged_recovery_exploratory.csv", index=False)
    error_rows = []
    for method in METHODS:
        for group in ["Control", "PD"]:
            part = recovery[recovery["group"] == group]
            error_rows.append({"method": method, "group": group,
                               "relative_error_median": part[f"error_{method}"].median(),
                               "heel_relative_error_median": part[f"heel_error_{method}"].median()})
    errors = pd.DataFrame(error_rows)
    errors.to_csv(METRICS_DIR / "merged_recovery_errors_exploratory.csv", index=False)
    print("\n== 2. Recovery error vs real sensors 1, 4, 7 (held-out subjects, median over subjects) ==")
    print(errors.pivot(index="method", columns="group",
                       values=["relative_error_median", "heel_relative_error_median"]).round(3).to_string())

    heel_rows = []
    for column in ["heel_real", *[f"heel_{m}" for m in METHODS]]:
        tests = compare_groups(recovery.dropna(subset=[column]), column)
        heel_rows.append({"heel_signal": column.removeprefix("heel_"),
                          "control_median": recovery.loc[recovery["group"] == "Control", column].median(),
                          "pd_median": recovery.loc[recovery["group"] == "PD", column].median(),
                          **{f"p_pd_higher_{s}": tests.loc[s, "p_pd_higher"] for s in tests.index},
                          "mann_whitney_p_All": tests.loc["All", "p_value"]})
    heel_table = pd.DataFrame(heel_rows)
    heel_table.to_csv(METRICS_DIR / "merged_heel_signal_exploratory.csv", index=False)
    print("\n== 3. Early-stance heel load (5-15 % of stance, body weight): real vs recovered; "
          "P(PD > Ctrl) < 0.5 = PD lower ==")
    print(heel_table.round(3).to_string(index=False))

    predictions = {v: pd.concat([
        pd.DataFrame({"repeat": r, "fold": f, "subject": subject_ids[test], "y": y[test],
                      "study": info["study"].to_numpy()[test], "score": res[1][v]})
        for (r, f, _, test), res in zip(splits, results)], ignore_index=True) for v in VARIANTS}
    score_table = pd.DataFrame({"y": subject_scores(predictions[VARIANTS[0]])["y"],
                                **{v: subject_scores(p)["score"] for v, p in predictions.items()}})
    boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)
    auc_rows = []
    for v, p in predictions.items():
        per_repeat = summarize(p)["auc"]
        per_study = summarize(p, by="study").groupby("study")["auc"].mean()
        d = boot[v] - boot[VARIANTS[0]]
        auc_rows.append({"variant": v, "auc_mean": per_repeat.mean(), "auc_sd": per_repeat.std(ddof=1),
                         "auc_pooled": boot[v].mean(), "ci_low": boot[v].quantile(0.025),
                         "ci_high": boot[v].quantile(0.975),
                         **{f"auc_{s}": per_study[s] for s in ["Ga", "Ju", "Si"]},
                         "difference_vs_real": d.mean(), "difference_ci_low": d.quantile(0.025),
                         "difference_ci_high": d.quantile(0.975)})
    auc_table = pd.DataFrame(auc_rows)
    auc_table.to_csv(METRICS_DIR / "merged_classifier_exploratory.csv", index=False)
    print("\n== 4. Insole-only classifier: real 6 sensors vs merged 4 signals vs recovered 6 sensors ==")
    print(auc_table.round(3).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_merged(sparsity, recovery, auc_table)
    print(f"\nEXPLORATORY. total time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
