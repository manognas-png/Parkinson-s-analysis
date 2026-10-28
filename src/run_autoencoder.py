"""Step 9: a small "gait deviation" autoencoder for a microcontroller.

Input: steady strides from the anatomical 6-sensor set (sensors 1, 4, 7 per foot),
each resampled to 100 points and divided by body weight -> 6 x 100 per stride.
A 1D conv autoencoder (~3.9k parameters) is trained on CONTROL strides only;
a subject's deviation score is the median reconstruction error over its strides.

Run as a SLURM job (not on the login node):
    sbatch --account=bioinfmath540f26_class slurm/autoencoder.sbatch
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import bootstrap_auc, subject_scores, summarize
from src.gait import resample_strides, steady_stride_bounds
from src.load import DATA_DIR, FILENAME_PATTERN, load_demographics, load_trial, parse_filename
from src.plot_explore import FIGURE_DIR, GROUP_COLORS, METRICS_DIR, RESULTS_DIR, TEXT_COLOR, save
from src.run_sensor_selection import ANATOMICAL

N_JOBS = 16  # matches --cpus-per-task in slurm/autoencoder.sbatch
DEVICE = "cpu"  # 16 tiny models in parallel on CPU cores; "cuda" would train on a GPU (one at a time)
N_SPLITS = 5
N_REPEATS = 10
SEED = 0
N_POINTS = 100
SENSORS = ANATOMICAL[6]  # L1, L4, L7, R1, R4, R7
VALIDATION_FRACTION = 0.15  # of training-control subjects, for early stopping
MAX_EPOCHS = 1000  # first run stopped at the 200 cap while still improving
PATIENCE = 30  # epochs without validation improvement before stopping
PCA_COMPONENTS = 10
N_BOOT = 2000
GRAVITY = 9.81
EXPECTED_SUBJECTS = 165
SCORES = {"ae": "autoencoder", "ae_int8": "autoencoder (int8)", "mean_distance": "distance to control mean",
          "pca": f"PCA reconstruction ({PCA_COMPONENTS} comp.)"}
EXISTING_FEATURES = {
    "svd_features.csv": ["energy_mode1_centered", "effective_n_modes_centered"],
    "network_features.csv": ["heel_forefoot_corr", "mean_edge_weight"],
    "dmd_features.csv": ["dmd_dominant_freq", "dmd_reconstruction_error"],
    "steady_gait_cv.csv": ["swing_cv", "stride_cv"],
}
MCU = {"ram_bytes": 256 * 1024, "flash_bytes": 1024 * 1024, "clock_hz": 64e6,
       "cycles_per_mac_float32": 2.0, "cycles_per_mac_int8": 0.5}


def fail(message):
    sys.exit(f"ERROR: {message}")


def check_inputs():
    """Stop before any work if data, earlier feature files, or PyTorch are missing."""
    try:
        import torch  # noqa: F401
    except ImportError:
        fail("PyTorch is not installed in .venv. On the login node run:\n"
             "  source .venv/bin/activate && pip install -r requirements.txt")
    if not (DATA_DIR / "demographics.txt").exists():
        fail(f"missing {DATA_DIR / 'demographics.txt'}")
    files = sorted(p for p in DATA_DIR.glob("*_01.txt") if FILENAME_PATTERN.match(p.name))
    if len(files) != EXPECTED_SUBJECTS:
        fail(f"expected {EXPECTED_SUBJECTS} _01 trial files in {DATA_DIR}, found {len(files)}")
    for filename, columns in EXISTING_FEATURES.items():
        path = METRICS_DIR / filename
        if not path.exists():
            fail(f"missing {path} (needed for the feature correlations)")
        absent = [c for c in columns if c not in pd.read_csv(path, nrows=0).columns]
        if absent:
            fail(f"{path} is missing columns {absent}")
    demographics = load_demographics()
    needed = {"speed_01", "weight_kg", "hoehn_yahr", "updrs"}
    if not needed <= set(demographics.columns):
        fail(f"demographics is missing {sorted(needed - set(demographics.columns))}")
    return files, demographics


# ---------- data ----------


def subject_strides(path, weight_kg):
    """Steady strides (n, 6, 100) of the 6 sensors, in units of body weight.

    Heel strikes come from the sum of the kept LEFT sensors (L1 + L4 + L7): a
    6-sensor insole has no total-force channel. Body weight = weight_kg * g; if
    weight is missing, the mean total vertical force over the trial is used (in
    steady walking it averages to body weight).
    """
    info = parse_filename(path)
    trial = load_trial(path)
    time_s = trial["time"].to_numpy()
    signals = trial[SENSORS].to_numpy().T
    proxy = trial[[s for s in SENSORS if s.startswith("L")]].sum(axis=1).to_numpy()
    mean_total_force = (trial["L_total"] + trial["R_total"]).mean()
    has_weight = np.isfinite(weight_kg)
    body_weight_n = weight_kg * GRAVITY if has_weight else mean_total_force
    strides = resample_strides(signals, time_s, steady_stride_bounds(proxy, time_s), N_POINTS) / body_weight_n
    return info["subject_id"], strides.astype(np.float32), has_weight, mean_total_force


# ---------- one cross-validation fold ----------


def run_fold(X, stride_subject, subject_ids, is_control, train, test, seed):
    """Train on training-fold controls; score every test-fold subject with all methods."""
    import torch

    from src.models import FakeQuantAutoencoder, squared_errors, train_autoencoder

    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    train_controls = subject_ids[train][is_control[train]]
    n_val = max(1, int(round(VALIDATION_FRACTION * len(train_controls))))
    val_subjects = set(rng.choice(train_controls, n_val, replace=False))
    fit_subjects = set(train_controls) - val_subjects
    fit_mask = np.isin(stride_subject, list(fit_subjects))
    val_mask = np.isin(stride_subject, list(val_subjects))

    model, best_epoch, val_loss = train_autoencoder(X[fit_mask], X[val_mask], seed, max_epochs=MAX_EPOCHS,
                                                    patience=PATIENCE, device=DEVICE)
    calibration = X[fit_mask][rng.choice(fit_mask.sum(), min(2000, fit_mask.sum()), replace=False)]
    quantized = FakeQuantAutoencoder(model, calibration)

    # Non-neural baselines, fitted on the same training controls (fit + validation)
    train_flat = X[fit_mask | val_mask].reshape((fit_mask | val_mask).sum(), -1).astype(np.float64)
    mean_stride = train_flat.mean(axis=0)
    centered = train_flat - mean_stride
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    components = eigenvectors[:, np.argsort(eigenvalues)[::-1][:PCA_COMPONENTS]]

    records, maps = [], {}
    for subject in subject_ids[test]:
        strides = X[stride_subject == subject]
        if len(strides) == 0:
            continue
        flat = strides.reshape(len(strides), -1).astype(np.float64) - mean_stride
        errors_ae = squared_errors(model, strides)
        errors_q = squared_errors(quantized, strides)
        records.append({
            "subject": subject,
            "ae": float(np.median(errors_ae.mean(axis=(1, 2)))),
            "ae_int8": float(np.median(errors_q.mean(axis=(1, 2)))),
            "mean_distance": float(np.median((flat ** 2).mean(axis=1))),
            "pca": float(np.median(((flat - flat @ components @ components.T) ** 2).mean(axis=1))),
            "n_strides": len(strides),
        })
        maps[subject] = errors_ae.mean(axis=0)  # (6, 100): where this subject's error sits
    return records, maps, {"best_epoch": best_epoch, "val_loss": val_loss, "n_fit_strides": int(fit_mask.sum())}


# ---------- figures ----------


def plot_scores(subjects):
    """Figure 16: per-subject deviation scores by study and group, network vs baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, score in zip(axes, ["ae", "mean_distance", "pca"]):
        for s, study in enumerate(["Ga", "Ju", "Si"]):
            for g, group in enumerate(["Control", "PD"]):
                values = subjects.loc[(subjects["study"] == study) & (subjects["group"] == group), score].dropna()
                x = s + (g - 0.5) * 0.4
                color = GROUP_COLORS[group]
                ax.boxplot(values, positions=[x], widths=0.3, patch_artist=True, showfliers=False,
                           boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
                           whiskerprops={"color": color}, capprops={"color": color},
                           medianprops={"color": TEXT_COLOR})
                ax.scatter(x + np.random.default_rng(0).uniform(-0.07, 0.07, len(values)), values,
                           s=9, color=color, alpha=0.7, linewidths=0, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Ga", "Ju", "Si"])
        ax.set_title(SCORES[score], loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Deviation score (median stride error, body-weight units², log)")
    for group in ["Control", "PD"]:
        axes[2].scatter([], [], color=GROUP_COLORS[group], s=40, label=group)
    axes[2].legend(frameon=False)
    fig.suptitle("Gait deviation scores (models trained on control strides only; held-out subjects)",
                 x=0.06, ha="left")
    fig.tight_layout()
    save(fig, "16_autoencoder_scores.png")


def plot_where(excess, mean_strides):
    """Figure 17: where PD deviates (excess error, sensor x gait phase) and mean strides."""
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(3, 6, height_ratios=[1.1, 1, 1])
    ax = fig.add_subplot(grid[0, :])
    limit = np.abs(excess).max()
    image = ax.imshow(excess, aspect="auto", cmap="BrBG_r", vmin=-limit, vmax=limit,
                      extent=[0, 100, len(SENSORS) - 0.5, -0.5])
    ax.set_yticks(range(len(SENSORS)))
    ax.set_yticklabels(SENSORS)
    ax.set_xlabel("Gait cycle (% from left heel strike)")
    ax.grid(False)
    ax.set_title("Extra reconstruction error in PD (PD mean − control mean, held-out subjects)", loc="left")
    fig.colorbar(image, ax=ax, label="Δ squared error", pad=0.01)
    phase = np.arange(N_POINTS)
    for k, sensor in enumerate(SENSORS):
        ax_s = fig.add_subplot(grid[1 + k // 3, (k % 3) * 2:(k % 3) * 2 + 2])
        for group in ["Control", "PD"]:
            curves = mean_strides[group][:, k, :]
            mean, sd = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
            ax_s.plot(phase, mean, color=GROUP_COLORS[group], linewidth=2, label=group)
            ax_s.fill_between(phase, mean - sd, mean + sd, color=GROUP_COLORS[group], alpha=0.15, linewidth=0)
        ax_s.set_title(sensor, loc="left")
        ax_s.set_xlim(0, 99)
        if k % 3 == 0:
            ax_s.set_ylabel("force / body weight")
    ax_s.legend(frameon=False, fontsize=8)
    fig.suptitle("Where the PD deviation is: sensors and gait phases (anatomical 6-sensor set)", x=0.06, ha="left")
    fig.tight_layout()
    save(fig, "17_autoencoder_where.png")


# ---------- main ----------


def main():
    start = time.time()
    files, demographics = check_inputs()
    weights = demographics.set_index("subject_id")["weight_kg"]

    # 1. Strides
    extracted = Parallel(n_jobs=N_JOBS)(
        delayed(subject_strides)(p, weights.get(parse_filename(p)["subject_id"], np.nan)) for p in files)
    subject_ids = np.array([e[0] for e in extracted])
    X = np.concatenate([e[1] for e in extracted if len(e[1])])
    stride_subject = np.concatenate([[e[0]] * len(e[1]) for e in extracted if len(e[1])])
    info = pd.DataFrame({"subject_id": subject_ids, "n_strides": [len(e[1]) for e in extracted],
                         "has_weight": [e[2] for e in extracted],
                         "mean_total_force_n": [e[3] for e in extracted]})
    meta = pd.DataFrame([parse_filename(p) for p in files])[["subject_id", "study", "group"]]
    info = info.merge(meta, on="subject_id").merge(
        demographics[["subject_id", "speed_01", "weight_kg", "hoehn_yahr", "updrs"]], on="subject_id", how="left")
    both = info.dropna(subset=["weight_kg"])
    rho, _ = spearmanr(both["weight_kg"] * GRAVITY, both["mean_total_force_n"])
    print(f"{len(X)} steady strides from {(info['n_strides'] > 0).sum()} subjects "
          f"(median {info['n_strides'].median():.0f} per subject; none for {list(info.loc[info['n_strides'] == 0, 'subject_id'])})")
    print(f"body weight from demographics for {info['has_weight'].sum()} subjects; mean total force used for "
          f"{list(info.loc[~info['has_weight'], 'subject_id'])}; Spearman(weight*g, mean total force) = {rho:.2f}")
    print(f"strides loaded in {time.time() - start:.0f} s\n")

    # 2. Cross-validation: same subject-wise splits as every earlier step
    y = (info["group"] == "PD").astype(int).to_numpy()
    is_control = y == 0
    jobs = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, subject_ids)):
            jobs.append((repeat, fold, train, test))
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_fold)(X, stride_subject, subject_ids, is_control, train, test, SEED + 100 * repeat + fold)
        for repeat, fold, train, test in jobs)
    training = pd.DataFrame([r[2] for r in results])
    print(f"training done ({time.time() - start:.0f} s): best epoch median {training['best_epoch'].median():.0f} "
          f"[{training['best_epoch'].min()}-{training['best_epoch'].max()}], validation MSE median "
          f"{training['val_loss'].median():.2e}, training strides per fold median {training['n_fit_strides'].median():.0f}")

    rows, maps = [], {}
    for (repeat, fold, _, _), (records, fold_maps, _) in zip(jobs, results):
        rows += [{"repeat": repeat, "fold": fold, **r} for r in records]
        for subject, m in fold_maps.items():
            maps.setdefault(subject, []).append(m)
    scores = pd.DataFrame(rows).merge(info[["subject_id", "study", "group"]], left_on="subject", right_on="subject_id")
    scores["y"] = (scores["group"] == "PD").astype(int)

    # 3. Per-subject scores (mean over repeats) and AUC
    subjects = scores.groupby("subject_id")[list(SCORES)].mean().reset_index().merge(info, on="subject_id")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    subjects.to_csv(METRICS_DIR / "autoencoder_scores.csv", index=False)

    auc_rows, columns = [], {}
    for score in SCORES:
        predictions = scores.rename(columns={score: "score"})[["repeat", "fold", "subject", "y", "study", "score"]]
        per_repeat = summarize(predictions)["auc"]
        per_study = summarize(predictions, by="study").groupby("study")["auc"].mean()
        columns[score] = subject_scores(predictions)
        auc_rows.append({"score": score, "auc_mean": per_repeat.mean(), "auc_sd_repeats": per_repeat.std(ddof=1),
                         **{f"auc_{s}": per_study.get(s, np.nan) for s in ["Ga", "Ju", "Si"]}})
    score_table = pd.DataFrame({"y": columns["ae"]["y"], **{s: c["score"] for s, c in columns.items()}})
    boot = bootstrap_auc(score_table, n_boot=N_BOOT, seed=SEED)
    auc_table = pd.DataFrame(auc_rows)
    auc_table["auc_pooled"] = [boot[s].mean() for s in SCORES]
    auc_table["ci_low"] = [boot[s].quantile(0.025) for s in SCORES]
    auc_table["ci_high"] = [boot[s].quantile(0.975) for s in SCORES]
    auc_table.to_csv(METRICS_DIR / "autoencoder_auc.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== AUC of the per-subject deviation score (higher score = more PD-like) ==")
    print(auc_table.round(3).to_string(index=False))
    print("\n== Paired bootstrap differences in pooled AUC ==")
    for a, b in [("ae", "mean_distance"), ("ae", "pca"), ("ae_int8", "ae")]:
        d = boot[a] - boot[b]
        print(f"  {a} - {b}: {d.mean():+.3f} (95% CI {d.quantile(0.025):+.3f} to {d.quantile(0.975):+.3f}); "
              f"{a} better in {(d > 0).mean():.0%} of draws")

    # 4. Raw vs speed-adjusted comparison (Huber, controls only)
    from src.plot_svd import compare_groups
    from src.speed_adjust import speed_adjusted_residuals

    comparison_rows = []
    for score in SCORES:
        subjects[f"{score}_adjusted"], coefficients = speed_adjusted_residuals(subjects, score)
        raw = compare_groups(subjects.dropna(subset=[score]), score)
        adjusted = compare_groups(subjects.dropna(subset=[f"{score}_adjusted"]), f"{score}_adjusted")
        for study in raw.index:
            comparison_rows.append({"score": score, "study": study,
                                    "raw": f"{raw.loc[study, 'p_pd_higher']:.2f} ({raw.loc[study, 'p_value']:.3f})",
                                    "adjusted": f"{adjusted.loc[study, 'p_pd_higher']:.2f} "
                                                f"({adjusted.loc[study, 'p_value']:.3f})"})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(METRICS_DIR / "autoencoder_comparison.csv", index=False)
    view = comparison.pivot(index="score", columns="study", values=["raw", "adjusted"]).reindex(list(SCORES))
    view = view.swaplevel(axis=1)[[(s, k) for s in ["All", "Ga", "Ju", "Si"] for k in ["raw", "adjusted"]]]
    print("\n== P(PD > Ctrl) (Mann-Whitney p): raw vs adjusted for speed + study (Huber, controls only) ==")
    print(view.to_string())

    # 5. Severity and existing features
    patients = subjects[subjects["group"] == "PD"]
    print("\n== Spearman correlation with severity, PD only: rho (p) ==")
    for score in ["ae", "mean_distance", "pca"]:
        parts = []
        for name, part in [("All", patients), *patients.groupby("study")]:
            for clinical in ["hoehn_yahr", "updrs"]:
                pair = part[[score, clinical]].dropna()
                r, p = spearmanr(pair[score], pair[clinical])
                parts.append(f"{name} {clinical} {r:+.2f} ({p:.3f})")
        print(f"  {score:14s} " + " | ".join(parts))

    existing = subjects[["subject_id", "group", "ae", "speed_01"]]
    for filename, columns_needed in EXISTING_FEATURES.items():
        table = pd.read_csv(METRICS_DIR / filename)
        existing = existing.merge(table[["subject_id", *columns_needed]], on="subject_id", how="left")
    print("\n== Spearman correlation of the autoencoder score with existing features: rho (p) ==")
    for feature in ["speed_01", *[c for cs in EXISTING_FEATURES.values() for c in cs]]:
        parts = []
        for name, part in [("all", existing), ("controls", existing[existing["group"] == "Control"]),
                           ("PD", existing[existing["group"] == "PD"])]:
            pair = part[["ae", feature]].dropna()
            r, p = spearmanr(pair["ae"], pair[feature])
            parts.append(f"{name} {r:+.2f} ({p:.3f})")
        print(f"  {feature:28s} " + " | ".join(parts))

    # 6. Where is the deviation? (held-out error maps, averaged over repeats)
    subject_maps = {s: np.mean(m, axis=0) for s, m in maps.items()}
    group_of = info.set_index("subject_id")["group"]
    by_group = {g: np.stack([m for s, m in subject_maps.items() if group_of[s] == g]) for g in ["Control", "PD"]}
    excess = by_group["PD"].mean(axis=0) - by_group["Control"].mean(axis=0)
    positive = np.clip(excess, 0, None)
    print("\n== Where PD's extra error sits (share of positive excess error) ==")
    print("  by sensor: " + ", ".join(f"{s} {v:.0%}" for s, v in zip(SENSORS, positive.sum(axis=1) / positive.sum())))
    bins = positive.reshape(len(SENSORS), 10, 10).sum(axis=2) / positive.sum()
    print("  by sensor x 10% phase bin (rows: sensors; columns: 0-10%, ..., 90-100% of the cycle from left heel strike):")
    print(pd.DataFrame(bins, index=SENSORS, columns=[f"{10 * i}-{10 * i + 10}" for i in range(10)])
          .map(lambda v: f"{v:.1%}").to_string())

    mean_strides = {g: np.stack([X[stride_subject == s].mean(axis=0) for s in info.loc[
        (info["group"] == g) & (info["n_strides"] > 0), "subject_id"]]) for g in ["Control", "PD"]}
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_scores(subjects)
    plot_where(excess, mean_strides)

    # 7. Microcontroller fit (estimates)
    from src.models import GaitAutoencoder, cost_report

    cost = cost_report(GaitAutoencoder(n_channels=len(SENSORS)), n_channels=len(SENSORS), n_points=N_POINTS)
    ms_float = cost["macs_per_stride"] * MCU["cycles_per_mac_float32"] / MCU["clock_hz"] * 1000
    ms_int8 = cost["macs_per_stride"] * MCU["cycles_per_mac_int8"] / MCU["clock_hz"] * 1000
    auc_float = auc_table.set_index("score").loc["ae", "auc_mean"]
    auc_int8 = auc_table.set_index("score").loc["ae_int8", "auc_mean"]
    mcu_rows = [
        {"quantity": "parameters", "float32": cost["parameters"], "int8": cost["parameters"]},
        {"quantity": "MACs per stride", "float32": cost["macs_per_stride"], "int8": cost["macs_per_stride"]},
        {"quantity": "flash (bytes)", "float32": cost["flash_bytes_float32"], "int8": cost["flash_bytes_int8"]},
        {"quantity": "flash (% of 1 MB)", "float32": 100 * cost["flash_bytes_float32"] / MCU["flash_bytes"],
         "int8": 100 * cost["flash_bytes_int8"] / MCU["flash_bytes"]},
        {"quantity": "RAM (bytes)", "float32": cost["ram_bytes_float32"], "int8": cost["ram_bytes_int8"]},
        {"quantity": "RAM (% of 256 KB)", "float32": 100 * cost["ram_bytes_float32"] / MCU["ram_bytes"],
         "int8": 100 * cost["ram_bytes_int8"] / MCU["ram_bytes"]},
        {"quantity": "time per stride at 64 MHz (ms, estimate)", "float32": ms_float, "int8": ms_int8},
        {"quantity": "AUC (mean over repeats)", "float32": auc_float, "int8": auc_int8},
    ]
    mcu = pd.DataFrame(mcu_rows)
    mcu.to_csv(METRICS_DIR / "autoencoder_mcu.csv", index=False)
    print("\n== Microcontroller fit (estimates; reference MCU: 256 KB RAM, 1 MB flash, 64 MHz) ==")
    print(mcu.round(3).to_string(index=False))
    print(f"  assumptions: {MCU['cycles_per_mac_float32']} cycles/MAC float32 (FPU), "
          f"{MCU['cycles_per_mac_int8']} cycles/MAC int8 (SIMD kernels); RAM includes a raw "
          "buffer of 2 s x 6 sensors of int16 readings")
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
