"""Train, check and export the insole-only model for the firmware.

1. Extract every recording with the device's exact steps (src/firmware_reference.py).
2. Cross-validated AUC of this exact pipeline (10 x 5 subject-wise splits, as in Step 10).
3. Final model on all subjects: ridge speed and ridge log-TUG (scalers folded in), healthy
   template (mean control steady stride), logistic regression on cross-fitted estimates.
4. Storage: the three 600-value arrays as int16 + one scale each; 15 other numbers as floats.
5. Write firmware/include/model_params.h, firmware/model_params.json, firmware/include/
   demo_recordings.h (GaCo01 and GaPt29, demo pair 3) and firmware/reference_outputs.csv
   (Python reference results with the stored parameters, for checking the C code).

Run as part of slurm/firmware.sbatch (not on the login node).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import fit_model, summarize  # noqa: E402
from src.firmware_reference import FEATURES, device_input, extract, gait_features, score  # noqa: E402
from src.load import DATA_DIR, FILENAME_PATTERN, load_demographics, parse_filename  # noqa: E402
from src.run_insole_regression import insole_estimates, ridge_search  # noqa: E402
from src.run_ml import make_models  # noqa: E402

FIRMWARE = Path(__file__).resolve().parent
N_JOBS = 16
N_SPLITS, N_REPEATS, SEED = 5, 10, 0
DEMO_SUBJECTS = [("GaCo01", "healthy"), ("GaPt29", "Parkinson's")]


def load_subject(path):
    info = parse_filename(path)
    extracted = extract(device_input(path))
    return info["subject_id"], extracted


def training_arrays(subjects, extracted):
    """Steady, analysed strides of every subject: X (n, 6, 100), F (n, 601), stride_subject."""
    X, F, owner = [], [], []
    for k, subject in enumerate(subjects):
        e = extracted[subject]
        keep = e["steady"] & np.isfinite(e["logdur"])
        X.append(e["X"][keep])
        F.append(np.hstack([e["X"][keep].reshape(keep.sum(), 6 * 100), e["logdur"][keep][:, None]]))
        owner.append(np.full(keep.sum(), k))
    return np.concatenate(X), np.concatenate(F), np.concatenate(owner)


def fold_auc(X, F, owner, targets, info, train, test, seed):
    is_control = (info["group"] == "Control").to_numpy()
    y = (info["group"] == "PD").astype(int).to_numpy()
    table = info.join(insole_estimates(X, F, owner, targets, is_control, train, test, seed))
    model = fit_model(clone(make_models(seed)["logistic"]), table.loc[train, FEATURES], y[train],
                      info["subject_id"].to_numpy()[train])
    return model.predict_proba(table.loc[test, FEATURES])[:, 1]


def folded(scaler, coef, intercept):
    w = np.ravel(coef) / scaler.scale_
    return w, float(np.ravel(intercept)[0] - np.sum(w * scaler.mean_))


def int16_array(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(values)) / 32767)
    return np.clip(np.round(values / scale), -32767, 32767).astype(np.int16), scale


def c_int16(name, values):
    rows = [", ".join(str(int(v)) for v in values[i:i + 20]) for i in range(0, len(values), 20)]
    return f"static const int16_t {name}[{len(values)}] = {{\n  " + ",\n  ".join(rows) + "\n};\n"


def c_doubles(name, values):
    return f"static const double {name}[{len(values)}] = {{" + ", ".join(repr(float(v)) for v in values) + "};\n"


def main():
    start = time.time()
    demographics = load_demographics()
    files = sorted(p for p in DATA_DIR.glob("*_01.txt") if FILENAME_PATTERN.match(p.name))
    loaded = dict(Parallel(n_jobs=N_JOBS)(delayed(load_subject)(p) for p in files))
    meta = pd.DataFrame([parse_filename(p) for p in files])[["subject_id", "study", "group"]]
    gait = pd.DataFrame([{"subject_id": s, **gait_features(e["cov"])} for s, e in loaded.items()])
    info = meta.merge(gait, on="subject_id").merge(demographics[["subject_id", "speed_01", "tug_s"]],
                                                   on="subject_id", how="left")
    subjects = info["subject_id"].tolist()
    X, F, owner = training_arrays(subjects, loaded)
    targets = np.column_stack([info["speed_01"], np.log(info["tug_s"])])
    y = (info["group"] == "PD").astype(int).to_numpy()
    print(f"extracted {len(subjects)} recordings ({len(X)} steady strides) with the device's steps "
          f"in {time.time() - start:.0f} s")

    # 2. Cross-validated AUC of this exact pipeline
    splits = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, info["subject_id"])):
            splits.append((repeat, fold, train, test))
    scores = Parallel(n_jobs=N_JOBS)(delayed(fold_auc)(X, F, owner, targets, info, tr, te, SEED + 100 * r + f)
                                     for r, f, tr, te in splits)
    predictions = pd.concat([pd.DataFrame({"repeat": r, "fold": f, "subject": info["subject_id"].to_numpy()[te],
                                           "y": y[te], "score": s})
                             for (r, f, _, te), s in zip(splits, scores)], ignore_index=True)
    auc = summarize(predictions)["auc"]
    print(f"cross-validated AUC of the firmware pipeline (float64): {auc.mean():.3f} +/- {auc.std(ddof=1):.3f} "
          f"(Step 10b raw insole: 0.812)")

    # 3. Final model on all subjects
    all_subjects = np.arange(len(subjects))
    is_control = (info["group"] == "Control").to_numpy()
    table = info.join(insole_estimates(X, F, owner, targets, is_control, all_subjects, np.array([], dtype=int), SEED))
    classifier = fit_model(clone(make_models(SEED)["logistic"]), table[FEATURES], y, info["subject_id"].to_numpy())
    params = {}
    for k, name in enumerate(["speed", "tug"]):
        rows = np.isfinite(targets[:, k][owner])
        pipe = ridge_search().fit(F[rows], targets[:, k][owner[rows]], groups=owner[rows]).best_estimator_
        w, b = folded(pipe["scale"], pipe["ridge"].coef_, pipe["ridge"].intercept_)
        params[f"{name}_w"], params[f"{name}_logdur_w"], params[f"{name}_b"] = w[:600], float(w[600]), b
    params["template"] = X[is_control[owner]].mean(axis=0).ravel()
    pipe = classifier.best_estimator_
    params["medians"] = pipe["impute"].statistics_
    params["logit_w"], params["logit_b"] = folded(pipe["scale"], pipe["model"].coef_, pipe["model"].intercept_)

    # 4. Storage
    stored, header = {}, []
    for name in ["speed_w", "tug_w", "template"]:
        q, scale = int16_array(params[name])
        stored[name] = q.astype(np.float64) * scale
        header += [c_int16(f"GM_{name.upper()}", q), f"static const double GM_{name.upper()}_SCALE = {scale!r};\n"]
    for name in ["speed_logdur_w", "speed_b", "tug_logdur_w", "tug_b", "logit_b"]:
        stored[name] = params[name]
        header.append(f"static const double GM_{name.upper()} = {params[name]!r};\n")
    for name in ["medians", "logit_w"]:
        stored[name] = np.asarray(params[name], dtype=np.float64)
        header.append(c_doubles(f"GM_{name.upper()}", stored[name]))
    n_int16, n_float = 3 * 600, 5 + 12
    (FIRMWARE / "include").mkdir(exist_ok=True)
    (FIRMWARE / "include" / "model_params.h").write_text(
        "/* Insole-only PD model for the firmware. Written by firmware/export_model.py; do not edit.\n"
        f"   {n_int16} int16 values (3 arrays x 600) + {n_float} floating-point values.\n"
        "   Features: est_speed, est_log_tug, template_dev, energy_mode1, effective_modes, heel_forefoot. */\n"
        "#ifndef MODEL_PARAMS_H\n#define MODEL_PARAMS_H\n#include <stdint.h>\n\n" + "\n".join(header) + "\n#endif\n")
    (FIRMWARE / "model_params.json").write_text(json.dumps(
        {k: (np.asarray(v).tolist() if np.ndim(v) else v) for k, v in stored.items()}))
    print(f"model: {n_int16} int16 + {n_float} float values = {2 * n_int16 + 4 * n_float} bytes "
          f"(float32 for the {n_float}); written to firmware/include/model_params.h")

    # 5. Reference outputs (stored parameters) and demo recordings
    reference = pd.DataFrame([{"subject_id": s, "group": g, **score(loaded[s], stored)}
                              for s, g in zip(subjects, info["group"])])
    reference.to_csv(FIRMWARE / "reference_outputs.csv", index=False)
    print(f"reference outputs written; in-sample AUC with stored parameters "
          f"{roc_auc_score(reference['group'] == 'PD', reference['logit']):.3f} (optimistic: trained on these subjects)")
    stored_vs_float = np.max(np.abs(reference["logit"] - pd.Series(
        [score(loaded[s], {**params, "template": params["template"]})["logit"] for s in subjects])))
    print(f"max |logit| change from int16 storage: {stored_vs_float:.4f}")

    demo = ["typedef struct { const char *name; const char *group; long n; const int16_t (*data)[6]; } gm_recording;\n"]
    for k, (subject, group) in enumerate(DEMO_SUBJECTS):
        samples = device_input(DATA_DIR / f"{subject}_01.txt")
        rows = ",".join("{" + ",".join(str(int(v)) for v in row) + "}" for row in samples)
        demo.append(f"static const int16_t GM_DEMO_{k}[{len(samples)}][6] = {{{rows}}};\n")
    demo.append("static const gm_recording GM_DEMO[] = {"
                + ", ".join(f'{{"{s}", "{g}", sizeof(GM_DEMO_{k}) / sizeof(GM_DEMO_{k}[0]), GM_DEMO_{k}}}'
                            for k, (s, g) in enumerate(DEMO_SUBJECTS)) + "};\n")
    demo.append(f"#define GM_DEMO_COUNT {len(DEMO_SUBJECTS)}\n")
    (FIRMWARE / "include" / "demo_recordings.h").write_text(
        "/* Demo recordings (6 sensors L1, L4, L7, R1, R4, R7; int16, 0.1 N; 100 Hz), PhysioNet\n"
        "   Gait in Parkinson's Disease. Written by firmware/export_model.py; do not edit. */\n"
        "#ifndef DEMO_RECORDINGS_H\n#define DEMO_RECORDINGS_H\n#include <stdint.h>\n" + "".join(demo) + "#endif\n")
    print(f"demo recordings: {', '.join(s for s, _ in DEMO_SUBJECTS)}; total {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
