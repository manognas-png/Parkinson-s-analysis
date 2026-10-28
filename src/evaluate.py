"""Subject-wise model evaluation, reusable for any model with predict_proba.

Works with one OR many rows per subject:
  * rows are split by `groups` (subject IDs), so a subject's rows are never in
    both the training and the test part;
  * predicted probabilities are averaged per subject before computing metrics,
    so every subject counts once, however many rows (trials, windows) it has.

Main entry points:
  repeated_group_cv     repeated StratifiedGroupKFold, out-of-fold predictions
  leave_one_group_out   train on all but one group (e.g. study), test on it
  summarize             subject-level metrics per repeat (optionally per subgroup)
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._search import BaseSearchCV

THRESHOLD = 0.5  # probability cut-off for the thresholded metrics
METRICS = ["auc", "balanced_accuracy", "sensitivity", "specificity"]


def binary_metrics(y_true, score, threshold=THRESHOLD):
    """AUC from scores; balanced accuracy, sensitivity, specificity at `threshold`.

    Label 1 = positive class (PD). AUC is NaN if only one class is present.
    """
    y_true = np.asarray(y_true)
    predicted = (np.asarray(score) >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "auc": roc_auc_score(y_true, score) if both_classes else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted) if both_classes else np.nan,
        "sensitivity": recall_score(y_true, predicted, pos_label=1) if (y_true == 1).any() else np.nan,
        "specificity": recall_score(y_true, predicted, pos_label=0) if (y_true == 0).any() else np.nan,
    }


def fit_model(model, X, y, groups):
    """Fit; hyperparameter searches also get `groups` so their inner CV is subject-wise."""
    if isinstance(model, BaseSearchCV):
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


def _fit_and_predict(model, X, y, groups, train, test, permutation_repeats, seed):
    """Fit a fresh copy on `train` rows; return test-row scores (and importances)."""
    fitted = fit_model(clone(model), X.iloc[train], y[train], groups[train])
    scores = fitted.predict_proba(X.iloc[test])[:, 1]

    importances = None
    if permutation_repeats:
        # Row-level AUC drop on the held-out fold when one feature is shuffled
        result = permutation_importance(fitted, X.iloc[test], y[test], scoring="roc_auc",
                                        n_repeats=permutation_repeats, random_state=seed, n_jobs=1)
        importances = result.importances_mean
    return scores, importances


def _check_inputs(X, y, groups):
    y = np.asarray(y)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y and groups must have the same number of rows")
    labels_per_group = pd.Series(y).groupby(groups).nunique()
    if (labels_per_group > 1).any():
        raise ValueError(f"Some groups have more than one label: {list(labels_per_group[labels_per_group > 1].index)}")
    return y, groups


def _predictions_frame(X, y, groups, row_info):
    frame = pd.DataFrame({"row": np.arange(len(X)), "subject": groups, "y": y})
    if row_info is not None:
        frame = pd.concat([frame, row_info.reset_index(drop=True)], axis=1)
    return frame


def repeated_group_cv(model, X, y, groups, n_splits=5, n_repeats=10, seed=0,
                      n_jobs=1, permutation_repeats=0, row_info=None):
    """Repeated StratifiedGroupKFold: folds keep subjects together and balance labels.

    Each repeat reshuffles subjects into folds (random_state = seed + repeat), so
    two calls with the same seed and rows use identical splits (paired comparisons).

    Returns
      predictions: one row per (repeat, data row) with columns repeat, fold, row,
                   subject, y, score (+ columns of `row_info`, e.g. study)
      importances: DataFrame (repeat, fold, one column per feature), or None
    """
    y, groups = _check_inputs(X, y, groups)

    jobs = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(X, y, groups)):
            jobs.append((repeat, fold, train, test))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_predict)(model, X, y, groups, train, test, permutation_repeats, seed + repeat)
        for repeat, fold, train, test in jobs
    )

    base = _predictions_frame(X, y, groups, row_info)
    parts, importance_rows = [], []
    for (repeat, fold, _, test), (scores, importances) in zip(jobs, results):
        part = base.iloc[test].copy()
        part.insert(0, "fold", fold)
        part.insert(0, "repeat", repeat)
        part["score"] = scores
        parts.append(part)
        if importances is not None:
            importance_rows.append({"repeat": repeat, "fold": fold, **dict(zip(X.columns, importances))})

    predictions = pd.concat(parts, ignore_index=True)
    counts = predictions.groupby("repeat")["row"].agg(["count", "nunique"])
    assert (counts["count"] == len(X)).all() and (counts["nunique"] == len(X)).all(), \
        "every row must be predicted exactly once per repeat"
    importances = pd.DataFrame(importance_rows) if importance_rows else None
    return predictions, importances


def leave_one_group_out(model, X, y, groups, test_groups, row_info=None):
    """Train on all rows except one value of `test_groups` (e.g. a study), test on it.

    `groups` (subject IDs) is still used for the inner CV of hyperparameter searches.
    Returns predictions with a `held_out` column naming the tested group.
    """
    y, groups = _check_inputs(X, y, groups)
    test_groups = np.asarray(test_groups)
    base = _predictions_frame(X, y, groups, row_info)

    parts = []
    for held_out in np.unique(test_groups):
        test = np.flatnonzero(test_groups == held_out)
        train = np.flatnonzero(test_groups != held_out)
        scores, _ = _fit_and_predict(model, X, y, groups, train, test, 0, 0)
        part = base.iloc[test].copy()
        part.insert(0, "held_out", held_out)
        part["score"] = scores
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def subject_level(predictions, keys=("repeat",)):
    """Average scores over each subject's rows (within each value of `keys`)."""
    keys = [k for k in keys if k in predictions.columns]
    extra = [c for c in predictions.columns if c not in {*keys, "subject", "score", "row", "fold"}]
    aggregations = {"score": "mean", **{c: "first" for c in extra}}
    return predictions.groupby([*keys, "subject"], as_index=False).agg(aggregations)


def summarize(predictions, keys=("repeat",), by=None, threshold=THRESHOLD):
    """Subject-level metrics for each value of `keys` (and of `by`, e.g. 'study')."""
    subjects = subject_level(predictions, keys)
    group_columns = [k for k in keys if k in subjects.columns] + ([by] if by else [])
    rows = []
    for key, subset in subjects.groupby(group_columns):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_columns, key)), "n_subjects": len(subset),
                     **binary_metrics(subset["y"], subset["score"], threshold)})
    return pd.DataFrame(rows)


def subject_scores(predictions):
    """One score per subject: mean over its rows and over all repeats."""
    return predictions.groupby("subject").agg(y=("y", "first"), score=("score", "mean"))


def bootstrap_auc(score_table, n_boot=2000, seed=0):
    """Stratified subject bootstrap of AUC for several score columns at once.

    score_table: index = subject, column 'y' plus one score column per model/variant.
    PD and control subjects are resampled separately (so both are always present),
    and every column uses the same resampled subjects, so differences and ratios
    between columns are paired. Returns an (n_boot x columns) DataFrame of AUCs.

    The interval reflects which subjects happened to be in the sample, not the
    variation from retraining the model.
    """
    rng = np.random.default_rng(seed)
    y = score_table["y"].to_numpy()
    columns = [c for c in score_table.columns if c != "y"]
    scores = score_table[columns].to_numpy()
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = np.empty((n_boot, len(columns)))
    for b in range(n_boot):
        sample = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        for k in range(len(columns)):
            aucs[b, k] = roc_auc_score(y[sample], scores[sample, k])
    return pd.DataFrame(aucs, columns=columns)
