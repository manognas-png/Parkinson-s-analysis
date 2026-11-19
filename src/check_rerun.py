"""After a full rerun: check the headline numbers and compare every metrics CSV with the
committed version (git HEAD).

Run from the project root (light; the last step of slurm/full_rerun.sbatch):
    python -m src.check_rerun
"""

import io
import subprocess
import sys

import numpy as np
import pandas as pd

from src.plot_explore import METRICS_DIR, RESULTS_DIR

# (label, expected value as reported, decimals it was reported to, function reading the new value)
def _ml_speed(d):
    return d.query("feature_set == '1 speed' and model == 'logistic' and evaluation == 'repeated_cv' "
                   "and subset == 'All' and metric == 'auc'")["mean"].item()


def _row(file, column, **where):
    d = pd.read_csv(METRICS_DIR / file)
    for k, v in where.items():
        d = d[d[k] == v]
    return d[column].item()


def _speed_r2():
    d = pd.read_csv(METRICS_DIR / "insole_regression.csv", header=[0, 1], index_col=[0, 1, 2])
    return d.loc[("speed", "ridge_strides", "All"), ("r2", "mean")]


HEADLINES = [
    ("speed-only AUC (Step 7, logistic, mean over repeats)", 0.771, 3,
     lambda: _ml_speed(pd.read_csv(METRICS_DIR / "ml_results.csv"))),
    ("COP10 16 sensors, pooled bootstrap AUC", 0.884, 3, lambda: _row("cop_auc.csv", "auc_pooled", model="COP10, 16 sensors")),
    ("COP10 16 sensors, mean over repeats", 0.869, 3, lambda: _row("cop_auc.csv", "auc_mean", model="COP10, 16 sensors")),
    ("anatomical 6-sensor AUC (gait)", 0.753, 3,
     lambda: _row("supervised_selection.csv", "auc_mean", variant="anatomical6", feature_set="gait")),
    ("anatomical 6 ratio to all 16", 1.05, 2,
     lambda: _row("supervised_selection.csv", "ratio_to_all16", variant="anatomical6", feature_set="gait")),
    ("anatomical 6 ratio CI low", 0.95, 2,
     lambda: _row("supervised_selection.csv", "ratio_ci_low", variant="anatomical6", feature_set="gait")),
    ("anatomical 6 ratio CI high", 1.15, 2,
     lambda: _row("supervised_selection.csv", "ratio_ci_high", variant="anatomical6", feature_set="gait")),
    ("insole-only AUC (body-weight-normalized)", 0.823, 3,
     lambda: _row("insole_classifier.csv", "auc_mean",
                  feature_set="est. speed + est. TUG + template + 6-sensor gait")),
    ("insole-only AUC, raw newtons (no body weight)", 0.812, 3,
     lambda: _row("insole_robustness.csv", "auc_all", feature_set="insole (raw)")),
    ("int16 insole AUC", 0.8232, 4, lambda: _row("insole_quantized.csv", "auc_mean", precision="int16")),
    ("speed estimate R2 (ridge)", 0.67, 2, _speed_r2),
]


def committed_csv(path):
    """The committed version of a metrics file, or None if it is not in git HEAD."""
    relative = path.relative_to(RESULTS_DIR.parent).as_posix()
    shown = subprocess.run(["git", "show", f"HEAD:{relative}"], capture_output=True, text=True)
    return pd.read_csv(io.StringIO(shown.stdout)) if shown.returncode == 0 else None


def compare_files():
    rows = []
    for path in sorted(METRICS_DIR.glob("*.csv")):
        new, old = pd.read_csv(path), committed_csv(path)
        if old is None:
            rows.append({"file": path.name, "status": "new file", "max_abs_diff": np.nan})
            continue
        if new.shape != old.shape or list(new.columns) != list(old.columns):
            rows.append({"file": path.name, "status": f"shape/columns differ {old.shape} -> {new.shape}",
                         "max_abs_diff": np.nan})
            continue
        numeric = new.select_dtypes("number").columns
        diff = (new[numeric] - old[numeric]).abs().to_numpy(dtype=float)
        text_same = new.drop(columns=numeric).astype(str).equals(old.drop(columns=numeric).astype(str))
        largest = np.nanmax(diff) if diff.size and np.isfinite(diff).any() else 0.0
        rows.append({"file": path.name, "status": "identical" if largest < 1e-9 and text_same
                     else ("text differs" if not text_same else "numbers differ"), "max_abs_diff": largest})
    committed = subprocess.run(["git", "ls-files", "results/metrics/*.csv"], capture_output=True, text=True).stdout.split()
    produced = {p.name for p in METRICS_DIR.glob("*.csv")}
    rows += [{"file": c.split("/")[-1], "status": "MISSING (not regenerated)", "max_abs_diff": np.nan}
             for c in committed if c.split("/")[-1] not in produced]
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 200)
    print("== Headline numbers: reported vs this run ==")
    failures = 0
    for label, expected, decimals, read in HEADLINES:
        try:
            value = float(read())
        except Exception as error:  # missing file or row
            print(f"  ERROR  {label}: {error}")
            failures += 1
            continue
        ok = round(value, decimals) == round(expected, decimals)
        failures += not ok
        print(f"  {'ok   ' if ok else 'DIFF '}  {label:52s} reported {expected:<7} now {value:.{decimals + 2}f}")

    print("\n== Every metrics file vs the committed version (git HEAD) ==")
    table = compare_files()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    changed = table[table["status"] != "identical"]
    print(f"\n{len(table) - len(changed)} of {len(table)} files identical; {failures} headline number(s) differ")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
