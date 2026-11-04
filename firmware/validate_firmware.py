"""Check the firmware's C code against the Python reference on every recording.

Compiles firmware/lib/gait_model/gait_model.c with the cluster's gcc twice:
  - double precision (must match the Python reference within rounding)
  - single precision, as on the Pico (reports how much float32 arithmetic changes results)
runs both on all 165 _01 recordings, and compares with firmware/reference_outputs.csv.

Run as part of slurm/firmware.sbatch, after firmware/export_model.py.
"""

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.firmware_reference import FEATURES  # noqa: E402
from src.load import DATA_DIR  # noqa: E402

FIRMWARE = Path(__file__).resolve().parent
BUILD = FIRMWARE / "host" / "build"
TOLERANCE_DOUBLE = 1e-6  # logit and features, double build vs Python


def compile_host(name, extra_flags):
    gcc = shutil.which("gcc")
    if gcc is None:
        sys.exit("ERROR: gcc not found (try: module load gcc)")
    BUILD.mkdir(parents=True, exist_ok=True)
    binary = BUILD / name
    command = [gcc, "-O2", "-std=c99", "-Wall", "-Wextra", *extra_flags, f"-I{FIRMWARE / 'include'}",
               f"-I{FIRMWARE / 'lib' / 'gait_model'}", str(FIRMWARE / "lib" / "gait_model" / "gait_model.c"),
               str(FIRMWARE / "host" / "run_host.c"), "-lm", "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"ERROR: compiling {name} failed:\n{result.stderr}")
    if result.stderr.strip():
        print(f"compiler warnings ({name}):\n{result.stderr}")
    return binary


def run(binary, subject):
    out = subprocess.run([str(binary), str(DATA_DIR / f"{subject}_01.txt")], capture_output=True, text=True, check=True)
    data = json.loads(out.stdout.replace("nan", "NaN"))
    return {"subject_id": subject, "n_strides": data["n_strides"], "n_steady": data["n_steady"],
            **dict(zip(FEATURES, data["features"])), "logit": data["logit"], "probability": data["probability"]}


def compare(reference, results, label):
    merged = reference.merge(results, on="subject_id", suffixes=("_py", "_c"))
    row = {"build": label, "recordings": len(merged),
           "strides_identical": int((merged["n_strides_py"] == merged["n_strides_c"]).sum()),
           "steady_identical": int((merged["n_steady_py"] == merged["n_steady_c"]).sum())}
    for column in [*FEATURES, "logit", "probability"]:
        a, b = merged[f"{column}_py"].to_numpy(float), merged[f"{column}_c"].to_numpy(float)
        both_missing = np.isnan(a) & np.isnan(b)
        diff = np.where(both_missing, 0, np.abs(a - b))
        row[column] = float(np.nanmax(np.where(np.isnan(diff), np.inf, diff)))
    row["auc_in_sample"] = roc_auc_score(merged["group"] == "PD", merged["logit_c"])
    return row


def main():
    reference = pd.read_csv(FIRMWARE / "reference_outputs.csv")
    builds = {"double": compile_host("run_host_double", []), "float (as on the Pico)":
              compile_host("run_host_float", ["-DGM_USE_FLOAT"])}
    rows = []
    for label, binary in builds.items():
        with ThreadPoolExecutor(8) as pool:
            results = pd.DataFrame(list(pool.map(lambda s: run(binary, s), reference["subject_id"])))
        results.to_csv(BUILD / f"outputs_{label.split()[0]}.csv", index=False)
        rows.append(compare(reference, results, label))
    table = pd.DataFrame(rows).set_index("build").T
    print(f"reference AUC (in-sample, stored parameters): "
          f"{roc_auc_score(reference['group'] == 'PD', reference['logit']):.4f}")
    print("\n== C firmware vs Python reference, all recordings: counts and max |difference| ==")
    print(table.to_string(float_format=lambda v: f"{v:.3g}"))
    double = rows[0]
    passed = (double["strides_identical"] == double["recordings"] and double["steady_identical"] == double["recordings"]
              and all(double[c] <= TOLERANCE_DOUBLE for c in [*FEATURES, "logit", "probability"]))
    print("\nVALIDATION PASSED (double build matches Python)" if passed else "\nVALIDATION FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
