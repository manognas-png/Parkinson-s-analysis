"""Loading utilities for the PhysioNet "Gait in Parkinson's Disease" dataset."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SAMPLING_RATE_HZ = 100

TRIAL_COLUMNS = (
    ["time"]
    + [f"L{i}" for i in range(1, 9)]
    + [f"R{i}" for i in range(1, 9)]
    + ["L_total", "R_total"]
)

# e.g. "GaPt13_10.txt" -> study="Ga", group="Pt", number="13", trial="10"
FILENAME_PATTERN = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d{2})_(\d{2})\.txt$")

# Demographics columns: original name -> clearer name
DEMOGRAPHIC_COLUMNS = {
    "ID": "subject_id",
    "Study": "study",
    "Group": "group",
    "Subjnum": "subject_number",
    "Gender": "sex",
    "Age": "age",
    "Height": "height_m",
    "Weight": "weight_kg",
    "HoehnYahr": "hoehn_yahr",
    "UPDRS": "updrs",
    "UPDRSM": "updrs_motor",
    "TUAG": "tug_s",
    "Speed_01": "speed_01",
    "Speed_02": "speed_02",
    "Speed_03": "speed_03",
    "Speed_04": "speed_04",
    "Speed_05": "speed_05",
    "Speed_06": "speed_06",
    "Speed_07": "speed_07",
    "Speed_10": "speed_10",
}


def load_trial(path):
    """Read one trial file into a DataFrame with named columns."""
    return pd.read_csv(path, sep="\t", header=None, names=TRIAL_COLUMNS)


def parse_filename(path):
    """Split a trial filename into study, group, subject_id and trial number."""
    name = Path(path).name
    match = FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Not a trial filename: {name}")

    study, group_code, number, trial = match.groups()
    return {
        "filename": name,
        "study": study,
        "group": "PD" if group_code == "Pt" else "Control",
        "subject_id": f"{study}{group_code}{number}",
        "trial": int(trial),
    }


def load_demographics(path=DATA_DIR / "demographics.txt"):
    """Read demographics.txt into a clean DataFrame (one row per subject)."""
    # Parse by hand: rows have trailing tabs of varying length, and the
    # file ends with many blank lines.
    with open(path) as f:
        lines = [line.rstrip("\r\n") for line in f]

    header = lines[0].split("\t")[: len(DEMOGRAPHIC_COLUMNS)]
    rows = [
        line.split("\t")[: len(header)]
        for line in lines[1:]
        if line.strip()
    ]
    df = pd.DataFrame(rows, columns=header).rename(columns=DEMOGRAPHIC_COLUMNS)

    # Numeric columns: "NaN" strings become real missing values
    numeric_columns = [c for c in df.columns if c not in ("subject_id", "study")]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Typo in the source file: "Juc010" should be "JuCo10"
    df["subject_id"] = df["subject_id"].replace({"Juc010": "JuCo10"})

    # Readable labels instead of 1/2 codes
    df["group"] = df["group"].map({1: "PD", 2: "Control"})
    df["sex"] = df["sex"].map({1: "male", 2: "female"})

    # Ju heights are in cm; everyone else's are in meters
    is_ju = df["study"] == "Ju"
    df.loc[is_ju, "height_m"] = df.loc[is_ju, "height_m"] / 100

    # Hoehn & Yahr doesn't apply to controls (Ga used 0, Ju/Si used NaN)
    df.loc[df["group"] == "Control", "hoehn_yahr"] = np.nan

    return df


def build_summary(data_dir=DATA_DIR):
    """One row per trial file: filename info, recording length, demographics."""
    records = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        if not FILENAME_PATTERN.match(path.name):
            continue  # skip format.txt, demographics.txt, etc.

        info = parse_filename(path)
        trial = load_trial(path)
        info["is_dual_task"] = info["trial"] == 10
        info["n_samples"] = len(trial)
        info["duration_s"] = trial["time"].iloc[-1] - trial["time"].iloc[0]
        records.append(info)

    summary = pd.DataFrame(records)

    demographics = load_demographics(Path(data_dir) / "demographics.txt")
    demographics = demographics.drop(columns=["study", "group"])
    return summary.merge(demographics, on="subject_id", how="left")
