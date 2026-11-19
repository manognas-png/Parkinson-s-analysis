"""Choose which recordings the Pico replays: writes firmware/include/demo_recordings.h.

Usage (from the project root, takes a few seconds; then rebuild with `pio run -d firmware`):
    python firmware/make_demo.py GaCo01 GaPt29 JuPt05 SiCo10
    python firmware/make_demo.py --list          # every subject with the model's output

Each recording uses about 12 bytes per sample (about 150 kB for 2 minutes); the Pico has 2 MB of
flash, so about 10 recordings fit. The button on the Pico steps through them in the order given.
The expected outputs come from firmware/reference_outputs.csv (Python reference, stored model).
Note: every subject was used to train the final model, so these show that the device reproduces the
model, not how well it does on new people (that is the cross-validated AUC, 0.814).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.firmware_reference import device_input  # noqa: E402
from src.load import DATA_DIR, parse_filename  # noqa: E402

FIRMWARE = Path(__file__).resolve().parent
FLASH_BUDGET = 1_600_000  # bytes for recordings, leaving room for the program and model
LABELS = {"Control": "healthy", "PD": "Parkinson's"}


def write_header(subjects):
    """Write demo_recordings.h for these subject ids (trial 01 of each); return the bytes used."""
    demo = ["typedef struct { const char *name; const char *group; long n; const int16_t (*data)[6]; } gm_recording;\n"]
    total = 0
    for k, subject in enumerate(subjects):
        path = DATA_DIR / f"{subject}_01.txt"
        if not path.exists():
            raise SystemExit(f"no recording {path.name} in data/raw (use --list to see the subject ids)")
        samples = device_input(path)
        total += samples.nbytes
        rows = ",".join("{" + ",".join(str(int(v)) for v in row) + "}" for row in samples)
        demo.append(f"static const int16_t GM_DEMO_{k}[{len(samples)}][6] = {{{rows}}};\n")
    if total > FLASH_BUDGET:
        raise SystemExit(f"{total / 1e6:.2f} MB of recordings is too much for the Pico's flash; choose fewer")
    groups = [LABELS[parse_filename(DATA_DIR / f"{s}_01.txt")["group"]] for s in subjects]
    demo.append("static const gm_recording GM_DEMO[] = {"
                + ", ".join(f'{{"{s}", "{g}", sizeof(GM_DEMO_{k}) / sizeof(GM_DEMO_{k}[0]), GM_DEMO_{k}}}'
                            for k, (s, g) in enumerate(zip(subjects, groups))) + "};\n")
    demo.append(f"#define GM_DEMO_COUNT {len(subjects)}\n")
    (FIRMWARE / "include" / "demo_recordings.h").write_text(
        "/* Demo recordings (6 sensors L1, L4, L7, R1, R4, R7; int16, 0.1 N; 100 Hz), PhysioNet\n"
        "   Gait in Parkinson's Disease. Written by firmware/make_demo.py; do not edit. */\n"
        "#ifndef DEMO_RECORDINGS_H\n#define DEMO_RECORDINGS_H\n#include <stdint.h>\n" + "".join(demo) + "#endif\n")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subjects", nargs="*", help="subject ids such as GaCo01 JuPt05")
    parser.add_argument("--list", action="store_true", help="show every subject and the expected model output")
    args = parser.parse_args()
    reference = pd.read_csv(FIRMWARE / "reference_outputs.csv").set_index("subject_id")
    reference["prediction"] = (reference["probability"] > 0.5).map({True: "PD-like", False: "typical"})
    columns = ["group", "n_strides", "n_steady", "probability", "prediction"]
    if args.list or not args.subjects:
        with pd.option_context("display.max_rows", None):
            print(reference[columns].round(3).to_string())
        return
    subjects = list(dict.fromkeys(args.subjects))  # drop repeats, keep order
    total = write_header(subjects)
    print(reference.reindex(subjects)[columns].round(3).to_string())
    print(f"\nwrote firmware/include/demo_recordings.h: {len(subjects)} recordings, {total / 1e3:.0f} kB")
    print("now rebuild: pio run -d firmware   (then restart the simulator)")


if __name__ == "__main__":
    main()
