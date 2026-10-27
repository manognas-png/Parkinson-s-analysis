# pd-gait-networks

Course project analyzing gait in Parkinson's disease (PD) using foot-pressure
sensor recordings from the PhysioNet **Gait in Parkinson's Disease** dataset.

## Dataset

- **Source:** <https://physionet.org/content/gaitpd/1.0.0/>
- **Subjects:** 165 total (93 PD patients, 72 healthy controls) from three studies:

  | Study | Description | PD | Control |
  |---|---|---|---|
  | Ga | Yogev et al. 2005 — dual-task walking | 29 | 18 |
  | Ju | Hausdorff et al. 2007 — rhythmic auditory stimulation | 29 | 25 |
  | Si | Frenkel-Toledo et al. 2005 — treadmill walking | 35 | 29 |

- **Recordings:** vertical ground reaction force (N) from 8 insole sensors
  under each foot, plus total force per foot, sampled at **100 Hz** (19 columns
  per trial file; see `data/raw/format.txt`).
- **File names:** `<Study><Group><Subject>_<Walk>.txt`, e.g. `GaPt13_10.txt`
  - `Co` = control, `Pt` = PD patient
  - Walk `01` = normal walk; walk `10` (Ga only) = dual-task walk (serial-7
    subtraction while walking)
- **Demographics:** `data/raw/demographics.txt` (age, sex, height, weight,
  Hoehn & Yahr stage, UPDRS, Timed Up-and-Go, gait speed).

### Known data quirks

- Demographics row `Juc010` is a typo for `JuCo10`, which has no trial files.
- Ju heights are in cm; Ga/Si heights are in meters.
- Hoehn & Yahr for controls is `0` in Ga but `NaN` in Ju/Si.
- The `Speed_10` column does not match which `_10` files exist — use the files.

### Getting the data

The raw data is not tracked in git. Download it into `data/raw/`:

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw https://physionet.org/files/gaitpd/1.0.0/
cd data/raw && sha256sum -c --quiet SHA256SUMS.txt   # no output = all files OK
```

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```
pd-gait-networks/
├── data/
│   ├── raw/          # original PhysioNet files (not tracked)
│   └── processed/    # derived datasets (not tracked)
├── notebooks/        # exploratory Jupyter notebooks
├── results/
│   ├── figures/      # plots
│   └── metrics/      # evaluation results
├── slurm/            # cluster job scripts
├── src/              # reusable Python code
└── requirements.txt
```

## Notes

- Subjects have multiple trials, so train/test splits should be made **by
  subject**, not by trial, to avoid leakage.
- Study is confounded with protocol and available trials; be careful that
  models don't just learn which study a recording came from.

## References

- Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220, 2000.
- Yogev G, et al. Dual tasking, gait rhythmicity, and Parkinson's disease. *Eur J Neurosci* 22:1248–1256, 2005.
- Hausdorff JM, et al. Rhythmic auditory stimulation modulates gait variability in Parkinson's disease. *Eur J Neurosci* 26:2369–2375, 2007.
- Frenkel-Toledo S, et al. Treadmill walking as an external pacemaker to improve gait rhythm and stability in Parkinson's disease. *Mov Disord* 20(9):1109–1114, 2005.
