"""Python reference for the insole firmware (firmware/lib/gait_model/gait_model.c).

Mirrors exactly what the device does, so device and training cannot drift apart:
  - input: the 6 anatomical sensors (L1, L4, L7, R1, R4, R7) as int16 in units of 0.1 N
  - heel strikes on the left (L1 + L4 + L7 > 20 N after >= 10 samples at or below 20 N)
  - each stride resampled to 100 points in sample-index time (no time column on a device);
    strides longer than MAX_STRIDE_SAMPLES are recorded (for the steady rule) but not analysed
  - covariance of the 6 sensors over the whole recording -> gait features
  - at the end: steady-stride rule (rolling median of 7, +/-10 %, runs >= 10, trim 2) and
    medians of the per-stride estimates over steady strides
"""

import numpy as np
import pandas as pd

from src.load import load_trial

SENSORS = ["L1", "L4", "L7", "R1", "R4", "R7"]
FS = 100
THRESHOLD_N = 20.0
MIN_SWING = 10
POINTS = 100
MAX_STRIDE_SAMPLES = 400  # 4 s; longer strides are counted but not analysed
STEADY_WINDOW, STEADY_MIN_RUN, STEADY_TRIM = 7, 10, 2
HEEL_FOREFOOT_PAIRS = [(0, 1), (0, 2), (3, 4), (3, 5)]
FEATURES = ["est_speed", "est_log_tug", "template_dev", "energy_mode1", "effective_modes", "heel_forefoot"]


def device_input(path):
    """The 6 sensors as the device sees them: int16, 0.1 N (round half to even, like C rint)."""
    values = load_trial(path)[SENSORS].to_numpy()
    return np.clip(np.rint(values * 10), -32768, 32767).astype(np.int16)


def extract(samples):
    """Strides, steady mask and covariance from int16 samples (n, 6). Same steps as gm_push/gm_finish."""
    x = samples.astype(np.float64) / 10.0
    proxy = x[:, 0] + x[:, 1] + x[:, 2]
    off = proxy <= THRESHOLD_N
    strikes, run = [], 0
    for i in range(len(x)):
        if not off[i] and i > 0 and off[i - 1] and run >= MIN_SWING:
            strikes.append(i)
        run = run + 1 if off[i] else 0
    strides, logdur, times = [], [], []
    for a, b in zip(strikes[:-1], strikes[1:]):
        d = b - a
        times.append(d / FS)
        if d > MAX_STRIDE_SAMPLES:
            strides.append(np.full((6, POINTS), np.nan))
            logdur.append(np.nan)
            continue
        grid = np.arange(POINTS) * (d / POINTS)
        segment = x[a:b + 1]
        strides.append(np.stack([np.interp(grid, np.arange(d + 1), segment[:, c]) for c in range(6)]))
        logdur.append(np.log(d / FS))
    times = np.array(times)
    samples = np.rint(times * FS).astype(int)
    sums = x.sum(axis=0)
    cross = x.T @ x
    cov = cross - np.outer(sums, sums) / len(x)
    return {"times": times, "X": np.array(strides).reshape(len(strides), 6, POINTS), "logdur": np.array(logdur),
            "steady": steady_mask(samples), "cov": cov}


def steady_mask(samples):
    """Rolling median of 7 (centred, partial at the ends), within 10 %, runs >= 10, trim 2.

    Exact integers on stride lengths in samples: |d - m| <= m / 10  <=>  10 |2d - 2m| <= 2m
    (2m is an integer), so the result cannot depend on floating-point rounding.
    """
    n = len(samples)
    local = pd.Series(samples, dtype=float).rolling(STEADY_WINDOW, center=True, min_periods=1).median().to_numpy()
    twice_median = np.rint(2 * local).astype(int)
    steady = 10 * np.abs(2 * samples - twice_median) <= twice_median
    mask = np.zeros(n, dtype=bool)
    start = 0
    while start < n:
        if not steady[start]:
            start += 1
            continue
        end = start
        while end < n and steady[end]:
            end += 1
        if end - start >= STEADY_MIN_RUN:
            mask[start + STEADY_TRIM:end - STEADY_TRIM] = True
        start = end
    return mask


def gait_features(cov):
    """Mode-1 energy, effective modes and heel-forefoot coupling from the 6 x 6 covariance."""
    energies = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)
    fractions = energies / energies.sum()
    d = np.sqrt(np.diag(cov))
    coupling = np.mean([abs(cov[i, j]) / (d[i] * d[j]) for i, j in HEEL_FOREFOOT_PAIRS])
    return {"energy_mode1": fractions[0], "effective_modes": 1 / np.sum(fractions ** 2), "heel_forefoot": coupling}


def flat_inputs(X, logdur):
    """Ridge inputs per stride: 600 resampled values (sensor-major) + log duration."""
    return np.hstack([X.reshape(len(X), 6 * POINTS), logdur[:, None]])


def score(extracted, params):
    """Features and logistic logit for one recording with (dequantized) device parameters."""
    X, logdur, steady = extracted["X"], extracted["logdur"], extracted["steady"]
    flat = X.reshape(len(X), 6 * POINTS)
    speed = flat @ params["speed_w"] + logdur * params["speed_logdur_w"] + params["speed_b"]
    tug = flat @ params["tug_w"] + logdur * params["tug_logdur_w"] + params["tug_b"]
    template = np.reshape(params["template"], (6, POINTS))  # stored flat, sensor-major, as in C
    deviation = ((X - template) ** 2).mean(axis=(1, 2))
    keep = steady & np.isfinite(speed)
    features = {
        "est_speed": np.median(speed[keep]) if keep.any() else np.nan,
        "est_log_tug": np.median(tug[keep]) if keep.any() else np.nan,
        "template_dev": np.median(deviation[keep]) if keep.any() else np.nan,
        **gait_features(extracted["cov"]),
    }
    values = np.array([features[f] for f in FEATURES])
    values = np.where(np.isnan(values), params["medians"], values)
    logit = float(values @ params["logit_w"] + params["logit_b"])
    return {**features, "logit": logit, "probability": 1 / (1 + np.exp(-logit)),
            "n_strides": len(extracted["times"]), "n_steady": int(steady.sum())}
