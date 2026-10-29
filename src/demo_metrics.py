"""Per-stride metrics for the software demo: the Python reference that demo/engine.js
must reproduce.

Uses the same rules as the analysis code: heel strikes and steady strides from
src/gait.py (summed foot force crossing 20 N, minimum swing 0.1 s, steady-segment rule),
stance from heel strike to the first sample at or below 20 N (kept if 0.3-2.0 s, as in
src/cop.py). Detection always runs on newtons; reported forces are divided by body
weight when it is given.
"""

import numpy as np

from src.features_math import SENSOR_POSITIONS
from src.gait import CONTACT_THRESHOLD_N, resample_strides, steady_stride_bounds
from src.load import SAMPLING_RATE_HZ

GRAVITY = 9.81
STANCE_POINTS = 101  # stance normalized to 0..100 %
STRIDE_POINTS = 100  # template strides, as in Steps 9-10
MIN_STANCE_S, MAX_STANCE_S = 0.30, 2.00
VERSIONS = {
    "16": {foot: [f"{foot}{i}" for i in range(1, 9)] for foot in ["L", "R"]},
    "6": {foot: [f"{foot}1", f"{foot}4", f"{foot}7"] for foot in ["L", "R"]},
}
METRICS = ["stride_time", "cadence", "swing_time", "heel_load_early", "cop_05_15", "cop_25_35",
           "template_deviation"]


def template_channels(version):
    """Sensor order of a template stride: left foot then right foot."""
    return VERSIONS[version]["L"] + VERSIONS[version]["R"]


def ap_position(signals, sensors):
    """Front-to-back centre of pressure per sample, 0 = heel row, 1 = toe (sensor y from format.txt)."""
    forces = np.clip(signals, 0, None)
    y = np.array([SENSOR_POSITIONS[s][1] for s in sensors], dtype=float)
    total = forces.sum(axis=0)
    q = np.full(total.shape, np.nan)
    ok = total > 1e-8
    q[ok] = (forces[:, ok] * y[:, None]).sum(axis=0) / total[ok]
    return (q + 800) / 1600


def stance_end(force, time, start, end, threshold=CONTACT_THRESHOLD_N):
    """(first, last) sample indices of the stance in the stride [start, end), or None."""
    first, stride_end = np.searchsorted(time, [start, end])
    below = np.flatnonzero(force[first:stride_end] <= threshold)
    if len(below) == 0:
        return None
    last = first + int(below[0])
    duration = (last - first) / SAMPLING_RATE_HZ
    return (first, last) if MIN_STANCE_S <= duration <= MAX_STANCE_S else None


def resample_stance(values, first, last):
    """Linear interpolation of samples first..last-1 onto 101 points (0-100 % of stance)."""
    grid = np.linspace(first, last - 1, STANCE_POINTS)
    return np.interp(grid, np.arange(first, last), values[first:last])


def trial_metrics(trial, body_weight_kg=None, templates=None):
    """Per-stride metrics for both versions and both feet.

    trial: DataFrame with time and L1..R8. templates: {"16": array, "6": array}
    (channels x 100, body-weight units) or None. Returns a list of dicts.
    """
    time = trial["time"].to_numpy()
    scale = body_weight_kg * GRAVITY if body_weight_kg else 1.0
    rows = []
    for version, feet in VERSIONS.items():
        for foot, sensors in feet.items():
            signals = trial[sensors].to_numpy().T
            force = signals.sum(axis=0)
            q = ap_position(signals, sensors)
            heel = signals[0]
            channels = trial[template_channels(version)].to_numpy().T
            for start, end in steady_stride_bounds(force, time):
                stance = stance_end(force, time, start, end)
                if stance is None:
                    continue
                first, last = stance
                q_stance = resample_stance(q, first, last)
                heel_stance = resample_stance(heel, first, last)
                deviation = np.nan
                if foot == "L" and templates is not None and body_weight_kg:
                    stride = resample_strides(channels, time, [(start, end)], STRIDE_POINTS)[0] / scale
                    deviation = float(np.mean((stride - templates[version]) ** 2))
                rows.append({
                    "version": version, "foot": foot, "heel_strike": float(start),
                    "stride_time": float(end - start),
                    "cadence": float(120 / (end - start)),
                    "swing_time": float(end - time[last]),
                    "heel_load_early": float(heel_stance[5:16].mean() / scale),
                    "cop_05_15": float(q_stance[5:16].mean()),
                    "cop_25_35": float(q_stance[25:36].mean()),
                    "template_deviation": deviation,
                })
    return rows


def template_strides(trial, version, body_weight_kg):
    """Steady strides (left heel strike to left heel strike) of all version sensors / body weight."""
    time = trial["time"].to_numpy()
    left = trial[VERSIONS[version]["L"]].to_numpy().T.sum(axis=0)
    channels = trial[template_channels(version)].to_numpy().T
    bounds = steady_stride_bounds(left, time)
    return resample_strides(channels, time, bounds, STRIDE_POINTS) / (body_weight_kg * GRAVITY)
