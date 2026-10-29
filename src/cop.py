"""Stance-indexed anterior-posterior centre of pressure (AP-COP) features.

Follows Sifat et al. 2026, "A Compact Stance-Indexed Anterior-Posterior COP
Representation for Parkinson's Disease Classification from Plantar VGRF"
(arXiv:2609.22956): AP-COP10 = mean AP position and left-right asymmetry of the
AP-COP in five stance windows. Here stances come from our steady strides.
"""

import numpy as np
from scipy.signal import savgol_filter

from src.features_math import SENSOR_POSITIONS
from src.gait import CONTACT_THRESHOLD_N, steady_stride_bounds
from src.load import SAMPLING_RATE_HZ

STANCE_POINTS = 101  # 0..100 % of stance
WINDOWS = [(5, 15), (25, 35), (45, 55), (65, 75), (85, 95)]  # % of stance
MIN_STANCE_S, MAX_STANCE_S = 0.30, 2.00
MIN_VALID_SAMPLES = 30
COP_FEATURES = ([f"cop_pos_{a:02d}_{b:02d}" for a, b in WINDOWS]
                + [f"cop_asym_{a:02d}_{b:02d}" for a, b in WINDOWS])


def ap_position(signals, y_coordinates):
    """AP-COP per sample in [0, 1]: force-weighted mean sensor y, (q + 800) / 1600.

    signals: (sensors, samples) in newtons. NaN where the summed force is ~0.
    """
    forces = np.clip(signals, 0, None)
    total = forces.sum(axis=0)
    weighted = (forces * np.asarray(y_coordinates)[:, None]).sum(axis=0)
    q = np.full(total.shape, np.nan)
    has_force = total > 1e-8
    q[has_force] = weighted[has_force] / total[has_force]
    return (q + 800) / 1600


def stance_bounds(force, time, threshold=CONTACT_THRESHOLD_N):
    """(first, last) sample indices of the stance in each steady stride of one foot.

    Stance = heel strike until the force first drops to the contact threshold.
    """
    time = np.asarray(time)
    force = np.asarray(force)
    bounds = []
    for start, end in steady_stride_bounds(force, time):
        first, stride_end = np.searchsorted(time, [start, end])
        below = np.flatnonzero(force[first:stride_end] <= threshold)
        if len(below) == 0:
            continue
        last = first + below[0]
        duration = (last - first) / SAMPLING_RATE_HZ
        if MIN_STANCE_S <= duration <= MAX_STANCE_S:
            bounds.append((first, last))
    return bounds


def stance_traces(q, bounds):
    """AP-COP of each stance resampled to 101 points (0-100 %) and smoothed (Savitzky-Golay 11, 2)."""
    traces = []
    for first, last in bounds:
        segment = q[first:last]
        valid = np.isfinite(segment)
        if valid.sum() < MIN_VALID_SAMPLES:
            continue
        x = np.linspace(0, 100, len(segment))
        trace = np.interp(np.linspace(0, 100, STANCE_POINTS), x[valid], segment[valid])
        traces.append(savgol_filter(trace, 11, 2))
    return np.array(traces).reshape(len(traces), STANCE_POINTS)


def window_medians(traces):
    """Per window: mean of each stance's trace over the window, then median over stances."""
    if len(traces) == 0:
        return np.full(len(WINDOWS), np.nan)
    return np.array([np.median(traces[:, a:b + 1].mean(axis=1)) for a, b in WINDOWS])


def cop10(q_left, q_right):
    """Bilateral position (mean of feet) and Robinson-type asymmetry per window."""
    position = (q_left + q_right) / 2
    asymmetry = np.abs(q_left - q_right) / (0.5 * (np.abs(q_left) + np.abs(q_right)))
    return dict(zip(COP_FEATURES, np.r_[position, asymmetry]))


def trial_cop_features(trial, sensors_per_foot, force_per_foot=None):
    """AP-COP10 for one trial from the given sensors on each foot.

    sensors_per_foot: {"L": [...], "R": [...]} sensor names used for the COP.
    force_per_foot: {"L": column, "R": column} total-force channel used to find stances;
    default is the sum of the selected sensors (what a reduced insole would have).
    Returns (features, mean stance trace per foot).
    """
    time = trial["time"].to_numpy()
    medians, mean_traces = {}, {}
    for foot, sensors in sensors_per_foot.items():
        signals = trial[sensors].to_numpy().T
        force = (trial[force_per_foot[foot]].to_numpy() if force_per_foot else signals.sum(axis=0))
        q = ap_position(signals, [SENSOR_POSITIONS[s][1] for s in sensors])
        traces = stance_traces(q, stance_bounds(force, time))
        medians[foot] = window_medians(traces)
        mean_traces[foot] = traces.mean(axis=0) if len(traces) else np.full(STANCE_POINTS, np.nan)
    return cop10(medians["L"], medians["R"]), mean_traces
