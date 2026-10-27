"""Gait event detection and stride-time features from foot force signals."""

import numpy as np
import pandas as pd

from src.load import SAMPLING_RATE_HZ

# A foot counts as "on the ground" when its total force is above this (Newtons).
# Swing-phase force in this dataset is exactly 0 N, so a small value suffices.
CONTACT_THRESHOLD_N = 20

# The foot must have been off the ground at least this long before a heel
# strike counts. Rejects flickers around the threshold (real swing ~0.35 s).
MIN_SWING_S = 0.1

# Steady-walking segments (see steady_stride_mask)
STEADY_WINDOW = 7  # strides in the rolling median
STEADY_TOLERANCE = 0.10  # max deviation from the rolling median
STEADY_MIN_RUN = 10  # consecutive steady strides needed for a segment
STEADY_TRIM = 2  # strides dropped from each end of a segment


def detect_heel_strikes(force, time, threshold=CONTACT_THRESHOLD_N, min_swing_s=MIN_SWING_S):
    """Return the times at which `force` rises above `threshold` after a swing."""
    force = np.asarray(force)
    time = np.asarray(time)
    off_ground = force <= threshold

    # Candidate heel strikes: sample i is on the ground, sample i-1 was not
    rises = np.flatnonzero(off_ground[:-1] & ~off_ground[1:]) + 1

    # Keep only those preceded by a real swing phase
    n_swing = int(min_swing_s * SAMPLING_RATE_HZ)
    strikes = [i for i in rises if i >= n_swing and off_ground[i - n_swing:i].all()]
    return time[strikes]


def stride_times(heel_strike_times):
    """Time between consecutive heel strikes of the same foot (seconds)."""
    return np.diff(heel_strike_times)


def swing_times(force, time, heel_strike_times, threshold=CONTACT_THRESHOLD_N):
    """Swing time for each stride: last toe-off before the next heel strike -> that heel strike.

    Returns one value per stride (same length as stride_times), NaN if no toe-off was found.
    """
    force = np.asarray(force)
    time = np.asarray(time)
    off_ground = force <= threshold

    # Toe-offs: sample i is off the ground, sample i-1 was on it
    toe_off_times = time[np.flatnonzero(~off_ground[:-1] & off_ground[1:]) + 1]

    swings = np.full(len(heel_strike_times) - 1, np.nan)
    for k, (start, end) in enumerate(zip(heel_strike_times[:-1], heel_strike_times[1:])):
        j = np.searchsorted(toe_off_times, end) - 1  # last toe-off before `end`
        if j >= 0 and toe_off_times[j] >= start:
            swings[k] = end - toe_off_times[j]
    return swings


def steady_stride_mask(
    strides,
    window=STEADY_WINDOW,
    tolerance=STEADY_TOLERANCE,
    min_run=STEADY_MIN_RUN,
    trim=STEADY_TRIM,
):
    """True for strides inside steady walking segments.

    A stride is "steady" if it is within +/- `tolerance` of the rolling median
    of the surrounding `window` strides. Runs of at least `min_run` steady
    strides form a segment; `trim` strides are dropped from each end of it
    (the speeding-up / slowing-down edges).
    """
    strides = pd.Series(strides)
    local_median = strides.rolling(window, center=True, min_periods=1).median()
    is_steady = ((strides - local_median).abs() <= tolerance * local_median).to_numpy()

    mask = np.zeros(len(strides), dtype=bool)
    for start, end in runs_of_true(is_steady):
        if end - start >= min_run:
            mask[start + trim:end - trim] = True
    return mask


def runs_of_true(flags):
    """(start, end) index pairs of consecutive True values; `end` is exclusive."""
    padded = np.concatenate([[False], flags, [False]]).astype(int)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def coefficient_of_variation(values):
    """Standard deviation as a percentage of the mean (NaNs ignored; NaN if < 2 values)."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan
    return 100 * np.std(values, ddof=1) / np.mean(values)


def steady_window(force, time, max_seconds=10):
    """Start/end time of a steady walking stretch: the longest steady segment,
    cut to at most `max_seconds`. Also returns the stride frequency (Hz) there.

    Returns (None, None, nan) if the trial has no steady segment.
    """
    heel_strikes = detect_heel_strikes(force, time)
    strides = stride_times(heel_strikes)
    segments = runs_of_true(steady_stride_mask(strides))
    if not segments:
        return None, None, np.nan

    first, last = max(segments, key=lambda seg: seg[1] - seg[0])
    start = heel_strikes[first]
    end = min(heel_strikes[last], start + max_seconds)  # stride `last - 1` ends at heel strike `last`
    return start, end, 1 / np.median(strides[first:last])
