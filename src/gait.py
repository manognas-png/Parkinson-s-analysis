"""Gait event detection and stride-time features from foot force signals."""

import numpy as np

from src.load import SAMPLING_RATE_HZ

# A foot counts as "on the ground" when its total force is above this (Newtons).
# Swing-phase force in this dataset is exactly 0 N, so a small value suffices.
CONTACT_THRESHOLD_N = 20

# The foot must have been off the ground at least this long before a heel
# strike counts. Rejects flickers around the threshold (real swing ~0.35 s).
MIN_SWING_S = 0.1

# Strides further than this fraction from the subject's median stride time
# are treated as turns, pauses or detection errors and removed.
OUTLIER_FRACTION = 0.3


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


def outlier_mask(strides, fraction=OUTLIER_FRACTION):
    """True for strides within +/- `fraction` of the median stride time."""
    median = np.median(strides)
    return np.abs(strides - median) <= fraction * median


def coefficient_of_variation(values):
    """Standard deviation as a percentage of the mean."""
    return 100 * np.std(values, ddof=1) / np.mean(values)
