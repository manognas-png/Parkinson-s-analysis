"""Linear-algebra features of the 16-sensor gait signals: SVD, sensor networks, DMD."""

import numpy as np
from scipy.signal import decimate

SENSOR_COLUMNS = [f"L{i}" for i in range(1, 9)] + [f"R{i}" for i in range(1, 9)]


def sensor_matrix(trial):
    """16 x T matrix: one row per sensor (L1..L8, R1..R8), one column per sample."""
    return trial[SENSOR_COLUMNS].to_numpy().T


def center_rows(X):
    """Subtract each row's mean, so every sensor varies around 0."""
    return X - X.mean(axis=1, keepdims=True)


def scale_rows(X):
    """Divide each row by its standard deviation (rows with zero spread are left as is)."""
    std = X.std(axis=1, keepdims=True)
    return X / np.where(std > 0, std, 1)


def preprocess(X, standardize=False):
    """Center each sensor; optionally also scale it to unit variance."""
    X = center_rows(X)
    return scale_rows(X) if standardize else X


def svd(X):
    """Thin SVD: X = U @ diag(s) @ Vt.

    U  (16 x 16): spatial patterns over sensors ("modes"), one per column
    s  (16,):     singular values, largest first (how strong each mode is)
    Vt (16 x T):  time course of each mode, one per row

    Computed through the smaller Gram matrix: if X = U S Vt then
    X Xt = U S^2 Ut, so U and S^2 are its eigenvectors and eigenvalues,
    and Vt = S^-1 Ut X. For tall matrices, use Xt X = V S^2 Vt instead
    ("method of snapshots"). Much faster than np.linalg.svd here, which is
    slow under multithreaded OpenBLAS on the cluster.
    """
    if X.shape[0] > X.shape[1]:
        V, s, Ut = svd(X.T)
        return Ut.T, s, V.T

    eigenvalues, U = np.linalg.eigh(X @ X.T)  # ascending order
    order = np.argsort(eigenvalues)[::-1]
    U = U[:, order]
    s = np.sqrt(np.clip(eigenvalues[order], 0, None))  # tiny negatives -> 0

    Vt = np.zeros((len(s), X.shape[1]))
    nonzero = s > s[0] * 1e-12
    Vt[nonzero] = (U[:, nonzero].T @ X) / s[nonzero, None]
    return U, s, Vt


def energy_fractions(s):
    """Share of the total energy (sum of squares of X) carried by each mode."""
    energy = s**2
    return energy / energy.sum()


def n_components_for_energy(s, target=0.9):
    """Smallest k such that the first k modes hold at least `target` of the energy."""
    cumulative = np.cumsum(energy_fractions(s))
    return int(np.searchsorted(cumulative, target) + 1)


def effective_n_modes(s):
    """1 / sum(energy fraction^2): 1 if one mode holds everything, 16 if all are equal.

    A continuous alternative to counting modes up to a fixed energy threshold.
    """
    return 1 / np.sum(energy_fractions(s) ** 2)


def fix_signs(U, reference=None):
    """Choose a sign for each column of U (SVD signs are arbitrary).

    Without a reference: make each column's largest-magnitude entry positive.
    With a reference U: flip columns so they point the same way as the reference's.
    """
    U = U.copy()
    for k in range(U.shape[1]):
        if reference is None:
            sign = np.sign(U[np.argmax(np.abs(U[:, k])), k])
        else:
            sign = np.sign(U[:, k] @ reference[:, k])
        U[:, k] *= sign if sign != 0 else 1
    return U


def svd_features(trial, target=0.9, standardize=False):
    """Summary numbers for one trial: modes for `target` energy, energy in mode 1,
    effective number of modes."""
    _, s, _ = svd(preprocess(sensor_matrix(trial), standardize))
    return {
        "n_components_90": n_components_for_energy(s, target),
        "energy_mode1": energy_fractions(s)[0],
        "effective_n_modes": effective_n_modes(s),
    }


# ---- Sensor correlation networks ----

# Sensor (x, y) positions inside the insoles, from format.txt (arbitrary units;
# y runs from heel, -800, to toe, +800)
SENSOR_POSITIONS = {
    "L1": (-500, -800), "L2": (-700, -400), "L3": (-300, -400), "L4": (-700, 0),
    "L5": (-300, 0), "L6": (-700, 400), "L7": (-300, 400), "L8": (-500, 800),
    "R1": (500, -800), "R2": (700, -400), "R3": (300, -400), "R4": (700, 0),
    "R5": (300, 0), "R6": (700, 400), "R7": (300, 400), "R8": (500, 800),
}

IS_LEFT = np.array([name.startswith("L") for name in SENSOR_COLUMNS])
IS_HEEL = np.array([int(name[1:]) <= 3 for name in SENSOR_COLUMNS])  # sensors 1-3


def correlation_matrix(X):
    """Pearson correlation between rows: Z Zt / T, where Z is X standardized per row.

    This is the matrix whose eigenvectors / eigenvalues the standardized SVD finds
    (eigenvalues = s^2 / T).
    """
    Z = preprocess(X, standardize=True)
    return Z @ Z.T / X.shape[1]


def correlation_network(X):
    """Weighted adjacency matrix A: |correlation| between sensors, zero diagonal."""
    A = np.abs(correlation_matrix(X))
    np.fill_diagonal(A, 0)
    return A


def laplacian(A):
    """Graph Laplacian L = D - A, with D = diag(node strengths)."""
    return np.diag(A.sum(axis=1)) - A


def normalized_laplacian(A):
    """L_norm = I - D^-1/2 A D^-1/2 (eigenvalues between 0 and 2).

    Isolated nodes (no edges) get an all-zero row and column.
    """
    degree = A.sum(axis=1)
    connected = degree > 0
    d_inv_sqrt = np.zeros_like(degree)
    d_inv_sqrt[connected] = 1 / np.sqrt(degree[connected])
    return np.diag(connected.astype(float)) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


def threshold_network(A):
    """Keep only edges stronger than the median edge weight; the rest become 0."""
    cutoff = np.median(A[np.triu_indices(len(A), k=1)])
    return np.where(A > cutoff, A, 0)


def fiedler(A):
    """Second-smallest Laplacian eigenvalue and its eigenvector.

    Sign of the vector is arbitrary; it is fixed so the mean over the heel
    sensors (1-3, both feet) is positive.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian(A))  # ascending
    vector = eigenvectors[:, 1]
    if vector[IS_HEEL].mean() < 0:
        vector = -vector
    return eigenvalues[1], vector


def von_neumann_entropy(A):
    """Entropy of the normalized Laplacian's eigenvalue distribution.

    rho = L_norm / trace(L_norm) has non-negative eigenvalues summing to 1, like
    probabilities; S = -sum(p log p). Max is log(n - 1) for evenly weighted complete graphs.
    (trace(L_norm) = number of non-isolated nodes.)
    """
    eigenvalues = np.linalg.eigvalsh(normalized_laplacian(A))
    p = np.clip(eigenvalues, 0, None) / eigenvalues.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def network_features(trial):
    """Network summary numbers for one trial."""
    A = correlation_network(sensor_matrix(trial))
    A_thresholded = threshold_network(A)
    off_diagonal = ~np.eye(len(A), dtype=bool)
    same_foot = np.equal.outer(IS_LEFT, IS_LEFT)
    heel_to_forefoot = same_foot & np.not_equal.outer(IS_HEEL, IS_HEEL)
    return {
        "fiedler_value": fiedler(A)[0],
        "von_neumann_entropy": von_neumann_entropy(A),
        "mean_edge_weight": A[off_diagonal].mean(),
        "within_foot_corr": A[same_foot & off_diagonal].mean(),
        "left_right_corr": A[~same_foot].mean(),
        "heel_forefoot_corr": A[heel_to_forefoot].mean(),
        "fiedler_value_thresholded": fiedler(A_thresholded)[0],
        "entropy_thresholded": von_neumann_entropy(A_thresholded),
    }


# ---- Dynamic Mode Decomposition ----


def downsample(X, factor):
    """Low-pass filter, then keep every `factor`-th sample (avoids aliasing)."""
    return decimate(X, factor, axis=1, zero_phase=True)


def delay_embed(X, n_delays):
    """Stack `n_delays` time-shifted copies of X (Hankel matrix).

    Row block k holds X shifted by k samples, so each column is the recent
    history [x(t), x(t+1), ..., x(t+n_delays-1)] instead of a single snapshot.
    """
    n_columns = X.shape[1] - n_delays + 1
    return np.vstack([X[:, k:k + n_columns] for k in range(n_delays)])


def dmd(H, rank, dt):
    """Exact DMD: find A with H[:, 1:] ~ A @ H[:, :-1], restricted to `rank` SVD modes.

    Returns eigenvalues (per step), frequencies (Hz), growth rates (1/s), unit-norm
    modes (columns) and their amplitudes in the first snapshot.
    """
    X1, X2 = H[:, :-1], H[:, 1:]
    U, s, Vt = svd(X1)
    U, s, V = U[:, :rank], s[:rank], Vt[:rank].T

    A_tilde = U.T @ X2 @ V / s  # A projected onto the top `rank` directions
    eigenvalues, W = np.linalg.eig(A_tilde)
    modes = X2 @ V / s @ W
    modes /= np.linalg.norm(modes, axis=0)

    amplitudes = np.linalg.lstsq(modes, X1[:, 0], rcond=None)[0]
    omega = np.log(eigenvalues) / dt  # continuous-time rates
    return {
        "eigenvalues": eigenvalues,
        "frequencies": omega.imag / (2 * np.pi),
        "growth_rates": omega.real,
        "modes": modes,
        "amplitudes": amplitudes,
    }


def dmd_reconstruct(result, n_steps, stabilize=False):
    """x_k = sum_i modes_i * amplitude_i * eigenvalue_i^k, for k = 0..n_steps-1.

    stabilize=True moves eigenvalues with |lambda| > 1 onto the unit circle, so a
    slightly growing (spurious) mode stays sustained instead of blowing up.
    """
    eigenvalues = result["eigenvalues"]
    if stabilize:
        eigenvalues = eigenvalues / np.maximum(np.abs(eigenvalues), 1)
    powers = np.vander(eigenvalues, n_steps, increasing=True)
    return ((result["modes"] * result["amplitudes"]) @ powers).real


def dmd_features(X, dt, n_delays, rank, harmonic_tolerance=0.05, min_frequency=0.3):
    """Summary numbers from DMD of a (sensors x time) window, already centered."""
    H = delay_embed(X, n_delays)
    result = dmd(H, rank, dt)
    frequencies = np.abs(result["frequencies"])
    amplitude = np.abs(result["amplitudes"])

    # Dominant mode: largest amplitude among oscillating modes
    candidates = np.flatnonzero(frequencies > min_frequency)
    dominant = candidates[np.argmax(amplitude[candidates])]
    f0 = frequencies[dominant]

    # Modes at f0, 2 f0, 3 f0, ... (within +/- tolerance * f0), both conjugates
    harmonic_number = np.round(frequencies / f0)
    is_harmonic = (harmonic_number >= 1) & (np.abs(frequencies - harmonic_number * f0) <= harmonic_tolerance * f0)

    n_sensors = X.shape[0]
    reconstruction = dmd_reconstruct(result, H.shape[1], stabilize=True)[:n_sensors]
    error = np.linalg.norm(H[:n_sensors] - reconstruction) / np.linalg.norm(H[:n_sensors])

    return {
        "dmd_max_abs_eigenvalue": np.abs(result["eigenvalues"]).max(),  # diagnostic: > 1 = growing mode
        "dmd_dominant_freq": f0,
        "dmd_dominant_abs_eigenvalue": np.abs(result["eigenvalues"][dominant]),
        "dmd_dominant_growth_rate": result["growth_rates"][dominant],
        "dmd_harmonic_amplitude_fraction": amplitude[is_harmonic].sum() / amplitude.sum(),
        "dmd_reconstruction_error": error,
    }
