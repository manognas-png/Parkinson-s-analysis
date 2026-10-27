"""Linear-algebra features of the 16-sensor gait signals (SVD / PCA)."""

import numpy as np

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

    Computed through the small 16 x 16 Gram matrix: if X = U S Vt then
    X Xt = U S^2 Ut, so U and S^2 are its eigenvectors and eigenvalues,
    and Vt = S^-1 Ut X. Much faster than np.linalg.svd for 16 x 12000
    matrices (which is slow under multithreaded OpenBLAS on the cluster).
    """
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
    """Two summary numbers for one trial: modes for `target` energy, energy in mode 1."""
    _, s, _ = svd(preprocess(sensor_matrix(trial), standardize))
    return {
        "n_components_90": n_components_for_energy(s, target),
        "energy_mode1": energy_fractions(s)[0],
    }
