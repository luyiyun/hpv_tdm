from __future__ import annotations

import numpy as np


def compute_rho(
    agebins: np.ndarray,
    sex_window: int = 10,
    decline_rate: float = 0.05,
    n: int = 100,
    bounds: tuple[int, int] = (13, 60),
) -> np.ndarray:
    mat = np.eye(n)
    for offset in range(1, sex_window + 1):
        row = np.arange(0, n - offset)
        col = np.arange(offset, n)
        value = 1 - decline_rate * offset
        mat[row, col] = value
        mat[col, row] = value

    lower_bound, upper_bound = bounds
    mat[:, :lower_bound] = 0
    mat[:, upper_bound:] = 0
    mat[:lower_bound, :] = 0
    mat[upper_bound:, :] = 0

    coarse_bins = np.r_[agebins[:-1].astype(int), n]
    num_ages = len(coarse_bins) - 1
    result = np.zeros((num_ages, num_ages), dtype=float)
    for row_index, (row0, row1) in enumerate(zip(coarse_bins[:-1], coarse_bins[1:])):
        for col_index, (col0, col1) in enumerate(
            zip(coarse_bins[:-1], coarse_bins[1:])
        ):
            result[row_index, col_index] = mat[row0:row1, col0:col1].sum()

    row_sum = result.sum(axis=1, keepdims=True)
    return np.divide(result, row_sum, out=np.zeros_like(result), where=row_sum != 0)
