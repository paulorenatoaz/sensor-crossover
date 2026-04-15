# ---------------------------------------------------------------------------
# labeling.py — Construct binary labels from the reference sensor
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd


def create_labels(matrix: pd.DataFrame, R: int) -> tuple[pd.Series, float]:
    """Create binary labels from reference sensor R using the median threshold.

    Parameters
    ----------
    matrix : DataFrame (epoch × moteid)
    R : int, mote ID of the reference sensor

    Returns
    -------
    Y : pd.Series of int (0 or 1), indexed by epoch
    median_threshold : float
    """
    x_r = matrix[R]
    median_threshold = float(x_r.median())

    Y = (x_r > median_threshold).astype(int)

    n_pos = int(Y.sum())
    n_neg = len(Y) - n_pos
    print(f"Labels: class 0 = {n_neg}, class 1 = {n_pos}  "
          f"(median threshold = {median_threshold:.2f}°C)")
    return Y, median_threshold
