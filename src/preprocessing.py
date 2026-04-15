# ---------------------------------------------------------------------------
# preprocessing.py — Temporal synchronization and gap filling
# ---------------------------------------------------------------------------

import pandas as pd
import config as cfg


def pivot_to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format data into a (epoch × moteid) temperature matrix.

    Each row is one epoch (time step); each column is one sensor.
    """
    matrix = df.pivot_table(
        index="epoch", columns="moteid", values="temperature", aggfunc="mean"
    )
    matrix = matrix.sort_index()
    return matrix


def fill_small_gaps(matrix: pd.DataFrame, max_gap: int = cfg.MAX_GAP_FILL) -> pd.DataFrame:
    """Forward-fill gaps of at most *max_gap* consecutive NaNs per sensor."""
    return matrix.ffill(limit=max_gap)


def filter_epochs(
    matrix: pd.DataFrame, required_sensors: list[int]
) -> pd.DataFrame:
    """Drop epochs where any of *required_sensors* is still NaN."""
    mask = matrix[required_sensors].notna().all(axis=1)
    filtered = matrix.loc[mask]
    print(
        f"Kept {len(filtered)}/{len(matrix)} epochs "
        f"(required sensors: {required_sensors})"
    )
    return filtered
