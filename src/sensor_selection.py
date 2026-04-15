# ---------------------------------------------------------------------------
# sensor_selection.py — Profile sensors and select R, A, B candidates
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
import config as cfg


def compute_sensor_stats(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor statistics: count, mean, std, fraction of missing epochs."""
    total = len(matrix)
    stats = pd.DataFrame({
        "count": matrix.count(),
        "mean": matrix.mean(),
        "std": matrix.std(),
        "missing_frac": 1 - matrix.count() / total,
    })
    stats.index.name = "moteid"
    return stats.sort_index()


def compute_correlation_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation among all sensors."""
    return matrix.corr()


def select_reference(stats: pd.DataFrame) -> int:
    """Select reference sensor R: low missing data, high variability."""
    candidates = stats[stats["missing_frac"] < cfg.MISSING_FRAC_MAX_REF]
    if candidates.empty:
        # Relax threshold
        candidates = stats.nsmallest(5, "missing_frac")
    # Among candidates, pick the one with highest std
    R = int(candidates["std"].idxmax())
    print(f"Reference sensor R = {R}  "
          f"(missing={stats.loc[R, 'missing_frac']:.3f}, std={stats.loc[R, 'std']:.2f})")
    return R


def select_sensor_A(
    corr_matrix: pd.DataFrame, R: int, stats: pd.DataFrame
) -> int:
    """Select primary feature sensor A: high corr with R, low missing data."""
    candidates = stats[
        (stats["missing_frac"] < cfg.MISSING_FRAC_MAX_FEAT)
        & (stats.index != R)
    ]
    corrs_with_R = corr_matrix.loc[candidates.index, R]
    # Filter by minimum correlation
    valid = corrs_with_R[corrs_with_R >= cfg.CORR_A_MIN]
    if valid.empty:
        # Relax: pick top-3 by correlation
        valid = corrs_with_R.nlargest(3)
    A = int(valid.idxmax())
    print(f"Sensor A = {A}  (corr(A,R)={corr_matrix.loc[A, R]:.3f})")
    return A


def select_sensor_B(
    corr_matrix: pd.DataFrame,
    R: int,
    A: int,
    stats: pd.DataFrame,
    scenario: str,
) -> int:
    """Select secondary feature sensor B for a given scenario.

    Scenarios: 'high-correlation', 'mid-correlation', 'low-correlation'.
    """
    candidates = stats[
        (stats["missing_frac"] < cfg.MISSING_FRAC_MAX_FEAT)
        & (~stats.index.isin([R, A]))
    ].index

    corr_R = corr_matrix.loc[candidates, R]
    corr_A = corr_matrix.loc[candidates, A]

    if scenario == "high-correlation":
        mask = (corr_A >= cfg.CORR_B_HIGH_A_MIN) & (corr_R >= cfg.CORR_B_HIGH_R_MIN)
        if mask.sum() == 0:
            # Relax: pick highest corr(B, A)
            B = int(corr_A.idxmax())
        else:
            B = int(corr_A[mask].idxmax())

    elif scenario == "mid-correlation":
        lo, hi = cfg.CORR_B_MID_R_RANGE
        mask = (corr_R >= lo) & (corr_R <= hi) & (corr_A <= cfg.CORR_B_MID_A_MAX)
        if mask.sum() == 0:
            # Relax to wider range
            mask = (corr_R >= 0.3) & (corr_R <= 0.9) & (corr_A <= 0.85)
        if mask.sum() == 0:
            B = int(corr_R.idxmin())
        else:
            # Among valid, pick the one with highest corr(B, R) for informativeness
            B = int(corr_R[mask].idxmax())

    elif scenario == "low-correlation":
        mask = (corr_R <= cfg.CORR_B_LOW_R_MAX) & (corr_A <= cfg.CORR_B_LOW_A_MAX)
        if mask.sum() == 0:
            B = int(corr_R.idxmin())
        else:
            # Among valid, pick the one with lowest corr(B, R)
            B = int(corr_R[mask].idxmin())
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    print(
        f"Sensor B ({scenario}) = {B}  "
        f"(corr(B,R)={corr_matrix.loc[B, R]:.3f}, "
        f"corr(B,A)={corr_matrix.loc[B, A]:.3f})"
    )
    return B


def select_all_sensors(matrix: pd.DataFrame) -> dict:
    """Run full sensor selection pipeline.

    Returns dict with keys: R, A, B_high_corr, B_mid_corr, B_low_corr,
    plus metadata (correlations, stats).
    """
    stats = compute_sensor_stats(matrix)
    corr = compute_correlation_matrix(matrix)

    R = select_reference(stats)
    A = select_sensor_A(corr, R, stats)
    B_high = select_sensor_B(corr, R, A, stats, "high-correlation")
    B_mid = select_sensor_B(corr, R, A, stats, "mid-correlation")
    B_low = select_sensor_B(corr, R, A, stats, "low-correlation")

    sensors = {
        "R": R, "A": A,
        "B_high_corr": B_high,
        "B_mid_corr": B_mid,
        "B_low_corr": B_low,
    }

    print("\n=== Sensor Selection Summary ===")
    for key, sid in sensors.items():
        print(f"  {key:20s} = mote {sid:2d}  "
              f"(corr_R={corr.loc[sid, R]:.3f}, "
              f"corr_A={corr.loc[sid, A]:.3f}, "
              f"missing={stats.loc[sid, 'missing_frac']:.3f})")
    print()

    return {**sensors, "_stats": stats, "_corr": corr}
