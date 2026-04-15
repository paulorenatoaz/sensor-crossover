# ---------------------------------------------------------------------------
# experiment.py — Train/test splitting, Monte Carlo loop, crossover estimation
# ---------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import config as cfg


def stratified_split(
    matrix: pd.DataFrame,
    Y: pd.Series,
    test_frac: float = cfg.TEST_FRACTION,
    seed: int = cfg.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified random split into training pool and test set.

    Returns (X_pool, Y_pool, X_test, Y_test) as numpy arrays.
    """
    X_all = matrix.values
    Y_all = Y.values

    X_pool, X_test, Y_pool, Y_test = train_test_split(
        X_all, Y_all, test_size=test_frac, random_state=seed, stratify=Y_all,
    )

    print(f"Stratified split: pool={len(X_pool)}, test={len(X_test)}")
    print(f"  Pool class balance:  0={int((Y_pool==0).sum())} / 1={int((Y_pool==1).sum())}")
    print(f"  Test class balance:  0={int((Y_test==0).sum())} / 1={int((Y_test==1).sum())}")
    return X_pool, Y_pool, X_test, Y_test


def _get_feature_indices(
    matrix_columns: pd.Index, sensor_ids: list[int]
) -> list[int]:
    """Map sensor mote IDs to column indices in the matrix."""
    return [matrix_columns.get_loc(s) for s in sensor_ids]


def _make_model(model_name: str):
    if model_name == "svm":
        return LinearSVC(C=cfg.SVM_C, max_iter=cfg.SVM_MAX_ITER, dual="auto")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_trial(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    n: int,
    feature_idx: list[int],
    model_name: str,
    rng: np.random.Generator,
) -> float:
    """Run one trial: subsample, standardize, train, evaluate.

    Returns classification error rate on the test set.
    """
    # Stratified subsample of size n from the pool
    pool_size = len(X_pool)
    indices = np.arange(pool_size)

    # Stratified: sample proportionally from each class
    idx_0 = indices[Y_pool == 0]
    idx_1 = indices[Y_pool == 1]
    n_0 = n // 2
    n_1 = n - n_0
    chosen_0 = rng.choice(idx_0, size=min(n_0, len(idx_0)), replace=False)
    chosen_1 = rng.choice(idx_1, size=min(n_1, len(idx_1)), replace=False)
    chosen = np.concatenate([chosen_0, chosen_1])

    X_tr = X_pool[chosen][:, feature_idx]
    Y_tr = Y_pool[chosen]
    X_te = X_test[:, feature_idx]

    # Standardize using training statistics only
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-12] = 1.0  # avoid division by zero

    Z_tr = (X_tr - mu) / sigma
    Z_te = (X_te - mu) / sigma

    # Train
    model = _make_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(Z_tr, Y_tr)
            preds = model.predict(Z_te)
            error = float(np.mean(preds != Y_test))
        except Exception:
            error = 0.5  # chance level on failure

    return error


def run_experiment(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    feature_idx: list[int],
    n_values: list[int] = cfg.N_VALUES,
    n_reps: int = cfg.N_REPS,
    model_name: str = "svm",
    seed: int = cfg.RANDOM_SEED,
) -> pd.DataFrame:
    """Run Monte Carlo experiment over all sample sizes.

    Returns DataFrame with columns [n, rep, error].
    """
    rng = np.random.default_rng(seed)
    records = []
    for n in n_values:
        for rep in range(n_reps):
            err = run_single_trial(
                X_pool, Y_pool, X_test, Y_test,
                n, feature_idx, model_name, rng,
            )
            records.append({"n": n, "rep": rep, "error": err})
    return pd.DataFrame(records)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate: mean and std of error per n."""
    agg = results.groupby("n")["error"].agg(["mean", "std"]).reset_index()
    agg.columns = ["n", "mean_error", "std_error"]
    return agg


def fit_crossover_model(
    summary_d1: pd.DataFrame, summary_d2: pd.DataFrame
) -> dict:
    """Fit Δ(n) = G + B/n + C/n² and return coefficients + crossover points.

    Returns dict with keys: G, B, C, disc, roots (list of n* values),
    r2, ns, deltas, deltas_fit.
    """
    ns = summary_d1["n"].values.astype(float)
    deltas = summary_d1["mean_error"].values - summary_d2["mean_error"].values
    x = 1.0 / ns

    coeffs = np.polyfit(x, deltas, 2)
    C, B, G = coeffs
    disc = B**2 - 4 * G * C

    roots = []
    if disc > 0:
        for r in [(-B + np.sqrt(disc)) / (2 * C), (-B - np.sqrt(disc)) / (2 * C)]:
            if r > 0 and 1 / r >= 2:
                roots.append(round(1 / r, 1))
    roots.sort()

    delta_pred = G + B * x + C * x**2
    ss_res = np.sum((deltas - delta_pred) ** 2)
    ss_tot = np.sum((deltas - np.mean(deltas)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "G": G, "B": B, "C": C,
        "disc": disc,
        "roots": roots,
        "r2": r2,
        "ns": ns.tolist(),
        "deltas": deltas.tolist(),
        "deltas_fit": delta_pred.tolist(),
    }


def estimate_crossover(
    summary_d1: pd.DataFrame, summary_d2: pd.DataFrame
) -> list[float]:
    """Estimate all n* crossover points by linear interpolation of sign changes.

    Finds where the d=1 and d=2 error curves actually cross by detecting
    sign changes in delta = E(d=1) - E(d=2) and interpolating between
    adjacent sample sizes.

    Returns a sorted list of n* values (may be empty, 1, or 2 elements).
    """
    ns = summary_d1["n"].values.astype(float)
    delta = summary_d1["mean_error"].values - summary_d2["mean_error"].values

    crossings = []
    for i in range(len(delta) - 1):
        if delta[i] * delta[i + 1] < 0:  # sign change
            t = -delta[i] / (delta[i + 1] - delta[i])
            n_star = ns[i] + t * (ns[i + 1] - ns[i])
            crossings.append(round(float(n_star), 1))

    return sorted(crossings)


def compute_delta_stats(
    raw_d1: pd.DataFrame, raw_d2: pd.DataFrame
) -> pd.DataFrame:
    """Compute paired Δ statistics at each sample size.

    Parameters
    ----------
    raw_d1, raw_d2 : DataFrames from run_experiment() with [n, rep, error].
        Both must use the same (n, rep) pairs (shared RNG seed).

    Returns
    -------
    DataFrame with columns:
        n, mean_delta, std_delta, ci_lo, ci_hi, p_value, cohens_d,
        frac_d2_wins
    """
    merged = raw_d1.merge(raw_d2, on=["n", "rep"], suffixes=("_d1", "_d2"))
    merged["delta"] = merged["error_d1"] - merged["error_d2"]

    records = []
    for n_val, grp in merged.groupby("n"):
        deltas = grp["delta"].values
        mean_d = deltas.mean()
        std_d = deltas.std(ddof=1)
        se = std_d / np.sqrt(len(deltas))

        # Wilcoxon signed-rank test (paired, nonparametric)
        try:
            _, p_val = wilcoxon(deltas)
        except ValueError:
            p_val = 1.0

        records.append({
            "n": int(n_val),
            "mean_delta": mean_d,
            "std_delta": std_d,
            "ci_lo": mean_d - 1.96 * se,
            "ci_hi": mean_d + 1.96 * se,
            "p_value": p_val,
            "cohens_d": mean_d / std_d if std_d > 0 else 0.0,
            "frac_d2_wins": float((deltas > 0).sum()) / len(deltas),
        })

    return pd.DataFrame(records)
