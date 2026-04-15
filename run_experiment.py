#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_experiment.py — Main orchestrator
# ---------------------------------------------------------------------------

import os
import sys
import json

import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from src.data_loader import download_data, load_raw_data, load_mote_locations
from src.preprocessing import pivot_to_matrix, fill_small_gaps, filter_epochs
from src.sensor_selection import select_all_sensors, compute_correlation_matrix
from src.labeling import create_labels
from src.experiment import (
    stratified_split, run_experiment, summarize_results, estimate_crossover,
    compute_delta_stats, _get_feature_indices,
)
from src.plotting import (
    plot_crossover, plot_scenario_comparison, plot_correlation_vs_crossover,
    plot_delta_with_ci,
)
from src.report import generate_report
from src.dataset_report import generate_dataset_report


def main():
    print("=" * 60)
    print("  Sensor Crossover — Real-Data Experimental Validation")
    print("=" * 60)

    # ── Stage 1-2: Data acquisition & cleaning ────────────────────────
    print("\n[1/8] Downloading data...")
    download_data()

    print("\n[2/8] Loading and cleaning data...")
    df = load_raw_data()
    n_raw_readings = len(df)

    # ── Stage 3: Synchronization ──────────────────────────────────────
    print("\n[3/8] Building synchronized time-sensor matrix...")
    matrix = pivot_to_matrix(df)
    matrix = fill_small_gaps(matrix)
    n_epochs_before = len(matrix)
    print(f"  Matrix shape before filtering: {matrix.shape}")

    # ── Stage 4: Sensor selection ─────────────────────────────────────
    print("\n[4/8] Selecting sensors...")
    sensors = select_all_sensors(matrix)

    # Get all required sensors and filter matrix to epochs where all are present
    required = [
        sensors["R"], sensors["A"],
        sensors["B_high_corr"], sensors["B_mid_corr"], sensors["B_low_corr"],
    ]
    matrix = filter_epochs(matrix, required)
    print(f"  Matrix shape after filtering: {matrix.shape}")

    # ── Stage 5: Label construction ───────────────────────────────────
    print("\n[5/8] Creating binary labels...")
    Y, median_thr = create_labels(matrix, sensors["R"])

    # ── Stage 6: Train/test split ─────────────────────────────────────
    print("\n[6/8] Splitting data (stratified)...")
    X_pool, Y_pool, X_test, Y_test = stratified_split(matrix, Y)

    # ── Stage 7-8-9: Experiments ──────────────────────────────────────
    print("\n[7/8] Running experiments...")

    scenarios = {
        "high-correlation":  sensors["B_high_corr"],
        "mid-correlation":   sensors["B_mid_corr"],
        "low-correlation":   sensors["B_low_corr"],
    }

    A_idx = _get_feature_indices(matrix.columns, [sensors["A"]])

    summaries = {}
    n_stars = {}
    raw_results = {}
    delta_stats = {}

    for scenario_name, B_id in scenarios.items():
        print(f"\n  ── Scenario: {scenario_name} ──")

        AB_idx = _get_feature_indices(matrix.columns, [sensors["A"], B_id])

        # d = 1
        print(f"    d=1 (sensor A={sensors['A']}) ...")
        raw_d1 = run_experiment(
            X_pool, Y_pool, X_test, Y_test,
            feature_idx=A_idx, model_name="svm",
        )
        sum_d1 = summarize_results(raw_d1)

        # d = 2
        print(f"    d=2 (sensors A={sensors['A']}, B={B_id}) ...")
        raw_d2 = run_experiment(
            X_pool, Y_pool, X_test, Y_test,
            feature_idx=AB_idx, model_name="svm",
        )
        sum_d2 = summarize_results(raw_d2)

        # Crossover
        n_star_list = estimate_crossover(sum_d1, sum_d2)
        if n_star_list:
            stars_str = ", ".join(f"{v:.1f}" for v in n_star_list)
            print(f"    → Estimated crossover n* ≈ {stars_str}")
        else:
            print(f"    → No crossover detected in this range")

        # Paired delta statistics (Wilcoxon, CI)
        dstats = compute_delta_stats(raw_d1, raw_d2)

        summaries[scenario_name] = {"d1": sum_d1, "d2": sum_d2}
        n_stars[scenario_name] = n_star_list
        raw_results[scenario_name] = {"d1": raw_d1, "d2": raw_d2}
        delta_stats[scenario_name] = dstats

    # ── Stage 10: Visualization ───────────────────────────────────────
    print("\n[8/8] Generating figures...")
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    os.makedirs(cfg.TABLES_DIR, exist_ok=True)

    # Plot 1 — One crossover plot per scenario
    for scenario_name, data in summaries.items():
        plot_crossover(
            data["d1"], data["d2"],
            scenario_name=scenario_name.replace("-", " ").title(),
            n_stars=n_stars[scenario_name],
            save_path=os.path.join(
                cfg.FIGURES_DIR, f"crossover_{scenario_name}.pdf"
            ),
        )

    # Plot 2 — Scenario comparison
    plot_scenario_comparison(
        summaries,
        save_path=os.path.join(cfg.FIGURES_DIR, "scenario_comparison.pdf"),
    )

    # Plot 3 — Correlation vs crossover
    corr = sensors["_corr"]
    R = sensors["R"]
    corr_values = [corr.loc[scenarios[s], R] for s in scenarios]
    # Use first root for each scenario for the scatter plot
    n_star_values = [n_stars[s][0] if n_stars[s] else None for s in scenarios]
    labels_list = list(scenarios.keys())
    plot_correlation_vs_crossover(
        corr_values, n_star_values, labels_list,
        save_path=os.path.join(cfg.FIGURES_DIR, "corr_vs_crossover.pdf"),
    )

    # Plot 4 — Delta with confidence intervals
    plot_delta_with_ci(
        delta_stats,
        n_stars=n_stars,
        save_path=os.path.join(cfg.FIGURES_DIR, "delta_ci.pdf"),
    )

    # ── Save delta_stats ──────────────────────────────────────────────
    dstats_rows = []
    for sc_name, dstats_df in delta_stats.items():
        for _, row in dstats_df.iterrows():
            dstats_rows.append({"scenario": sc_name, **row.to_dict()})
    dstats_all = pd.DataFrame(dstats_rows)
    dstats_path = os.path.join(cfg.TABLES_DIR, "delta_stats.csv")
    dstats_all.to_csv(dstats_path, index=False)
    print(f"Delta stats saved to {dstats_path}")

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    rows = []
    for scenario_name in scenarios:
        d1 = summaries[scenario_name]["d1"]
        d2 = summaries[scenario_name]["d2"]
        for _, row in d1.iterrows():
            rows.append({
                "model": "svm", "scenario": scenario_name, "d": 1,
                "n": int(row["n"]),
                "mean_error": round(row["mean_error"], 4),
                "std_error": round(row["std_error"], 4),
            })
        for _, row in d2.iterrows():
            rows.append({
                "model": "svm", "scenario": scenario_name, "d": 2,
                "n": int(row["n"]),
                "mean_error": round(row["mean_error"], 4),
                "std_error": round(row["std_error"], 4),
            })

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.TABLES_DIR, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Print condensed tables
    for scenario_name in scenarios:
        d1 = summaries[scenario_name]["d1"]
        d2 = summaries[scenario_name]["d2"]
        n_s = n_stars[scenario_name]
        n_s_str = ", ".join(f"{v:.0f}" for v in n_s) if n_s else "N/A"
        print(f"\n  {scenario_name.upper()} (B=mote {scenarios[scenario_name]}, n*={n_s_str})")
        print(f"  {'n':>6s}  {'err(d=1)':>12s}  {'err(d=2)':>12s}  {'Δ':>8s}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}")
        for i in range(len(d1)):
            n = int(d1.iloc[i]["n"])
            e1 = d1.iloc[i]["mean_error"]
            e2 = d2.iloc[i]["mean_error"]
            delta = e1 - e2
            print(f"  {n:6d}  {e1:8.4f}±{d1.iloc[i]['std_error']:.4f}"
                  f"  {e2:8.4f}±{d2.iloc[i]['std_error']:.4f}"
                  f"  {delta:+.4f}")

    # Save metadata (including correlations for the report)
    A = sensors["A"]
    correlations = {}
    for sc_name, B_id in scenarios.items():
        correlations[sc_name] = {
            "rho_BR": round(float(corr.loc[B_id, R]), 4),
            "rho_BA": round(float(corr.loc[B_id, A]), 4),
            "rho_AR": round(float(corr.loc[A, R]), 4),
        }

    meta = {
        "sensors": {k: int(v) for k, v in sensors.items() if not k.startswith("_")},
        "median_threshold": median_thr,
        "correlations": correlations,
        "n_star": {k: v for k, v in n_stars.items()},
        "config": {
            "n_values": cfg.N_VALUES,
            "n_reps": cfg.N_REPS,
            "svm_C": cfg.SVM_C,
            "test_fraction": cfg.TEST_FRACTION,
            "seed": cfg.RANDOM_SEED,
        },
    }
    meta_path = os.path.join(cfg.RESULTS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nMetadata saved to {meta_path}")

    # ── Stage 9: HTML reports ─────────────────────────────────────────
    print("\n── Generating HTML reports ──")
    report_path = generate_report()

    locs = load_mote_locations()
    dataset_report_path = generate_dataset_report(
        matrix=matrix,
        stats=sensors["_stats"],
        corr=corr,
        locs=locs,
        sensors=sensors,
        median_thr=median_thr,
        n_raw_readings=n_raw_readings,
        n_epochs_before=n_epochs_before,
        n_epochs_after=len(matrix),
    )

    print("\n✓ Done. Figures in", cfg.FIGURES_DIR)
    print("  Report at", report_path)
    print("  Dataset report at", dataset_report_path)


if __name__ == "__main__":
    main()
