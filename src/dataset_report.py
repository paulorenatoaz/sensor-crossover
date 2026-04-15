# ---------------------------------------------------------------------------
# dataset_report.py — Generate self-contained HTML dataset report
# ---------------------------------------------------------------------------

import base64
import io
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from src.report import CSS


def _fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _make_floor_plan(locs: pd.DataFrame, stats: pd.DataFrame,
                     corr: pd.DataFrame, selected: dict) -> str:
    """Create a floor plan scatter plot of sensors, return as base64."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # All sensors as small gray dots
    merged = locs.merge(stats, left_on="moteid", right_index=True, how="inner")
    ax.scatter(merged["x"], merged["y"], s=30, c="#cbd5e1", edgecolors="#94a3b8",
               zorder=2, linewidths=0.5)

    # Labels for all
    for _, row in merged.iterrows():
        ax.annotate(str(int(row["moteid"])), (row["x"], row["y"]),
                     fontsize=6, ha="center", va="bottom",
                     xytext=(0, 4), textcoords="offset points", color="#64748b")

    # Highlight selected sensors
    colors = {
        "R": "#ef4444", "A": "#3b82f6",
        "B_high_corr": "#22c55e", "B_mid_corr": "#f59e0b", "B_low_corr": "#8b5cf6",
    }
    labels_map = {
        "R": "R (reference)", "A": "A (primary)",
        "B_high_corr": "B high-corr", "B_mid_corr": "B mid-corr", "B_low_corr": "B low-corr",
    }
    for key, sid in selected.items():
        loc = locs[locs["moteid"] == sid]
        if loc.empty:
            continue
        x, y = float(loc["x"].iloc[0]), float(loc["y"].iloc[0])
        ax.scatter(x, y, s=120, c=colors[key], edgecolors="black",
                   linewidths=1.2, zorder=5, label=labels_map[key])
        ax.annotate(f"mote {sid}", (x, y), fontsize=7, fontweight="bold",
                     ha="center", va="bottom", xytext=(0, 8),
                     textcoords="offset points", color=colors[key])

    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.set_title("Intel Lab — Sensor Floor Plan")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_corr_heatmap(corr: pd.DataFrame, selected: dict) -> str:
    """Create a correlation heatmap for the selected sensors, return as base64."""
    sel_ids = [selected[k] for k in ["R", "A", "B_high_corr", "B_mid_corr", "B_low_corr"]]
    sel_labels = [f"mote {s}" for s in sel_ids]
    sub = corr.loc[sel_ids, sel_ids].values

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sub, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(sel_labels)))
    ax.set_xticklabels(sel_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sel_labels)))
    ax.set_yticklabels(sel_labels, fontsize=8)

    # Annotate cells
    for i in range(len(sel_ids)):
        for j in range(len(sel_ids)):
            val = sub[i, j]
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson ρ")
    ax.set_title("Correlation Matrix — Selected Sensors")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_temp_distribution(matrix: pd.DataFrame, selected: dict) -> str:
    """Create temperature distribution plot for selected sensors."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {
        "R": "#ef4444", "A": "#3b82f6",
        "B_high_corr": "#22c55e", "B_mid_corr": "#f59e0b", "B_low_corr": "#8b5cf6",
    }
    labels_map = {
        "R": "R", "A": "A",
        "B_high_corr": "B high", "B_mid_corr": "B mid", "B_low_corr": "B low",
    }
    for key in ["R", "A", "B_high_corr", "B_mid_corr", "B_low_corr"]:
        sid = selected[key]
        if sid in matrix.columns:
            data = matrix[sid].dropna()
            ax.hist(data, bins=50, alpha=0.5, color=colors[key],
                    label=f"{labels_map[key]} (mote {sid})", density=True)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.set_title("Temperature Distributions — Selected Sensors")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


EXTRA_CSS = """
.highlight-row { background: #eff6ff !important; font-weight: 600; }
.stats-table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.82rem; }
.stats-table th, .stats-table td {
    padding: 0.35rem 0.6rem; border: 1px solid var(--border); text-align: right;
}
.stats-table th { background: var(--accent); color: white; font-weight: 600; position: sticky; top: 0; }
.stats-table td:first-child { text-align: center; font-weight: 600; }
.stats-table tbody tr:nth-child(even) { background: var(--card-bg); }
.criteria-box {
    background: var(--card-bg); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 4px;
    padding: 1rem 1.2rem; margin: 1rem 0; font-size: 0.9rem;
}
.criteria-box code { background: #e2e8f0; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.85em; }
"""


def generate_dataset_report(
    matrix: pd.DataFrame,
    stats: pd.DataFrame,
    corr: pd.DataFrame,
    locs: pd.DataFrame,
    sensors: dict,
    median_thr: float,
    n_raw_readings: int,
    n_epochs_before: int,
    n_epochs_after: int,
    output_path: str | None = None,
):
    """Generate a self-contained HTML dataset report.

    Parameters
    ----------
    matrix : filtered epoch×sensor DataFrame
    stats : per-sensor statistics from compute_sensor_stats
    corr : full correlation matrix
    locs : mote locations DataFrame
    sensors : dict with R, A, B_high_corr, B_mid_corr, B_low_corr
    median_thr : median threshold for label creation
    n_raw_readings : total clean readings from raw data
    n_epochs_before : matrix rows before epoch filtering
    n_epochs_after : matrix rows after epoch filtering
    output_path : where to save the HTML
    """
    output_path = output_path or os.path.join(cfg.RESULTS_DIR, "dataset.html")

    selected = {k: sensors[k] for k in ["R", "A", "B_high_corr", "B_mid_corr", "B_low_corr"]}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Generate figures
    floor_plan_img = _make_floor_plan(locs, stats, corr, selected)
    corr_heatmap_img = _make_corr_heatmap(corr, selected)
    temp_dist_img = _make_temp_distribution(matrix, selected)

    # Compute some useful numbers
    n_sensors = len(stats)
    n_selected_sensors = len(matrix.columns)
    n_epochs = len(matrix)
    ref_mean = stats.loc[selected["R"], "mean"]
    ref_std = stats.loc[selected["R"], "std"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Report — Intel Lab Sensor Data</title>
<style>{CSS}{EXTRA_CSS}</style>
</head>
<body>

<h1>Dataset Report — Intel Lab Sensor Data</h1>
<p class="subtitle">
    Detailed data profiling and sensor selection rationale<br>
    Generated {timestamp} &mdash; <a href="index.html">&larr; Back to Experiment Report</a>
</p>

<nav class="nav">
    <a href="#source">Data Source</a>
    <a href="#processing">Processing</a>
    <a href="#sensors">All Sensors</a>
    <a href="#floorplan">Floor Plan</a>
    <a href="#correlations">Correlations</a>
    <a href="#selection">Selection</a>
    <a href="#selected">Selected Sensors</a>
</nav>

<h2 id="source">Data Source</h2>

<div class="meta-grid">
    <div class="meta-card">
        <div class="label">Dataset</div>
        <div class="value">Intel Lab</div>
        <div>Indoor sensor network deployment at Intel Berkeley Research Lab (2004)</div>
    </div>
    <div class="meta-card">
        <div class="label">Source</div>
        <div class="value">MIT CSAIL</div>
        <div><a href="http://db.csail.mit.edu/labdata/labdata.html">db.csail.mit.edu/labdata</a></div>
    </div>
    <div class="meta-card">
        <div class="label">Sensors</div>
        <div class="value">{n_sensors} motes</div>
        <div>Mica2Dot wireless sensor nodes measuring temperature, humidity, light, voltage</div>
    </div>
    <div class="meta-card">
        <div class="label">Modality used</div>
        <div class="value">Temperature</div>
        <div>Readings in &deg;C, filtered to [{cfg.TEMP_MIN}, {cfg.TEMP_MAX}]</div>
    </div>
</div>

<p>
The Intel Berkeley Research Lab dataset contains readings from 54 sensors deployed across
a lab environment. Each sensor (mote) reports temperature, humidity, light, and voltage
at approximately 31-second intervals. We use only the <strong>temperature</strong> modality
to study classification performance as a function of feature dimensionality.
</p>

<h2 id="processing">Data Processing Pipeline</h2>

<div class="meta-grid">
    <div class="meta-card">
        <div class="label">Raw readings</div>
        <div class="value">{n_raw_readings:,}</div>
        <div>After cleaning (valid mote IDs, plausible temperatures)</div>
    </div>
    <div class="meta-card">
        <div class="label">Epochs (before filter)</div>
        <div class="value">{n_epochs_before:,}</div>
        <div>Time-synchronized rows (epoch &times; sensor matrix)</div>
    </div>
    <div class="meta-card">
        <div class="label">Epochs (after filter)</div>
        <div class="value">{n_epochs_after:,}</div>
        <div>Keeping only epochs where all 5 selected sensors are present</div>
    </div>
    <div class="meta-card">
        <div class="label">Gap filling</div>
        <div class="value">&le; {cfg.MAX_GAP_FILL} epochs</div>
        <div>Forward-fill small gaps per sensor before filtering</div>
    </div>
</div>

<div class="criteria-box">
    <strong>Cleaning steps:</strong>
    <ol style="margin-top:0.5rem;">
        <li>Parse raw tab-separated readings; drop malformed rows</li>
        <li>Filter mote IDs to valid range [{cfg.VALID_MOTE_RANGE[0]}, {cfg.VALID_MOTE_RANGE[1]}]</li>
        <li>Filter temperatures to [{cfg.TEMP_MIN}&deg;C, {cfg.TEMP_MAX}&deg;C]</li>
        <li>Pivot to epoch &times; sensor matrix</li>
        <li>Forward-fill gaps &le; {cfg.MAX_GAP_FILL} consecutive NaN epochs</li>
        <li>Drop epochs missing any of the 5 required sensors</li>
    </ol>
</div>

<h3>Label Construction</h3>
<p>
Binary labels are created by splitting on the <strong>median temperature</strong> of the
reference sensor R (mote {selected['R']}): class 0 if T &le; {median_thr:.2f}&deg;C,
class 1 if T &gt; {median_thr:.2f}&deg;C. This yields approximately balanced classes
(50/50 by construction).
</p>

<h2 id="sensors">All Sensor Statistics</h2>

<p>Per-sensor statistics computed on the full matrix ({n_epochs_before:,} epochs, before filtering).
Highlighted rows are the selected sensors.</p>
"""

    # All-sensor stats table
    sel_ids = set(selected.values())
    html += """<table class="stats-table">
    <thead><tr>
        <th>Mote</th><th>Readings</th><th>Missing %</th>
        <th>Mean (&deg;C)</th><th>Std (&deg;C)</th>
        <th>&rho;(&#183;, R)</th><th>&rho;(&#183;, A)</th><th>Role</th>
    </tr></thead>
    <tbody>
"""
    R_id, A_id = selected["R"], selected["A"]
    role_map = {v: k for k, v in selected.items()}

    for sid in sorted(stats.index):
        s = stats.loc[sid]
        rho_r = corr.loc[sid, R_id] if sid != R_id else 1.0
        rho_a = corr.loc[sid, A_id] if sid != A_id else 1.0
        role = role_map.get(sid, "")
        role_display = {
            "R": "R (reference)", "A": "A (primary)",
            "B_high_corr": "B<sub>high</sub>",
            "B_mid_corr": "B<sub>mid</sub>",
            "B_low_corr": "B<sub>low</sub>",
        }.get(role, "")
        row_cls = ' class="highlight-row"' if sid in sel_ids else ""
        html += f"""<tr{row_cls}>
            <td>{int(sid)}</td>
            <td>{int(s['count']):,}</td>
            <td>{s['missing_frac']*100:.1f}%</td>
            <td>{s['mean']:.2f}</td>
            <td>{s['std']:.2f}</td>
            <td>{rho_r:.3f}</td>
            <td>{rho_a:.3f}</td>
            <td>{role_display}</td>
        </tr>\n"""

    html += "</tbody></table>\n"

    # Floor plan
    html += f"""
<h2 id="floorplan">Sensor Floor Plan</h2>
<p>Spatial layout of the 54 motes in the Intel Berkeley Research Lab.
Selected sensors are highlighted with color coding.</p>
<div class="figure-row single">
    <img src="{floor_plan_img}" alt="Sensor floor plan">
</div>
"""

    # Correlations
    html += f"""
<h2 id="correlations">Correlation Structure</h2>
<p>Pearson correlation matrix among the 5 selected sensors.
High correlation between A and B indicates redundancy; low correlation indicates complementary information.</p>
<div class="figure-row single">
    <img src="{corr_heatmap_img}" alt="Correlation heatmap">
</div>
"""

    # Temperature distributions
    html += f"""
<h3>Temperature Distributions</h3>
<p>Kernel density estimates of temperature readings for each selected sensor.
Sensors with similar distributions are likely correlated; those with different shapes
or offsets carry complementary information for classification.</p>
<div class="figure-row single">
    <img src="{temp_dist_img}" alt="Temperature distributions">
</div>
"""

    # Selection criteria
    html += f"""
<h2 id="selection">Sensor Selection Criteria</h2>

<p>Sensors are selected automatically based on data quality and inter-sensor correlation structure.
The goal is to create three experimental scenarios with different levels of feature redundancy.</p>

<div class="criteria-box">
    <strong>Step 1 — Reference sensor R</strong><br>
    Criteria: <code>missing_frac &lt; {cfg.MISSING_FRAC_MAX_REF}</code>, then highest standard deviation.<br>
    Rationale: R must be reliable (low missing data) and informative (high variance gives a meaningful binary split).
</div>

<div class="criteria-box">
    <strong>Step 2 — Primary feature A</strong><br>
    Criteria: <code>missing_frac &lt; {cfg.MISSING_FRAC_MAX_FEAT}</code>,
    <code>&rho;(A, R) &ge; {cfg.CORR_A_MIN}</code>, then highest &rho;(A, R).<br>
    Rationale: A is the single best feature for predicting the label derived from R.
    High correlation with R ensures good baseline classification with d=1.
</div>

<div class="criteria-box">
    <strong>Step 3a — B<sub>high</sub> (high-correlation scenario)</strong><br>
    Criteria: <code>&rho;(B, A) &ge; {cfg.CORR_B_HIGH_A_MIN}</code> and
    <code>&rho;(B, R) &ge; {cfg.CORR_B_HIGH_R_MIN}</code>.<br>
    Rationale: B is nearly redundant with A. Adding it should help little asymptotically
    but hurt at small n (curse of dimensionality) &mdash; producing a <strong>late crossover</strong>.
</div>

<div class="criteria-box">
    <strong>Step 3b — B<sub>mid</sub> (mid-correlation scenario)</strong><br>
    Criteria: <code>{cfg.CORR_B_MID_R_RANGE[0]} &le; &rho;(B, R) &le; {cfg.CORR_B_MID_R_RANGE[1]}</code>
    and <code>&rho;(B, A) &le; {cfg.CORR_B_MID_A_MAX}</code>.<br>
    Rationale: B has moderate correlation — not fully redundant, not fully complementary.
    This is the regime where <strong>double crossover</strong> may occur:
    d=2 starts worse (small n), becomes better (mid n), then the advantage saturates or reverses.
</div>

<div class="criteria-box">
    <strong>Step 3c — B<sub>low</sub> (low-correlation scenario)</strong><br>
    Criteria: <code>&rho;(B, R) &le; {cfg.CORR_B_LOW_R_MAX}</code> and
    <code>&rho;(B, A) &le; {cfg.CORR_B_LOW_A_MAX}</code>.<br>
    Rationale: B provides genuinely new information. Adding it should help quickly,
    producing an <strong>early crossover</strong> at small n.
</div>
"""

    # Selected sensors detail
    html += """
<h2 id="selected">Selected Sensors — Detail</h2>
"""
    detail_rows = [
        ("R (reference)", "R"),
        ("A (primary feature)", "A"),
        ("B<sub>high</sub> (high-correlation)", "B_high_corr"),
        ("B<sub>mid</sub> (mid-correlation)", "B_mid_corr"),
        ("B<sub>low</sub> (low-correlation)", "B_low_corr"),
    ]
    for label, key in detail_rows:
        sid = selected[key]
        s = stats.loc[sid]
        rho_r = corr.loc[sid, R_id] if sid != R_id else 1.0
        rho_a = corr.loc[sid, A_id] if sid != A_id else 1.0
        loc_row = locs[locs["moteid"] == sid]
        loc_str = f"({float(loc_row['x'].iloc[0])}, {float(loc_row['y'].iloc[0])})" if not loc_row.empty else "unknown"

        html += f"""
<div class="scenario-section">
    <h3>{label} &mdash; Mote {sid}</h3>
    <table class="fit-table">
        <tr><td>Mote ID</td><td>{sid}</td></tr>
        <tr><td>Location (x, y)</td><td>{loc_str}</td></tr>
        <tr><td>Valid readings</td><td>{int(s['count']):,}</td></tr>
        <tr><td>Missing fraction</td><td>{s['missing_frac']*100:.1f}%</td></tr>
        <tr><td>Mean temperature</td><td>{s['mean']:.2f} &deg;C</td></tr>
        <tr><td>Std deviation</td><td>{s['std']:.2f} &deg;C</td></tr>
        <tr><td>&rho;(mote {sid}, R={R_id})</td><td>{rho_r:.4f}</td></tr>
        <tr><td>&rho;(mote {sid}, A={A_id})</td><td>{rho_a:.4f}</td></tr>
    </table>
</div>
"""

    # Footer
    html += f"""
<footer>
    Dataset Report &mdash; Intel Lab Sensor Data &mdash; {timestamp}<br>
    <a href="index.html">&larr; Back to Experiment Report</a>
</footer>

</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dataset report saved to {output_path}")
    return output_path
