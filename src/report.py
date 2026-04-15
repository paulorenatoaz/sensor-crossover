# ---------------------------------------------------------------------------
# report.py — Generate self-contained HTML report for GitHub Pages
# ---------------------------------------------------------------------------

import base64
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config as cfg


def _img_to_base64(path: str) -> str:
    """Read an image file and return a base64-encoded data URI."""
    ext = os.path.splitext(path)[1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "svg": "image/svg+xml"}.get(
        ext.lstrip("."), "image/png"
    )
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _interpolate_crossover(df: pd.DataFrame, scenario: str) -> list[float]:
    """Find crossover points by linear interpolation of the actual error vectors."""
    sc = df[(df["model"] == "svm") & (df["scenario"] == scenario)]
    d1 = sc[sc["d"] == 1].sort_values("n").reset_index(drop=True)
    d2 = sc[sc["d"] == 2].sort_values("n").reset_index(drop=True)
    if len(d1) == 0 or len(d2) == 0:
        return []
    ns = d1["n"].values.astype(float)
    delta = d1["mean_error"].values - d2["mean_error"].values
    crossings = []
    for i in range(len(delta) - 1):
        if delta[i] * delta[i + 1] < 0:
            t = -delta[i] / (delta[i + 1] - delta[i])
            crossings.append(round(float(ns[i] + t * (ns[i + 1] - ns[i])), 1))
    return sorted(crossings)


def _build_error_table_html(df: pd.DataFrame, scenario: str) -> str:
    """Build an HTML error table for one scenario."""
    sc = df[(df["model"] == "svm") & (df["scenario"] == scenario)]
    d1 = sc[sc["d"] == 1].sort_values("n").reset_index(drop=True)
    d2 = sc[sc["d"] == 2].sort_values("n").reset_index(drop=True)

    rows_html = ""
    for i in range(len(d1)):
        n = int(d1.loc[i, "n"])
        e1 = d1.loc[i, "mean_error"]
        s1 = d1.loc[i, "std_error"]
        e2 = d2.loc[i, "mean_error"]
        s2 = d2.loc[i, "std_error"]
        delta = e1 - e2
        winner_cls = "positive" if delta > 0 else "negative"
        rows_html += f"""<tr>
            <td>{n}</td>
            <td>{e1:.4f} &pm; {s1:.4f}</td>
            <td>{e2:.4f} &pm; {s2:.4f}</td>
            <td class="{winner_cls}">{delta:+.4f}</td>
        </tr>\n"""

    return f"""<table class="data-table">
        <thead><tr>
            <th>n</th><th>E(d=1)</th><th>E(d=2)</th><th>&Delta; (d=1 &minus; d=2)</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>"""


CSS = """
:root {
    --bg: #ffffff;
    --text: #1a1a2e;
    --accent: #2563eb;
    --border: #e2e8f0;
    --card-bg: #f8fafc;
    --green: #16a34a;
    --red: #dc2626;
    --gray: #64748b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}
h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
h2 { font-size: 1.4rem; margin-top: 2.5rem; margin-bottom: 1rem;
     padding-bottom: 0.3rem; border-bottom: 2px solid var(--accent); }
h3 { font-size: 1.15rem; margin-top: 1.5rem; margin-bottom: 0.5rem; color: var(--accent); }
p, li { margin-bottom: 0.5rem; }
.subtitle { color: var(--gray); font-size: 0.95rem; margin-bottom: 2rem; }
.meta-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem; margin: 1rem 0 2rem;
}
.meta-card {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem;
}
.meta-card .label { font-size: 0.8rem; color: var(--gray); text-transform: uppercase; letter-spacing: 0.05em; }
.meta-card .value { font-size: 1.3rem; font-weight: 600; margin-top: 0.2rem; }
.scenario-section {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0;
}
.figure-row {
    display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; margin: 1rem 0;
}
.figure-row img { max-width: 480px; width: 100%; border-radius: 4px; border: 1px solid var(--border); }
.figure-row.single img { max-width: 600px; }
table.data-table, table.fit-table {
    width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem;
}
table.data-table th, table.data-table td,
table.fit-table td {
    padding: 0.45rem 0.75rem; border: 1px solid var(--border); text-align: right;
}
table.data-table th { background: var(--accent); color: white; font-weight: 600; }
table.data-table tbody tr:nth-child(even) { background: var(--card-bg); }
table.fit-table td:first-child { text-align: left; font-weight: 500; width: 45%; }
td.positive { color: var(--green); font-weight: 600; }
td.negative { color: var(--red); font-weight: 600; }
.summary-table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
.summary-table th, .summary-table td {
    padding: 0.5rem 0.8rem; border: 1px solid var(--border); text-align: center;
}
.summary-table th { background: var(--accent); color: white; }
.summary-table tbody tr:nth-child(even) { background: var(--card-bg); }
.nav { position: sticky; top: 0; background: white; z-index: 10;
       border-bottom: 1px solid var(--border); padding: 0.5rem 0; margin-bottom: 1.5rem; }
.nav a { margin-right: 1.2rem; text-decoration: none; color: var(--accent); font-size: 0.9rem; }
.nav a:hover { text-decoration: underline; }
.badge {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
    font-size: 0.8rem; font-weight: 600;
}
.badge-single { background: #dbeafe; color: #1e40af; }
.badge-double { background: #fef3c7; color: #92400e; }
.badge-none   { background: #f1f5f9; color: #475569; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
         color: var(--gray); font-size: 0.85rem; text-align: center; }
"""


def generate_report(
    results_csv: str | None = None,
    metadata_json: str | None = None,
    figures_dir: str | None = None,
    output_path: str | None = None,
):
    """Generate a self-contained HTML report from experiment results.

    All images are embedded as base64 so the report is a single file.
    """
    results_csv = results_csv or os.path.join(cfg.TABLES_DIR, "results.csv")
    metadata_json = metadata_json or os.path.join(cfg.RESULTS_DIR, "metadata.json")
    figures_dir = figures_dir or cfg.FIGURES_DIR
    output_path = output_path or os.path.join(cfg.RESULTS_DIR, "index.html")

    df = pd.read_csv(results_csv)
    with open(metadata_json) as f:
        meta = json.load(f)

    sensors = meta["sensors"]
    config = meta["config"]
    n_stars = meta["n_star"]
    correlations = meta.get("correlations", {})

    scenarios = ["high-correlation", "mid-correlation", "low-correlation"]

    # ── Header ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sensor Crossover — Experiment Report</title>
<style>{CSS}</style>
</head>
<body>

<h1>Finite-Sample Crossover in Sensor Classification</h1>
<p class="subtitle">
    Intel Lab Dataset &mdash; Real-data validation of the fewer-features phenomenon<br>
    Generated {timestamp}
</p>

<nav class="nav">
    <a href="#motivation">Motivation</a>
    <a href="#methodology">Methodology</a>
    <a href="#setup">Setup</a>
    <a href="#summary">Summary</a>
    <a href="#results">Results</a>
    <a href="#comparison">Comparison</a>
    <a href="#analysis">Analysis</a>
    <a href="dataset.html">Dataset Report &rarr;</a>
</nav>

<h2 id="motivation">Motivation</h2>

<p>
In classification tasks, adding more features is generally expected to improve
accuracy &mdash; more information should yield better decisions. However, when
training data is <strong>limited</strong>, the opposite can happen: additional
features increase the effective dimensionality of the problem, demanding more
samples to estimate decision boundaries reliably. This creates a
<strong>finite-sample crossover</strong> where a lower-dimensional classifier
temporarily outperforms a higher-dimensional one.
</p>

<p>
This phenomenon is well known in asymptotic theory (the &ldquo;curse of
dimensionality&rdquo;), but its <em>practical manifestation</em> &mdash; the
specific sample size <strong>n*</strong> at which the benefit of an extra
feature overtakes the cost of estimating it &mdash; has received less empirical
attention on real sensor data.
</p>

<p>
This experiment investigates the crossover using <strong>real temperature
readings</strong> from the Intel Berkeley Research Lab. We fix one primary
sensor&nbsp;(A) and vary a second sensor&nbsp;(B) across three correlation
regimes &mdash; high, mid, and low &mdash; to answer:
</p>
<ol>
    <li>At what sample size <em>n*</em> does the 2-sensor classifier surpass
        the 1-sensor classifier?</li>
    <li>How does the correlation between sensors A and B affect <em>n*</em>?</li>
    <li>Can a <strong>double crossover</strong> occur, where the 2-sensor model
        is first better, then worse, then better again?</li>
</ol>

<h2 id="methodology">Methodology</h2>

<p>
We frame the problem as a binary classification task derived from real sensor
readings. The pipeline has four stages:
</p>

<h3>1. Data &amp; Label Construction</h3>
<p>
From the Intel Lab dataset (54 temperature motes, ~2.3&thinsp;M readings over
38 days), we build a synchronized time&times;sensor matrix after removing
outliers and filling small gaps (&le;&thinsp;3 consecutive NaNs). A
<strong>reference mote R</strong> defines binary classes via a median split of
its temperature readings: class&nbsp;0 if temperature &le; median, class&nbsp;1
otherwise.
</p>

<h3>2. Sensor Role Assignment</h3>
<p>
Sensors are automatically assigned to roles based on pairwise Pearson
correlation:
</p>
<ul>
    <li><strong>A</strong> (primary feature) &mdash; the mote most correlated
        with R (&rho; &ge; {cfg.CORR_A_MIN}).</li>
    <li><strong>B<sub>high</sub></strong> &mdash; high correlation with A
        (&rho;<sub>BA</sub> &ge; {cfg.CORR_B_HIGH_A_MIN}), acting as a
        near-redundant second feature.</li>
    <li><strong>B<sub>mid</sub></strong> &mdash; moderate range
        ({cfg.CORR_B_MID_R_RANGE[0]} &le; &rho;<sub>BR</sub> &le;
        {cfg.CORR_B_MID_R_RANGE[1]}), where the utility of the extra sensor
        is ambiguous.</li>
    <li><strong>B<sub>low</sub></strong> &mdash; low correlation
        (&rho;<sub>BR</sub> &le; {cfg.CORR_B_LOW_R_MAX}), providing a
        complementary but noisy feature.</li>
</ul>

<h3>3. Monte Carlo Evaluation</h3>
<p>
For each scenario (high / mid / low correlation) and dimensionality
(<em>d</em>&thinsp;=&thinsp;1 using sensor A only, or
<em>d</em>&thinsp;=&thinsp;2 using sensors A+B), we draw <em>n</em> training
samples uniformly from the pool ({cfg.N_REPS} repetitions per setting). The
classifier is evaluated on a fixed stratified hold-out set
({int(cfg.TEST_FRACTION*100)}% of the data). We record the mean and standard
deviation of classification error across repetitions.
</p>

<h3>4. Crossover Detection</h3>
<p>
The crossover point <em>n*</em> is estimated by linear interpolation: we
compute &Delta;(n) = E(d=1) &minus; E(d=2) at each sample size and locate
sign changes between consecutive points. Where &Delta; changes from negative
to positive, the 2-sensor model begins to outperform the 1-sensor model, and
vice versa.
</p>

"""

    # ── Setup section ─────────────────────────────────────────────────
    html += f"""
<h2 id="setup">Experiment Setup</h2>

<div class="meta-grid">
    <div class="meta-card">
        <div class="label">Dataset</div>
        <div class="value">Intel Lab</div>
        <div>54 temperature sensors, ~55K synchronized epochs</div>
    </div>
    <div class="meta-card">
        <div class="label">Classifier</div>
        <div class="value">Linear SVM</div>
        <div>C={config['svm_C']}, scikit-learn LinearSVC</div>
    </div>
    <div class="meta-card">
        <div class="label">Sample sizes</div>
        <div class="value">{config['n_values'][0]} &ndash; {config['n_values'][-1]}</div>
        <div>{len(config['n_values'])} values: {', '.join(str(v) for v in config['n_values'])}</div>
    </div>
    <div class="meta-card">
        <div class="label">Repetitions</div>
        <div class="value">{config['n_reps']}</div>
        <div>Monte Carlo trials per (n, d, scenario)</div>
    </div>
    <div class="meta-card">
        <div class="label">Test fraction</div>
        <div class="value">{config['test_fraction']}</div>
        <div>Stratified hold-out</div>
    </div>
</div>

<h3>Sensor Selection</h3>"""

    rho_AR = correlations.get("high-correlation", {}).get("rho_AR", "&mdash;")

    html += f"""
<table class="summary-table">
    <thead><tr>
        <th>Role</th><th>Mote ID</th><th>&rho;(&middot;, R)</th><th>&rho;(&middot;, A)</th><th>Description</th>
    </tr></thead>
    <tbody>
        <tr><td>R (reference / label)</td><td>{sensors['R']}</td>
            <td>1.000</td><td>{rho_AR}</td>
            <td>Median-split to create binary classes</td></tr>
        <tr><td>A (primary feature)</td><td>{sensors['A']}</td>
            <td>{rho_AR}</td><td>1.000</td>
            <td>Highest correlation with R</td></tr>"""

    b_rows = [
        ("B<sub>high</sub>", "B_high_corr", "high-correlation", "High correlation with A &mdash; near-redundant feature"),
        ("B<sub>mid</sub>", "B_mid_corr", "mid-correlation", "Moderate correlation with A &mdash; ambiguous utility"),
        ("B<sub>low</sub>", "B_low_corr", "low-correlation", "Low correlation with A &mdash; complementary feature"),
    ]
    for label, key, sc_name, desc in b_rows:
        corr_sc = correlations.get(sc_name, {})
        rho_br = corr_sc.get("rho_BR", "&mdash;")
        rho_ba = corr_sc.get("rho_BA", "&mdash;")
        html += f"""
        <tr><td>{label}</td><td>{sensors[key]}</td>
            <td>{rho_br}</td><td>{rho_ba}</td>
            <td>{desc}</td></tr>"""

    html += """
    </tbody>
</table>
"""

    # ── Summary section ───────────────────────────────────────────────
    html += """
<h2 id="summary">Results Summary</h2>
<table class="summary-table">
    <thead><tr>
        <th>Scenario</th><th>B sensor</th>
        <th>&rho;(B, R)</th><th>&rho;(B, A)</th>
        <th>n* (crossover)</th><th>Type</th>
    </tr></thead>
    <tbody>
"""
    for sc in scenarios:
        b_key = {"high-correlation": "B_high_corr", "mid-correlation": "B_mid_corr",
                 "low-correlation": "B_low_corr"}[sc]
        interp_roots = _interpolate_crossover(df, sc)
        n_roots = len(interp_roots)
        badge_cls = {0: "badge-none", 1: "badge-single", 2: "badge-double"}.get(n_roots, "badge-double")
        ctype = {0: "No crossover", 1: "Single", 2: "Double"}.get(n_roots, f"{n_roots}x")
        roots_str = ", ".join(f"{r:.1f}" for r in interp_roots) if interp_roots else "N/A"
        corr_sc = correlations.get(sc, {})
        rho_br = corr_sc.get("rho_BR", "&mdash;")
        rho_ba = corr_sc.get("rho_BA", "&mdash;")

        html += f"""<tr>
            <td>{sc}</td>
            <td>mote {sensors[b_key]}</td>
            <td>{rho_br}</td>
            <td>{rho_ba}</td>
            <td>{roots_str}</td>
            <td><span class="badge {badge_cls}">{ctype}</span></td>
        </tr>\n"""

    html += "</tbody></table>\n"

    # ── Detail sections per scenario ──────────────────────────────────
    html += '\n<h2 id="results">SVM Results by Scenario</h2>\n'

    for sc in scenarios:
        interp_roots = _interpolate_crossover(df, sc)
        n_roots = len(interp_roots)
        badge_cls = {0: "badge-none", 1: "badge-single", 2: "badge-double"}.get(n_roots, "badge-double")
        ctype = {0: "No crossover", 1: "Single", 2: "Double"}.get(n_roots, f"{n_roots}x")

        html += f"""
<div class="scenario-section">
    <h3>{sc.replace('-', ' ').title()}
        <span class="badge {badge_cls}">{ctype}</span>
    </h3>
"""
        # Crossover figure
        fig_path = os.path.join(figures_dir, f"crossover_{sc}.png")
        if not os.path.exists(fig_path):
            fig_path = os.path.join(figures_dir, f"crossover_{sc}_svm.png")
        if os.path.exists(fig_path):
            img_data = _img_to_base64(fig_path)
            html += f'<div class="figure-row single"><img src="{img_data}" alt="Crossover plot {sc}"></div>\n'

        # Error table
        html += _build_error_table_html(df, sc)

        html += "</div>\n"

    # ── Comparison section ────────────────────────────────────────────
    html += '\n<h2 id="comparison">Scenario Comparison</h2>\n'
    fig_path = os.path.join(figures_dir, "scenario_comparison.png")
    if not os.path.exists(fig_path):
        fig_path = os.path.join(figures_dir, "scenario_comparison_svm.png")
    if os.path.exists(fig_path):
        img_data = _img_to_base64(fig_path)
        html += f'<div class="figure-row single"><img src="{img_data}" alt="Scenario comparison"></div>\n'

    # Correlation vs crossover
    fig_path = os.path.join(figures_dir, "corr_vs_crossover.png")
    if not os.path.exists(fig_path):
        fig_path = os.path.join(figures_dir, "corr_vs_crossover_svm.png")
    if os.path.exists(fig_path):
        img_data = _img_to_base64(fig_path)
        html += f'<div class="figure-row single"><img src="{img_data}" alt="Correlation vs crossover"></div>\n'

    # ── Analysis section ──────────────────────────────────────────────

    # Load delta stats if available
    dstats_path = os.path.join(cfg.TABLES_DIR, "delta_stats.csv")
    dstats_df = None
    if os.path.exists(dstats_path):
        dstats_df = pd.read_csv(dstats_path)

    html += """
<h2 id="analysis">Analysis: Double Crossover</h2>

<h3>Statistical Significance of Error Differences</h3>
<p>
For each sample size <em>n</em> and scenario, the paired error difference
&Delta; = E(d=1) &minus; E(d=2) was computed on every Monte Carlo trial
(same training subsample, same test set, different feature sets).
The figure below shows the mean &Delta; with its 95% confidence interval.
Points where the CI includes zero (non-significant) are marked with an
open ring.  Wilcoxon signed-rank <em>p</em>-values confirm the results
non-parametrically.
</p>
"""

    # Delta CI figure
    fig_path = os.path.join(figures_dir, "delta_ci.png")
    if os.path.exists(fig_path):
        img_data = _img_to_base64(fig_path)
        html += f'<div class="figure-row single"><img src="{img_data}" alt="Delta with CI"></div>\n'

    # Significance table per scenario
    if dstats_df is not None:
        for sc in scenarios:
            sc_ds = dstats_df[dstats_df["scenario"] == sc].sort_values("n")
            if sc_ds.empty:
                continue
            html += f"<h4>{sc.replace('-', ' ').title()}</h4>\n"
            html += """<table class="data-table">
<thead><tr>
    <th>n</th>
    <th>&Delta; mean</th>
    <th>95% CI</th>
    <th>Wilcoxon <em>p</em></th>
    <th>Cohen's <em>d</em></th>
    <th>d=2 wins</th>
    <th>Significance</th>
</tr></thead>
<tbody>
"""
            for _, row in sc_ds.iterrows():
                n_val = int(row["n"])
                mean_d = row["mean_delta"]
                ci_lo = row["ci_lo"]
                ci_hi = row["ci_hi"]
                p_val = row["p_value"]
                cd = row["cohens_d"]
                frac_w = row["frac_d2_wins"]

                # Significance stars
                if p_val < 0.001:
                    sig_str = '<span style="color:#d62728;">***</span>'
                elif p_val < 0.01:
                    sig_str = '<span style="color:#d62728;">**</span>'
                elif p_val < 0.05:
                    sig_str = '<span style="color:#d62728;">*</span>'
                else:
                    sig_str = '<span style="color:#999;">n.s.</span>'

                # Who wins
                delta_cls = "positive" if mean_d > 0 else "negative"

                html += f"""<tr>
    <td>{n_val}</td>
    <td class="{delta_cls}">{mean_d:+.4f}</td>
    <td>[{ci_lo:+.4f}, {ci_hi:+.4f}]</td>
    <td>{p_val:.4f}</td>
    <td>{cd:+.3f}</td>
    <td>{frac_w:.1%}</td>
    <td>{sig_str}</td>
</tr>\n"""
            html += "</tbody></table>\n"

    # ── Footer ────────────────────────────────────────────────────────
    html += f"""
<footer>
    Sensor Crossover Experiment Report &mdash; {timestamp}<br>
    Config: {config['n_reps']} reps, n &isin; {{{', '.join(str(v) for v in config['n_values'])}}},
    SVM C={config['svm_C']}, seed={config['seed']}
</footer>

</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report saved to {output_path}")
    return output_path
