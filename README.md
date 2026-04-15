# Finite-Sample Crossover in Sensor Classification

Empirical validation of the **fewer-features phenomenon**: when training data
is scarce, a classifier using *one* sensor can outperform a classifier using
*two*, even though the extra sensor carries useful information.

**[View the experiment report →](https://paulorenatoaz.github.io/sensor-crossover/)**

## Key findings

| Scenario | Correlation ρ(B,R) | Crossover n* | Type |
|---|---|---|---|
| High-correlation | 0.89 | 106 | Single |
| Mid-correlation | 0.66 | 4, 78 | Double |
| Low-correlation | 0.38 | 14 | Single |

- At small *n*, the 2-sensor (d=2) SVM suffers from higher variance and is
  outperformed by the 1-sensor (d=1) model.
- The crossover point *n\** depends on the correlation structure between sensors.
- The **mid-correlation** scenario exhibits a rare *double crossover*,
  explained by a second-order model Δ(n) = G + B/n + C/n².

## Quick start

```bash
# Install the package (creates the `sensor-crossover` command)
pip install -e .

# Run the full experiment (downloads data, trains models, generates reports)
sensor-crossover run

# Check that reports are ready for GitHub Pages
sensor-crossover publish
```

Or without installing:

```bash
pip install -r requirements.txt
python3 cli.py run
```

## Project structure

```
cli.py               # Entry point: sensor-crossover run / publish
run_experiment.py     # Experiment orchestrator (9 stages)
config.py             # All hyperparameters and paths
pyproject.toml        # Package metadata & console script
requirements.txt      # Python dependencies
src/
  data_loader.py      # Download & load Intel Lab dataset
  preprocessing.py    # Pivot, gap-fill, epoch filtering
  sensor_selection.py # Automated sensor role assignment
  labeling.py         # Binary label from median split
  experiment.py       # Monte Carlo trials & crossover estimation
  plotting.py         # Publication-quality matplotlib figures
  report.py           # Self-contained HTML experiment report
  dataset_report.py   # Self-contained HTML dataset report
results/              # Generated reports & figures (served by GitHub Pages)
```

## Dataset

[Intel Berkeley Research Lab](http://db.csail.mit.edu/labdata/labdata.html) —
54 temperature sensors, ~2.3M readings over 38 days. The pipeline
automatically downloads, cleans, and synchronizes the data.

## Method

1. **Sensor selection** — One reference mote (R) defines binary classes via
   median split; one primary feature (A) is the most correlated sensor; three
   secondary sensors (B) span high / mid / low correlation regimes.
2. **Monte Carlo evaluation** — For each sample size *n* ∈ {2, …, 1024},
   500 random draws from the training pool are classified by a linear SVM.
   Error is measured on a held-out stratified test set.
3. **Crossover detection** — n\* is estimated by linear interpolation of sign
   changes in Δ(n) = E(d=1) − E(d=2).

## License

MIT
