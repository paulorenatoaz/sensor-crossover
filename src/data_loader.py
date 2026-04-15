# ---------------------------------------------------------------------------
# data_loader.py — Download, parse, and clean the Intel Lab Sensor Dataset
# ---------------------------------------------------------------------------

import gzip
import os
import requests
import pandas as pd
import config as cfg


def download_file(url: str, dest: str) -> None:
    """Download a file if it does not already exist."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    print(f"  → saved to {dest}")


def download_data() -> None:
    """Download both data.txt.gz and mote_locs.txt."""
    download_file(cfg.DATA_URL, cfg.DATA_FILE)
    download_file(cfg.LOCS_URL, cfg.LOCS_FILE)


def load_raw_data() -> pd.DataFrame:
    """Load and clean the raw sensor data.

    Returns
    -------
    pd.DataFrame with columns [datetime, epoch, moteid, temperature]
    """
    col_names = [
        "date", "time", "epoch", "moteid",
        "temperature", "humidity", "light", "voltage",
    ]
    with gzip.open(cfg.DATA_FILE, "rt") as f:
        df = pd.read_csv(
            f, sep=r"\s+", header=None, names=col_names,
            na_values=["", "NA"], on_bad_lines="skip",
        )

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], errors="coerce"
    )
    df = df.dropna(subset=["datetime"])

    # Coerce numeric columns
    for col in ["epoch", "moteid", "temperature"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch", "moteid", "temperature"])
    df["epoch"] = df["epoch"].astype(int)
    df["moteid"] = df["moteid"].astype(int)

    # Filter valid mote IDs
    lo, hi = cfg.VALID_MOTE_RANGE
    df = df[(df["moteid"] >= lo) & (df["moteid"] <= hi)]

    # Filter plausible temperatures
    df = df[
        (df["temperature"] >= cfg.TEMP_MIN)
        & (df["temperature"] <= cfg.TEMP_MAX)
    ]

    # Keep only needed columns, sorted
    df = df[["datetime", "epoch", "moteid", "temperature"]].sort_values(
        "datetime"
    )
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} clean readings from {df['moteid'].nunique()} sensors")
    return df


def load_mote_locations() -> pd.DataFrame:
    """Load sensor (mote) x-y coordinates.

    Returns
    -------
    pd.DataFrame with columns [moteid, x, y]
    """
    locs = pd.read_csv(
        cfg.LOCS_FILE, sep=r"\s+", header=None, names=["moteid", "x", "y"]
    )
    return locs
