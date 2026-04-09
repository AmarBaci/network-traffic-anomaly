from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".zip"}


def resolve_data_path(data_path: str | None = None) -> Path:
    """Resolve the dataset path from CLI argument or environment defaults."""
    if data_path:
        path = Path(data_path)
    else:
        data_dir = os.getenv("DATA_DIR", "data")
        raw_file = os.getenv("RAW_DATA_FILE", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.zip")
        path = Path(data_dir) / raw_file

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported dataset format '{path.suffix}'. Expected one of: {sorted(SUPPORTED_SUFFIXES)}"
        )

    return path


def load_dataframe(data_path: str | None = None) -> pd.DataFrame:
    """Load a CSV or ZIP-compressed CSV into a DataFrame."""
    path = resolve_data_path(data_path)

    if path.suffix.lower() == ".zip":
        df = pd.read_csv(path, compression="zip", low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {path}")

    return df
