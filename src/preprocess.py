from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


METADATA_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "SimillarHTTP",
    "Inbound",
}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning that is safe for CIC-style traffic tables."""
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    # Drop columns that contain no usable values.
    cleaned = cleaned.dropna(axis=1, how="all")
    return cleaned


def build_features_and_labels(
    df: pd.DataFrame,
    label_column: str = "Label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Create numeric feature matrix X and binary anomaly labels y."""
    if label_column not in df.columns:
        raise ValueError(
            f"Missing label column '{label_column}'. Available columns: {list(df.columns)[:10]}..."
        )

    labels = df[label_column].astype(str).str.strip().str.lower()
    y = (labels != "benign").astype(int)

    columns_to_drop = {label_column, *METADATA_COLUMNS}
    X = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(X.median(numeric_only=True))

    # Keep numeric-only columns after coercion.
    X = X.select_dtypes(include=[np.number])

    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable numeric features found after preprocessing.")

    return X, y


def split_train_test_for_anomaly(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Train on normal traffic only; test on held-out normal + all anomalies."""
    normal_mask = y == 0
    anomaly_mask = y == 1

    if anomaly_mask.sum() == 0:
        raise ValueError("No anomaly samples found in the provided dataset.")

    X_normal = X[normal_mask]

    X_train_normal, X_test_normal = train_test_split(
        X_normal,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_test = pd.concat([X_test_normal, X[anomaly_mask]], axis=0)
    y_test = pd.concat(
        [
            pd.Series(np.zeros(len(X_test_normal), dtype=int), index=X_test_normal.index),
            pd.Series(np.ones(anomaly_mask.sum(), dtype=int), index=X[anomaly_mask].index),
        ],
        axis=0,
    )

    X_test = X_test.sort_index()
    y_test = y_test.sort_index()

    return X_train_normal, X_test, y_test
