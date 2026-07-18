from __future__ import annotations

import re
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

DEFAULT_LEAKAGE_PATTERNS = [
    r"(^|_)label($|_)",
    r"attack",
    r"anomaly",
    r"malicious",
    r"target",
    r"class",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning that is safe for CIC-style traffic tables."""
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    # Drop columns that contain no usable values.
    cleaned = cleaned.dropna(axis=1, how="all")
    return cleaned


def validate_schema(
    df: pd.DataFrame,
    label_column: str = "Label",
    required_columns: list[str] | None = None,
    min_rows: int = 100,
) -> dict[str, object]:
    """Validate basic dataset schema and return diagnostics."""
    if min_rows > 0 and len(df) < min_rows:
        raise ValueError(f"Dataset has too few rows ({len(df)}). Expected at least {min_rows}.")

    if label_column not in df.columns:
        raise ValueError(f"Missing label column '{label_column}'.")

    missing_required = []
    for column in required_columns or []:
        if column not in df.columns:
            missing_required.append(column)

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    label_distribution = (
        df[label_column].astype(str).str.strip().str.lower().value_counts(dropna=False).to_dict()
    )
    duplicate_rows = int(df.duplicated().sum())
    missing_ratio = float(df.isna().mean().mean()) if len(df) > 0 else 0.0

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "global_missing_ratio": missing_ratio,
        "label_distribution": label_distribution,
    }


def find_leakage_columns(
    columns: list[str],
    label_column: str,
    leakage_patterns: list[str] | None = None,
) -> list[str]:
    """Find suspicious columns likely to leak labels into features."""
    patterns = leakage_patterns or DEFAULT_LEAKAGE_PATTERNS
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]

    leaked = []
    label_lc = label_column.strip().lower()
    for col in columns:
        col_lc = col.strip().lower()
        if col_lc == label_lc:
            continue
        if any(rx.search(col_lc) for rx in compiled):
            leaked.append(col)
    return leaked


def build_data_quality_summary(df: pd.DataFrame, top_n: int = 10) -> dict[str, object]:
    """Build a compact quality summary for reporting."""
    missing_ratio_series = df.isna().mean().sort_values(ascending=False)
    top_missing = {
        str(col): float(ratio)
        for col, ratio in missing_ratio_series.head(top_n).items()
        if ratio > 0
    }

    constant_columns = [
        str(col)
        for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "top_missing_columns": top_missing,
        "constant_columns_count": int(len(constant_columns)),
        "constant_columns_sample": constant_columns[:top_n],
    }


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


def build_features_and_labels_with_report(
    df: pd.DataFrame,
    label_column: str = "Label",
    max_missing_ratio: float = 0.5,
    leakage_patterns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Create X/y plus preprocessing diagnostics for reporting."""
    if label_column not in df.columns:
        raise ValueError(
            f"Missing label column '{label_column}'. Available columns: {list(df.columns)[:10]}..."
        )

    labels = df[label_column].astype(str).str.strip().str.lower()
    y = (labels != "benign").astype(int)

    columns_to_drop = {label_column, *METADATA_COLUMNS}
    leakage_columns = find_leakage_columns(list(df.columns), label_column, leakage_patterns)
    columns_to_drop.update(leakage_columns)

    X = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    missing_ratio = X.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    if high_missing_cols:
        X = X.drop(columns=high_missing_cols)

    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=[np.number])

    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable numeric features found after preprocessing.")

    report = {
        "original_feature_columns": int(df.shape[1] - 1),
        "final_feature_columns": int(X.shape[1]),
        "dropped_metadata_or_label_columns": sorted(c for c in columns_to_drop if c in df.columns),
        "dropped_high_missing_columns": sorted(high_missing_cols),
        "dropped_constant_columns": sorted(constant_cols),
        "dropped_leakage_columns": sorted(leakage_columns),
    }

    return X, y, report


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


def split_train_val_test_for_anomaly(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train on normal-only data and create val/test sets with normal+anomaly samples."""
    if val_size < 0 or test_size <= 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size must be >=0, test_size > 0, and val_size + test_size < 1.")

    normal_mask = y == 0
    anomaly_mask = y == 1

    if anomaly_mask.sum() == 0:
        raise ValueError("No anomaly samples found in the provided dataset.")

    X_normal = X[normal_mask]
    X_anomaly = X[anomaly_mask]

    holdout_ratio = val_size + test_size
    X_train, X_normal_holdout = train_test_split(
        X_normal,
        test_size=holdout_ratio,
        random_state=random_state,
        shuffle=True,
    )

    if val_size == 0:
        X_val_normal = X_normal_holdout.iloc[0:0]
        X_test_normal = X_normal_holdout
    else:
        relative_test_ratio = test_size / holdout_ratio
        X_val_normal, X_test_normal = train_test_split(
            X_normal_holdout,
            test_size=relative_test_ratio,
            random_state=random_state,
            shuffle=True,
        )

    if val_size == 0:
        X_val_anomaly = X_anomaly.iloc[0:0]
        X_test_anomaly = X_anomaly
    else:
        relative_test_ratio = test_size / holdout_ratio
        X_val_anomaly, X_test_anomaly = train_test_split(
            X_anomaly,
            test_size=relative_test_ratio,
            random_state=random_state,
            shuffle=True,
        )

    X_val = pd.concat([X_val_normal, X_val_anomaly], axis=0).sort_index()
    y_val = pd.concat(
        [
            pd.Series(np.zeros(len(X_val_normal), dtype=int), index=X_val_normal.index),
            pd.Series(np.ones(len(X_val_anomaly), dtype=int), index=X_val_anomaly.index),
        ],
        axis=0,
    ).sort_index()

    X_test = pd.concat([X_test_normal, X_test_anomaly], axis=0).sort_index()
    y_test = pd.concat(
        [
            pd.Series(np.zeros(len(X_test_normal), dtype=int), index=X_test_normal.index),
            pd.Series(np.ones(len(X_test_anomaly), dtype=int), index=X_test_anomaly.index),
        ],
        axis=0,
    ).sort_index()

    return X_train, X_val, y_val, X_test, y_test
