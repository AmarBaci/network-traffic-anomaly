"""Apache Spark preprocessing pipeline for CIC-IDS2017 network traffic data.

Replicates the same cleaning and splitting protocol used by preprocess.py but
runs on a Spark cluster, making the pipeline horizontally scalable to datasets
that exceed single-node memory.

Outputs (written to --output-dir as Parquet):
    train_normal/   - normal-traffic rows only (model training set)
    val/            - mixed normal + anomaly (threshold calibration)
    test/           - mixed normal + anomaly (held-out evaluation)
    quality_report.json

Usage
-----
Local mode (single machine):
    python src/spark_preprocess.py

Cluster / Spark-submit:
    spark-submit --master yarn src/spark_preprocess.py \\
        --data-path data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.zip \\
        --output-dir results/spark_splits
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ---------------------------------------------------------------------------
# Constants — must stay in sync with preprocess.py
# ---------------------------------------------------------------------------
METADATA_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "SimillarHTTP",
    "Inbound",
}

LABEL_COLUMN = "Label"

# Constant columns identified during Phase 2 data quality analysis.
KNOWN_CONSTANT_COLUMNS = [
    "Bwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bulk Rate",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd URG Flags",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_zip_to_csv(zip_path: str, tmp_dir: str = "/tmp") -> str:
    """Extract the first CSV from a ZIP archive and return its path."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found inside {zip_path}")
        target = csv_names[0]
        zf.extract(target, path=tmp_dir)
        return str(Path(tmp_dir) / target)


def _build_spark_session(app_name: str = "NetworkTrafficPreprocess") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        # Increase shuffle partitions for larger datasets when running on a cluster.
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def load_csv(spark: SparkSession, csv_path: str):
    """Read CSV with header inference and return a Spark DataFrame."""
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .csv(csv_path)
    )


def clean_columns(df):
    """Strip whitespace from column names (mirrors pandas clean_dataframe)."""
    renamed = {c: c.strip() for c in df.columns}
    for old, new in renamed.items():
        if old != new:
            df = df.withColumnRenamed(old, new)
    return df


def replace_inf(df):
    """Replace Inf / -Inf with null so they can be imputed with the median."""
    for col_name in df.columns:
        if df.schema[col_name].dataType == DoubleType():
            df = df.withColumn(
                col_name,
                F.when(F.col(col_name).isin([float("inf"), float("-inf")]), None)
                .otherwise(F.col(col_name)),
            )
    return df


def drop_non_feature_columns(df):
    """Drop the label, metadata, and known constant columns."""
    to_drop = ({LABEL_COLUMN} | METADATA_COLUMNS | set(KNOWN_CONSTANT_COLUMNS))
    existing = set(df.columns)
    df = df.drop(*[c for c in to_drop if c in existing])
    return df


def impute_with_median(df, feature_cols: list[str]):
    """Replace null values in numeric columns with the per-column approximate median."""
    # approxQuantile is an O(n) pass; relative error 0.01 gives <1% deviation.
    medians = df.approxQuantile(feature_cols, [0.5], relativeError=0.01)
    median_map = {col: vals[0] for col, vals in zip(feature_cols, medians)}

    for col_name, median_val in median_map.items():
        if median_val is not None:
            df = df.withColumn(
                col_name,
                F.when(F.col(col_name).isNull(), median_val).otherwise(F.col(col_name)),
            )
    return df, median_map


def cast_to_double(df, feature_cols: list[str]):
    """Cast all feature columns to DoubleType for the ML pipeline."""
    for col_name in feature_cols:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
    return df


def build_label_column(df):
    """Add a binary 'anomaly' column: 0 = benign, 1 = attack."""
    df = df.withColumn(
        "anomaly",
        F.when(F.lower(F.trim(F.col(LABEL_COLUMN))) == "benign", 0).otherwise(1),
    )
    return df


def assemble_and_scale(df, feature_cols: list[str]):
    """Assemble feature columns into a vector and apply StandardScaler."""
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    df = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    return df, scaler_model


def split_train_val_test(df, val_size: float, test_size: float, seed: int):
    """Split into train (normal-only), val, and test.

    Mirrors the logic of split_train_val_test_for_anomaly in preprocess.py:
    - Train: normal rows only
    - Val / Test: stratified mix of normal + anomaly
    """
    train_size = 1.0 - val_size - test_size

    normal_df = df.filter(F.col("anomaly") == 0)
    anomaly_df = df.filter(F.col("anomaly") == 1)

    # Normal rows: split into train / val_normal / test_normal
    train_n, holdout_n = normal_df.randomSplit(
        [train_size, val_size + test_size], seed=seed
    )
    val_n, test_n = holdout_n.randomSplit(
        [val_size / (val_size + test_size), test_size / (val_size + test_size)],
        seed=seed,
    )

    # Anomaly rows: split into val_anomaly / test_anomaly
    val_a, test_a = anomaly_df.randomSplit(
        [val_size / (val_size + test_size), test_size / (val_size + test_size)],
        seed=seed,
    )

    train_normal = train_n
    val_df = val_n.union(val_a)
    test_df = test_n.union(test_a)

    return train_normal, val_df, test_df


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def build_quality_report(
    df_raw,
    df_train,
    df_val,
    df_test,
    feature_cols: list[str],
    median_map: dict[str, float],
) -> dict[str, Any]:
    total_rows = df_raw.count()
    label_dist = (
        df_raw.groupBy(F.lower(F.trim(F.col(LABEL_COLUMN))).alias("label"))
        .count()
        .collect()
    )

    return {
        "total_rows": total_rows,
        "total_columns": len(df_raw.columns),
        "label_distribution": {row["label"]: row["count"] for row in label_dist},
        "feature_columns_used": len(feature_cols),
        "dropped_metadata_columns": sorted(METADATA_COLUMNS & set(df_raw.columns)),
        "dropped_constant_columns": sorted(
            set(KNOWN_CONSTANT_COLUMNS) & set(df_raw.columns)
        ),
        "imputed_columns_count": sum(1 for v in median_map.values() if v is not None),
        "train_rows_normal": df_train.count(),
        "val_rows": df_val.count(),
        "val_anomalies": df_val.filter(F.col("anomaly") == 1).count(),
        "test_rows": df_test.count(),
        "test_anomalies": df_test.filter(F.col("anomaly") == 1).count(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spark preprocessing pipeline for CIC-IDS2017 network traffic data."
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to the raw CSV or ZIP file. Falls back to DATA_DIR/RAW_DATA_FILE env vars.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/spark_splits",
        help="Directory to write Parquet splits and the quality report.",
    )
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Randomly sample this many rows before processing (useful for local testing).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve input path
    if args.data_path:
        raw_path = args.data_path
    else:
        data_dir = os.getenv("DATA_DIR", "data")
        raw_file = os.getenv(
            "RAW_DATA_FILE",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.zip",
        )
        raw_path = str(Path(data_dir) / raw_file)

    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    # Unzip if needed (Spark can read CSV directly from a local path but not ZIP)
    if raw_path.lower().endswith(".zip"):
        print(f"[1/7] Extracting ZIP: {raw_path}")
        csv_path = _extract_zip_to_csv(raw_path)
    else:
        csv_path = raw_path
        print(f"[1/7] Using CSV: {csv_path}")

    spark = _build_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    print(f"       Spark version: {spark.version}")

    # ------------------------------------------------------------------ load
    print("[2/7] Loading CSV into Spark DataFrame...")
    df = load_csv(spark, csv_path)
    if args.sample_rows:
        fraction = min(1.0, args.sample_rows / df.count())
        df = df.sample(fraction=fraction, seed=args.random_seed)
        print(f"       Sampled to ~{args.sample_rows} rows (fraction={fraction:.4f})")

    df_raw = df  # keep original for quality report

    # ------------------------------------------------------------------ clean
    print("[3/7] Cleaning column names and replacing Inf values...")
    df = clean_columns(df)
    df = build_label_column(df)
    df = replace_inf(df)

    # ------------------------------------------------------------------ features
    print("[4/7] Dropping metadata, label, and constant columns...")
    df = drop_non_feature_columns(df)

    # Select only numeric columns (mirrors select_dtypes in pandas version)
    numeric_types = {"IntegerType", "LongType", "FloatType", "DoubleType"}
    feature_cols = [
        f.name
        for f in df.schema.fields
        if type(f.dataType).__name__ in numeric_types and f.name != "anomaly"
    ]
    df = cast_to_double(df, feature_cols)

    print(f"       Feature columns after cleaning: {len(feature_cols)}")

    # ------------------------------------------------------------------ impute
    print("[5/7] Imputing missing values with per-column median...")
    df, median_map = impute_with_median(df, feature_cols)

    # ------------------------------------------------------------------ scale
    print("[6/7] Assembling feature vector and applying StandardScaler...")
    df, scaler_model = assemble_and_scale(df, feature_cols)

    # ------------------------------------------------------------------ split
    print("[7/7] Splitting into train / val / test and writing Parquet...")
    train_df, val_df, test_df = split_train_val_test(
        df, val_size=args.val_size, test_size=args.test_size, seed=args.random_seed
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write only the scaled feature vector + anomaly label
    output_cols = ["features", "anomaly"]
    train_df.select(output_cols).write.mode("overwrite").parquet(
        str(out / "train_normal")
    )
    val_df.select(output_cols).write.mode("overwrite").parquet(str(out / "val"))
    test_df.select(output_cols).write.mode("overwrite").parquet(str(out / "test"))

    # ------------------------------------------------------------------ report
    report = build_quality_report(
        df_raw=df_raw,
        df_train=train_df,
        df_val=val_df,
        df_test=test_df,
        feature_cols=feature_cols,
        median_map=median_map,
    )

    report_path = out / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\nSpark preprocessing complete.")
    print(f"  Train (normal only) : {report['train_rows_normal']:,} rows")
    print(f"  Val                 : {report['val_rows']:,} rows  "
          f"({report['val_anomalies']:,} anomalies)")
    print(f"  Test                : {report['test_rows']:,} rows  "
          f"({report['test_anomalies']:,} anomalies)")
    print(f"  Quality report      : {report_path}")
    print(f"  Parquet splits      : {out}/")

    spark.stop()


if __name__ == "__main__":
    main()
