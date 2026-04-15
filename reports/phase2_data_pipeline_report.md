# Phase 2 Data Pipeline Report

## Schema Validation

- Rows: 30000
- Columns: 79
- Duplicate rows: 105
- Global missing ratio: 0.0000
- Label distribution: {'ddos': 17101, 'benign': 12899}

## Feature Engineering and Leakage Checks

- Original feature columns: 78
- Final feature columns: 68
- Dropped leakage columns: None detected
- Dropped high-missing columns: None
- Dropped constant columns: Bwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd PSH Flags, Bwd URG Flags, CWE Flag Count, Fwd Avg Bulk Rate, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd URG Flags

## Train/Validation/Test Protocol

- Train rows (normal only): 7739
- Validation rows: 5565 (anomalies: 4275)
- Test rows: 16696 (anomalies: 12826)
- Validation size: 0.1
- Test size: 0.3

## Data Quality Snapshot

- Constant columns count: 10
- Top missing columns: {'Flow Packets/s': 0.0002, 'Flow Bytes/s': 0.0002}

Phase 2 status: Data pipeline checks and split protocol are active and reproducible via config.
