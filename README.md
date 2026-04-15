# Self-Supervised Pattern Recognition for Large-Scale Network Traffic

Research project for a Master's Pattern Recognition course.

This repository explores self-supervised deep learning for anomaly detection in large-scale network traffic, with a focus on reducing dependence on labeled attack data.

## Core Idea

Traditional intrusion and anomaly detection pipelines often require many labeled attack examples. In real-world systems, most traffic is unlabeled and attack patterns evolve quickly.

This project aims to:

- Learn traffic representations from unlabeled data using self-supervised learning
- Detect anomalous traffic patterns automatically
- Improve detection performance while lowering manual labeling effort

## Research Context

- Domain: Big-data cybersecurity monitoring systems
- Research gap: Many methods assume access to labeled attack datasets
- Contribution target: Contrastive self-supervised representation learning on traffic flows/windows

## Current Status

This repository is in the early setup phase.

Already included:

- Initial Python dependencies
- Data directory and starter dataset archive
- Project skeleton for source code and notebooks

Planned next milestones:

- Data ingestion and preprocessing pipeline
- Baseline anomaly model (non-SSL)
- Self-supervised encoder training
- Evaluation pipeline and reproducible experiments

## Project Structure

```
network-traffic-anomaly/
  data/          # Raw and intermediate data artifacts (ignored if large)
  notebooks/     # Exploratory analysis and experiment notebooks
  src/           # Source code (pipelines, models, training, evaluation)
  requirements.txt
  README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AmarBaci/network-traffic-anomaly.git
cd network-traffic-anomaly
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare data

Place dataset files in `data/`.

Current dataset sample in the repository:

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.zip`

Note: Large datasets and packet captures are intentionally excluded from Git.

### 5. Run Phase 1 baseline hardening

Run both baseline models:

```bash
./.venv/bin/python src/baseline_isolation_forest.py
./.venv/bin/python src/baseline_one_class_svm.py
```

Generate comparison report:

```bash
./.venv/bin/python src/summarize_runs.py
```

Key outputs:

- Latest model metrics JSON files:
  - `results/baseline_isolation_forest_metrics.json`
  - `results/baseline_one_class_svm_metrics.json`
- Per-run metrics snapshots in `results/metrics/`
- Run logs:
  - `results/experiment_runs.jsonl`
  - `results/experiment_runs.csv`
- Phase 1 report:
  - `reports/phase1_baseline_report.md`

Optional analysis notebook:

- `notebooks/phase1_baseline_analysis.ipynb`

Configuration and reproducibility controls are centralized in:

- `configs/phase1_baselines.json`

## Planned Methodology (High Level)

1. Parse and clean network traffic records
2. Build flow/window-level features and augmentations
3. Pretrain a self-supervised encoder (contrastive objective)
4. Fit anomaly detector in embedding space
5. Evaluate with anomaly detection metrics

Expected metrics:

- ROC-AUC
- PR-AUC
- F1 score (thresholded)
- False positive rate

## Reference Literature

1. Self-Supervised Anomaly Pattern Detection for Large-Scale Industrial Data  
	https://doi.org/10.1016/j.neucom.2022.09.069
2. Discovering Unknowns: Context-Enhanced Anomaly Detection  
	https://doi.org/10.1016/j.patcog.2022.108860
3. Enhancing Anomaly Detection Performance: Deep Learning Models  
	https://doi.org/10.31436/iiumej.v26i2.3287

## Notes for First-Time Contributors

- This is an academic research codebase under active development.
- Interfaces and file layout may evolve during early iterations.
- Keep experiments reproducible (fixed seeds, saved configs, tracked metrics).

## License

No license has been added yet.

If you plan to share this beyond private use, add a license file (for example, MIT or Apache-2.0) based on your advisor/course guidance.
