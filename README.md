# Self-Supervised Pattern Recognition for Large-Scale Network Traffic

Master's Pattern Recognition course project. Develops a contrastive self-supervised encoder for network traffic anomaly detection, trained on unlabeled normal traffic only and evaluated against classical unsupervised baselines.

## Results

| Model | ROC-AUC | PR-AUC | F1 | FPR |
|---|---|---|---|---|
| Supervised RF (labeled upper bound) | 0.9999 | 0.9999 | 0.9925 | 0.0000 |
| **Contrastive SSL (ours)** | **0.7923** | **0.9302** | 0.1245 | 0.0016 |
| Isolation Forest | 0.7555 | 0.8879 | 0.1101 | 0.0132 |
| One-Class SVM | 0.6521 | 0.9008 | 0.1237 | 0.0003 |

Dataset: CIC-IDS2017 DDoS capture (30,000 rows, 68 features). All models trained on normal traffic only; evaluated on the same held-out test set.

## Project Structure

```
network-traffic-anomaly/
├── src/nta/                          # installable package (pip install -e .)
│   ├── contrastive_ssl.py            # SSL model training (main experiment)
│   ├── baseline_isolation_forest.py  # Isolation Forest baseline
│   ├── baseline_one_class_svm.py     # One-Class SVM baseline
│   ├── supervised_upper_bound.py     # Labeled RF ceiling comparison
│   ├── ablation_study.py             # Component ablation (loss / augmentation)
│   ├── cross_attack_eval.py          # Generalisation to unseen attack types
│   ├── spark_preprocess.py           # Spark-based scalable preprocessing pipeline
│   ├── contrastive_net.py            # Encoder, projection head, decoder
│   ├── contrastive_augment.py        # Tabular augmentation (noise, mask, jitter)
│   ├── preprocess.py                 # Cleaning, feature engineering, splits
│   ├── evaluate.py                   # ROC-AUC, PR-AUC, F1, FPR
│   ├── generate_plots.py             # ROC/PR curves, training loss, t-SNE
│   ├── dashboard_api.py              # FastAPI backend
│   └── dashboard_backend.py          # Dashboard payload builder
├── configs/
│   ├── contrastive.json              # SSL hyperparameters
│   └── baselines.json                # Baseline hyperparameters
├── data/                             # Dataset files (large files not tracked)
├── results/                          # Metrics JSON, checkpoints, run logs
├── reports/figures/                  # ROC/PR curves, t-SNE plots
├── frontend/                         # React dashboard (Vite)
├── pyproject.toml                    # Package metadata / build config
└── requirements.txt                  # Pinned dependency versions
```

## Setup

```bash
git clone https://github.com/AmarBaci/network-traffic-anomaly.git
cd network-traffic-anomaly
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .            # install the `nta` package (editable)
```

Scripts are run as modules from the repo root, e.g. `python -m nta.contrastive_ssl`.

Place the dataset file in `data/`:
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.zip` (CIC-IDS2017)

## Running Experiments

All scripts read from `configs/` and log every run to `results/experiment_runs.jsonl`.

**Spark preprocessing** (scalable alternative to preprocess.py)
```bash
# Local mode
python -m nta.spark_preprocess

# Cluster (spark-submit)
spark-submit --master yarn src/nta/spark_preprocess.py \
    --data-path data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.zip \
    --output-dir results/spark_splits
```
Writes train/val/test Parquet splits and `results/spark_splits/quality_report.json`.

**Baselines**
```bash
python -m nta.baseline_isolation_forest
python -m nta.baseline_one_class_svm
```

**SSL model**
```bash
python -m nta.contrastive_ssl
```
Saves checkpoint and centroid to `results/contrastive_ssl_checkpoint.pt`.

**Supervised upper bound**
```bash
python -m nta.supervised_upper_bound
```

**Ablation study** (5 configurations, 50 epochs each)
```bash
python -m nta.ablation_study
# Use --epochs 100 to match full training length
```

**Cross-attack generalisation** (requires additional CIC-IDS2017 files)
```bash
python -m nta.cross_attack_eval --new-data data/<attack_file>.csv
```
Download additional files from [Kaggle — CICIDS2017](https://www.kaggle.com/datasets/cicdataset/cicids2017).

**Plots**
```bash
python -m nta.generate_plots
```
Outputs ROC/PR curves, training loss curve, and t-SNE projection to `reports/figures/`.

## Dashboard

Start the API and frontend in two terminals:

```bash
# Terminal 1
python -m uvicorn nta.dashboard_api:app --reload

# Terminal 2
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`. Use the Refresh button to rebuild the dashboard payload after running new experiments.

## Reproducibility

All experiments use `random_seed = 42` and deterministic PyTorch operations. Hyperparameters are stored in `configs/` and every run is appended to `results/experiment_runs.jsonl` with its full configuration and metrics.
