# Self-Supervised Pattern Recognition for Large-Scale Network Traffic

Master's Pattern Recognition course project. Develops a contrastive self-supervised encoder for network traffic anomaly detection, trained on unlabeled normal traffic only and evaluated against classical unsupervised baselines.

## Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| Supervised RF (labeled upper bound) | 0.9999 | 0.9999 | 1.0000 | 0.9852 | 0.9925 | 0.0000 |
| **Contrastive SSL (ours)** | **0.7923** | **0.9302** | 0.9930 | 0.0664 | 0.1245 | 0.0016 |
| Isolation Forest | 0.7555 | 0.8879 | 0.9363 | 0.0585 | 0.1101 | 0.0132 |
| One-Class SVM | 0.6521 | 0.9008 | 0.9988 | 0.0660 | 0.1237 | 0.0003 |

Dataset: CIC-IDS2017 DDoS capture (30,000 rows, 68 features). The unsupervised models are trained on normal traffic only; the supervised Random Forest uses labels and serves as a ceiling. All models are evaluated on the same held-out test set.

### Reading the results

- **ROC-AUC is the main comparison.** It measures how well each model ranks attacks above normal traffic, independent of any threshold. The contrastive encoder ranks best among the label-free models (0.79 vs 0.76 and 0.65).
- **The low F1 comes from the threshold policy, not the ranking.** The alert threshold is set at the 95th percentile of validation scores, so only the most anomalous ~5% of traffic is flagged. Alerts are almost always correct (precision 0.99, FPR 0.16%), but only ~7% of attack flows are caught. That suits a low-false-alarm setting; a lower threshold trades precision for recall.
- **PR-AUC is inflated by class balance.** About 77% of the test set is attack traffic (12,826 of 16,696 rows), so a random scorer would already reach a PR-AUC of ~0.77. Compare PR-AUC values against that baseline rather than against 0.
- **Next step:** calibrate the threshold on normal-only validation traffic to target a fixed false-positive rate, which is closer to how an unsupervised detector would be deployed.

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
├── tests/                            # pytest unit tests
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

## Testing

Unit tests cover the evaluation metrics, preprocessing and train/val/test splitting, and the tabular augmentations. They use small synthetic data, so the dataset isn't needed:

```bash
pip install pytest
python -m pytest
```

## Reproducibility

All experiments use `random_seed = 42` and deterministic PyTorch operations. Hyperparameters are stored in `configs/` and every run is appended to `results/experiment_runs.jsonl` with its full configuration and metrics.
