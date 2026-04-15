# Phase 1 Experiment Report

## Baseline Comparison Table

| model | roc_auc | pr_auc | precision | recall | f1 | fpr | n_samples | n_anomalies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| isolation_forest | 0.7555 | 0.8879 | 0.9363 | 0.0585 | 0.1101 | 0.0132 | 16696 | 12826 |
| one_class_svm | 0.6521 | 0.9008 | 0.9988 | 0.0660 | 0.1237 | 0.0003 | 16696 | 12826 |

## First Clean Experiment Report Section

We compared two unsupervised baselines trained on normal traffic and evaluated on held-out normal plus anomalous traffic.
The current best F1 among latest runs is from model: one_class_svm.
This phase establishes a reproducible benchmark so self-supervised methods can be evaluated against a stable baseline in the next phase.