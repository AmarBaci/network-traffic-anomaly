# Phase 1 Experiment Report

## Baseline Comparison Table

| model | roc_auc | pr_auc | precision | recall | f1 | fpr | n_samples | n_anomalies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| isolation_forest | 0.7543 | 0.9145 | 0.9542 | 0.0585 | 0.1103 | 0.0124 | 20971 | 17101 |
| one_class_svm | 0.6531 | 0.9209 | 0.9990 | 0.0613 | 0.1155 | 0.0003 | 20971 | 17101 |

## First Clean Experiment Report Section

We compared two unsupervised baselines trained on normal traffic and evaluated on held-out normal plus anomalous traffic.
The current best F1 among latest runs is from model: one_class_svm.
This phase establishes a reproducible benchmark so self-supervised methods can be evaluated against a stable baseline in the next phase.