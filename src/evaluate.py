from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold_quantile: float = 0.95,
) -> dict[str, Any]:
    """Compute anomaly detection metrics from continuous anomaly scores."""
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be in (0, 1).")

    y_true = np.asarray(y_true).astype(int)
    anomaly_scores = np.asarray(anomaly_scores).astype(float)

    threshold = float(np.quantile(anomaly_scores, threshold_quantile))
    y_pred = (anomaly_scores >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, anomaly_scores))
        pr_auc = float(average_precision_score(y_true, anomaly_scores))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "n_samples": int(len(y_true)),
        "n_anomalies": int((y_true == 1).sum()),
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    """Create readable multiline output for terminal logs."""
    return "\n".join(
        [
            "Evaluation Metrics",
            f"  Samples:      {metrics['n_samples']}",
            f"  Anomalies:    {metrics['n_anomalies']}",
            f"  ROC-AUC:      {metrics['roc_auc']:.4f}",
            f"  PR-AUC:       {metrics['pr_auc']:.4f}",
            f"  Precision:    {metrics['precision']:.4f}",
            f"  Recall:       {metrics['recall']:.4f}",
            f"  F1:           {metrics['f1']:.4f}",
            f"  FPR:          {metrics['fpr']:.4f}",
            f"  Threshold q:  {metrics['threshold_quantile']:.2f}",
            f"  Threshold:    {metrics['threshold']:.6f}",
        ]
    )
