from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUNS_JSONL = "results/experiment_runs.jsonl"
DEFAULT_OUTPUT_JSON = "artifacts/dashboard/latest_dashboard_data.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Run log file not found: {target}")

    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"Run log file is empty: {target}")
    return records


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_per_model(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("run_id")
        .groupby("model", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def build_dashboard_payload(runs_jsonl: str = DEFAULT_RUNS_JSONL, history_limit: int = 30) -> dict[str, Any]:
    records = load_jsonl_records(runs_jsonl)
    df = pd.DataFrame(records)

    required = {"run_id", "model", "roc_auc", "pr_auc", "f1", "fpr"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required run columns: {missing}")

    latest = _latest_per_model(df)

    latest_models: list[dict[str, Any]] = []
    for _, row in latest.iterrows():
        latest_models.append(
            {
                "run_id": row.get("run_id"),
                "model": row.get("model"),
                "roc_auc": _to_float(row.get("roc_auc")),
                "pr_auc": _to_float(row.get("pr_auc")),
                "precision": _to_float(row.get("precision")),
                "recall": _to_float(row.get("recall")),
                "f1": _to_float(row.get("f1")),
                "fpr": _to_float(row.get("fpr")),
                "threshold": _to_float(row.get("threshold")),
                "n_samples": int(row.get("n_samples")) if pd.notna(row.get("n_samples")) else None,
                "n_anomalies": int(row.get("n_anomalies")) if pd.notna(row.get("n_anomalies")) else None,
            }
        )

    leaderboard = sorted(
        latest_models,
        key=lambda x: (x.get("f1") is not None, x.get("f1") or -1.0),
        reverse=True,
    )

    recent_runs_df = df.sort_values("run_id", ascending=False).head(history_limit)
    recent_runs: list[dict[str, Any]] = []
    for _, row in recent_runs_df.iterrows():
        recent_runs.append(
            {
                "run_id": row.get("run_id"),
                "model": row.get("model"),
                "roc_auc": _to_float(row.get("roc_auc")),
                "pr_auc": _to_float(row.get("pr_auc")),
                "f1": _to_float(row.get("f1")),
                "fpr": _to_float(row.get("fpr")),
            }
        )

    latest_row = df.sort_values("run_id").tail(1).iloc[0].to_dict()
    schema_report = latest_row.get("schema_report")
    preprocess_report = latest_row.get("preprocess_report")
    quality_summary = latest_row.get("quality_summary")

    payload = {
        "generated_at_utc": utc_now_iso(),
        "source": {
            "runs_jsonl": str(runs_jsonl),
            "history_limit": history_limit,
            "total_runs": int(len(df)),
        },
        "summary": {
            "best_model_by_f1": leaderboard[0]["model"] if leaderboard else None,
            "model_count": len(latest_models),
        },
        "latest_models": latest_models,
        "leaderboard": leaderboard,
        "recent_runs": recent_runs,
        "data_quality": {
            "schema_report": schema_report,
            "preprocess_report": preprocess_report,
            "quality_summary": quality_summary,
        },
    }

    return payload


def export_dashboard_payload(
    output_json: str = DEFAULT_OUTPUT_JSON,
    runs_jsonl: str = DEFAULT_RUNS_JSONL,
    history_limit: int = 30,
) -> Path:
    payload = build_dashboard_payload(runs_jsonl=runs_jsonl, history_limit=history_limit)

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path
