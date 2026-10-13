import json

from nta.dashboard_backend import build_dashboard_payload


def write_runs(path, runs):
    path.write_text("\n".join(json.dumps(r) for r in runs) + "\n", encoding="utf-8")


def run(run_id, model, roc_auc, f1):
    return {"run_id": run_id, "model": model, "roc_auc": roc_auc, "pr_auc": 0.5, "f1": f1, "fpr": 0.01}


def test_best_model_is_highest_roc_auc_among_unsupervised_models(tmp_path):
    runs_file = tmp_path / "runs.jsonl"
    write_runs(
        runs_file,
        [
            run("001", "isolation_forest", 0.75, 0.30),  # highest F1, lower ROC-AUC
            run("002", "contrastive_ssl", 0.79, 0.12),
            run("003", "supervised_rf", 0.99, 0.99),  # ceiling, must not win
            run("004", "ablation_full_model", 0.95, 0.20),  # variant, must not win
        ],
    )

    payload = build_dashboard_payload(runs_jsonl=str(runs_file))

    assert payload["summary"]["best_model_by_roc_auc"] == "contrastive_ssl"
    assert [m["model"] for m in payload["leaderboard"]][:2] == ["supervised_rf", "ablation_full_model"]


def test_latest_run_per_model_is_used(tmp_path):
    runs_file = tmp_path / "runs.jsonl"
    write_runs(
        runs_file,
        [
            run("001", "contrastive_ssl", 0.90, 0.10),
            run("002", "one_class_svm", 0.80, 0.10),
            run("003", "contrastive_ssl", 0.70, 0.10),  # newer, worse run replaces the old one
        ],
    )

    payload = build_dashboard_payload(runs_jsonl=str(runs_file))

    assert payload["summary"]["best_model_by_roc_auc"] == "one_class_svm"
