from __future__ import annotations

import argparse

from dashboard_backend import DEFAULT_OUTPUT_JSON, DEFAULT_RUNS_JSONL, export_dashboard_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stable dashboard JSON payload from run logs.")
    parser.add_argument("--runs-jsonl", type=str, default=DEFAULT_RUNS_JSONL, help="Path to run log JSONL")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON, help="Path to output dashboard JSON")
    parser.add_argument("--history-limit", type=int, default=30, help="Max recent runs to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_dashboard_payload(
        output_json=args.output_json,
        runs_jsonl=args.runs_jsonl,
        history_limit=args.history_limit,
    )
    print(f"Saved dashboard payload to: {output_path}")


if __name__ == "__main__":
    main()
