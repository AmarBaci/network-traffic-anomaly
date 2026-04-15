from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from dashboard_backend import DEFAULT_OUTPUT_JSON, export_dashboard_payload


app = FastAPI(title="Network Traffic Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dashboard/data")
def get_dashboard_data(
    output_json: str = Query(DEFAULT_OUTPUT_JSON, description="Path to precomputed dashboard JSON"),
    auto_build: bool = Query(True, description="Build dashboard payload when file is missing"),
) -> dict[str, Any]:
    target = Path(output_json)

    if not target.exists():
        if not auto_build:
            raise HTTPException(status_code=404, detail=f"Dashboard payload not found: {target}")
        export_dashboard_payload(output_json=str(target))

    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard payload: {exc}") from exc


@app.post("/dashboard/refresh")
def refresh_dashboard_backend(skip_train: bool = False) -> dict[str, str]:
    cmd = [sys.executable, "src/refresh_dashboard_backend.py"]
    if skip_train:
        cmd.append("--skip-train")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Refresh command failed: {exc}") from exc

    return {"status": "ok", "message": "Dashboard backend refreshed."}
