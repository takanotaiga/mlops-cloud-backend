from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

# Import only within this CLI process, not from the manager
from backend_module.database import DataBaseManager
from ml_module.registry import get_model


def _get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _build_db_manager() -> DataBaseManager:
    surreal_endpoint = _get_env("SURREAL_ENDPOINT", "ws://192.168.1.25:65303/rpc")
    surreal_username = _get_env("SURREAL_USERNAME", "root")
    surreal_password = _get_env("SURREAL_PASSWORD", "root")
    surreal_namespace = _get_env("SURREAL_NAMESPACE", "mlops")
    surreal_database = _get_env("SURREAL_DATABASE", "cloud_ui")
    return DataBaseManager(
        endpoint_url=str(surreal_endpoint),
        username=str(surreal_username),
        password=str(surreal_password),
        namespace=str(surreal_namespace),
        database=str(surreal_database),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ML group processing in a separate process")
    ap.add_argument("--input", required=True, help="Path to input JSON containing job and group info")
    ap.add_argument("--work-dir", required=True, help="Working directory for intermediate outputs")
    ap.add_argument("--result", required=True, help="Path to write result JSON")
    args = ap.parse_args()

    in_path = Path(args.input)
    work_dir = Path(args.work_dir)
    result_path = Path(args.result)
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        spec: Dict[str, Any] = json.load(f)

    model_name = spec.get("model")
    model_source = spec.get("modelSource")
    task_type = spec.get("taskType")
    group = spec.get("group") or []

    model = get_model(model_name, model_source, task_type)
    if model is None:
        err = {"error": f"Unsupported model combination: model={model_name}, source={model_source}, taskType={task_type}"}
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(err, f, ensure_ascii=False, indent=2)
        return 2

    dbm = _build_db_manager()

    try:
        res = model.process_group(dbm, group, str(work_dir))
    except Exception as e:
        err = {"error": f"process_group failed: {e}"}
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(err, f, ensure_ascii=False, indent=2)
        return 1

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

