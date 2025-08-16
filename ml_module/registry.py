from __future__ import annotations

from typing import List, Protocol, Optional, Dict, Any
from backend_module.database import DataBaseManager


class MLModel(Protocol):
    def process_group(self, db_manager: DataBaseManager, file_group: List[Dict[str, Any]], work_dir: str) -> Dict[str, Any]:
        ...


def get_model(model: Optional[str], model_source: Optional[str], task_type: Optional[str]) -> Optional[MLModel]:
    """Return a model adapter instance for the given job fields, or None if unsupported."""
    if model == "samurai-ulr" and model_source == "internet" and task_type == "one-shot-object-detection":
        from .model_samurai_ulr import SamuraiULRModel  # lazy import
        return SamuraiULRModel()
    return None


def run_inference_task(
    *,
    db_manager: DataBaseManager,
    job_id: str,
    task_type: Optional[str],
    model: Optional[str],
    model_source: Optional[str],
    file_group: List[Dict[str, Any]],
    work_dir: str,
) -> Optional[Dict[str, Any]]:
    """Select a model and run inference for a single file group.

    Handles debug printing here (moved from caller).
    """
    # Debug prints moved here from manager
    if task_type == "one-shot-object-detection":
        print("==== Start one shot object detection ====")
        print("ML Model:", model)
        dbg = {
            "jobId": str(job_id),
            "taskType": task_type,
            "model": model,
            "modelSource": model_source,
            "group": file_group,
        }
        print(dbg)
        if model == "samurai-ulr":
            print("Select Model: SAMURAI Ultra Long Range")
    else:
        print("[ERROR]", task_type, "is unknown task type.")
        return None

    adapter = get_model(model, model_source, task_type)
    if adapter is None:
        print("[ERROR] No model adapter for:", model, model_source, task_type)
        return None

    # Execute model-specific processing
    return adapter.process_group(db_manager, file_group, work_dir)
