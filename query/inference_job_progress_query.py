from __future__ import annotations

from typing import Optional, Dict, Any, List

from backend_module.database import DataBaseManager
from query.utils import first_result


def get_progress(db_manager: DataBaseManager, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch `progress` object from an inference_job.

    Returns a dict like {"steps": [...], "current_key": "..."} or None.
    """
    res = db_manager.query(
        "SELECT VALUE progress FROM inference_job WHERE id = <record> $ID LIMIT 1",
        {"ID": job_id},
    )
    return first_result(res)


def set_progress(db_manager: DataBaseManager, job_id: str, progress: Dict[str, Any]) -> Any:
    """Replace `progress` field for an inference_job."""
    return db_manager.query(
        "UPDATE inference_job SET progress = $PROG WHERE id = <record> $ID",
        {"ID": job_id, "PROG": progress},
    )


def init_steps_if_absent(
    db_manager: DataBaseManager,
    job_id: str,
    steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Initialize progress.steps if not present.

    Steps should be a list of {key, label, state} where state is one of
    "pending" | "running" | "completed" | "faild".
    Returns the resulting progress object.
    """
    cur = get_progress(db_manager, job_id) or {}
    if not isinstance(cur, dict):
        cur = {}
    if not cur.get("steps"):
        cur = {"steps": steps, "current_key": None}
        set_progress(db_manager, job_id, cur)
    return cur


def update_step_state(
    db_manager: DataBaseManager,
    job_id: str,
    *,
    step_key: str,
    state: str,
    set_current: bool = True,
) -> Dict[str, Any]:
    """Update a single step's state by rewriting the full progress object."""
    prog = get_progress(db_manager, job_id) or {}
    steps = list(prog.get("steps") or [])
    updated = False
    for s in steps:
        if isinstance(s, dict) and s.get("key") == step_key:
            s["state"] = state
            updated = True
            break
    if not updated:
        steps.append({"key": step_key, "label": step_key, "state": state})
    prog["steps"] = steps
    if set_current:
        prog["current_key"] = step_key
    set_progress(db_manager, job_id, prog)
    return prog

