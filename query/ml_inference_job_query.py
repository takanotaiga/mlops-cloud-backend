from typing import List
from backend_module.database import DataBaseManager
from query.utils import first_result, extract_results


def get_queued_job(db_manager: DataBaseManager) -> List[dict]:
    """取得可能な queued ジョブの配列を返す。"""
    payload = db_manager.query(
        "SELECT * FROM inference_job WHERE status = 'ProcessWaiting';"
    )
    return extract_results(payload)

def get_linked_file(db_manager: DataBaseManager, job_id: str) -> List[str]:
    payload = db_manager.query(
        "RETURN array::concat((SELECT VALUE key FROM file WHERE dataset INSIDE array::flatten((SELECT VALUE datasets FROM inference_job WHERE id = <record> $JOB_ID)) AND mime !~ 'video/'), (SELECT VALUE key FROM encoded_segment WHERE file.dataset INSIDE array::flatten((SELECT VALUE datasets FROM inference_job WHERE id = <record> $JOB_ID)) AND file.mime ~ 'video/'));",
        {"JOB_ID" : job_id}
    )
    return extract_results(payload)


def get_merge_groups(db_manager: DataBaseManager, job_id: str) -> List[dict]:
    """Fetch merge_group rows for datasets referenced by an inference_job."""
    payload = db_manager.query(
        "SELECT * FROM merge_group WHERE dataset INSIDE (array::flatten((SELECT VALUE datasets FROM inference_job WHERE id = <record> $JOB_ID)));",
        {"JOB_ID": job_id},
    )
    return extract_results(payload)


# ---------------- Status Update API ----------------

ALLOWED_STATES = {"ProcessWaiting", "ProcessRunning", "Completed", "Faild"}
ALLOWED_NEXT = {
    "ProcessWaiting": {"ProcessRunning"},
    "ProcessRunning": {"Completed"},
    "Completed": set(),
    "Faild": set(),
}


def set_inference_job_status(db_manager: DataBaseManager, job_id: str, new_state: str):
    if new_state not in ALLOWED_STATES:
        raise ValueError(f"Unknown state: {new_state}")

    cur_res = db_manager.query(
        "SELECT VALUE status FROM inference_job WHERE id = <record> $ID LIMIT 1",
        {"ID": job_id},
    )
    current = first_result(cur_res)
    if current is None:
        raise ValueError(f"Inference job not found: {job_id}")
    if current == new_state:
        return {"id": job_id, "status": current, "updated": False}

    if new_state == "Faild":
        upd = db_manager.query(
            "UPDATE inference_job SET status = $NEW WHERE id = <record> $ID",
            {"ID": job_id, "NEW": new_state},
        )
        return {"id": job_id, "status": new_state, "updated": True, "previous": current, "result": upd}

    allowed = ALLOWED_NEXT.get(current, set())
    if new_state not in allowed:
        raise ValueError(f"Invalid transition: {current} -> {new_state}")

    upd = db_manager.query(
        "UPDATE inference_job SET status = $NEW WHERE id = <record> $ID AND status = $CUR",
        {"ID": job_id, "NEW": new_state, "CUR": current},
    )
    if first_result(upd) is None:
        raise ValueError("Concurrent update detected; state changed by another process")
    return {"id": job_id, "status": new_state, "updated": True, "previous": current, "result": upd}


def has_in_progress_job(db_manager: DataBaseManager) -> bool:
    """進行中の推論ジョブが存在するかを返す。"""
    payload = db_manager.query(
        "SELECT VALUE count() FROM inference_job WHERE status = 'ProcessRunning';"
    )
    cnt = first_result(payload)
    try:
        return int(cnt or 0) > 0
    except Exception:
        return False
