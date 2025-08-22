from typing import List
from backend_module.database import DataBaseManager
from query.utils import first_result, extract_results


def queue_unhls_video_jobs(db_manager: DataBaseManager):
    """Enqueue HLS jobs for videos that don't have a corresponding hls_job yet."""
    db_manager.query(
        "INSERT INTO hls_job (SELECT time::now() AS created_at, id AS file, 'queued' AS status FROM file WHERE encode INSIDE ['video-none','video-merge'] AND mime ~ 'video/' AND id NOTINSIDE (SELECT VALUE file FROM hls_job));"
    )


def get_queued_job(db_manager: DataBaseManager) -> List[dict]:
    payload = db_manager.query(
        "SELECT * FROM hls_job WHERE status = 'queued';"
    )
    return extract_results(payload)


def has_in_progress_job(db_manager: DataBaseManager) -> bool:
    payload = db_manager.query(
        "SELECT VALUE count() FROM hls_job WHERE status = 'in_progress';"
    )
    cnt = first_result(payload)
    try:
        return int(cnt or 0) > 0
    except Exception:
        return False


# ---------------- Status Update API ----------------

class JobNotFound(Exception):
    pass


class InvalidJobTransition(Exception):
    pass


ALLOWED_STATES = {"queued", "in_progress", "faild", "complete"}
ALLOWED_NEXT = {
    "queued": {"in_progress"},
    "in_progress": {"complete"},
    "complete": set(),
    "faild": set(),
}


def set_hls_job_status(db_manager: DataBaseManager, job_id: str, new_state: str):
    """
    Validate and update hls_job state with optimistic concurrency.
    Allowed: queued -> in_progress -> complete; faild is always allowed.
    """
    if new_state not in ALLOWED_STATES:
        raise InvalidJobTransition(f"Unknown state: {new_state}")

    cur_res = db_manager.query(
        "SELECT VALUE status FROM hls_job WHERE id = <record> $ID LIMIT 1",
        {"ID": job_id},
    )
    current = first_result(cur_res)
    if current is None:
        raise JobNotFound(f"HLS job not found: {job_id}")

    if current == new_state:
        return {"id": job_id, "status": current, "updated": False}

    if new_state == "faild":
        upd = db_manager.query(
            "UPDATE hls_job SET status = $NEW WHERE id = <record> $ID",
            {"ID": job_id, "NEW": new_state},
        )
        return {"id": job_id, "status": new_state, "updated": True, "previous": current, "result": upd}

    allowed = ALLOWED_NEXT.get(current, set())
    if new_state not in allowed:
        raise InvalidJobTransition(f"Invalid transition: {current} -> {new_state}")

    upd = db_manager.query(
        "UPDATE hls_job SET status = $NEW WHERE id = <record> $ID AND status = $CUR",
        {"ID": job_id, "NEW": new_state, "CUR": current},
    )
    if first_result(upd) is None:
        raise InvalidJobTransition("Concurrent update detected; state changed by another process")

    return {"id": job_id, "status": new_state, "updated": True, "previous": current, "result": upd}

