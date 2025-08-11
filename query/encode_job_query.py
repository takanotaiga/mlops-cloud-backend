from typing import Any, Optional
from backend_module.database import DataBaseManager


def queue_unencoded_video_jobs(db_manager: DataBaseManager):
    db_manager.query(
        "INSERT INTO encode_job (SELECT time::now() AS created_at, id AS file, 'queued' AS status FROM file WHERE encode = 'video-none' AND mime ~ 'video/' AND id NOTINSIDE (SELECT VALUE file FROM encode_job));"
    )


def get_queued_job(db_manager: DataBaseManager):
    res = db_manager.query(
        """
        SELECT 
            *
        FROM 
            encode_job
        WHERE 
            status = 'queued'
        ORDER 
            BY created_at ASC
        LIMIT 
            math::max([
                $limit_in_progress - array::len((SELECT id FROM encode_job WHERE status = 'in_progress')),
                0
            ]);
        """,
        {"limit_in_progress": 3},
    )
    return res


# ---------------- Status Update API ----------------

class JobNotFound(Exception):
    pass


class InvalidJobTransition(Exception):
    pass


def _first_result(payload: Any) -> Optional[Any]:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "result" in first:
            results = first.get("result") or []
            return results[0] if results else None
        return first
    return None


ALLOWED_STATES = {"queued", "in_progress", "faild", "complete"}
ALLOWED_NEXT = {
    "queued": {"in_progress"},
    "in_progress": {"complete"},
    "complete": set(),
    "faild": set(),
}


def set_encode_job_status(db_manager: DataBaseManager, job_id: str, new_state: str):
    """
    状態遷移をバリデーションして更新する。
    - 許可: queued -> in_progress -> complete
    - 失敗: faild はどの状態からでも可
    - 同一状態は何もしない（エラーにしない）
    失敗時は InvalidJobTransition を送出。存在しない場合は JobNotFound。
    """
    if new_state not in ALLOWED_STATES:
        raise InvalidJobTransition(f"Unknown state: {new_state}")

    # 現在ステータス取得
    cur_res = db_manager.query(
        "SELECT VALUE status FROM encode_job WHERE id = <record> $ID LIMIT 1",
        {"ID": job_id},
    )
    current = _first_result(cur_res)
    if current is None:
        raise JobNotFound(f"Encode job not found: {job_id}")

    # 変更不要
    if current == new_state:
        return {"id": job_id, "status": current, "updated": False}

    # faild は常に許可
    if new_state == "faild":
        upd = db_manager.query(
            "UPDATE encode_job SET status = $NEW WHERE id = <record> $ID",
            {"ID": job_id, "NEW": new_state},
        )
        return {"id": job_id, "status": new_state, "updated": True, "previous": current, "result": upd}

    # 通常の遷移チェック
    allowed = ALLOWED_NEXT.get(current, set())
    if new_state not in allowed:
        raise InvalidJobTransition(f"Invalid transition: {current} -> {new_state}")

    # 競合防止のため、期待する現在状態を条件に含める
    upd = db_manager.query(
        "UPDATE encode_job SET status = $NEW WHERE id = <record> $ID AND status = $CUR",
        {"ID": job_id, "NEW": new_state, "CUR": current},
    )
    # 更新できなかった場合（並行更新等）はブロック
    if _first_result(upd) is None:
        raise InvalidJobTransition("Concurrent update detected; state changed by another process")

    return {"id": job_id, "status": new_state, "updated": True, "previous": current, "result": upd}
