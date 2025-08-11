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