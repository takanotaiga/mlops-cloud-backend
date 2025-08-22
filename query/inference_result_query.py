from typing import List, Optional
from backend_module.database import DataBaseManager


def insert_inference_result(
    db_manager: DataBaseManager,
    *,
    job_id: str,
    dataset: Optional[str],
    files: List[str],
    key: str,
    bucket: str,
    size: int,
    labels: List[str],
    meta: dict,
):
    """Insert a new inference_result row with minimal metadata."""
    return db_manager.query(
        """
        INSERT INTO inference_result {
            job: <record> $JOB,
            dataset: $DATASET,
            files: $FILES,
            key: $KEY,
            bucket: $BUCKET,
            size: $SIZE,
            labels: $LABELS,
            createdAt: time::now(),
            meta: $META
        };
        """,
        {
            "JOB": job_id,
            "DATASET": dataset,
            "FILES": [f if str(f).startswith("file:") else f"file:{f}" for f in files],
            "KEY": key,
            "BUCKET": bucket,
            "SIZE": size,
            "LABELS": labels,
            "META": meta,
        },
    )


def get_s3key(db_manager: DataBaseManager, inference_result_id: str) -> str:
    """Return the `key` for a given inference_result record id.

    Raises KeyError if not found.
    """
    res = db_manager.query(
        "SELECT VALUE key FROM inference_result WHERE id = <record> $ID LIMIT 1",
        {"ID": inference_result_id},
    )
    # Lazy import to avoid circulars
    from query.utils import first_result

    key = first_result(res)
    if key is None:
        raise KeyError(f"Inference result not found: {inference_result_id}")
    return key
