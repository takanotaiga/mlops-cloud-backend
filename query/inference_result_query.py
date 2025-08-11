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

