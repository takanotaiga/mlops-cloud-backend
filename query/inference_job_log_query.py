from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, List, Optional

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.uuid_tools import get_uuid
from query.utils import extract_results, first_result
from query.utils import rid_leaf

LOG_DB_RETENTION = 10_000
LOG_ARCHIVE_CHUNK_SIZE = 10_000


def append_job_log(
    db_manager: DataBaseManager,
    *,
    job_id: str,
    source: str,
    stream: str,
    message: str,
    seq: int,
    archive_uploader: Optional[MinioS3Uploader] = None,
    retention: int = LOG_DB_RETENTION,
    archive_chunk_size: int = LOG_ARCHIVE_CHUNK_SIZE,
) -> Any:
    """Persist one inference job log line.

    The job relation is mandatory so UI queries can isolate logs per job and
    never mix concurrent MLX/CV output from other jobs.
    """
    result = db_manager.query(
        """
        CREATE inference_job_log CONTENT {
            job: <record> $JOB,
            source: $SOURCE,
            stream: $STREAM,
            message: $MESSAGE,
            seq: $SEQ,
            createdAt: time::now()
        };
        """,
        {
            "JOB": job_id,
            "SOURCE": source,
            "STREAM": stream,
            "MESSAGE": message,
            "SEQ": seq,
        },
    )
    if archive_uploader is not None:
        archive_job_logs_if_needed(
            db_manager,
            uploader=archive_uploader,
            job_id=job_id,
            retention=retention,
            archive_chunk_size=archive_chunk_size,
        )
    return result


def list_job_logs(
    db_manager: DataBaseManager,
    *,
    job_id: str,
) -> List[dict]:
    payload = db_manager.query(
        """
        SELECT source, stream, message, seq, createdAt
        FROM inference_job_log
        WHERE job = <record> $JOB
        ORDER BY createdAt ASC, seq ASC;
        """,
        {"JOB": job_id},
    )
    return extract_results(payload)


def list_job_log_archives(
    db_manager: DataBaseManager,
    *,
    job_id: str,
) -> List[dict]:
    payload = db_manager.query(
        """
        SELECT bucket, key, rowCount, firstSeq, lastSeq, firstCreatedAt, lastCreatedAt, createdAt
        FROM inference_job_log_archive
        WHERE job = <record> $JOB
        ORDER BY firstCreatedAt ASC, firstSeq ASC, createdAt ASC;
        """,
        {"JOB": job_id},
    )
    return extract_results(payload)


def count_job_logs(db_manager: DataBaseManager, *, job_id: str) -> int:
    payload = db_manager.query(
        """
        SELECT count() AS count
        FROM inference_job_log
        WHERE job = <record> $JOB
        GROUP ALL;
        """,
        {"JOB": job_id},
    )
    row = first_result(payload) or {}
    try:
        return int(row.get("count") or 0)
    except Exception:
        return 0


def _format_log_archive_line(row: dict) -> str:
    created_at = str(row.get("createdAt") or "")
    source = str(row.get("source") or "log")
    stream = str(row.get("stream") or "stdout")
    seq = row.get("seq")
    message = str(row.get("message") or "").replace("\r", "").replace("\n", "\\n")
    return f"[{created_at}] [{source}/{stream}] [#{seq}] {message}"


def archive_job_logs_if_needed(
    db_manager: DataBaseManager,
    *,
    uploader: MinioS3Uploader,
    job_id: str,
    retention: int = LOG_DB_RETENTION,
    archive_chunk_size: int = LOG_ARCHIVE_CHUNK_SIZE,
) -> Optional[dict]:
    """Move oldest log chunks to S3 .log files when DB retention is exceeded."""
    if retention <= 0 or archive_chunk_size <= 0:
        return None
    if count_job_logs(db_manager, job_id=job_id) <= retention:
        return None

    rows = extract_results(
        db_manager.query(
            """
            SELECT id, source, stream, message, seq, createdAt
            FROM inference_job_log
            WHERE job = <record> $JOB
            ORDER BY createdAt ASC, seq ASC
            LIMIT $LIMIT;
            """,
            {"JOB": job_id, "LIMIT": archive_chunk_size},
        )
    )
    if len(rows) < archive_chunk_size:
        return None

    first = rows[0]
    last = rows[-1]
    with tempfile.TemporaryDirectory(prefix="job-log-archive-") as td:
        local_path = Path(td) / "job_logs.log"
        local_path.write_text("\n".join(_format_log_archive_line(row) for row in rows) + "\n", encoding="utf-8")
        key = (
            f"inference/{rid_leaf(job_id)}/logs/"
            f"job_logs_{first.get('seq', 0)}_{last.get('seq', 0)}_{get_uuid(8)}.log"
        )
        upload = uploader.upload_file_as(str(local_path), key)
        if upload.status != S3Info.SUCCESS:
            raise RuntimeError(f"Upload job log archive failed: {upload.error}")

        archive = first_result(
            db_manager.query(
                """
                CREATE inference_job_log_archive CONTENT {
                    job: <record> $JOB,
                    bucket: $BUCKET,
                    key: $KEY,
                    rowCount: $ROW_COUNT,
                    firstSeq: $FIRST_SEQ,
                    lastSeq: $LAST_SEQ,
                    firstCreatedAt: $FIRST_CREATED_AT,
                    lastCreatedAt: $LAST_CREATED_AT,
                    createdAt: time::now(),
                    meta: {
                        artifact: 'job_log',
                        contentType: 'text/plain'
                    }
                };
                """,
                {
                    "JOB": job_id,
                    "BUCKET": uploader.bucket,
                    "KEY": key,
                    "ROW_COUNT": len(rows),
                    "FIRST_SEQ": first.get("seq"),
                    "LAST_SEQ": last.get("seq"),
                    "FIRST_CREATED_AT": first.get("createdAt"),
                    "LAST_CREATED_AT": last.get("createdAt"),
                },
            )
        )
        db_manager.query(
            """
            DELETE inference_job_log
            WHERE id IN $IDS;
            """,
            {"IDS": [row.get("id") for row in rows if row.get("id")]},
        )
        return archive if isinstance(archive, dict) else None


def get_job_id_for_inference_result(
    db_manager: DataBaseManager,
    inference_result_id: str,
) -> Optional[str]:
    payload = db_manager.query(
        "SELECT VALUE job FROM inference_result WHERE id = <record> $ID LIMIT 1;",
        {"ID": inference_result_id},
    )
    value = first_result(payload)
    return str(value) if value else None
