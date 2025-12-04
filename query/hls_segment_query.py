from backend_module.database import DataBaseManager
from query.utils import extract_results


def insert_hls_segment(
    db_manager: DataBaseManager,
    *,
    file_id,
    key: str,
    size: int,
    bucket: str,
    meta: dict,
):
    """Insert one HLS segment row into DB."""
    return db_manager.query(
        """
        INSERT INTO hls_segment {
            file: <record> $FILE,
            key: $KEY,
            type: 'video-hls-segment',
            size: $SIZE,
            uploadedAt: time::now(),
            bucket: $BUCKET,
            meta: $META
        };
        """,
        {
            "FILE": file_id,
            "KEY": key,
            "SIZE": size,
            "BUCKET": bucket,
            "META": meta,
        },
    )


def get_segments_with_file_by_keys(db_manager: DataBaseManager, keys: list[str]):
    """Return hls_segment rows for given keys with joined file info."""
    return db_manager.query(
        """
        SELECT
            key,
            meta,
            file,
            file.name AS file_name,
            file.dataset AS file_dataset
        FROM hls_segment
        WHERE key INSIDE $KEYS;
        """,
        {"KEYS": keys},
    )


def list_segments_for_file(db_manager: DataBaseManager, file_id: str):
    """Return ordered HLS segment rows (including init) for a file."""
    res = db_manager.query(
        """
        SELECT key, bucket, meta, meta.index AS idx
        FROM hls_segment
        WHERE file = <record> $FILE
        ORDER BY idx ASC;
        """,
        {"FILE": file_id},
    )
    return extract_results(res)
