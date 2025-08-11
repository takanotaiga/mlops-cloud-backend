from backend_module.database import DataBaseManager


def insert_encoded_segment(
    db_manager: DataBaseManager,
    *,
    file_id,
    key: str,
    size: int,
    bucket: str,
    meta: dict,
):
    """エンコード済みセグメント1件をDBへ登録する。"""
    return db_manager.query(
        """
        INSERT INTO encoded_segment {
            file: <record> $FILE,
            key: $KEY,
            type: 'video-segment',
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
    """Return encoded_segment rows for given keys with joined file info.

    Each row includes: key, meta, file (rid), and file name/dataset for grouping.
    """
    return db_manager.query(
        """
        SELECT
            key,
            meta,
            file,
            file.name AS file_name,
            file.dataset AS file_dataset
        FROM encoded_segment
        WHERE key INSIDE $KEYS;
        """,
        {"KEYS": keys},
    )
