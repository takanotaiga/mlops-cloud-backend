from backend_module.database import DataBaseManager


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

