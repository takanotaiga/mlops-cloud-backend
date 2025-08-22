from backend_module.database import DataBaseManager


def insert_hls_playlist(
    db_manager: DataBaseManager,
    *,
    file_id,
    key: str,
    size: int,
    bucket: str,
    meta: dict,
):
    """Insert HLS playlist (m3u8) metadata into DB."""
    return db_manager.query(
        """
        INSERT INTO hls_playlist {
            file: <record> $FILE,
            key: $KEY,
            type: 'video-hls-playlist',
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

