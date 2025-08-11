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
