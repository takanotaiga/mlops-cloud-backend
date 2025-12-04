from backend_module.database import DataBaseManager
from query.utils import first_result


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


def get_playlists_with_file_by_keys(db_manager: DataBaseManager, keys: list[str]):
    """Return hls_playlist rows for given keys with joined file info."""
    return db_manager.query(
        """
        SELECT
            key,
            file,
            file.name AS file_name,
            file.dataset AS file_dataset
        FROM hls_playlist
        WHERE key INSIDE $KEYS;
        """,
        {"KEYS": keys},
    )


def get_playlist_for_file(db_manager: DataBaseManager, file_id: str):
    """Return the first playlist row for a given file id, or None."""
    res = db_manager.query(
        """
        SELECT *
        FROM hls_playlist
        WHERE file = <record> $FILE
        LIMIT 1;
        """,
        {"FILE": file_id},
    )
    return first_result(res)
