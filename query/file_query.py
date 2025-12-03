from backend_module.database import DataBaseManager
from query.utils import first_result, extract_results


class FileRecordNotFound(Exception):
    """Raised when a file record cannot be found in the DB."""


def get_file(db_manager: DataBaseManager, file_id):
    res = db_manager.query(
        "SELECT * FROM file WHERE id = <record> $ID LIMIT 1",
        {"ID": file_id},
    )
    row = first_result(res)
    if row is None:
        raise FileRecordNotFound(f"File not found: {file_id}")
    return row


def get_s3key(db_manager: DataBaseManager, file_id):
    res = db_manager.query(
        "SELECT VALUE key FROM file WHERE id = <record> $ID LIMIT 1",
        {"ID": file_id},
    )
    key = first_result(res)
    if key is None:
        raise FileRecordNotFound(f"File not found: {file_id}")
    return key


def list_videos_missing_thumbs(db_manager: DataBaseManager, *, limit: int = 50):
    """Return file rows for videos missing a thumbnail key."""
    res = db_manager.query(
        """
        SELECT id, key, dataset, name, thumbKey
        FROM file
        WHERE mime ~ 'video/' AND (thumbKey = NONE OR thumbKey = null OR string::len(thumbKey) = 0)
        LIMIT $LIMIT;
        """,
        {"LIMIT": limit},
    )
    # return raw rows; caller handles iteration
    from query.utils import extract_results
    return extract_results(res)


def set_thumb_key(db_manager: DataBaseManager, file_id, thumb_key: str):
    """Update the file.thumbKey for a given file record."""
    return db_manager.query(
        """
        UPDATE file SET thumbKey = $THUMB WHERE id = <record> $ID RETURN AFTER;
        """,
        {"ID": file_id, "THUMB": thumb_key},
    )


def get_file_by_dataset_and_name(db_manager: DataBaseManager, dataset: str, name: str):
    """Return a single file row for an exact dataset + name match."""
    res = db_manager.query(
        "SELECT * FROM file WHERE dataset = $DATASET AND name = $NAME LIMIT 1;",
        {"DATASET": dataset, "NAME": name},
    )
    return first_result(res)


def get_files_by_names(db_manager: DataBaseManager, dataset: str, names: list[str]):
    """Return file rows for the given dataset whose name is in names."""
    res = db_manager.query(
        "SELECT * FROM file WHERE dataset = $DATASET AND name INSIDE $NAMES;",
        {"DATASET": dataset, "NAMES": names},
    )
    return extract_results(res)


def insert_file_record(
    db_manager: DataBaseManager,
    *,
    dataset: str,
    key: str,
    name: str,
    mime: str,
    size: int,
    bucket: str,
    encode: str,
    thumb_key: str | None = None,
    meta: dict | None = None,
):
    """Insert a new file row and return Surreal's response."""
    return db_manager.query(
        """
        INSERT INTO file {
            dataset: $DATASET,
            key: $KEY,
            name: $NAME,
            mime: $MIME,
            size: $SIZE,
            bucket: $BUCKET,
            encode: $ENCODE,
            thumbKey: $THUMB,
            uploadedAt: time::now(),
            meta: $META
        };
        """,
        {
            "DATASET": dataset,
            "KEY": key,
            "NAME": name,
            "MIME": mime,
            "SIZE": size,
            "BUCKET": bucket,
            "ENCODE": encode,
            "THUMB": thumb_key,
            "META": meta if meta is not None else {},
        },
    )


def mark_file_dead(db_manager: DataBaseManager, file_id: str):
    """Set dead flag on a file record (cleanup task will remove it)."""
    return db_manager.query(
        """
        UPDATE file SET dead = true WHERE id = <record> $ID;
        """,
        {"ID": file_id},
    )
