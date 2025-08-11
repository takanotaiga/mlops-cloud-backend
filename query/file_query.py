from typing import Any, Optional
from backend_module.database import DataBaseManager


class FileRecordNotFound(Exception):
    """Raised when a file record cannot be found in the DB."""


def _first_result(payload: Any) -> Optional[Any]:
    """
    Extract the first logical result from SurrealDB's query() response.
    Accepts both envelope-style (with 'result') and raw list responses.
    """
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "result" in first:
            results = first.get("result") or []
            return results[0] if results else None
        # Fallback: assume already a list of rows/values
        return first
    return None


def get_file(db_manager: DataBaseManager, file_id):
    res = db_manager.query(
        "SELECT * FROM file WHERE id = <record> $ID LIMIT 1",
        {"ID": file_id},
    )
    row = _first_result(res)
    if row is None:
        raise FileRecordNotFound(f"File not found: {file_id}")
    return row


def get_s3key(db_manager: DataBaseManager, file_id):
    res = db_manager.query(
        "SELECT VALUE key FROM file WHERE id = <record> $ID LIMIT 1",
        {"ID": file_id},
    )
    key = _first_result(res)
    if key is None:
        raise FileRecordNotFound(f"File not found: {file_id}")
    return key
