from backend_module.database import DataBaseManager
from query.utils import first_result


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
