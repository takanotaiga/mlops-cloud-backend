from typing import List
from backend_module.database import DataBaseManager
from query.utils import extract_results


def get_key_bboxes_for_file(db_manager: DataBaseManager, file_id: str, *, category: str = "sam2_key_bbox") -> List[dict]:
    """Return annotation rows containing normalized bbox seeds for a given file.

    Each row should include: id, label, x1, y1, x2, y2, category, dataset, file
    """
    payload = db_manager.query(
        """
        SELECT * FROM annotation
        WHERE category = $CAT AND file = <record> $FILE
        ORDER BY id ASC;
        """,
        {"CAT": category, "FILE": file_id},
    )
    return extract_results(payload)


def get_key_bboxes_for_files(db_manager: DataBaseManager, file_ids: List[str], *, category: str = "sam2_key_bbox") -> List[dict]:
    """Return annotations for multiple files."""
    payload = db_manager.query(
        """
        SELECT * FROM annotation
        WHERE category = $CAT AND file INSIDE $FILES
        ORDER BY file ASC, id ASC;
        """,
        {"CAT": category, "FILES": [f"file:{fid}" if not str(fid).startswith("file:") else fid for fid in file_ids]},
    )
    return extract_results(payload)

