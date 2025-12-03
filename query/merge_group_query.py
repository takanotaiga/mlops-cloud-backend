from backend_module.database import DataBaseManager
from query.utils import extract_results


def list_pending_merge_groups(db_manager: DataBaseManager):
    """Return merge_group rows that have not been processed into a merged file."""
    return extract_results(
        db_manager.query(
            """
            SELECT * FROM merge_group
            WHERE mergedFile = NONE OR mergedFile = null;
            """
        )
    )


def set_merged_file(db_manager: DataBaseManager, merge_group_id: str, file_id: str):
    """Mark merge_group as merged and link the created file record."""
    return db_manager.query(
        """
        UPDATE merge_group SET mergedFile = <record> $FILE, mergedAt = time::now()
        WHERE id = <record> $ID;
        """,
        {"FILE": file_id, "ID": merge_group_id},
    )


def mark_merge_group_dead(db_manager: DataBaseManager, merge_group_id: str):
    """Set dead flag on merge_group to allow cleanup to delete it later."""
    return db_manager.query(
        """
        UPDATE merge_group SET dead = true WHERE id = <record> $ID;
        """,
        {"ID": merge_group_id},
    )
