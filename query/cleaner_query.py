from typing import List

from backend_module.database import DataBaseManager
from query.utils import extract_results


def list_dead_file_records(db_manager: DataBaseManager) -> List[dict]:
    """Return [{ id, key, thumbKey? }] for file rows marked as dead."""
    payload = db_manager.query(
        "SELECT id, key, thumbKey FROM file WHERE dead = true;"
    )
    rows = extract_results(payload)
    # Ensure structure and filter out empties
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id"), "key": r.get("key"), "thumbKey": r.get("thumbKey")})
    return out


def delete_file_record(db_manager: DataBaseManager, file_id: str):
    """Delete a single file record by id."""
    return db_manager.query(
        "DELETE file WHERE id = <record> $ID;",
        {"ID": file_id},
    )


# -------------- Orphan cleanup (no file linkage) --------------

def list_orphan_annotations(db_manager: DataBaseManager) -> List[dict]:
    """Return [{ id, key }] for annotation rows whose file.id is NONE."""
    payload = db_manager.query(
        "SELECT key, id FROM annotation WHERE file.id = NONE;"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id"), "key": r.get("key")})
    return out


def delete_annotation_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE annotation WHERE id = <record> $ID;",
        {"ID": rid},
    )


def list_orphan_encode_jobs(db_manager: DataBaseManager) -> List[dict]:
    """Return [{ id }] for encode_job rows whose file.id is NONE."""
    payload = db_manager.query(
        "SELECT id FROM encode_job WHERE file.id = NONE;"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id")})
    return out


def delete_encode_job_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE encode_job WHERE id = <record> $ID;",
        {"ID": rid},
    )


def list_orphan_encoded_segments(db_manager: DataBaseManager) -> List[dict]:
    """Return [{ id, key }] for encoded_segment rows whose file.id is NONE."""
    payload = db_manager.query(
        "SELECT id, key FROM encoded_segment WHERE file.id = NONE;"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id"), "key": r.get("key")})
    return out


def delete_encoded_segment_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE encoded_segment WHERE id = <record> $ID;",
        {"ID": rid},
    )


def list_orphan_hls_jobs(db_manager: DataBaseManager) -> List[dict]:
    payload = db_manager.query(
        "SELECT id FROM hls_job WHERE file.id = NONE;"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id")})
    return out


def delete_hls_job_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE hls_job WHERE id = <record> $ID;",
        {"ID": rid},
    )


def list_orphan_hls_playlists(db_manager: DataBaseManager) -> List[dict]:
    payload = db_manager.query(
        "SELECT id, key FROM hls_playlist WHERE file.id = NONE;"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id"), "key": r.get("key")})
    return out


def delete_hls_playlist_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE hls_playlist WHERE id = <record> $ID;",
        {"ID": rid},
    )


def list_orphan_hls_segments(db_manager: DataBaseManager) -> List[dict]:
    payload = db_manager.query(
        "SELECT id, key FROM hls_segment WHERE file.id = NONE;"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id"), "key": r.get("key")})
    return out


def delete_hls_segment_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE hls_segment WHERE id = <record> $ID;",
        {"ID": rid},
    )


# -------------- Inference cleanup --------------

def list_dead_inference_jobs(db_manager: DataBaseManager) -> List[dict]:
    payload = db_manager.query(
        "SELECT id FROM inference_job WHERE dead = true;"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id")})
    return out


def delete_inference_job_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE inference_job WHERE id = <record> $ID;",
        {"ID": rid},
    )


def list_orphan_inference_results(db_manager: DataBaseManager) -> List[dict]:
    payload = db_manager.query(
        "SELECT id, key FROM inference_result WHERE job.id = NONE;"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id"), "key": r.get("key")})
    return out


def delete_inference_result_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE inference_result WHERE id = <record> $ID;",
        {"ID": rid},
    )


# -------------- Dataset-orphaned cleanup --------------

def list_orphan_labels(db_manager: DataBaseManager) -> List[dict]:
    payload = db_manager.query(
        "SELECT id FROM label WHERE dataset NOT IN (SELECT VALUE dataset FROM file);"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id")})
    return out


def delete_label_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE label WHERE id = <record> $ID;",
        {"ID": rid},
    )


def list_orphan_merge_groups(db_manager: DataBaseManager) -> List[dict]:
    payload = db_manager.query(
        "SELECT id FROM merge_group WHERE dataset NOT IN (SELECT VALUE dataset FROM file);"
    )
    rows = extract_results(payload)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("id"):
            out.append({"id": r.get("id")})
    return out


def delete_merge_group_record(db_manager: DataBaseManager, rid: str):
    return db_manager.query(
        "DELETE merge_group WHERE id = <record> $ID;",
        {"ID": rid},
    )
