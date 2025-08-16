from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from query.utils import rid_leaf
from query.inference_result_query import insert_inference_result


@dataclass
class GroupUploadItem:
    index: int  # 1-based group index
    dataset: Optional[str]
    file_ids: List[str]
    video_path: Optional[str]
    schema_json_path: Optional[str] = None
    group_parquet: Optional[str] = None
    results_artifacts: Optional[List[Dict[str, Any]]] = None  # per-file artifacts (e.g., parquet)


def upload_group_results(
    *,
    db_manager: DataBaseManager,
    uploader: MinioS3Uploader,
    job_id: str,
    items: List[GroupUploadItem],
):
    """Upload per-group results to S3 and register in DB.

    - Uploads the main result video per group under
      inference/<job_leaf>/group_###.mp4 and inserts inference_result.
    - Optionally uploads schema JSON and group-level parquet.
    - Optionally uploads per-file artifacts if provided (e.g., parquet by file).
    """
    job_leaf = rid_leaf(job_id)

    for it in items:
        # Upload main video
        if it.video_path and Path(it.video_path).exists():
            key = f"inference/{job_leaf}/group_{it.index:03d}.mp4"
            up = uploader.upload_file_as(it.video_path, key)
            if up.status == S3Info.SUCCESS:
                size = Path(it.video_path).stat().st_size
                insert_inference_result(
                    db_manager,
                    job_id=job_id,
                    dataset=it.dataset,
                    files=it.file_ids,
                    key=key,
                    bucket=uploader.bucket,
                    size=size,
                    labels=[],
                    meta={
                        "groupIndex": it.index,
                        "sourceFiles": it.file_ids,
                        "artifact": "plot_video",
                    },
                )

        # Upload schema JSON
        if it.schema_json_path and Path(it.schema_json_path).exists():
            key = f"inference/{job_leaf}/group_{it.index:03d}_schema.json"
            up = uploader.upload_file_as(it.schema_json_path, key)
            if up.status == S3Info.SUCCESS:
                size = Path(it.schema_json_path).stat().st_size
                insert_inference_result(
                    db_manager,
                    job_id=job_id,
                    dataset=it.dataset,
                    files=it.file_ids,
                    key=key,
                    bucket=uploader.bucket,
                    size=size,
                    labels=[],
                    meta={
                        "groupIndex": it.index,
                        "sourceFiles": it.file_ids,
                        "artifact": "schema_json",
                        "contentType": "application/json",
                    },
                )

        # Upload group-level parquet
        if it.group_parquet and Path(it.group_parquet).exists():
            key = f"inference/{job_leaf}/group_{it.index:03d}/overall_results.parquet"
            up = uploader.upload_file_as(it.group_parquet, key)
            if up.status == S3Info.SUCCESS:
                size = Path(it.group_parquet).stat().st_size
                insert_inference_result(
                    db_manager,
                    job_id=job_id,
                    dataset=it.dataset,
                    files=it.file_ids,
                    key=key,
                    bucket=uploader.bucket,
                    size=size,
                    labels=[],
                    meta={
                        "groupIndex": it.index,
                        "sourceFiles": it.file_ids,
                        "artifact": "results_parquet",
                        "contentType": "application/x-parquet",
                    },
                )

        # Upload per-file artifacts (if provided)
        if it.results_artifacts:
            for art in it.results_artifacts:
                p = art.get("parquet")
                fid = str(art.get("file_id")) if art.get("file_id") is not None else None
                if not p or not fid or not Path(p).exists():
                    continue
                key = f"inference/{job_leaf}/group_{it.index:03d}/{Path(fid).name}_results.parquet"
                up = uploader.upload_file_as(p, key)
                if up.status == S3Info.SUCCESS:
                    size = Path(p).stat().st_size
                    insert_inference_result(
                        db_manager,
                        job_id=job_id,
                        dataset=it.dataset,
                        files=[fid],
                        key=key,
                        bucket=uploader.bucket,
                        size=size,
                        labels=[],
                        meta={
                            "groupIndex": it.index,
                            "sourceFiles": [fid],
                            "artifact": "results_parquet",
                            "contentType": "application/x-parquet",
                        },
                    )

