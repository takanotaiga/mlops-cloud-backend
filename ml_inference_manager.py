import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.config import load_surreal_config, load_s3_config
from backend_module.uuid_tools import get_uuid
from query import ml_inference_job_query
from query.encoded_segment_query import get_segments_with_file_by_keys
from query.utils import extract_results
 
import mimetypes
from collections import defaultdict
import re
import json


def _get_env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    """Read an environment variable with optional default and required flag."""
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise ValueError(f"Environment variable {name} is required")
    return val


class MLInferenceRunner:
    """
    Simple polling runner for ML inference tasks.
    This sets up DB and S3 clients similarly to video_manager.TaskRunner,
    but leaves the actual ML processing steps as placeholders.
    """

    def __init__(self, interval: Optional[float] = None):
        # Polling interval seconds
        self.interval = float(_get_env("POLL_INTERVAL", str(interval if interval is not None else 5)))

        # SurrealDB configuration (compose-friendly)
        sconf = load_surreal_config()
        self.db_manager = DataBaseManager(
            endpoint_url=sconf["endpoint_url"],
            username=sconf["username"],
            password=sconf["password"],
            namespace=sconf["namespace"],
            database=sconf["database"],
        )

        # S3/MinIO configuration (compose-friendly)
        mconf = load_s3_config()
        self.uploader = MinioS3Uploader(
            endpoint_url=mconf["endpoint_url"],
            access_key=mconf["access_key"],
            secret_key=mconf["secret_key"],
            bucket=mconf["bucket"],
            region_name=mconf["region_name"],
            multipart_threshold_bytes=mconf["multipart_threshold_bytes"],
            multipart_chunksize_bytes=mconf["multipart_chunksize_bytes"],
            part_concurrency=mconf["part_concurrency"],
            addressing_style=mconf["addressing_style"],
        )

        self.next_time = time.time()
        # Keep per-job work directories for debugging when true-ish
        self.keep_work_dir = _get_env("KEEP_WORK_DIR", "0").lower() in ("1", "true", "yes", "on")

    def dataset_download(
        self,
        job_id: str,
        work_dir: Path,
    ) -> tuple[dict[str, list[dict]], dict[str, str], dict[str, str], list[list[list[object]]]]:
        """
        Download all files linked to the job's datasets and prepare
        preprocessed structures for downstream inference.

        Returns a tuple:
          - file_segments: {file_id: [{key, local_path, index}, ...]}
          - file_name_by_id: {file_id: file_name}
          - file_dataset_by_id: {file_id: dataset_id}
          - grouped: [[ [segment_path, seq], ... ], ...] group lists
        """
        print("==== Start Dataset Download ====")
        linked_file = ml_inference_job_query.get_linked_file(self.db_manager, job_id)
        download_result = self.uploader.download_files(keys=linked_file, dest_dir=work_dir)
        key_to_local = {
            r.key: r.local_path
            for r in download_result
            if r.status == S3Info.SUCCESS and r.local_path
        }
        print("==== Complete ====")

        print("==== build groups ====")
        def _is_video(path: str) -> bool:
            mtype, _ = mimetypes.guess_type(path)
            return (mtype or "").startswith("video/")

        video_keys = [k for k, lp in key_to_local.items() if _is_video(lp)]
        if not video_keys:
            print("No video files in this batch; nothing to group.")
            return {}, {}, {}, []

        seg_payload = get_segments_with_file_by_keys(self.db_manager, video_keys)
        seg_rows = extract_results(seg_payload)

        print("[dataset_download]", "Aggregate segments by original file id")
        file_segments: dict[str, list[dict]] = defaultdict(list)
        file_name_by_id: dict[str, str] = {}
        file_dataset_by_id: dict[str, str] = {}
        for row in seg_rows:
            key = row.get("key")
            meta = row.get("meta") or {}
            file_rid = str(row.get("file"))
            file_name = row.get("file_name")
            file_dataset = row.get("file_dataset")
            if key not in key_to_local:
                continue  # skip not downloaded
            file_name_by_id[file_rid] = file_name
            file_dataset_by_id[file_rid] = file_dataset
            file_segments[file_rid].append(
                {
                    "key": key,
                    "local_path": key_to_local[key],
                    "index": (meta.get("index") if isinstance(meta, dict) else None),
                }
            )

        # Sort segments within each file by numeric meta.index,
        # with a robust fallback to the local filename pattern out_###.
        def _seg_sort_key(seg: dict):
            idx = seg.get("index")
            try:
                idx_num = int(idx)
            except Exception:
                idx_num = None
            if idx_num is not None:
                return (0, idx_num)
            # fallback: parse from local filename like out_003-xxxx.mp4
            base = os.path.basename(seg.get("local_path") or "")
            m = re.search(r"out_(\d+)", base)
            if m:
                return (1, int(m.group(1)))
            return (2, base)

        for segs in file_segments.values():
            segs.sort(key=_seg_sort_key)

        print("[dataset_download]", "Fetch merge groups for this job")
        mg_rows = ml_inference_job_query.get_merge_groups(self.db_manager, job_id)
        merge_groups = mg_rows  # list[dict]

        print("[dataset_download]", "Build dataset->name->file_id mapping")
        files_by_dataset_name: dict[str, dict[str, str]] = defaultdict(dict)
        for fid, name in file_name_by_id.items():
            dataset = file_dataset_by_id.get(fid)
            if dataset and name:
                files_by_dataset_name[dataset][name] = fid

        used_files: set[str] = set()
        grouped: list[list[list[object]]] = []

        print("[dataset_download]", "Compose groups for merge definitions")
        for mg in merge_groups:
            dataset = mg.get("dataset")
            members = mg.get("members") or []
            # Resolve member names to file ids (filtering missing)
            member_fids = [files_by_dataset_name.get(dataset, {}).get(name) for name in members]
            member_fids = [fid for fid in member_fids if fid]
            # Sort concatenation order by original file name
            member_fids.sort(key=lambda fid: (file_name_by_id.get(fid) or ""))

            seq = 1
            group_list: list[list[object]] = []
            for fid in member_fids:
                used_files.add(fid)
                for seg in file_segments.get(fid, []):
                    group_list.append([seg["local_path"], seq])
                    seq += 1
            if group_list:
                grouped.append(group_list)

        print("[dataset_download]", "Add independent videos (not in any merge group)")
        for fid, segs in file_segments.items():
            if fid in used_files:
                continue
            group_list = [[seg["local_path"], i] for i, seg in enumerate(segs, start=1)]
            if group_list:
                grouped.append(group_list)

        print("==== Complete ====")

        return file_segments, file_name_by_id, file_dataset_by_id, grouped


    def _build_file_groups(
        self,
        *,
        grouped: list[list[list[object]]],
        file_segments: dict[str, list[dict]],
        file_dataset_by_id: dict[str, str],
        file_name_by_id: dict[str, str],
    ) -> list[list[dict]]:
        """
        Build per-group file descriptors from grouped segment paths.

        Args:
            grouped: Grouped segments where each group is a list of [segment_path, seq].
            file_segments: Mapping file_id -> list of segment dicts containing "local_path".
            file_dataset_by_id: Mapping file_id -> dataset id.
            file_name_by_id: Mapping file_id -> original file name.

        Returns:
            A list of groups; each group is a list of dicts with keys
            "file_id", "dataset", "name", and "segments" (list of paths).
        """
        file_groups: list[list[dict]] = []
        for _, group in enumerate(grouped, start=1):
            file_ids_in_group: List[str] = []
            # Determine member file ids in order of first occurrence within the group
            for p, _ in group:
                for fid, segs in file_segments.items():
                    if any(s.get("local_path") == p for s in segs):
                        if fid not in file_ids_in_group:
                            file_ids_in_group.append(fid)
                        break

            file_group: list[dict] = []
            for fid in file_ids_in_group:
                seg_paths = [s["local_path"] for s in file_segments.get(fid, [])]
                if not seg_paths:
                    continue
                file_group.append(
                    {
                        "file_id": fid,
                        "dataset": file_dataset_by_id.get(fid),
                        "name": file_name_by_id.get(fid),
                        "segments": seg_paths,
                    }
                )
            if not file_group:
                continue

            file_groups.append(file_group)

        return file_groups

    def task_main(self):
        """
        One polling tick.
        This is where you would:
          - enqueue inference jobs for eligible inputs
          - fetch queued jobs from DB
          - download model inputs from object storage
          - run ML inference (placeholder)
          - upload results and write metadata back to DB
          - update job state
        The concrete schema/queries are intentionally omitted.
        """

        jobs = ml_inference_job_query.get_queued_job(self.db_manager)  # placeholder until queries are defined

        def _process_job(job: dict):
            job_id = job.get("id")
            task_type = job.get("taskType")
            model = job.get("model")
            model_source = job.get("modelSource")
            job_options = job.get("options") or {}
            # options may be JSON string; try to decode
            if isinstance(job_options, str):
                try:
                    job_options = json.loads(job_options)
                except Exception:
                    job_options = {}
            
            # Basic guard
            if not job_id or not task_type:
                return
            # NOTE: We no longer import or call ml_module directly here.
            # All model execution must happen in a separate Python process.

            # Update status to running (required before Completed)
            try:
                ml_inference_job_query.set_inference_job_status(
                    self.db_manager, job_id, "ProcessRunning"
                )
            except Exception as _e:
                # If we cannot move to running, fail the job explicitly so the loop doesn't retry forever
                try:
                    ml_inference_job_query.set_inference_job_status(
                        self.db_manager, job_id, "Faild"
                    )
                except Exception:
                    pass
                raise

            # Prepare a per-job working directory
            work_dir = Path("work_infer") / get_uuid(16)
            try:
                # Download dataset and build groups
                (
                    file_segments,
                    file_name_by_id,
                    file_dataset_by_id,
                    grouped,
                ) = self.dataset_download(job_id=job_id, work_dir=work_dir)
                if not grouped:
                    print("No valid groups produced for this job.")
                    # We are already in ProcessRunning, so completing is valid
                    ml_inference_job_query.set_inference_job_status(
                        self.db_manager, job_id, "Completed"
                    )
                    return
                
                file_groups = self._build_file_groups(
                    grouped=grouped,
                    file_segments=file_segments,
                    file_dataset_by_id=file_dataset_by_id,
                    file_name_by_id=file_name_by_id,
                )
                
                from ml_module import registry
                from ml_module.postprocess import postprocess_video_paths
                from ml_module.uploader import upload_group_results, GroupUploadItem

                # 1) Run inference per group and collect result paths
                result_paths: list[str] = []
                group_contexts: list[dict] = []
                temp_datasets_to_cleanup: set[str] = set()
                for gi, fg in enumerate(file_groups, start=1):
                    g_work_dir = work_dir / f"g_{gi:03d}"
                    g_work_dir.mkdir(parents=True, exist_ok=True)
                    res = registry.run_inference_task(
                        db_manager=self.db_manager,
                        job_id=str(job_id),
                        task_type=str(task_type),
                        model=model,
                        model_source=model_source,
                        file_group=fg,
                        work_dir=str(g_work_dir),
                    )
                    out_path = None
                    schema_json_path = None
                    results_artifacts = None
                    group_parquet = None
                    video_description = None
                    schema_description = None
                    group_parquet_description = None
                    if isinstance(res, dict):
                        out_path = res.get("output_path")
                        schema_json_path = res.get("schema_json_path")
                        results_artifacts = res.get("results_artifacts")
                        group_parquet = res.get("group_parquet")
                        video_description = res.get("video_description")
                        schema_description = res.get("schema_description")
                        group_parquet_description = res.get("group_parquet_description")
                        try:
                            for p in (res.get("temp_datasets") or []):
                                if isinstance(p, str) and p:
                                    temp_datasets_to_cleanup.add(p)
                        except Exception:
                            pass
                    elif isinstance(res, str):
                        out_path = res
                    if out_path:
                        result_paths.append(out_path)
                    else:
                        result_paths.append("")  # placeholder to keep indexing stable
                    # derive dataset and file_ids for this group from fg
                    file_ids = [str(x.get("file_id")) for x in fg if x.get("file_id")]
                    dataset = str(fg[0].get("dataset")) if fg and fg[0].get("dataset") else None
                    group_contexts.append({
                        "index": gi,
                        "dataset": dataset,
                        "file_ids": file_ids,
                        "schema_json_path": schema_json_path,
                        "results_artifacts": results_artifacts,
                        "group_parquet": group_parquet,
                        "video_description": video_description,
                        "schema_description": schema_description,
                        "group_parquet_description": group_parquet_description,
                    })

                # 2) Postprocess all result videos (e.g., transcode)
                processed_paths = postprocess_video_paths([p for p in result_paths if p], str(work_dir / "post"))

                # 3) Upload to S3 and register DB records via module
                items: list[GroupUploadItem] = []
                pi = 0
                for ctx, orig in zip(group_contexts, result_paths):
                    vp: Optional[str] = None
                    if orig:
                        if pi < len(processed_paths):
                            vp = processed_paths[pi]
                            pi += 1
                    items.append(
                        GroupUploadItem(
                            index=ctx["index"],
                            dataset=ctx["dataset"],
                            file_ids=ctx["file_ids"],
                            video_path=vp,
                            video_description=ctx.get("video_description"),
                            schema_json_path=ctx.get("schema_json_path"),
                            schema_description=ctx.get("schema_description"),
                            group_parquet=ctx.get("group_parquet"),
                            group_parquet_description=ctx.get("group_parquet_description"),
                            results_artifacts=ctx.get("results_artifacts") or [],
                        )
                    )

                upload_group_results(
                    db_manager=self.db_manager,
                    uploader=self.uploader,
                    job_id=str(job_id),
                    items=items,
                )

                # 4) Mark job completed
                ml_inference_job_query.set_inference_job_status(self.db_manager, job_id, "Completed")
                return


            except Exception as e:
                try:
                    ml_inference_job_query.set_inference_job_status(self.db_manager, job_id, "Faild")
                except Exception:
                    pass
                print(f"Inference job {job_id} failed: {e}")
            finally:
                try:
                    if work_dir.exists() and not self.keep_work_dir:
                        # Best-effort cleanup of workspace and any temp datasets
                        import shutil
                        shutil.rmtree(work_dir, ignore_errors=True)
                        for d in list(temp_datasets_to_cleanup):
                            shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass

        if jobs:
            with ThreadPoolExecutor(max_workers=1) as ex:
                futures = [ex.submit(_process_job, job) for job in jobs]
                for _ in as_completed(futures):
                    pass

    def run(self):
        """Run the periodic polling loop."""
        while True:
            now = time.time()
            if now < self.next_time:
                time.sleep(self.next_time - now)

            start_time = time.time()
            self.task_main()
            end_time = time.time()

            self.next_time += self.interval

            # If a cycle took longer than the schedule, push the next tick
            if end_time - start_time > self.next_time:
                self.next_time = end_time + self.interval


if __name__ == "__main__":
    runner = MLInferenceRunner()
    runner.run()
