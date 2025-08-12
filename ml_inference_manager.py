import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from query import ml_inference_job_query
from query.encoded_segment_query import get_segments_with_file_by_keys
from query.utils import extract_results, rid_leaf
from query.inference_result_query import insert_inference_result
from ml_module.registry import get_model
from backend_module.encoder import transcode_video
import mimetypes
from collections import defaultdict
import re

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

        # SurrealDB configuration (env-driven with local-dev defaults)
        surreal_endpoint = _get_env("SURREAL_ENDPOINT", "ws://192.168.1.25:65303/rpc")
        surreal_username = _get_env("SURREAL_USERNAME", "root")
        surreal_password = _get_env("SURREAL_PASSWORD", "root")
        surreal_namespace = _get_env("SURREAL_NAMESPACE", "mlops")
        surreal_database = _get_env("SURREAL_DATABASE", "cloud_ui")

        # S3/MinIO configuration (env-driven with local-dev defaults)
        s3_endpoint = _get_env("S3_ENDPOINT", "http://192.168.1.25:65300")
        s3_access_key = _get_env("S3_ACCESS_KEY", "minioadmin")
        s3_secret_key = _get_env("S3_SECRET_KEY", "minioadmin")
        s3_bucket = _get_env("S3_BUCKET", "mlops-datasets")

        # Initialize database manager (thread-safe query)
        self.db_manager = DataBaseManager(
            endpoint_url=surreal_endpoint,
            username=surreal_username,
            password=surreal_password,
            namespace=surreal_namespace,
            database=surreal_database,
        )

        # Initialize object storage client with multipart settings similar to video_manager
        self.uploader = MinioS3Uploader(
            endpoint_url=s3_endpoint,
            access_key=s3_access_key,
            secret_key=s3_secret_key,
            bucket=s3_bucket,
            multipart_threshold_bytes=300 * 1024 * 1024,  # 300MB
            multipart_chunksize_bytes=64 * 1024 * 1024,   # 64MB
            part_concurrency=4,
        )

        self.next_time = time.time()
        # Keep per-job work directories for debugging when true-ish
        self.keep_work_dir = _get_env("KEEP_WORK_DIR", "0").lower() in ("1", "true", "yes", "on")

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
            linked_file = ml_inference_job_query.get_linked_file(self.db_manager, job_id)
            # Select model adapter based on job
            if not job_id or not task_type or len(linked_file) == 0:
                return
            model_adapter = get_model(model, model_source, task_type)
            if model_adapter is None:
                return

            # Update status to running
            try:
                ml_inference_job_query.set_inference_job_status(self.db_manager, job_id, "ProcessRunning")
            except Exception:
                pass

            # Prepare a per-job working directory
            work_dir = Path("work_infer") / str(job_id).split(":")[-1]
            try:
                download_result = self.uploader.download_files(
                    keys=linked_file,
                    dest_dir=work_dir
                )
                # Build key -> local_path mapping for successful downloads
                key_to_local = {r.key: r.local_path for r in download_result if r.status == S3Info.SUCCESS and r.local_path}

                # Determine file types by local path; collect video keys only
                def _is_video(path: str) -> bool:
                    mtype, _ = mimetypes.guess_type(path)
                    return (mtype or "").startswith("video/")

                video_keys = [k for k, lp in key_to_local.items() if _is_video(lp)]
                if not video_keys:
                    print("No video files in this batch; nothing to group.")
                    return

                # Fetch encoded_segment rows with joined file info for these keys
                seg_payload = get_segments_with_file_by_keys(self.db_manager, video_keys)
                seg_rows = extract_results(seg_payload)

                # Aggregate segments by original file id
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
                    file_segments[file_rid].append({
                        "key": key,
                        "local_path": key_to_local[key],
                        "index": (meta.get("index") if isinstance(meta, dict) else None)
                    })

                    print(meta.get("index") if isinstance(meta, dict) else None)
                    print(meta.get("index"))

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

                # Fetch merge groups for this job
                mg_rows = ml_inference_job_query.get_merge_groups(self.db_manager, job_id)
                merge_groups = mg_rows  # already a list of dicts

                # Build dataset->name->file_id mapping
                files_by_dataset_name: dict[str, dict[str, str]] = defaultdict(dict)
                for fid, name in file_name_by_id.items():
                    dataset = file_dataset_by_id.get(fid)
                    if dataset and name:
                        files_by_dataset_name[dataset][name] = fid

                used_files: set[str] = set()
                grouped: list[list[list[object]]] = []

                # Compose groups for merge definitions
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

                # Add independent videos (not in any merge group)
                for fid, segs in file_segments.items():
                    if fid in used_files:
                        continue
                    group_list = [[seg["local_path"], i] for i, seg in enumerate(segs, start=1)]
                    if group_list:
                        grouped.append(group_list)

                # Process each group via the selected model adapter
                for gi, group in enumerate(grouped, start=1):
                    # Determine member file ids in order of appearance
                    file_ids_in_group: List[str] = []
                    for p, _seq in group:
                        # find file id owning this segment
                        for fid, segs in file_segments.items():
                            if any(s.get("local_path") == p for s in segs):
                                if fid not in file_ids_in_group:
                                    file_ids_in_group.append(fid)
                                break
                    
                    work_gdir = work_dir / f"g_{gi:03d}"
                    work_gdir.mkdir(parents=True, exist_ok=True)
                    # Build model input group
                    model_group = []
                    for fid in file_ids_in_group:
                        seg_paths = [s["local_path"] for s in file_segments.get(fid, [])]
                        if not seg_paths:
                            continue
                        model_group.append({
                            "file_id": fid,
                            "dataset": file_dataset_by_id.get(fid),
                            "name": file_name_by_id.get(fid),
                            "segments": seg_paths,
                        })
                    if not model_group:
                        continue

                    model_res = model_adapter.process_group(self.db_manager, model_group, str(work_gdir))
                    final_path = model_res.get("output_path")
                    labels = model_res.get("labels", []) or []
                    schema_json_path = model_res.get("schema_json_path")
                    results_artifacts = model_res.get("results_artifacts", []) or []
                    if not final_path:
                        continue

                    # Optionally compress/transcode the final result to reduce size
                    enc_final_path = str(work_gdir / "group_tracked_enc.mp4")
                    try:
                        enc_final_path = transcode_video(final_path, output_path=enc_final_path)
                    except Exception as _e:
                        # If transcode fails, fall back to the original
                        enc_final_path = final_path

                    # Upload result video
                    dataset = None
                    if file_ids_in_group:
                        dataset = file_dataset_by_id.get(file_ids_in_group[0])
                    result_key = f"inference/{rid_leaf(job_id)}/group_{gi:03d}.mp4"
                    up = self.uploader.upload_file_as(enc_final_path, result_key)
                    if up.status != S3Info.SUCCESS:
                        raise RuntimeError(f"Upload inference result failed: {up.error}")

                    # Register inference_result row
                    size = Path(enc_final_path).stat().st_size
                    insert_inference_result(
                        self.db_manager,
                        job_id=job_id,
                        dataset=dataset,
                        files=file_ids_in_group,
                        key=result_key,
                        bucket=self.uploader.bucket,
                        size=size,
                        labels=sorted(set(labels)),
                        meta={
                            "groupIndex": gi,
                            "sourceFiles": file_ids_in_group,
                            "artifact": "plot_video",
                        },
                    )

                    # Upload schema JSON alongside the plot video and register to DB
                    try:
                        if schema_json_path and os.path.exists(schema_json_path):
                            schema_key = f"inference/{rid_leaf(job_id)}/group_{gi:03d}_schema.json"
                            up_js = self.uploader.upload_file_as(schema_json_path, schema_key)
                            if up_js.status != S3Info.SUCCESS:
                                raise RuntimeError(f"Upload schema JSON failed: {up_js.error}")
                            size_js = Path(schema_json_path).stat().st_size
                            insert_inference_result(
                                self.db_manager,
                                job_id=job_id,
                                dataset=dataset,
                                files=file_ids_in_group,
                                key=schema_key,
                                bucket=self.uploader.bucket,
                                size=size_js,
                                labels=[],
                                meta={
                                    "groupIndex": gi,
                                    "sourceFiles": file_ids_in_group,
                                    "artifact": "schema_json",
                                    "contentType": "application/json",
                                },
                            )
                    except Exception as _e:
                        # Fail schema upload softly without failing the whole job
                        print(f"Schema JSON upload/insert skipped due to error: {_e}")

                    # Upload per-file results artifacts (Parquet only)
                    for art in results_artifacts:
                        try:
                            fid = str(art.get("file_id"))
                            file_leaf = rid_leaf(fid)
                            # Parquet
                            pq_path = art.get("parquet")
                            if pq_path and os.path.exists(pq_path):
                                pq_key = f"inference/{rid_leaf(job_id)}/group_{gi:03d}/{file_leaf}_results.parquet"
                                up_pq = self.uploader.upload_file_as(pq_path, pq_key)
                                if up_pq.status == S3Info.SUCCESS:
                                    size_pq = Path(pq_path).stat().st_size
                                    insert_inference_result(
                                        self.db_manager,
                                        job_id=job_id,
                                        dataset=dataset,
                                        files=[fid],
                                        key=pq_key,
                                        bucket=self.uploader.bucket,
                                        size=size_pq,
                                        labels=[],
                                        meta={
                                            "groupIndex": gi,
                                            "sourceFiles": [fid],
                                            "artifact": "results_parquet",
                                            "contentType": "application/x-parquet",
                                        },
                                    )
                                else:
                                    print(f"Parquet upload failed for {pq_path}: {up_pq.error}")
                        except Exception as _e:
                            print(f"Artifact upload/insert skipped due to error: {_e}")

                # All groups done -> mark job completed
                ml_inference_job_query.set_inference_job_status(self.db_manager, job_id, "Completed")
            except Exception as e:
                try:
                    ml_inference_job_query.set_inference_job_status(self.db_manager, job_id, "Faild")
                except Exception:
                    pass
                print(f"Inference job {job_id} failed: {e}")
            finally:
                try:
                    if work_dir.exists() and not self.keep_work_dir:
                        # Best-effort cleanup
                        import shutil
                        shutil.rmtree(work_dir, ignore_errors=True)
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
