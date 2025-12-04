import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.config import load_surreal_config, load_s3_config
from backend_module.uuid_tools import get_uuid
from query import ml_inference_job_query
from query.utils import extract_results, first_result
from backend_module.progress_tracker import InferenceJobProgressTracker


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

    def _get_single_video_record(self, job_id: str) -> dict:
        """Validate job datasets and return the single video record."""
        ds_res = self.db_manager.query(
            "SELECT VALUE datasets FROM inference_job WHERE id = <record> $JOB_ID LIMIT 1;",
            {"JOB_ID": job_id},
        )
        datasets = first_result(ds_res) or []
        if not isinstance(datasets, list):
            raise ValueError("inference_job.datasets must be an array")
        datasets = [d for d in datasets if d]
        if not datasets:
            raise ValueError("inference_job must reference exactly one dataset (found none)")
        if len(datasets) > 1:
            raise ValueError("Only one dataset per inference_job is supported")
        dataset_id = str(datasets[0])

        video_rows = extract_results(
            self.db_manager.query(
                """
                SELECT id, key, name, dataset
                FROM file
                WHERE dataset = $DATASET AND mime ~ 'video/';
                """,
                {"DATASET": dataset_id},
            )
        )
        if not video_rows:
            raise ValueError("No video file found in dataset for inference job")
        if len(video_rows) > 1:
            raise ValueError("Multiple video files found; only one is supported")

        video_row = video_rows[0]
        key = video_row.get("key")
        if not key:
            raise ValueError("Target video file is missing object storage key")
        file_id = str(video_row.get("id"))
        file_name = video_row.get("name") or ""
        return {
            "dataset": dataset_id,
            "file_id": file_id,
            "file_name": file_name,
            "key": key,
        }

    def dataset_download(
        self,
        job_id: str,
        work_dir: Path,
        *,
        video_info: Optional[dict] = None,
    ) -> str:
        """
        Download the single MP4 video linked to the job and return its local path.
        """
        print("==== Start Dataset Download ====")
        info = video_info or self._get_single_video_record(job_id)
        key = info["key"]

        download_result = self.uploader.download_files(
            keys=[key],
            dest_dir=work_dir,
            keep_key_paths=False,
        )
        local_path: Optional[str] = None
        for r in download_result:
            if (
                r.key == key
                and getattr(r, "status", None) == S3Info.SUCCESS
                and getattr(r, "local_path", None)
            ):
                local_path = r.local_path
                break
        if not local_path:
            raise RuntimeError("Failed to download video file for inference")

        print("==== Complete ====")
        return local_path

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
            tracker: Optional[InferenceJobProgressTracker] = None
            temp_datasets_to_cleanup: set[str] = set()
            try:
                # Initialize progress tracker and default steps
                tracker = InferenceJobProgressTracker(self.db_manager, str(job_id))
                tracker.init_default_steps()

                # Discover target video and download it
                tracker.start("download")
                video_info = self._get_single_video_record(job_id)
                local_path = self.dataset_download(job_id=job_id, work_dir=work_dir, video_info=video_info)
                tracker.complete("download")

                file_group = {
                    "file_id": video_info["file_id"],
                    "dataset": video_info["dataset"],
                    "name": video_info["file_name"],
                    "segments": [local_path],
                }

                from ml_module import registry
                from ml_module.postprocess import postprocess_video_paths
                from ml_module.uploader import upload_group_results, GroupUploadItem

                # 1) Run inference for the single group and collect result paths
                g_work_dir = work_dir / "g_001"
                g_work_dir.mkdir(parents=True, exist_ok=True)
                res = registry.run_inference_task(
                    db_manager=self.db_manager,
                    job_id=str(job_id),
                    task_type=str(task_type),
                    model=model,
                    model_source=model_source,
                    file_group=[file_group],
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

                # 2) Postprocess all result videos (e.g., transcode)
                try:
                    tracker.start("postprocess")
                except Exception:
                    pass
                processed_paths = postprocess_video_paths([out_path] if out_path else [], str(work_dir / "post"))
                try:
                    tracker.complete("postprocess")
                except Exception:
                    pass

                # 3) Upload to S3 and register DB records via module
                video_path: Optional[str] = None
                if out_path:
                    video_path = processed_paths[0] if processed_paths else out_path

                item = GroupUploadItem(
                    index=1,
                    dataset=file_group["dataset"],
                    file_ids=[file_group["file_id"]],
                    video_path=video_path,
                    video_description=video_description,
                    schema_json_path=schema_json_path,
                    schema_description=schema_description,
                    group_parquet=group_parquet,
                    group_parquet_description=group_parquet_description,
                    results_artifacts=results_artifacts or [],
                )

                upload_group_results(
                    db_manager=self.db_manager,
                    uploader=self.uploader,
                    job_id=str(job_id),
                    items=[item],
                )
                # Upload step
                try:
                    tracker.start("upload")
                except Exception:
                    pass
                try:
                    tracker.complete("upload")
                except Exception:
                    pass

                # 4) Mark job completed
                ml_inference_job_query.set_inference_job_status(self.db_manager, job_id, "Completed")
                return


            except Exception as e:
                try:
                    ml_inference_job_query.set_inference_job_status(self.db_manager, job_id, "Faild")
                except Exception:
                    pass
                try:
                    # Best effort to mark the last started step as failed
                    tracker.fail("upload") if 'tracker' in locals() else None
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
