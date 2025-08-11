import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader


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
        surreal_namespace = _get_env("SURREAL_NAMESPACE", "test")
        surreal_database = _get_env("SURREAL_DATABASE", "test")

        # S3/MinIO configuration (env-driven with local-dev defaults)
        s3_endpoint = _get_env("S3_ENDPOINT", "http://192.168.1.25:65300")
        s3_access_key = _get_env("S3_ACCESS_KEY", "minioadmin")
        s3_secret_key = _get_env("S3_SECRET_KEY", "minioadmin")
        s3_bucket = _get_env("S3_BUCKET", "horus-bucket")

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

        # Example skeleton (commented out; adapt to your schema):
        # jobs = get_queued_inference_jobs(self.db_manager)
        jobs: List[dict] = []  # placeholder until queries are defined

        def _process_job(job: dict):
            job_id = job.get("id")
            file_id = job.get("file")
            if not job_id or not file_id:
                return

            # Prepare a per-job working directory
            work_dir = Path("work_infer") / str(job_id).split(":")[-1]
            try:
                # set_inference_job_status(self.db_manager, job_id, "in_progress")

                # Placeholder: download input(s) using self.uploader
                # key = get_input_key(self.db_manager, file_id)
                # local_src = work_dir / "input.bin"  # or appropriate filename
                # self.uploader.download_file(key, str(local_src))

                # Placeholder: run ML inference here
                # results = run_inference(local_src)

                # Placeholder: upload results and write DB metadata
                # up = self.uploader.upload_file(str(result_path), key_prefix=f"inference/{file_id}")
                # insert_inference_result(self.db_manager, file_id=file_id, key=up.key, bucket=self.uploader.bucket, meta={...})

                # set_inference_job_status(self.db_manager, job_id, "complete")
                pass
            except Exception as e:
                # set_inference_job_status(self.db_manager, job_id, "faild")
                print(f"Inference job {job_id} failed: {e}")
            finally:
                try:
                    if work_dir.exists():
                        # Best-effort cleanup
                        import shutil
                        shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass

        if jobs:
            with ThreadPoolExecutor(max_workers=2) as ex:
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

