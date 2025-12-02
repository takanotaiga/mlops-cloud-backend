# Repository Guidelines

## Project Structure & Module Organization
- `backend_module/`: Core modules (e.g., `object_storage.py` for S3/MinIO I/O, `database.py`, `uuid_tools.py`).
- `query/`: Query-related helpers and scripts.
- `docker/`: Container assets (e.g., `Dockerfile.cv`).
- Root scripts: `video_manager.py`, `test.py` for ad-hoc runs.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`.
- Install deps (example): `pip install -U pip boto3`.
- Run local script: `python test.py` (adjust as needed).
- Lint/format (if installed): `ruff .` and `black .`.
- Docker build (example): `docker build -f docker/Dockerfile.cv -t mlops-cv .`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, type hints where practical.
- Modules and functions: `snake_case`; classes: `PascalCase`.
- Keep functions cohesive; prefer small, testable units.
- Add docstrings for public methods (e.g., upload/download APIs in `object_storage.py`).

## Testing Guidelines
- Prefer `pytest`; name tests `test_*.py`.
- Place tests next to modules or under `tests/` mirroring package paths.
- Run: `pytest -q` (if added). For quick checks: `python test.py`.
- Aim for coverage on error paths (network failures, missing keys).

## Commit & Pull Request Guidelines
- Commits: clear, imperative subject (e.g., "Add streaming S3 downloader").
- Reference issues in the body (e.g., `Fixes #123`).
- PRs: include purpose, screenshots/logs if relevant, and steps to verify.
- Keep diffs focused; update docs when behavior changes.

## Security & Configuration Tips
- Do not hardcode secrets. Use environment variables for S3/MinIO:
  `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`.
- Use least-privilege credentials for buckets.
- Large transfers: tune concurrency and thresholds in `MinioS3Uploader` as needed.

## Architecture Overview
- `backend_module/object_storage.py` centralizes S3/MinIO uploads and downloads.
- Uses `boto3` TransferManager for multi-part, concurrent transfers to handle very large files efficiently.

## Code Walkthrough

### Runtime Orchestrator
- `video_manager.py`:
  - Polls SurrealDB for queued HLS jobs every `interval` seconds via `TaskRunner`.
  - For each job: download the source file from S3/MinIO, encode to HLS (playlist + init + fMP4 segments), upload assets, register playlist/segment metadata, then update job status.
  - Runs jobs with GPU/CPU executors (1 each) and cleans up per-job work directories under `work/` regardless of success/failure.

### Storage Layer
- `backend_module/object_storage.py`:
  - `MinioS3Uploader`: thin wrapper around `boto3.client('s3')` configured for MinIO-compatible endpoints.
  - Uploads: `upload_file()` and `upload_files()` with automatic multipart based on `TransferConfig` thresholds. Returns `UploadResult` with status in `S3Info`.
  - Downloads: `download_file()` and `download_files()`; pre-checks existence with `head_object`; supports keeping S3 key path structure. Returns `DownloadResult` with `S3Info`.
  - Content-Type inference via `mimetypes`; stable path-style addressing for MinIO.

### Database Layer
- `backend_module/database.py`:
  - `DataBaseManager`: thread-safe SurrealDB client wrapper using a lock around `query()` to guard multi-threaded access from the orchestrator.
  - Retries connection for up to ~5 seconds before raising.

### Encoding Layer
- `backend_module/encoder.py`:
  - `encode_to_hls(input_path, out_dir=None)`: produces HLS VOD (fMP4) with playlist + init + `seg_XXXXX.m4s`, NVENC-first with libx265 fallback.
  - `probe_video(path)`: `ffprobe` wrapper to extract duration, width/height, frame rate, and codec.

### Query Helpers
- `query/file_query.py`:
  - `get_file(...)`, `get_s3key(...)`: fetch file record or its S3 key; raise `FileRecordNotFound` if missing.
- `query/encoded_segment_query.py`:
  - `insert_encoded_segment(...)`: inserts one encoded segment record with file linkage, size, bucket, and rich `meta` (duration, index, time ranges, dimensions, fps, codec).
- `query/hls_job_query.py`:
  - `queue_unhls_video_jobs(...)`: enqueues HLS jobs for videos/results missing HLS.
  - `set_hls_job_status(...)`: guards state transitions (`queued -> in_progress -> complete`, `faild` always allowed).
- `query/utils.py`:
  - Response helpers `first_result(...)`, `extract_results(...)`, and Surreal record id utility `rid_leaf(...)`.

### UUID Utilities
- `backend_module/uuid_tools.py`: `get_uuid(length)` returns a lowercase alphanumeric UUID string cropped to the requested length.

### Ad-hoc Script
- `test.py`: small, local example to download/upload using `MinioS3Uploader` with hardcoded endpoints and credentials.

## End-to-End Job Flow
- Discover work: enqueue new `hls_job` for eligible `file`/`inference_result` records.
- Fetch queued HLS jobs and mark each `in_progress` before processing.
- Download the source object from S3/MinIO to a per-job `work/hls_<job_id>/` directory.
- Encode to HLS (playlist + init + fMP4 segments), upload under `hls/<file_id>/...`.
- Insert `hls_playlist`/`hls_segment` rows with timing and video metadata.
- Mark job `complete` on success; `faild` on any error. Always remove local `work/hls_<job_id>/`.

## Configuration
- Object storage: use environment variables and pass to `MinioS3Uploader`:
  - `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`.
- Database (SurrealDB): set and pass via env:
  - `SURREAL_ENDPOINT`, `SURREAL_USERNAME`, `SURREAL_PASSWORD`, `SURREAL_NAMESPACE`, `SURREAL_DATABASE`.
- Note: current `video_manager.py` and `test.py` contain hardcoded endpoints/credentials for local development. Replace with environment-driven configuration before deployment.

## Operational Notes
- Requirements: `ffmpeg` and `ffprobe` must be installed and available on PATH; NVENC is optional—CPU fallback is automatic.
- Concurrency:
  - Upload/download concurrency uses `TransferConfig.max_concurrency` (per-file parts) and per-call thread pools for multi-file parallelism.
  - Orchestrator runs up to two jobs in parallel; adjust `max_workers` as needed with DB contention in mind.
- Error handling:
  - S3 operations return rich status (`S3Info`) and error messages; callers must check.
  - Encode errors set job state to `faild` and proceed with cleanup.
- File system: job-scoped work areas under `work/` are created and cleaned per execution.

## How To Run
- Local ad-hoc: `python test.py` to try a simple upload/download.
- Orchestrator: `python video_manager.py` to start the polling worker.
- Docker (example): `docker build -f docker/Dockerfile.cv -t mlops-cv .`.

## Suggestions / TODOs
- Replace hardcoded configuration in `video_manager.py` and `test.py` with environment variables and a small config loader.
- Add minimal pytest coverage for:
  - `hls_job` state transitions (valid/invalid, concurrent update guard).
  - `object_storage` error paths (missing keys, timeouts mocked).
  - `probe_video` parsing with fixture JSON.
- Consider structured logging and per-job correlation IDs for observability.
