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
