import os
from typing import Optional, Dict, Any


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(val)
    except Exception:
        return default


def load_surreal_config() -> Dict[str, Any]:
    """Load SurrealDB connection settings from env.

    Supports compose-style names and legacy fallbacks:
      - Preferred: SURREAL_URL, SURREAL_NS, SURREAL_DB, SURREAL_USER, SURREAL_PASS
      - Legacy:   SURREAL_ENDPOINT, SURREAL_NAMESPACE, SURREAL_DATABASE, SURREAL_USERNAME, SURREAL_PASSWORD
    """
    endpoint = _get_env("SURREAL_URL", _get_env("SURREAL_ENDPOINT", "ws://localhost:8000/rpc"))
    namespace = _get_env("SURREAL_NS", _get_env("SURREAL_NAMESPACE", "mlops"))
    database = _get_env("SURREAL_DB", _get_env("SURREAL_DATABASE", "cloud_ui"))
    username = _get_env("SURREAL_USER", _get_env("SURREAL_USERNAME", "root"))
    password = _get_env("SURREAL_PASS", _get_env("SURREAL_PASSWORD", "root"))
    return {
        "endpoint_url": endpoint,
        "namespace": namespace,
        "database": database,
        "username": username,
        "password": password,
    }


def load_s3_config() -> Dict[str, Any]:
    """Load MinIO/S3 settings from env.

    Preferred envs (compose example):
      - MINIO_ENDPOINT_INTERNAL, MINIO_REGION, MINIO_ACCESS_KEY_ID, MINIO_SECRET_ACCESS_KEY,
        MINIO_BUCKET, MINIO_FORCE_PATH_STYLE
      - S3_MULTIPART_THRESHOLD_BYTES (optional)

    Legacy fallbacks:
      - S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET
    """
    endpoint = _get_env("MINIO_ENDPOINT_INTERNAL", _get_env("S3_ENDPOINT", "http://localhost:9000"))
    region = _get_env("MINIO_REGION", "us-east-1")
    access_key = _get_env("MINIO_ACCESS_KEY_ID", _get_env("S3_ACCESS_KEY", "minioadmin"))
    secret_key = _get_env("MINIO_SECRET_ACCESS_KEY", _get_env("S3_SECRET_KEY", "minioadmin"))
    bucket = _get_env("MINIO_BUCKET", _get_env("S3_BUCKET", "mlops-datasets"))
    path_style = _get_bool("MINIO_FORCE_PATH_STYLE", True)

    # Transfer tuning (optional)
    multipart_threshold = _get_int("S3_MULTIPART_THRESHOLD_BYTES", 300 * 1024 * 1024)
    multipart_chunksize = _get_int("S3_MULTIPART_CHUNKSIZE_BYTES", 64 * 1024 * 1024)
    part_concurrency = _get_int("S3_TRANSFER_CONCURRENCY", 4)

    return {
        "endpoint_url": endpoint,
        "region_name": region,
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": bucket,
        "addressing_style": "path" if path_style else "virtual",
        "multipart_threshold_bytes": multipart_threshold,
        "multipart_chunksize_bytes": multipart_chunksize,
        "part_concurrency": part_concurrency,
    }

