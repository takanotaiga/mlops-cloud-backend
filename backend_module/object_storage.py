import posixpath
import mimetypes
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend_module.uuid_tools import get_uuid
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError, BotoCoreError

from enum import IntEnum, auto
class S3Info(IntEnum):
    SUCCESS = auto()
    FILE_NOT_FOUND = auto()
    UPLOAD_FAILED = auto()
    DOWNLOAD_FAILED = auto()
    OBJECT_NOT_FOUND = auto()

@dataclass
class UploadResult:
    local_path: str
    key: Optional[str]
    status: S3Info
    error: Optional[str] = None


@dataclass
class DownloadResult:
    key: str
    local_path: Optional[str]
    status: S3Info
    error: Optional[str] = None

class MinioS3Uploader:
    """
    - __init__ でクライアント作成＆バケット存在確認
    - 300MB超は自動マルチパート（TransferConfig）
    - 複数ファイルはスレッドで並列アップロード
    """
    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        *,
        region_name: str = "us-east-1",
        ensure_bucket: bool = True,
        multipart_threshold_bytes: int = 300 * 1024 * 1024,  # 300MB
        multipart_chunksize_bytes: int = 64 * 1024 * 1024,   # 64MB
        part_concurrency: int = 4,                           # 各ファイル内のパート並列
        connect_timeout: int = 10,
        read_timeout: int = 300,
        max_attempts: int = 3,
        addressing_style: str = "path",  # MinIOはpathが安定
    ):
        self.bucket = bucket

        # --- ログイン（クライアント生成） ---
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
            config=BotoConfig(
                signature_version="s3v4",
                read_timeout=read_timeout,
                connect_timeout=connect_timeout,
                retries={"max_attempts": max_attempts, "mode": "standard"},
                s3={"addressing_style": addressing_style},
            ),
        )

        # --- バケット確認（必要なら作成） ---
        if ensure_bucket:
            try:
                self.s3.head_bucket(Bucket=self.bucket)
            except Exception:
                self.s3.create_bucket(Bucket=self.bucket)

        # --- 転送設定（300MB超→自動マルチパート） ---
        self.transfer_config = TransferConfig(
            multipart_threshold=multipart_threshold_bytes,
            multipart_chunksize=multipart_chunksize_bytes,
            max_concurrency=part_concurrency,
            use_threads=True,
        )

    # --------- public API ---------

    def upload_file(self, local_path: str, key_prefix: str = "") -> UploadResult:
        """
        単一ファイルをアップロード。成功時は S3キーを返す。
        """
        p = Path(local_path)
        if not p.exists():
            return UploadResult(str(p), None, S3Info.FILE_NOT_FOUND, "File not found")

        # Place UUID after the filename (before extension) to keep lexicographic
        # ordering by base name stable across variant uploads.
        stem = p.stem
        suffix = p.suffix  # including dot, e.g., '.mp4'
        key_name = f"{stem}-{get_uuid(16)}{suffix}"
        key = posixpath.join(key_prefix.strip("/"), key_name) if key_prefix else key_name

        extra_args = {"ContentType": self._infer_content_type(str(p))}

        try:
            # boto3.transfer が閾値超でマルチパートに自動切替
            self.s3.upload_file(
                Filename=str(p),
                Bucket=self.bucket,
                Key=key,
                ExtraArgs=extra_args,
                Config=self.transfer_config,
            )
            return UploadResult(str(p), key, S3Info.SUCCESS)
        except (ClientError, BotoCoreError) as e:
            return UploadResult(str(p), None, S3Info.UPLOAD_FAILED, str(e))
        except Exception as e:
            return UploadResult(str(p), None, S3Info.UPLOAD_FAILED, str(e))

    def upload_file_as(self, local_path: str, key: str) -> UploadResult:
        """Upload a single file using the exact provided key (no UUID suffixing)."""
        p = Path(local_path)
        if not p.exists():
            return UploadResult(str(p), None, S3Info.FILE_NOT_FOUND, "File not found")

        extra_args = {"ContentType": self._infer_content_type(str(p))}
        try:
            self.s3.upload_file(
                Filename=str(p),
                Bucket=self.bucket,
                Key=key,
                ExtraArgs=extra_args,
                Config=self.transfer_config,
            )
            return UploadResult(str(p), key, S3Info.SUCCESS)
        except (ClientError, BotoCoreError) as e:
            return UploadResult(str(p), None, S3Info.UPLOAD_FAILED, str(e))
        except Exception as e:
            return UploadResult(str(p), None, S3Info.UPLOAD_FAILED, str(e))

    def upload_files(
        self,
        files: Iterable[str],
        key_prefix: str = "",
        *,
        max_workers: int = 4,  # 複数ファイルを同時に上げる並列度
    ) -> List[UploadResult]:
        """
        複数ファイルを並列アップロード。各要素の UploadResult を返す。
        """
        results: List[UploadResult] = []
        paths = list(files)

        # boto3 の client は基本スレッドセーフ
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(self.upload_file, str(p), key_prefix): str(p) for p in paths}
            for fut in as_completed(fut_map):
                try:
                    res = fut.result()
                except Exception as e:
                    res = UploadResult(fut_map[fut], None, S3Info.UPLOAD_FAILED, str(e))
                results.append(res)
        return results

    # --------- download API ---------

    def download_file(
        self,
        key: str,
        local_path: str,
        *,
        overwrite: bool = True,
    ) -> DownloadResult:
        """
        指定キーをローカルへダウンロード（大容量対応）。

        - boto3 の TransferManager を利用し、閾値超は自動マルチパート並列
        - 既定では上書き（overwrite=True）。False の場合は既存ファイルで失敗
        """
        p = Path(local_path)
        if p.exists() and not overwrite:
            return DownloadResult(key, str(p), S3Info.DOWNLOAD_FAILED, "Local file exists")

        # 保存先ディレクトリを用意
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)

        # 事前に存在確認（NoSuchKey の早期検出）
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
        except (ClientError, BotoCoreError) as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            if code in {"404", "NoSuchKey", "NotFound"}:
                return DownloadResult(key, None, S3Info.OBJECT_NOT_FOUND, str(e))
            # head 失敗時も後続 download で再試行される可能性があるため続行せずエラー返却
            return DownloadResult(key, None, S3Info.DOWNLOAD_FAILED, str(e))

        try:
            self.s3.download_file(
                Bucket=self.bucket,
                Key=key,
                Filename=str(p),
                Config=self.transfer_config,
            )
            return DownloadResult(key, str(p), S3Info.SUCCESS)
        except (ClientError, BotoCoreError) as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            if code in {"404", "NoSuchKey", "NotFound"}:
                return DownloadResult(key, None, S3Info.OBJECT_NOT_FOUND, str(e))
            return DownloadResult(key, str(p), S3Info.DOWNLOAD_FAILED, str(e))
        except Exception as e:
            return DownloadResult(key, str(p), S3Info.DOWNLOAD_FAILED, str(e))

    def download_files(
        self,
        keys: Iterable[str],
        dest_dir: str,
        *,
        max_workers: int = 4,
        keep_key_paths: bool = False,
    ) -> List[DownloadResult]:
        """
        複数キーを並列ダウンロード。

        - keep_key_paths=True で S3 のキー階層をそのまま配下に再現
        - False の場合はファイル名のみを dest_dir 直下に保存
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        def _target_path(k: str) -> str:
            if keep_key_paths:
                return str(dest.joinpath(k))
            # キー末尾名のみ
            name = k.rstrip("/").split("/")[-1]
            return str(dest.joinpath(name))

        results: List[DownloadResult] = []
        ks = list(keys)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(self.download_file, k, _target_path(k)): k for k in ks}
            for fut in as_completed(fut_map):
                try:
                    res = fut.result()
                except Exception as e:
                    res = DownloadResult(fut_map[fut], None, S3Info.DOWNLOAD_FAILED, str(e))
                results.append(res)
        return results

    def stream_object(self, key: str, chunk_size: int = 8 * 1024 * 1024):
        """
        S3オブジェクトをチャンク単位でストリーミングするジェネレータ。
        大容量データを逐次処理する用途向け。
        """
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        except (ClientError, BotoCoreError) as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            if code in {"404", "NoSuchKey", "NotFound"}:
                return
            raise
        body = resp["Body"]
        for chunk in body.iter_chunks(chunk_size):
            if chunk:
                yield chunk

    # --------- helper ---------

    @staticmethod
    def _infer_content_type(path: str) -> str:
        ctype, _ = mimetypes.guess_type(path)
        return ctype or "application/octet-stream"
