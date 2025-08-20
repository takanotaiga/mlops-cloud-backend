import time
from pathlib import Path
import shutil
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.config import load_surreal_config, load_s3_config
from backend_module.encoder import encode_to_segments, probe_video, create_thumbnail
from backend_module import gpu_check
from query import encode_job_query, file_query
from query.encoded_segment_query import insert_encoded_segment
from query.utils import rid_leaf
 

class TaskRunner:
    def __init__(self, interval=5):
        self.interval = interval
        # SurrealDB from environment (compose-friendly)
        sconf = load_surreal_config()
        self.db_manager = DataBaseManager(
            endpoint_url=sconf["endpoint_url"],
            username=sconf["username"],
            password=sconf["password"],
            namespace=sconf["namespace"],
            database=sconf["database"],
        )

        # MinIO/S3 from environment (compose-friendly)
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

    def task_main(self):
        # 1) サムネイル作成（優先処理）
        if self._process_missing_thumbnails():
            # サムネイルを処理した場合は、このサイクルではここで終了（優先度を担保）
            return

        # 新規ジョブをエンキュー
        encode_job_query.queue_unencoded_video_jobs(self.db_manager)

        # キュー取得（SurrealDBの応答形式を吸収）
        jobs = encode_job_query.get_queued_job(self.db_manager)

        def _process_job(job: dict):
            job_id = job.get("id")
            file_id = job.get("file")
            if not job_id or not file_id:
                return

            work_dir = Path("work") / rid_leaf(job_id)
            try:
                # ステータスを in_progress へ
                encode_job_query.set_encode_job_status(self.db_manager, job_id, "in_progress")

                # S3キー取得
                s3_key = file_query.get_s3key(self.db_manager, file_id)

                # 作業ディレクトリ準備
                work_dir.mkdir(parents=True, exist_ok=True)
                filename = s3_key.rstrip("/").split("/")[-1]
                local_src = work_dir / filename

                # ダウンロード
                dl_res = self.uploader.download_file(s3_key, str(local_src))
                if dl_res.status != S3Info.SUCCESS:
                    raise RuntimeError(f"Download failed: {dl_res.error}")

                # エンコード（out/<uuid>/ に出力される）
                outputs: List[str] = encode_to_segments(str(local_src), out_dir=str(work_dir / "encoded"))

                # アップロード（encoded/<file_id>/ 配下に格納）
                upload_results = self.uploader.upload_files(outputs, key_prefix=f"encoded/{rid_leaf(file_id)}")
                if any(r.status != S3Info.SUCCESS for r in upload_results):
                    errs = ", ".join(f"{r.local_path}:{r.error}" for r in upload_results if r.status != S3Info.SUCCESS)
                    raise RuntimeError(f"Upload failed: {errs}")

                # DB 登録（各セグメントのメタ情報付与）
                seg_infos = [probe_video(p) for p in outputs]
                total = len(outputs)
                cumulative = 0.0
                # アップロード結果を local_path -> key で引けるようにする
                path_to_key = {r.local_path: r.key for r in upload_results if r.status == S3Info.SUCCESS and r.key}
                for idx, (local_path, info) in enumerate(zip(outputs, seg_infos), start=1):
                    up_key = path_to_key.get(local_path)
                    if not up_key:
                        raise RuntimeError(f"Uploaded key missing for {local_path}")
                    size = Path(local_path).stat().st_size
                    start_sec = cumulative
                    end_sec = cumulative + (info.get("durationSec") or 0.0)
                    cumulative = end_sec
                    meta = {
                        "durationSec": info.get("durationSec"),
                        "index": idx,  # 1-based, outputs の順序に一致
                        "total": total,
                        "startSec": start_sec,
                        "endSec": end_sec,
                        "startMin": start_sec / 60.0,
                        "endMin": end_sec / 60.0,
                        "width": info.get("width"),
                        "height": info.get("height"),
                        "nb_frames": info.get("nb_frames"),
                        "avg_frame_rate": info.get("avg_frame_rate"),
                        "codec_name": info.get("codec_name"),
                    }
                    insert_encoded_segment(
                        self.db_manager,
                        file_id=file_id,
                        key=up_key,
                        size=size,
                        bucket=self.uploader.bucket,
                        meta=meta,
                    )

                # ここまで完了したら complete へ
                encode_job_query.set_encode_job_status(self.db_manager, job_id, "complete")

            except Exception as e:
                # いずれの段階でも失敗時は faild へ
                try:
                    encode_job_query.set_encode_job_status(self.db_manager, job_id, "faild")
                except Exception:
                    pass
                # ログ代わりに出力
                print(f"Job {job_id} failed: {e}")
            finally:
                # 成功・失敗を問わずローカル作業ディレクトリを削除
                try:
                    if work_dir.exists():
                        shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass

        # 並列実行（最大2）
        if jobs:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = [ex.submit(_process_job, job) for job in jobs]
                for _ in as_completed(futures):
                    pass

    def _process_missing_thumbnails(self) -> bool:
        """Create and register thumbnails for videos missing thumbKey.

        Returns True if any thumbnails were processed (to prioritize over encoding).
        """
        rows = file_query.list_videos_missing_thumbs(self.db_manager, limit=16)
        if not rows:
            return False

        def _thumb_time(duration: float | None) -> float:
            # pick a safe time within the video
            if not duration or duration <= 0:
                return 1.0
            return max(0.5, min(5.0, duration * 0.1))

        for row in rows:
            file_id = row.get("id")
            s3_key = row.get("key")
            dataset = row.get("dataset")
            name = row.get("name")
            if not file_id or not s3_key or not dataset or not name:
                continue

            work_dir = Path("work_thumb") / rid_leaf(file_id)
            try:
                work_dir.mkdir(parents=True, exist_ok=True)
                local_src = work_dir / s3_key.rstrip("/").split("/")[-1]

                dl = self.uploader.download_file(s3_key, str(local_src))
                if dl.status != S3Info.SUCCESS:
                    raise RuntimeError(f"Download failed: {dl.error}")

                info = probe_video(str(local_src))
                ts = _thumb_time(info.get("durationSec"))
                thumb_local = work_dir / f"{name}.jpg"

                # ffmpeg呼び出しは encoder.py に委譲
                create_thumbnail(str(local_src), str(thumb_local), timestamp_sec=ts, width=640, quality=2)

                # Upload with deterministic key: <dataset>/.thumbs/<name>.jpg
                thumb_key = f"{dataset}/.thumbs/{name}.jpg"
                up = self.uploader.upload_file_as(str(thumb_local), thumb_key)
                if up.status != S3Info.SUCCESS:
                    raise RuntimeError(f"Upload thumb failed: {up.error}")

                # Register thumbKey on file
                file_query.set_thumb_key(self.db_manager, file_id, thumb_key)

            except Exception as e:
                print(f"Thumbnail generation failed for {file_id}: {e}")
            finally:
                try:
                    if work_dir.exists():
                        shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass

        return True

    def run(self):
        while True:
            now = time.time()
            if now < self.next_time:
                time.sleep(self.next_time - now)

            start_time = time.time()
            self.task_main()
            end_time = time.time()

            self.next_time += self.interval

            if end_time - start_time > self.next_time:
                self.next_time = end_time + self.interval


if __name__ == "__main__":
    if gpu_check.is_gpu_available():
        print("GPU detected ✅")
    else:
        print("No GPU detected ❌")
        gpu_check.exit_with_delay()

    runner = TaskRunner(interval=5)
    runner.run()
