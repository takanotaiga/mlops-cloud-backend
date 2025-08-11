import time
from pathlib import Path
from typing import List

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.encoder import encode_to_segments, probe_video
from query import encode_job_query, file_query
from query.encoded_segment_query import insert_encoded_segment
from query.utils import rid_leaf

class TaskRunner:
    def __init__(self, interval=5):
        self.interval = interval
        self.db_manager = DataBaseManager(
            endpoint_url="ws://192.168.1.25:65303/rpc",
            username="root",
            password="root",
            namespace="test",
            database="test"
        )
        self.uploader = MinioS3Uploader(
            endpoint_url="http://192.168.1.25:65300",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket="horus-bucket",
            multipart_threshold_bytes=300 * 1024 * 1024,  # 300MB
            multipart_chunksize_bytes=64 * 1024 * 1024,   # 128MB
            part_concurrency=4,
        )

        self.next_time = time.time()

    def task_main(self):
        # 新規ジョブをエンキュー
        encode_job_query.queue_unencoded_video_jobs(self.db_manager)

        # キュー取得（SurrealDBの応答形式を吸収）
        jobs = encode_job_query.get_queued_job(self.db_manager)

        for job in jobs:
            job_id = job.get("id")
            file_id = job.get("file")
            if not job_id or not file_id:
                continue

            try:
                # ステータスを in_progress へ
                encode_job_query.set_encode_job_status(self.db_manager, job_id, "in_progress")

                # S3キー取得
                s3_key = file_query.get_s3key(self.db_manager, file_id)

                # 作業ディレクトリ準備
                work_dir = Path("work") / rid_leaf(job_id)
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
                durations = [si.get("durationSec") or 0.0 for si in seg_infos]
                total = len(outputs)
                cumulative = 0.0
                for idx, (local_path, up_res, info) in enumerate(zip(outputs, upload_results, seg_infos)):
                    if up_res.key is None:
                        # 保険（ここまでに弾いているが念のため）
                        raise RuntimeError("Uploaded key missing")
                    size = Path(local_path).stat().st_size
                    start_sec = cumulative
                    end_sec = cumulative + (info.get("durationSec") or 0.0)
                    cumulative = end_sec
                    meta = {
                        "durationSec": info.get("durationSec"),
                        "index": idx + 1,  # 1-based
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
                        key=up_res.key,
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
    runner = TaskRunner(interval=5)
    runner.run()
