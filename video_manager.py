import time
import json
import math
import subprocess
from pathlib import Path
from typing import List

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.encoder import encode_to_segments
from query import encode_job_query, file_query

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
        raw = encode_job_query.get_queued_job(self.db_manager)
        jobs = self._extract_results(raw)

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
                work_dir = Path("work") / self._rid_leaf(job_id)
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
                upload_results = self.uploader.upload_files(outputs, key_prefix=f"encoded/{self._rid_leaf(file_id)}")
                if any(r.status != S3Info.SUCCESS for r in upload_results):
                    errs = ", ".join(f"{r.local_path}:{r.error}" for r in upload_results if r.status != S3Info.SUCCESS)
                    raise RuntimeError(f"Upload failed: {errs}")

                # DB 登録（各セグメントのメタ情報付与）
                seg_infos = [self._probe_video(p) for p in outputs]
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
                    self._insert_encoded_segment(
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

    @staticmethod
    def _extract_results(payload):
        """SurrealDBクエリ応答からレコード配列を抽出する。"""
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and "result" in first:
                return first.get("result") or []
            # 既にレコード配列が返っているケース
            return payload
        return []

    @staticmethod
    def _rid_leaf(rid) -> str:
        """Record ID をパス用に文字列化。'table:id' -> 'id' を返す。"""
        s = str(rid)
        return s.split(":", 1)[1] if ":" in s else s

    @staticmethod
    def _probe_video(path: str):
        """ffprobeで動画情報を取得し、必要メタを返す。"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = {}
        try:
            data = json.loads(proc.stdout or "{}")
        except Exception:
            data = {}

        # 代表ストリーム（video）を特定
        streams = data.get("streams") or []
        v = None
        for s in streams:
            if s.get("codec_type") == "video":
                v = s
                break
        fmt = data.get("format") or {}

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        duration = _to_float(fmt.get("duration"))
        if duration is None and v is not None:
            duration = _to_float(v.get("duration"))

        info = {
            "durationSec": duration,
            "width": v.get("width") if v else None,
            "height": v.get("height") if v else None,
            "nb_frames": None,
            "avg_frame_rate": None,
            "codec_name": v.get("codec_name") if v else None,
        }

        if v is not None:
            # フレーム数
            try:
                info["nb_frames"] = int(v.get("nb_frames")) if v.get("nb_frames") is not None else None
            except Exception:
                info["nb_frames"] = None
            # フレームレート
            afr = v.get("avg_frame_rate") or v.get("r_frame_rate")
            info["avg_frame_rate"] = afr

        return info

    def _insert_encoded_segment(self, *, file_id, key: str, size: int, bucket: str, meta: dict):
        """エンコード結果を DB に登録する。"""
        self.db_manager.query(
            """
            INSERT INTO encoded_segment {
                file: <record> $FILE,
                key: $KEY,
                type: 'video-segment',
                size: $SIZE,
                uploadedAt: time::now(),
                bucket: $BUCKET,
                meta: $META
            };
            """,
            {
                "FILE": file_id,
                "KEY": key,
                "SIZE": size,
                "BUCKET": bucket,
                "META": meta,
            },
        )


if __name__ == "__main__":
    runner = TaskRunner(interval=5)
    runner.run()
