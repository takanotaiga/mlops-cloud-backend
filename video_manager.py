import time
from pathlib import Path
import shutil
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.config import load_surreal_config, load_s3_config
from backend_module.encoder import encode_to_segments, probe_video, create_thumbnail, encode_to_hls
from query import encode_job_query, file_query
from query import inference_result_query
from query.encoded_segment_query import insert_encoded_segment
from query import hls_job_query
from query.hls_playlist_query import insert_hls_playlist
from query.hls_segment_query import insert_hls_segment
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

        # 新規ジョブをエンキュー（H264 MP4 / HLS）
        encode_job_query.queue_unencoded_video_jobs(self.db_manager)
        hls_job_query.queue_unhls_video_jobs(self.db_manager)

        # キュー取得（SurrealDBの応答形式を吸収）
        jobs = encode_job_query.get_queued_job(self.db_manager)
        hls_jobs = hls_job_query.get_queued_job(self.db_manager)

        def _process_job(job: dict, *, backend: str = "auto"):
            job_id = job.get("id")
            file_id = job.get("file")
            if not job_id or not file_id:
                return

            work_dir = Path("work") / rid_leaf(job_id)
            try:
                # ステータスを in_progress へ
                encode_job_query.set_encode_job_status(self.db_manager, job_id, "in_progress")

                # S3キー取得（file または inference_result に対応）
                sfile = str(file_id)
                if sfile.startswith("inference_result:"):
                    s3_key = inference_result_query.get_s3key(self.db_manager, file_id)
                else:
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
                outputs: List[str] = encode_to_segments(
                    str(local_src), out_dir=str(work_dir / "encoded"), backend=backend
                )

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

        # HLS ジョブ処理
        def _process_hls_job(job: dict, *, backend: str = "auto"):
            job_id = job.get("id")
            file_id = job.get("file")
            if not job_id or not file_id:
                return

            work_dir = Path("work") / ("hls_" + rid_leaf(job_id))
            try:
                # ステータスを in_progress へ
                hls_job_query.set_hls_job_status(self.db_manager, job_id, "in_progress")

                # S3キー取得（file または inference_result に対応）
                sfile = str(file_id)
                if sfile.startswith("inference_result:"):
                    s3_key = inference_result_query.get_s3key(self.db_manager, file_id)
                else:
                    s3_key = file_query.get_s3key(self.db_manager, file_id)

                # 作業ディレクトリ準備
                work_dir.mkdir(parents=True, exist_ok=True)
                filename = s3_key.rstrip("/").split("/")[-1]
                local_src = work_dir / filename

                # ダウンロード
                dl_res = self.uploader.download_file(s3_key, str(local_src))
                if dl_res.status != S3Info.SUCCESS:
                    raise RuntimeError(f"Download failed: {dl_res.error}")

                # HLS エンコード（out/<uuid>/hls に出力される）
                hls_out = encode_to_hls(
                    str(local_src), out_dir=str(work_dir / "encoded"), segment_time=6, backend=backend
                )
                playlist_path = hls_out["playlist"]
                seg_paths: List[str] = list(hls_out["segments"])  # includes init + .m4s
                out_root = Path(hls_out["out_dir"])

                # S3へアップロード（プレイリストとセグメントはファイル名を保持して上げる必要あり）
                key_prefix = f"hls/{rid_leaf(file_id)}"

                def _relkey(p: str) -> str:
                    return f"{key_prefix}/" + str(Path(p).relative_to(out_root)).replace("\\", "/")

                # まずプレイリスト
                playlist_key = f"{key_prefix}/index.m3u8"
                up_pl = self.uploader.upload_file_as(playlist_path, playlist_key)
                if up_pl.status != S3Info.SUCCESS:
                    raise RuntimeError(f"Upload playlist failed: {up_pl.error}")
                # Register playlist metadata
                try:
                    pl_size = Path(playlist_path).stat().st_size
                except Exception:
                    pl_size = 0
                insert_hls_playlist(
                    self.db_manager,
                    file_id=file_id,
                    key=playlist_key,
                    size=pl_size,
                    bucket=self.uploader.bucket,
                    meta={
                        "kind": "playlist",
                        "totalSegments": len([p for p in seg_paths if p.endswith('.m4s')]),
                    },
                )

                # セグメント群（init.mp4 と *.m4s）
                up_seg_results = []
                for p in seg_paths:
                    up = self.uploader.upload_file_as(p, _relkey(p))
                    up_seg_results.append(up)
                if any(r.status != S3Info.SUCCESS for r in up_seg_results):
                    errs = ", ".join(f"{r.local_path}:{r.error}" for r in up_seg_results if r.status != S3Info.SUCCESS)
                    raise RuntimeError(f"Upload segments failed: {errs}")

                # DB 登録（各HLSセグメントのメタ情報付与）
                total_segments = len([p for p in seg_paths if p.endswith('.m4s')])
                cumulative = 0.0
                for idx, p in enumerate(seg_paths):
                    key = _relkey(p)
                    size = Path(p).stat().st_size
                    if p.endswith(".m4s"):
                        info = probe_video(p)
                        dur = info.get("durationSec") or 0.0
                        start_sec = cumulative
                        end_sec = cumulative + dur
                        cumulative = end_sec
                        meta = {
                            "durationSec": dur,
                            "index": idx,  # includes init as 0, segments start from 1
                            "total": total_segments,
                            "kind": "segment",
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
                    else:
                        meta = {
                            "durationSec": 0.0,
                            "index": idx,
                            "total": total_segments,
                            "kind": "init",
                            "startSec": 0.0,
                            "endSec": 0.0,
                        }
                    insert_hls_segment(
                        self.db_manager,
                        file_id=file_id,
                        key=key,
                        size=size,
                        bucket=self.uploader.bucket,
                        meta=meta,
                    )

                # 完了
                hls_job_query.set_hls_job_status(self.db_manager, job_id, "complete")

            except Exception as e:
                try:
                    hls_job_query.set_hls_job_status(self.db_manager, job_id, "faild")
                except Exception:
                    pass
                print(f"HLS Job {job_id} failed: {e}")
            finally:
                try:
                    if work_dir.exists():
                        shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass

        # 新スケジューラ: GPU/CPU 各1並列で実行（GPU優先）
        pending: list[tuple[str, dict]] = []
        # HLS を先に並べ、次に通常エンコード
        for j in (hls_jobs or []):
            pending.append(("hls", j))
        for j in (jobs or []):
            pending.append(("encode", j))

        if not pending:
            return

        gpu_ex = ThreadPoolExecutor(max_workers=1)
        cpu_ex = ThreadPoolExecutor(max_workers=1)
        try:
            gpu_future = None
            cpu_future = None

            def _submit_next_to(resource: str):
                nonlocal gpu_future, cpu_future
                if not pending:
                    return
                kind, job = pending.pop(0)
                if resource == "gpu":
                    if kind == "hls":
                        gpu_future = gpu_ex.submit(_process_hls_job, job, backend="gpu")
                    else:
                        gpu_future = gpu_ex.submit(_process_job, job, backend="gpu")
                else:
                    if kind == "hls":
                        cpu_future = cpu_ex.submit(_process_hls_job, job, backend="cpu")
                    else:
                        cpu_future = cpu_ex.submit(_process_job, job, backend="cpu")

            # まずGPUへ、次にCPUへ投げる
            if gpu_future is None:
                _submit_next_to("gpu")
            if cpu_future is None:
                _submit_next_to("cpu")

            # 以降、どちらかが空いたら次を投入
            while (gpu_future is not None or cpu_future is not None) or pending:
                # 補充（先にGPU優先）
                if gpu_future is None and pending:
                    _submit_next_to("gpu")
                if cpu_future is None and pending:
                    _submit_next_to("cpu")

                actives = [f for f in [gpu_future, cpu_future] if f is not None]
                if not actives:
                    break
                done, _ = wait(actives, return_when=FIRST_COMPLETED)
                if gpu_future in done:
                    gpu_future = None
                if cpu_future in done:
                    cpu_future = None
        finally:
            gpu_ex.shutdown(wait=True)
            cpu_ex.shutdown(wait=True)

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
    runner = TaskRunner(interval=5)
    runner.run()
