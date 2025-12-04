import time
from pathlib import Path
import shutil
import mimetypes
from typing import List
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.config import load_surreal_config, load_s3_config
from backend_module.encoder import probe_video, create_thumbnail, encode_to_hls, concat_videos, hls_to_mp4
from backend_module.uuid_tools import get_uuid
from query import file_query
from query import inference_result_query
from query import hls_job_query
from query import merge_group_query
from query.hls_playlist_query import insert_hls_playlist, get_playlist_for_file
from query.hls_segment_query import insert_hls_segment, list_segments_for_file
from query.utils import rid_leaf, first_result
 

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
        # Merge multi-part videos defined in merge_group into a single video (copy concat)
        if self._process_merge_groups():
            return

        # 1) サムネイル作成（優先処理）
        if self._process_missing_thumbnails():
            # サムネイルを処理した場合は、このサイクルではここで終了（優先度を担保）
            return

        # Completed HLS jobs -> repackage HLS segments into a single MP4 (copy)
        if self._process_hls_repack_to_mp4():
            return

        # 新規ジョブをエンキュー（HLS のみ）
        hls_job_query.queue_unhls_video_jobs(self.db_manager)

        # キュー取得（SurrealDBの応答形式を吸収）
        hls_jobs = hls_job_query.get_queued_job(self.db_manager)

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
        pending: list[dict] = []
        for j in (hls_jobs or []):
            pending.append(j)

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
                job = pending.pop(0)
                if resource == "gpu":
                    gpu_future = gpu_ex.submit(_process_hls_job, job, backend="gpu")
                else:
                    cpu_future = cpu_ex.submit(_process_hls_job, job, backend="cpu")

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

    def _process_hls_repack_to_mp4(self) -> bool:
        """
        Convert completed HLS assets back into a single MP4 (copy) and swap the file key.

        Returns True if a file was processed (success or failure) to throttle the loop.
        """
        res = self.db_manager.query(
            """
            SELECT id, key, dataset, name, bucket, mime
            FROM file
            WHERE encode = 'video-none'
              AND id INSIDE (SELECT VALUE file FROM hls_job WHERE status = 'complete')
            LIMIT 1;
            """
        )
        target = first_result(res)
        if not target:
            return False

        file_id = target.get("id")
        orig_key = target.get("key")
        if not file_id or not orig_key:
            return False

        dataset = target.get("dataset")
        current_name = target.get("name") or Path(orig_key).name
        bucket = target.get("bucket") or self.uploader.bucket
        mime = target.get("mime") or "video/mp4"

        work_dir = Path("work_hls_repack") / rid_leaf(file_id)
        try:
            work_dir.mkdir(parents=True, exist_ok=True)

            playlist_row = get_playlist_for_file(self.db_manager, file_id)
            if not playlist_row or not playlist_row.get("key"):
                raise RuntimeError("Missing HLS playlist for file")
            playlist_key = playlist_row.get("key")

            seg_rows = list_segments_for_file(self.db_manager, file_id)
            segment_keys = [r.get("key") for r in seg_rows if isinstance(r, dict) and r.get("key")]
            if not segment_keys:
                raise RuntimeError("No HLS segments registered for file")

            dl_results = self.uploader.download_files(keys=[playlist_key, *segment_keys], dest_dir=work_dir)
            key_to_local = {
                r.key: r.local_path for r in dl_results if r.status == S3Info.SUCCESS and r.local_path
            }
            failed = [r.key for r in dl_results if r.status != S3Info.SUCCESS]
            if failed:
                raise RuntimeError(f"Download failed for: {failed}")

            playlist_local = key_to_local.get(playlist_key)
            if not playlist_local:
                raise RuntimeError("Downloaded playlist not found locally")

            stem = Path(current_name).stem or rid_leaf(file_id)
            out_local = work_dir / f"{stem}_hls.mp4"
            hls_to_mp4(str(playlist_local), str(out_local))
            if not out_local.exists():
                raise RuntimeError("HLS repack output missing")

            dataset_prefix = str(dataset).strip("/") if dataset else ""
            if not dataset_prefix and orig_key:
                dataset_prefix = str(Path(orig_key).parent).strip("/")
            if dataset_prefix == ".":
                dataset_prefix = ""
            new_name = out_local.name
            new_key = f"{dataset_prefix}/{new_name}" if dataset_prefix else new_name
            if orig_key == new_key:
                new_key = f"{dataset_prefix}/{stem}-{get_uuid(8)}.mp4" if dataset_prefix else f"{stem}-{get_uuid(8)}.mp4"

            upload = self.uploader.upload_file_as(str(out_local), new_key)
            if upload.status != S3Info.SUCCESS:
                raise RuntimeError(f"Upload repacked MP4 failed: {upload.error}")

            new_size = out_local.stat().st_size
            new_mime = mimetypes.guess_type(out_local.name)[0] or mime

            file_query.update_file_after_hls_repack(
                self.db_manager,
                file_id,
                new_key=new_key,
                name=new_name,
                size=new_size,
                mime=new_mime,
                bucket=bucket,
                encode="video-hls-repacked",
                source_key=orig_key,
            )

            del_res = self.uploader.delete_key(orig_key)
            if del_res.status != S3Info.SUCCESS:
                raise RuntimeError(f"Delete original video failed: {del_res.error}")

        except Exception as e:
            print(f"HLS repack failed for {file_id}: {e}")
        finally:
            try:
                if work_dir.exists():
                    shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

        return True

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

    def _process_merge_groups(self) -> bool:
        """
        Build a single concatenated video for each merge_group entry (copy, no re-encode).

        Returns True if any merge was performed to throttle the cycle.
        """
        groups = merge_group_query.list_pending_merge_groups(self.db_manager)
        if not groups:
            return False

        for mg in groups:
            mg_id = mg.get("id")
            dataset = mg.get("dataset")
            members = mg.get("members") or []
            if not dataset or not members:
                continue

            target_name = f"_merged_{members[0]}"

            # If a final file already exists, just mark the group as merged.
            existing = file_query.get_file_by_dataset_and_name(self.db_manager, dataset, target_name)
            if existing and existing.get("encode") == "video-none":
                try:
                    merge_group_query.set_merged_file(self.db_manager, mg_id, existing.get("id"))
                except Exception:
                    pass
                return True

            # Resolve member file rows in declared order
            rows = file_query.get_files_by_names(self.db_manager, dataset, members)
            by_name = {str(r.get("name")): r for r in rows if r.get("key")}
            ordered_rows = [by_name.get(str(n)) for n in members if by_name.get(str(n))]
            if len(ordered_rows) != len(members):
                # missing source files; try next group
                continue

            keys = [r["key"] for r in ordered_rows if r.get("key")]
            work_dir = Path("work_merge") / (rid_leaf(mg_id) if mg_id else get_uuid(8))
            merged_local = work_dir / target_name

            try:
                work_dir.mkdir(parents=True, exist_ok=True)
                dl_results = self.uploader.download_files(keys=keys, dest_dir=work_dir)
                key_to_local = {
                    r.key: r.local_path for r in dl_results if r.status == S3Info.SUCCESS and r.local_path
                }
                if len(key_to_local) != len(keys):
                    raise RuntimeError("One or more source videos failed to download")

                ordered_paths = [key_to_local[k] for k in keys]
                concat_videos(ordered_paths, str(merged_local))

                # Upload merged video with dataset/<first_member_name>
                target_key = f"{dataset}/{target_name}"
                up = self.uploader.upload_file_as(str(merged_local), target_key)
                if up.status != S3Info.SUCCESS:
                    raise RuntimeError(f"Upload failed: {up.error}")

                first_row = ordered_rows[0] if ordered_rows else None
                if not first_row:
                    raise RuntimeError("Missing first source row for merge group")

                # Copy thumbnail with prefixed name (do not reference original)
                new_thumb_key = None
                first_thumb = first_row.get("thumbKey")
                if first_thumb:
                    thumb_suffix = Path(first_thumb).suffix or ".jpg"
                    new_thumb_key = f"{dataset}/.thumbs/{Path(target_name).name}{thumb_suffix}"
                    thumb_local = work_dir / Path(first_thumb).name
                    dl_thumb = self.uploader.download_file(first_thumb, str(thumb_local))
                    if dl_thumb.status == S3Info.SUCCESS and dl_thumb.local_path:
                        up_thumb = self.uploader.upload_file_as(dl_thumb.local_path, new_thumb_key)
                        if up_thumb.status != S3Info.SUCCESS:
                            new_thumb_key = None
                    else:
                        new_thumb_key = None

                merged_size = merged_local.stat().st_size
                mime = first_row.get("mime") or mimetypes.guess_type(target_name)[0] or "video/mp4"
                bucket = first_row.get("bucket") or self.uploader.bucket
                ins = file_query.insert_file_record(
                    self.db_manager,
                    dataset=dataset,
                    key=target_key,
                    name=target_name,
                    mime=mime,
                    size=merged_size,
                    bucket=bucket,
                    encode="video-none",
                    thumb_key=new_thumb_key,
                    meta={
                        "mergeGroup": mg_id,
                        "members": members,
                        "mode": mg.get("mode"),
                    },
                )
                new_row = first_result(ins)
                new_file_id = new_row.get("id") if isinstance(new_row, dict) else None
                if new_file_id:
                    merge_group_query.set_merged_file(self.db_manager, mg_id, new_file_id)
                    merge_group_query.mark_merge_group_dead(self.db_manager, mg_id)
                    for row in ordered_rows:
                        fid = row.get("id")
                        if fid:
                            try:
                                file_query.mark_file_dead(self.db_manager, fid)
                            except Exception:
                                pass
                return True
            except Exception as e:
                print(f"Merge group {mg_id or dataset} failed: {e}")
            finally:
                try:
                    if work_dir.exists():
                        shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass

        return False

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
