import time
from typing import List

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader, S3Info
from backend_module.config import load_surreal_config, load_s3_config
from query.cleaner_query import (
    list_dead_file_records,
    delete_file_record,
    list_orphan_annotations,
    delete_annotation_record,
    list_orphan_encode_jobs,
    delete_encode_job_record,
    list_orphan_encoded_segments,
    delete_encoded_segment_record,
    list_orphan_hls_jobs,
    delete_hls_job_record,
    list_orphan_hls_playlists,
    delete_hls_playlist_record,
    list_orphan_hls_segments,
    delete_hls_segment_record,
    list_dead_inference_jobs,
    delete_inference_job_record,
    list_orphan_inference_results,
    delete_inference_result_record,
    list_orphan_labels,
    delete_label_record,
    list_orphan_merge_groups,
    delete_merge_group_record,
)


class TaskRunner:
    def __init__(self, interval: int = 5):
        self.interval = interval

        # SurrealDB from environment
        sconf = load_surreal_config()
        self.db = DataBaseManager(
            endpoint_url=sconf["endpoint_url"],
            username=sconf["username"],
            password=sconf["password"],
            namespace=sconf["namespace"],
            database=sconf["database"],
        )

        # MinIO/S3 from environment
        mconf = load_s3_config()
        self.s3 = MinioS3Uploader(
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
        # Define sequential cleanup tasks. For each task:
        # - fetch rows
        # - delete S3 objects if key is present
        # - delete DB records (only if S3 delete succeeded or no key)
        tasks = [
            {
                "name": "file",
                "fetch": lambda: list_dead_file_records(self.db),
                "delete": lambda rid: delete_file_record(self.db, rid),
                "has_key": True,
                # files can have multiple S3 objects (main key + thumbKey)
                "keys_from_row": lambda r: [
                    k for k in [r.get("key"), r.get("thumbKey")] if isinstance(k, str) and k
                ],
            },
            {
                "name": "annotation",
                "fetch": lambda: list_orphan_annotations(self.db),
                "delete": lambda rid: delete_annotation_record(self.db, rid),
                "has_key": True,
            },
            {
                "name": "encode_job",
                "fetch": lambda: list_orphan_encode_jobs(self.db),
                "delete": lambda rid: delete_encode_job_record(self.db, rid),
                "has_key": False,
            },
            {
                "name": "encoded_segment",
                "fetch": lambda: list_orphan_encoded_segments(self.db),
                "delete": lambda rid: delete_encoded_segment_record(self.db, rid),
                "has_key": True,
            },
            {
                "name": "hls_job",
                "fetch": lambda: list_orphan_hls_jobs(self.db),
                "delete": lambda rid: delete_hls_job_record(self.db, rid),
                "has_key": False,
            },
            {
                "name": "hls_playlist",
                "fetch": lambda: list_orphan_hls_playlists(self.db),
                "delete": lambda rid: delete_hls_playlist_record(self.db, rid),
                "has_key": True,
            },
            {
                "name": "hls_segment",
                "fetch": lambda: list_orphan_hls_segments(self.db),
                "delete": lambda rid: delete_hls_segment_record(self.db, rid),
                "has_key": True,
            },
            {
                "name": "inference_job",
                "fetch": lambda: list_dead_inference_jobs(self.db),
                "delete": lambda rid: delete_inference_job_record(self.db, rid),
                "has_key": False,
            },
            {
                "name": "inference_result",
                "fetch": lambda: list_orphan_inference_results(self.db),
                "delete": lambda rid: delete_inference_result_record(self.db, rid),
                "has_key": True,
            },
            {
                "name": "label",
                "fetch": lambda: list_orphan_labels(self.db),
                "delete": lambda rid: delete_label_record(self.db, rid),
                "has_key": False,
            },
            {
                "name": "merge_group",
                "fetch": lambda: list_orphan_merge_groups(self.db),
                "delete": lambda rid: delete_merge_group_record(self.db, rid),
                "has_key": False,
            },
        ]

        for t in tasks:
            name = t["name"]
            try:
                rows = t["fetch"]() or []
            except Exception as e:
                print(f"Query error ({name}): {e}")
                continue

            if not rows:
                continue

            key_success: set[str] = set()
            keys: List[str] = []
            if t["has_key"]:
                if "keys_from_row" in t:
                    # flatten multi-keys (e.g., file.key + file.thumbKey)
                    for r in rows:
                        try:
                            ks = t["keys_from_row"](r) or []
                        except Exception:
                            ks = []
                        for k in ks:
                            if isinstance(k, str) and k:
                                keys.append(k)
                else:
                    keys = [r.get("key") for r in rows if isinstance(r.get("key"), str) and r.get("key")]
                if keys:
                    try:
                        del_results = self.s3.delete_keys(keys)
                    except Exception as e:
                        print(f"Delete batch failed ({name}), falling back: {e}")
                        del_results = [self.s3.delete_key(k) for k in keys]
                    for r in del_results:
                        if r.status == S3Info.SUCCESS:
                            key_success.add(r.key)

            # Delete DB records
            rec_deleted = 0
            rec_failed = 0
            for r in rows:
                rid = r.get("id")
                if not rid:
                    continue
                if not t["has_key"]:
                    proceed = True
                else:
                    # Single-key or multi-key check
                    if "keys_from_row" in t:
                        ks = []
                        try:
                            ks = t["keys_from_row"](r) or []
                        except Exception:
                            ks = []
                        # If no keys present, proceed; otherwise require all present keys to have succeeded
                        present = [k for k in ks]
                        proceed = (not present) or all(k in key_success for k in present) or (not keys)
                    else:
                        k = r.get("key")
                        proceed = (not k) or (k in key_success) or (not keys)
                if proceed:
                    try:
                        t["delete"](rid)
                        rec_deleted += 1
                    except Exception as e:
                        rec_failed += 1
                        print(f"Record delete failed ({name}) {rid}: {e}")
            if rec_deleted or rec_failed:
                if rec_failed:
                    print(f"{name}: records deleted={rec_deleted}, failed={rec_failed}")
                else:
                    print(f"{name}: records deleted={rec_deleted}")

    def run(self):
        while True:
            now = time.time()
            if now < self.next_time:
                time.sleep(self.next_time - now)

            start = time.time()
            self.task_main()
            end = time.time()

            self.next_time += self.interval
            if end - start > self.next_time:
                self.next_time = end + self.interval


if __name__ == "__main__":
    TaskRunner(interval=5).run()
