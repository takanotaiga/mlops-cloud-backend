import time
from typing import List

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader
from backend_module.config import load_surreal_config, load_s3_config
from query.cleaner_query import (
    list_dead_file_records,
    delete_file_record,
    list_orphan_annotations,
    delete_annotation_record,
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
        reuslts = list_dead_file_records(self.db)
        for result in reuslts:
            s3_key = result["key"]
            s3_thumb_key = result["thumbKey"]
            record_id = result["id"]
            self.s3.delete_key(s3_key)
            self.s3.delete_key(s3_thumb_key)
            delete_file_record(self.db, record_id)
            print(result)

        reuslts = list_orphan_annotations(self.db)
        for result in reuslts:
            s3_key = result["key"]
            record_id = result["id"]
            self.s3.delete_key(s3_key)
            delete_annotation_record(self.db, record_id)
            print(result)

        reuslts = list_orphan_encoded_segments(self.db)
        for result in reuslts:
            s3_key = result["key"]
            self.s3.delete_key(s3_key)
            record_id = result["id"]
            delete_encoded_segment_record(self.db, record_id)
            print(result)

        reuslts = list_orphan_hls_jobs(self.db)
        for result in reuslts:
            record_id = result["id"]
            delete_hls_job_record(self.db, record_id)
            print(result)

        reuslts = list_orphan_hls_playlists(self.db)
        for result in reuslts:
            s3_key = result["key"]
            self.s3.delete_key(s3_key)
            record_id = result["id"]
            delete_hls_playlist_record(self.db, record_id)
            print(result)

        reuslts = list_orphan_hls_segments(self.db)
        for result in reuslts:
            s3_key = result["key"]
            self.s3.delete_key(s3_key)
            record_id = result["id"]
            delete_hls_segment_record(self.db, record_id)
            print(result)

        reuslts = list_dead_inference_jobs(self.db)
        for result in reuslts:
            record_id = result["id"]
            delete_inference_job_record(self.db, record_id)
            print(result)

        reuslts = list_orphan_inference_results(self.db)
        for result in reuslts:
            s3_key = result["key"]
            self.s3.delete_key(s3_key)
            record_id = result["id"]
            delete_inference_result_record(self.db, record_id)
            print(result)

        reuslts = list_orphan_labels(self.db)
        for result in reuslts:
            record_id = result["id"]
            delete_label_record(self.db, record_id)
            print(result)

        reuslts = list_orphan_merge_groups(self.db)
        for result in reuslts:
            record_id = result["id"]
            delete_merge_group_record(self.db, record_id)
            print(result)

    def run(self):
        while True:
            time.sleep(self.interval)
            self.task_main()


if __name__ == "__main__":
    TaskRunner(interval=5).run()
