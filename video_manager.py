import time
from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader
from query import encode_job_query

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
        encode_job_query.queue_unencoded_video_jobs(self.db_manager)
        encode_job_query.get_queued_job(self.db_manager)

        # resutl = self.db_manager.query(
        #     "SELECT * FROM encode_job;")
        # print(resutl)

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
