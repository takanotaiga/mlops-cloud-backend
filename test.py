from backend_module.object_storage import MinioS3Uploader

if __name__ == "__main__":
    uploader = MinioS3Uploader(
        endpoint_url="http://192.168.1.25:65300",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket="horus-bucket",
        multipart_threshold_bytes=300 * 1024 * 1024,  # 300MB
        multipart_chunksize_bytes=64 * 1024 * 1024,   # 64MB
        part_concurrency=4,                           # 各ファイル内のパート並列
    )

    # 単体
    # r = uploader.upload_file("/home/taiga/horus-runner/out.mp4", key_prefix="uploads/2025-08-10")
    # print(r)
    r = uploader.download_file(
        key="uploads/2025-08-10/0a8557c4b65f4319-out.mp4",
        local_path="/home/taiga/mlops-cloud-backend/out.mp4"
    )
    print(r)


    # 複数（外側の並列はここで指定）
    # rs = uploader.upload_files(
    #     [
    #         "/home/taiga/horus-runner/out/out_000.mp4", 
    #         "/home/taiga/horus-runner/out/out_001.mp4",
    #         "/home/taiga/horus-runner/out/out_002.mp4",
    #         "/home/taiga/horus-runner/out/out_003.mp4",
    #     ],
    #     key_prefix="uploads/2025-08-10",
    #     max_workers=3,  # 同時に3ファイルまで
    # )
    # for x in rs:
    #     print(x)
