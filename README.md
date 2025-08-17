# mlops-cloud-backend

このリポジトリにはエンコード実行ワーカー（`video_manager.py`）と推論実行ワーカー（`ml_inference_manager.py`）が含まれます。どちらも SurrealDB と MinIO/S3 の接続情報を環境変数から読み込みます。

以下に Docker 単体実行（docker run）と docker compose のサンプルを示します。

## Docker Run（単体実行）

先にイメージをビルドします。

- エンコードワーカー（FFmpeg 含む）
  - `docker build -f Dockerfile.cv -t mlops-video .`
- 推論ワーカー（CUDA/CuDNN 前提）
  - `docker build -f Dockerfile.mlx -t mlops-ml .`

実行例（各種環境変数は適宜変更してください）。

- エンコードワーカー（CPU/GPU どちらでも可）
  - `docker run --rm \
      -e SURREAL_URL=ws://database:8000/rpc \
      -e SURREAL_NS=mlops \
      -e SURREAL_DB=cloud_ui \
      -e SURREAL_USER=root \
      -e SURREAL_PASS=root \
      -e MINIO_ENDPOINT_INTERNAL=http://object-storage:9000 \
      -e MINIO_REGION=us-east-1 \
      -e MINIO_ACCESS_KEY_ID=minioadmin \
      -e MINIO_SECRET_ACCESS_KEY=minioadmin \
      -e MINIO_BUCKET=mlops-datasets \
      -e MINIO_FORCE_PATH_STYLE=true \
      -e S3_MULTIPART_THRESHOLD_BYTES=1000000000 \
      --name mlops-video mlops-video`

- 推論ワーカー（GPU 利用）
  - `docker run --rm --gpus all \
      -e SURREAL_URL=ws://database:8000/rpc \
      -e SURREAL_NS=mlops \
      -e SURREAL_DB=cloud_ui \
      -e SURREAL_USER=root \
      -e SURREAL_PASS=root \
      -e MINIO_ENDPOINT_INTERNAL=http://object-storage:9000 \
      -e MINIO_REGION=us-east-1 \
      -e MINIO_ACCESS_KEY_ID=minioadmin \
      -e MINIO_SECRET_ACCESS_KEY=minioadmin \
      -e MINIO_BUCKET=mlops-datasets \
      -e MINIO_FORCE_PATH_STYLE=true \
      -e S3_MULTIPART_THRESHOLD_BYTES=1000000000 \
      --name mlops-ml mlops-ml`

## docker compose（例）

最小構成の例です。SurrealDB と MinIO、バックエンドワーカー2種を同時に起動します。

```yaml
version: "3.9"
services:
  database:
    image: surrealdb/surrealdb:latest
    command: ["start", "--log", "info", "-A", "--user", "root", "--pass", "root", "memory"]
    ports:
      - "8000:8000"
    restart: unless-stopped

  object-storage:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"  # S3 API
      - "9001:9001"  # Console UI
    volumes:
      - minio-data:/data
    restart: unless-stopped

  mlops-video:
    image: mlops-video  # 事前に `docker build -f Dockerfile.cv -t mlops-video .`
    depends_on:
      - database
      - object-storage
    environment:
      SURREAL_URL: ws://database:8000/rpc
      SURREAL_NS: mlops
      SURREAL_DB: cloud_ui
      SURREAL_USER: root
      SURREAL_PASS: root
      MINIO_ENDPOINT_INTERNAL: http://object-storage:9000
      MINIO_REGION: us-east-1
      MINIO_ACCESS_KEY_ID: minioadmin
      MINIO_SECRET_ACCESS_KEY: minioadmin
      MINIO_BUCKET: mlops-datasets
      MINIO_FORCE_PATH_STYLE: "true"
      S3_MULTIPART_THRESHOLD_BYTES: "1000000000"
    restart: unless-stopped

  mlops-ml:
    image: mlops-ml  # 事前に `docker build -f Dockerfile.mlx -t mlops-ml .`
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    depends_on:
      - database
      - object-storage
    environment:
      SURREAL_URL: ws://database:8000/rpc
      SURREAL_NS: mlops
      SURREAL_DB: cloud_ui
      SURREAL_USER: root
      SURREAL_PASS: root
      MINIO_ENDPOINT_INTERNAL: http://object-storage:9000
      MINIO_REGION: us-east-1
      MINIO_ACCESS_KEY_ID: minioadmin
      MINIO_SECRET_ACCESS_KEY: minioadmin
      MINIO_BUCKET: mlops-datasets
      MINIO_FORCE_PATH_STYLE: "true"
      S3_MULTIPART_THRESHOLD_BYTES: "1000000000"
      POLL_INTERVAL: "5"
    restart: unless-stopped

volumes:
  minio-data:
```

補足
- それぞれのサービスは環境変数から設定を読み込みます（`backend_module/config.py`）。
- UI と併用する場合は、UI 側の compose に合わせて同じ Surreal/MinIO のエンドポイントを指定してください。
- 推論側は GPU を使う前提のイメージです。環境により `--gpus all`（docker run）や `deploy.resources.reservations.devices`（compose）の指定を調整してください。
