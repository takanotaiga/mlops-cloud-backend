# mlops-cloud-backend

Python worker repository for MLOps Cloud.

This repo does not expose the primary product API. Workers communicate with the UI through shared SurrealDB records and MinIO/S3 objects.

## Workers

| Worker | Command | Purpose |
|---|---|---|
| Video / CV | `uv run video_manager.py` | encode uploaded videos and inference result videos to HLS |
| Inference | `uv run ml_inference_manager.py` | run SAMURAI/SAM2/RT-DETR inference jobs |
| Cleaner | `uv run cleaner_manager.py` | remove `dead=true` files and orphan records from DB/S3 |
| Hardware metrics | `uv run hardware_metrics_manager.py` | collect metrics |

## Install

```bash
uv sync
```

GPU / inference dependencies:

```bash
uv sync --extra mlx
```

Python requirement: `>=3.11,<3.12`.

## Docker

Build base image:

```bash
docker build -f Dockerfile.base -t mlops-cloud-backend-base:dev .
```

Build GPU image:

```bash
docker build -f Dockerfile.gpu -t mlops-cloud-backend-gpu:dev .
```

Run GPU check:

```bash
docker run --rm --gpus all mlops-cloud-backend-gpu:dev nvidia-smi
```

Older names like `Dockerfile.cv` and `Dockerfile.mlx` are obsolete.

## Configuration

Preferred SurrealDB env:

```bash
SURREAL_URL=ws://database:8000/rpc
SURREAL_NS=mlops
SURREAL_DB=cloud_ui
SURREAL_USER=root
SURREAL_PASS=root
```

Preferred MinIO/S3 env:

```bash
MINIO_ENDPOINT_INTERNAL=http://object-storage:9000
MINIO_REGION=us-east-1
MINIO_ACCESS_KEY_ID=minioadmin
MINIO_SECRET_ACCESS_KEY=minioadmin
MINIO_BUCKET=mlops-datasets
MINIO_FORCE_PATH_STYLE=true
S3_MULTIPART_THRESHOLD_BYTES=1000000000
```

Legacy `SURREAL_ENDPOINT` / `S3_ENDPOINT` style variables are fallback only.

## Inference Notes

Current supported UI path:

- `taskType=one-shot-object-detection`
- `model=samurai-ulr` or `model=t260-ulr`
- one dataset containing exactly one video
- one-shot SAM2 bbox annotation as seed

`t260-ulr` uses the official SAM2.1 package for tracking and RF-DETR for detector fine-tuning/inference. RF-DETR TensorRT export is not enabled yet; T260 defaults to PyTorch FP16.

Runtime options:

- `inferenceBackend=tensorrt-fp16` (default for compatibility)
- `inferenceBackend=pytorch-fp16`
- `inferenceBackend=pytorch-fp32`
- `rtdetrEpochs` defaults to 4

TensorRT behavior can depend on GPU generation, CUDA, TensorRT and exported ONNX shape/options. Keep PyTorch fallback paths healthy and test with Phase4 when changing this area.

## HLS Notes

`video_manager.py` is responsible for HLS encoding both uploaded dataset videos and inference result videos. UI video preview expects HLS playlist records:

- `hls_playlist`
- `hls_segment`

The UI HLS route rewrites playlist segment URLs through `/api/storage/object`.

## Tests

Unit tests:

```bash
uv run pytest -q
```

Integration tests from the compose repo:

```bash
cd ../mlops-cloud
docker compose -f e2e/compose.phase2.yml up --build --abort-on-container-exit --exit-code-from backend-test backend-test
docker compose -f e2e/compose.phase2.yml down -v
```

GPU E2E:

```bash
cd ../mlops-cloud
docker compose -f e2e/compose.phase4.yml up --build --abort-on-container-exit --exit-code-from phase4-test phase4-test
docker compose -f e2e/compose.phase4.yml down -v
```

## Operational Caveats

- `Faild` and `StopInterrept` are existing status values. Preserve compatibility.
- Cleaner deletes DB/S3 asynchronously after UI soft delete.
- The former WebSocket terminal bridge was removed because it exposed host SSH access from the application surface.
- Do not commit real credentials.
