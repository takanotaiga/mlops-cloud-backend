# AGENTS.md

このリポジトリは MLOps Cloud の Python worker 群です。単一 API サーバーではなく、SurrealDB をポーリングし、MinIO/S3 上の object を処理する常駐プロセス群です。

## 主要 worker

| File | Role |
|---|---|
| `video_manager.py` | video / inference_result を HLS 化し `hls_*` records を作成 |
| `ml_inference_manager.py` | `inference_job` を処理し SAMURAI/SAM2/RT-DETR pipeline を実行 |
| `cleaner_manager.py` | `dead=true` file や orphan annotation を DB/S3 から削除 |
| `hardware_metrics_manager.py` | hardware metrics を収集 |
| `terminal_manager.py` | WebSocket terminal bridge |
| `system_manager.py` | host/system helper |

## 主要 module

| Path | Role |
|---|---|
| `backend_module/config.py` | env config loader |
| `backend_module/database.py` | SurrealDB wrapper |
| `backend_module/object_storage.py` | MinIO/S3 wrapper |
| `backend_module/encoder.py` | FFmpeg/HLS helper |
| `backend_module/progress_tracker.py` | inference progress update helper |
| `query/` | table-specific query helpers |
| `ml_module/` | SAMURAI/RT-DETR pipeline and CLI helpers |

## Environment

Python is pinned to `>=3.11,<3.12`. Use `uv`.

```bash
uv sync
uv sync --extra mlx
```

Preferred env names:

- `SURREAL_URL`, `SURREAL_NS`, `SURREAL_DB`, `SURREAL_USER`, `SURREAL_PASS`
- `MINIO_ENDPOINT_INTERNAL`, `MINIO_REGION`, `MINIO_ACCESS_KEY_ID`, `MINIO_SECRET_ACCESS_KEY`, `MINIO_BUCKET`, `MINIO_FORCE_PATH_STYLE`
- `S3_MULTIPART_THRESHOLD_BYTES`, `S3_MULTIPART_CHUNKSIZE_BYTES`, `S3_TRANSFER_CONCURRENCY`

Legacy fallbacks exist for `SURREAL_ENDPOINT`, `SURREAL_NAMESPACE`, `SURREAL_DATABASE`, `SURREAL_USERNAME`, `SURREAL_PASSWORD`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`. Do not add new code that only supports legacy names.

## Dockerfiles

Current Dockerfiles are:

- `Dockerfile.base`: non-GPU/base worker image
- `Dockerfile.gpu`: GPU inference/CV image

Do not introduce new references to old `Dockerfile.cv` or `Dockerfile.mlx`.

## Inference pipeline notes

- Current production path is `taskType=one-shot-object-detection`, `model=samurai-ulr`.
- Expected input is one dataset with exactly one video.
- UI can pass `inferenceBackend` as `tensorrt-fp16`, `pytorch-fp16`, or `pytorch-fp32`.
- UI can pass `rtdetrEpochs`; default remains 4.
- RT-DETR training should use pretrained `rtdetr-l.pt` when configured that way.
- TensorRT behavior is GPU/driver dependent. Keep PyTorch FP32/FP16 fallback paths working.
- Result videos should be uploaded as inference artifacts and then HLS encoded by `video_manager.py` / `cv-backend`.

## State values

`Faild` and `StopInterrept` are existing DB/status values. Do not silently rename them. If adding normalized spelling, keep compatibility mappings.

## Tests

Local unit tests:

```bash
uv run pytest -q
```

Integration tests live in `../mlops-cloud/e2e` and should be run from the `mlops-cloud` repo.

```bash
cd ../mlops-cloud
docker compose -f e2e/compose.phase2.yml up --build --abort-on-container-exit --exit-code-from backend-test backend-test
docker compose -f e2e/compose.phase2.yml down -v
```

GPU pipeline:

```bash
cd ../mlops-cloud
docker compose -f e2e/compose.phase4.yml up --build --abort-on-container-exit --exit-code-from phase4-test phase4-test
docker compose -f e2e/compose.phase4.yml down -v
```

Phase4 requires NVIDIA container runtime and can take minutes.

## Security / operations

- Do not hardcode real credentials.
- `terminal_manager.py` can bridge to host SSH. Treat it as sensitive and avoid exposing it without auth/network controls.
- Cleanup is asynchronous. UI deletes usually mark records `dead=true`; backend cleaner removes DB/S3 later.
- Work directories must be job-scoped and cleaned on failure.
- HLS output should register playlist/segments in DB and upload all referenced objects to S3.
