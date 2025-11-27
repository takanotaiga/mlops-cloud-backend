#!/usr/bin/env bash
set -euo pipefail

# Build the MLX image and run tests inside the container with repo and assets mounted.
# Usage: ./run_mlx_tests.sh ["custom command"]

IMAGE_NAME=${IMAGE_NAME:-mlops-cloud-mlx}
WORKDIR=${WORKDIR:-/workspace}
CMD=${1:-"pytest tests/test_encoder.py -q"}

docker build -f Dockerfile.mlx -t "$IMAGE_NAME" .

docker run --rm \
  -v "$(pwd)":"${WORKDIR}" \
  -w "${WORKDIR}" \
  -e PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}" \
  "$IMAGE_NAME" \
  bash -lc "$CMD"
