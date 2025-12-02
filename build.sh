#!/usr/bin/env bash
docker build -t mlops_cloud_backend_base:local_latest -f Dockerfile.base .
docker build -t mlops_cloud_backend_gpu:local_latest -f Dockerfile.gpu .
docker image list
