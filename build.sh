docker build -t taigatakano/mlops-cloud-backend-cv:latest -f ./Dockerfile.cv .
docker build -t taigatakano/mlops-cloud-backend-mlx:latest -f ./Dockerfile.mlx .

docker push taigatakano/mlops-cloud-backend-cv:latest
docker push taigatakano/mlops-cloud-backend-mlx:latest
