#!/bin/sh

# build fake data generator
cd fake-data-generator
docker build -t luntaixia/general-clf-faker .
docker push luntaixia/general-clf-faker

# test image
docker pull luntaixia/general-clf-faker
docker run -it --rm \
 -e SECRET_TOML_PATH=/app-mount/secrets.toml \
 -v /home/luntaixia/Downloads/docker-volume-mapping/general_clf_pipeline_volume/app-mount:/app-mount \
 luntaixia/general-clf-faker \
 python -m src.dag_run test_run