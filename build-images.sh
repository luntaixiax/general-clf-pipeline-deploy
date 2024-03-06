#!/bin/sh

# build fake data generator
cd fake-data-generator
docker build -t luntaixia/general-clf-faker .
docker push luntaixia/general-clf-faker