#! /bin/bash

# Placeholder script for setting up CVAT on server

cd ../
git clone https://github.com/cvat-ai/cvat.git
cd cvat

# need to set host to public IP address
# do everytime cvat is started
export CVAT_HOST=sunny.ece.ncsu.edu
# needs platform specification if not running on x86_64
# DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up -d
docker compose up -d

# create super user
# will need to enter details manually
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
