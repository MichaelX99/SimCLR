SHELL := /bin/bash

DOCKER_IMAGE = simclr_image
DOCKER_CONTAINER = simclr_container
DOCKER_HOME=/home/ubuntu

build:
	DOCKER_BUILDKIT=1 docker build -t ${DOCKER_IMAGE} \
	--build-arg UID=${shell id -u} --build-arg GID=${shell id -g} \
	 -f docker/Dockerfile .

# --gpus all

develop: build
	docker run \
	-it --rm \
	-e "TERM=xterm-256color" \
	-v ${PWD}/src:${DOCKER_HOME}/src \
	--name=${DOCKER_CONTAINER} \
	${DOCKER_IMAGE}
