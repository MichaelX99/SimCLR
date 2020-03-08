SHELL := /bin/bash

DOCKER_IMAGE = simclr_image
DOCKER_CONTAINER = simclr_container
DOCKER_HOME=/home/ubuntu

build:
	DOCKER_BUILDKIT=1 docker build -t ${DOCKER_IMAGE} \
	--build-arg UID=${shell id -u} --build-arg GID=${shell id -g} \
	 -f docker/Dockerfile .

develop: build
	docker run --gpus all \
	-it --rm \
	-e "TERM=xterm-256color" \
	-v ${PWD}/simclr:${DOCKER_HOME}/simclr \
	--name=${DOCKER_CONTAINER} \
	${DOCKER_IMAGE}
