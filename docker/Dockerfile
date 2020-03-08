# syntax = docker/dockerfile:1.0-experimental
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

RUN apt update && apt install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.7 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.7 get-pip.py && \
    rm get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

RUN pip install --upgrade https://storage.googleapis.com/jax-releases/cuda102/jaxlib-0.1.40-cp37-none-linux_x86_64.whl
RUN pip install --upgrade jax

ARG UID
ARG GID

RUN groupadd -g ${GID} ubuntu
RUN useradd -m -u ${UID} -g ${GID} ubuntu
USER ubuntu

WORKDIR /home/ubuntu
ADD setup.py /home/ubuntu
ADD README.md /home/ubuntu
ADD simclr /home/ubuntu/simclr

RUN pip install --user -e .
