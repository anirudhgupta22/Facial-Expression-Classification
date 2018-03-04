FROM continuumio/miniconda
MAINTAINER anirudh gupta <22anirudh.gupta@gmail.com>

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /tmp/requirements.txt

RUN set -eux \
    && apt-get update \
    && apt-get install -y libjpeg62-turbo-dev \
    && apt-get install -y libblas-dev \
    && conda install mkl-service \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && conda clean --all -y