FROM continuumio/miniconda
MAINTAINER anirudh gupta <22anirudh.gupta@gmail.com>

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN mkdir -p /root/.keras/

RUN ln -s /usr/src/app/nonsvn/.theanorc /root/.theanorc
RUN ln -s /usr/src/app/nonsvn/keras.json /root/.keras/keras.json

COPY requirements.txt /tmp/requirements.txt

RUN set -eux \
    && apt-get update \
    && apt-get install -y make gcc g++ libsnappy-dev \
    && apt-get install -y libjpeg62-turbo-dev \
    && apt-get install -y libblas-dev \
    && conda install mkl-service \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && conda clean --all -y