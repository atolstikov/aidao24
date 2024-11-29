FROM mambaorg/micromamba:noble

USER root

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      unzip \
      locales \
      gnupg \
      curl \
      make \
      ca-certificates; \
    apt-get clean -y; \
    rm -rf /var/lib/apt/lists/*; \
    locale-gen en_US.UTF-8; \
    update-locale

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libx11-6 \
    gdebi-core \
    libapparmor1  \
    libcurl4-openssl-dev \
    build-essential \
    gnupg2 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base python==3.10 pip -c conda-forge && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt \
   && rm -rf /root/.cache

RUN mkdir /opt/private_test/
COPY ./images /opt/private_test/
