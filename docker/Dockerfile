ARG cuda_version=11.8.0
ARG cudnn_version=8
ARG ubuntu=20.04
FROM nvcr.io/nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel-ubuntu${ubuntu}

LABEL maintainer "Tomoya Okazaki"

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PIP_DEFAULT_TIMEOUT=100
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"

RUN apt -y update && apt -y upgrade && \
    apt -y install --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        graphviz \
        graphviz-dev \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        python3-dev \
        python3-pip \
        wget \
        zip \
        unzip && \
    apt -y clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        setuptools==69.5.1 \
        wheel==0.43.0 && \
    python3 -m pip install --no-cache-dir \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip cache purge

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        check_orientation==0.0.5 \
        cmake==3.29.3 \
        einops==0.8.0 \
        h5py==3.11.0 \
        kneed==0.8.5 \
        kornia==0.7.2 \
        kornia-moons==0.2.9 \
        kornia-rs==0.1.3 \
        mmcv-full==1.7.2 \
        mmsegmentation==0.30.0 \
        opencv-python==4.9.0.80 \
        pandas==2.0.3 \
        pygraphviz==1.11 \
        scipy==1.10.1 \
        scikit-learn==1.3.2 \
        timm==1.0.3 \
        tqdm==4.66.4 \
        transformers==4.40.1 \
        yacs==0.1.8 && \
    python3 -m pip cache purge

# pycolmap
WORKDIR /home
RUN git clone https://github.com/colmap/colmap.git
WORKDIR /home/colmap/build

# https://github.com/colmap/colmap/issues/1822
RUN cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86

RUN ninja install

WORKDIR /home/colmap/pycolmap
RUN python3 -m pip install .

# lightglue
WORKDIR /home
RUN git clone https://github.com/cvg/LightGlue.git
WORKDIR /home/LightGlue
RUN python3 -m pip install -e .

WORKDIR /home/work