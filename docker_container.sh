#!/bin/sh
# -*- coding: utf-8 -*-

docker run \
    -it \
    --rm \
    --gpus all \
    --shm-size=29g \
    --name kaggle_IMC2024 \
    --volume $(pwd)/:/kaggle/ \
    --workdir /kaggle/working/ \
    tmyok/pytorch:image-matching-challenge-2024