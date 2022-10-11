#!/bin/bash

set -xe

APP_ROOT="$(dirname "$0")/.."

cd "$APP_ROOT/src/vendor/stable-diffusion"
conda env create -f environment.yaml

SECRET_FILE=/run/secrets/huggingface
HUGGINGFACE_USERNAME="$(head -n 1 "$SECRET_FILE")"
HUGGINGFACE_PASSWORD="$(tail -n 1 "$SECRET_FILE")"

mkdir -p models/ldm/stable-diffusion-v1
wget --progress=bar:force:noscroll --user="$HUGGINGFACE_USERNAME" --password="$HUGGINGFACE_PASSWORD" -O models/ldm/stable-diffusion-v1/model.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt

conda run -n ldm ../../../scripts/download_dep_models.py

mkdir -p /root/.cache/torch/hub/checkpoints
wget --progress=bar:force:noscroll -O /root/.cache/torch/hub/checkpoints/checkpoint_liberty_with_aug.pth https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth
