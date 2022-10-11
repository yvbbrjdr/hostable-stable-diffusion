#!/bin/bash

set -e

APP_ROOT="$(dirname "$0")/.."

echo "Make sure you have accepted the license agreement at https://huggingface.co/CompVis/stable-diffusion-v-1-4-original"
printf "Please enter your huggingface username: "
read username
printf "Please enter your huggingface password: "
read -s password
echo

SECRET_FILE="$(mktemp)"
trap "rm -f $SECRET_FILE" EXIT
echo "$username" > "$SECRET_FILE"
echo "$password" >> "$SECRET_FILE"

sudo DOCKER_BUILDKIT=1 docker build -t hostable-stable-diffusion \
    --secret id=huggingface,src="$SECRET_FILE" \
    "$APP_ROOT"
