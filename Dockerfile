FROM continuumio/anaconda3:latest

COPY . /workspace
WORKDIR /workspace

RUN --mount=type=secret,id=huggingface scripts/setup_env.sh
