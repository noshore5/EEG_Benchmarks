# syntax=docker/dockerfile:1
#
# SzCORE seizure-detection container for the CoherIQ TMC-T model.
# Build:  docker build -t ghcr.io/noshore5/eeg_benchmarks-szcore:0.1.0 Epilepsy/szcore
# Test:   docker run --rm -v "$PWD/in:/data" -v "$PWD/out:/output" \
#             -e INPUT=sample.edf -e OUTPUT=sample.tsv \
#             ghcr.io/noshore5/eeg_benchmarks-szcore:0.1.0

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

ARG UID=10001
RUN adduser --disabled-password --gecos "" --home "/nonexistent" \
    --shell "/sbin/nologin" --no-create-home --uid "${UID}" appuser

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Algorithm code + trained checkpoint (algo/model_weights.pt must exist --
# produced by Epilepsy/szcore/train_detector.py).
COPY algo/ ./algo/

USER appuser

VOLUME ["/data"]
VOLUME ["/output"]

ENV INPUT=""
ENV OUTPUT=""

CMD python3 -m algo "/data/${INPUT}" "/output/${OUTPUT}"
