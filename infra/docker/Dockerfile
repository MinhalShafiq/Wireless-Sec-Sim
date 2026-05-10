# Simulation worker image.
#
# CUDA + Sionna RT requires the OptiX runtime (libnvoptix.so.1), which
# ships with libnvidia-gl-<DRV>. When running with `docker run --gpus all
# --runtime=nvidia`, the host driver libs are bind-mounted in.
#
# Build:
#   docker build -t wireless-sec-sim:dev -f infra/docker/Dockerfile .
# Run (CPU):
#   docker run --rm -v $PWD/data:/app/data wireless-sec-sim:dev \
#     python -m scripts.run_scenario configs/munich_baseline.yaml
# Run (GPU):
#   docker run --rm --gpus all -e SIM_VARIANT=cuda_ad_mono_polarized \
#     -v $PWD/data:/app/data wireless-sec-sim:dev \
#     python -m scripts.run_scenario configs/munich_baseline.yaml

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SIM_VARIANT=llvm_ad_mono_polarized

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

COPY libs/    ./libs/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY pyproject.toml README.md ./

ENV PYTHONPATH=/app

ENTRYPOINT ["python3"]
CMD ["-m", "scripts.run_scenario", "configs/munich_baseline.yaml", "--no-render"]
