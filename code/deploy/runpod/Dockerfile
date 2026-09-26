# Base: PyTorch 2.6.0 + CUDA 12.6, cuDNN 9, Python 3.11.
#
# This image runs one RealLit benchmark cell per RunPod Serverless job
# (``deploy/runpod/handler.py``). The four paper classifiers and four
# generative imputers are scaffold stubs by default; the third-party
# packages they wrap are installed below so a reviewer can drop in the
# reference implementations without rebuilding the image.

ARG TARGET_PLATFORM=linux/amd64
FROM --platform=${TARGET_PLATFORM} pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Core deps + the third-party packages that the paper-method stubs lazily
# import (so a reviewer who replaces a stub with the real implementation
# does not need to rebuild the image).
RUN pip install --upgrade pip && pip install \
    "numpy>=1.26,<2.0" \
    "pandas>=2.2" \
    "scikit-learn>=1.4" \
    "scipy>=1.12" \
    "pyyaml>=6.0" \
    "tqdm>=4.66" \
    "jsonschema>=4.21" \
    "lightgbm>=4.0" \
    "miceforest>=5.7" \
    "tabpfn>=0.1.10" \
    "tabicl>=0.2.0" \
    "tabdpt>=0.0.4" \
    "requests>=2.31" \
    "runpod>=1.6" \
    "pytorch_lightning>=2.0"

# Bring the harness in and install in editable mode (RealLit.csv ships with it).
COPY . /workspace
RUN pip install -e .

# Build-time sanity check.
RUN python -c "import torch, tabpfn, tabicl, tabdpt, miceforest, runpod, pytorch_lightning; \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

CMD ["python", "-u", "deploy/runpod/handler.py"]
