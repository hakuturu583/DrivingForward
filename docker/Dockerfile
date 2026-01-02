FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG USERNAME=appuser
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-distutils \
    python3.8-dev \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace/DrivingForward

RUN python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --index-url https://download.pytorch.org/whl/cu113

# Install Python dependencies at build time
COPY requirements.txt ./requirements.txt
RUN python3.8 -m pip install -r requirements.txt

# Only copy the gaussian rasterizer submodule to speed up builds.
COPY models/gaussian/gaussian-splatting/submodules/diff-gaussian-rasterization \
     models/gaussian/gaussian-splatting/submodules/diff-gaussian-rasterization

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
RUN python3.8 -m pip install models/gaussian/gaussian-splatting/submodules/diff-gaussian-rasterization

RUN chown -R ${UID}:${GID} /workspace/DrivingForward

USER ${USERNAME}

CMD ["/bin/bash"]
