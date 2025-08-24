# Example of H3DGS Dockerfile
FROM nvcr.io/nvidia/pytorch:25.05-py3

ENV LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64"

# Set working directory
WORKDIR /app

# Copy Submodules
COPY ./submodules/ /app/submodules

# Copy requirements
COPY requirements.txt .

# Install system dependencies (including CMake and other libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake \
      libgl1 \
      libglew-dev \
      libglm-dev \
      libassimp-dev \
      libboost-all-dev \
      libgtk-3-dev \
      libopencv-dev \
      libglfw3-dev \
      libavdevice-dev \
      libavcodec-dev \
      libeigen3-dev \
      libxxf86vm-dev \
      libembree-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements
RUN pip install -r requirements.txt

COPY . /app/