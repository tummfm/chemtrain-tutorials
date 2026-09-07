#!/usr/bin/env bash
set -euo pipefail

LOCAL_CUDA="/usr/local/cuda-12.9"
export CUDA_HOME="${LOCAL_CUDA}"
export PATH="${LOCAL_CUDA}/bin:${PATH}"
export LD_LIBRARY_PATH="${LOCAL_CUDA}/lib64:${LOCAL_CUDA}/lib:${LD_LIBRARY_PATH:-}"

# Install chemtrain
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation \
  "OpenEquivariance[jax] @ git+https://github.com/PASSIONLab/OpenEquivariance@c55b9fbf4507a5e3494291992d2b30c7a106d1fb#subdirectory=openequivariance"
python -m pip install --no-build-isolation \
  "OpenEquivariance_extjax @ git+https://github.com/PASSIONLab/OpenEquivariance@c55b9fbf4507a5e3494291992d2b30c7a106d1fb#subdirectory=openequivariance_extjax"
