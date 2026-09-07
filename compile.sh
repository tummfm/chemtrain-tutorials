#!/usr/bin/env bash
set -euo pipefail

wdir=$(pwd)
MPI="/opt/openmpi-4.1.8-cuda"
ARCH=AMPERE80
LOCAL_CUDA="/usr/local/cuda-12.9"
BAZEL_CACHE_ROOT="${TMPDIR:-/tmp}/chemAiML26-${UID}"

mkdir -p software external "${BAZEL_CACHE_ROOT}/output" "${BAZEL_CACHE_ROOT}/bazelisk"

# Keep Bazelisk downloads off the project filesystem. Bazel's output tree is
# redirected separately through build.py's --output_path option below.
export BAZELISK_HOME="${BAZEL_CACHE_ROOT}/bazelisk"

initialize_repo() {
  local url=$1
  local revision=$2
  local directory=$3

  if [[ -e "${directory}" ]]; then
    if ! git -C "${directory}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "error: ${directory} exists but is not a Git repository" >&2
      return 1
    fi
  else
    git clone --no-checkout --filter=blob:none "${url}" "${directory}"
  fi

  git -C "${directory}" fetch --depth=1 "${url}" "${revision}"
  git -C "${directory}" checkout --detach FETCH_HEAD
  git -C "${directory}" submodule update --init --recursive
}

initialize_repo \
  https://github.com/tummfm/chemtrain \
  7a07083dfef30e7ad0bd8d6e12deae1c3edd3538 \
  external/chemtrain
initialize_repo \
  https://github.com/tummfm/lammps \
  553ca5e3acd69bc6dacf26aca17481970eb45486 \
  external/lammps

# Compile pjrt plugin and compile lammps with kokkos
cd external/chemtrain/chemtrain-deploy || exit
python build.py \
    --install_location "${wdir}/software/lib" \
    --output_path "${BAZEL_CACHE_ROOT}/output" \
    --enable_cuda \
    --build_gpu_pjrt_plugin \
    --ffi_provider_target="@openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so" \
    --cuda_version 12.9.1 \
    --cudnn_version 9.8.0 \
    --cuda_compute_capabilities "sm_80,sm_86,sm_90"
cd "$wdir"

# Compile lammps
export PATH="${MPI}/bin:${LOCAL_CUDA}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI}/lib:${LOCAL_CUDA}/lib:${LD_LIBRARY_PATH:-}"

cd external/lammps && mkdir -p build && cd build || exit
cmake -S ../cmake \
  -C ../cmake/presets/kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_INSTALL_PREFIX="${wdir}/software" \
  -D BUILD_MPI=ON \
  -D PKG_CHEMTRAIN-DEPLOY=ON \
  -D CHEMTRAIN_DEPLOY_ROOT="$wdir/external/chemtrain/chemtrain-deploy" \
  -D CHEMTRAIN_DEPLOY_LIBCONNECTOR="$wdir/software/lib/libconnector.so" \
  -D "Kokkos_ARCH_${ARCH}"=ON

cmake --build . --parallel
cmake --build . --target install --parallel
