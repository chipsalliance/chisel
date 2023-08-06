#!/usr/bin/env bash
##===- utils/run-tests-docker.sh - Run tests in docker -------*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# This script should be run in the docker container started in the
# 'run-docker.sh' script.
#
##===----------------------------------------------------------------------===##

set -e

UTILS_DIR=$(dirname "$BASH_SOURCE[0]")

if [ ! -e llvm/build_20.04 ]; then
  echo "=== Building MLIR"
  $UTILS_DIR/build-llvm.sh build_20.04 build_20.04/install
fi

echo "=== Building CIRCT"
cmake -Bdocker_build \
  -DMLIR_DIR=llvm/build_20.04/lib/cmake/mlir \
  -DLLVM_DIR=llvm/build_20.04/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DVERILATOR_PATH=/usr/bin/verilator \
  -DCAPNP_PATH=/usr \
  -DCMAKE_BUILD_TYPE=DEBUG

cmake --build docker_build -j$(nproc) --target check-circt
cmake --build docker_build -j$(nproc) --target check-circt-integration
