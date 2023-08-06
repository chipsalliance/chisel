#!/usr/bin/env bash
##===- utils/build-release.sh - Build Release ----------------*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script builds circt and makes a release tarball in $/tmp.
#
##===----------------------------------------------------------------------===##

mkdir -p "$(dirname "$BASH_SOURCE[0]")/../tmp"
TMP_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../tmp" && pwd)

echo $TMP_DIR
cd $TMP_DIR

mkdir -p circt && \
mkdir -p llvm && \
mkdir -p install && \
cd llvm && \
cmake ../../llvm/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_PROJECTS='mlir' \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_STATIC_LINK_CXX_STDLIB=ON \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" && \
cmake --build . --target install -- -j$(nproc)  && \
cd ../circt && \
cmake ../.. \
    -DMLIR_DIR=../llvm/lib/cmake/mlir \
    -DLLVM_DIR=../llvm/lib/cmake/llvm \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLLVM_STATIC_LINK_CXX_STDLIB=ON \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DVERILATOR_DISABLE=ON \
    -DLLVM_EXTERNAL_LIT=../llvm/bin \
    -DLLVM_ENABLE_TERMINFO=OFF && \
cmake --build . --target install -- -j$(nproc) && \
cd ../install && \
tar --transform 's,^,circt-release/,' -czf ../circt-release.tgz .

echo "Done."
