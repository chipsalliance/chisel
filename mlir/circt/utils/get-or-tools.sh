#!/usr/bin/env bash
##===- utils/get-or-tools.sh - Install OR-Tools --------------*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script downloads, compiles, and installs OR-Tools into $/ext.
#
##===----------------------------------------------------------------------===##

OR_TOOLS_VER=9.5

mkdir -p "$(dirname "$BASH_SOURCE[0]")/../ext"
EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)

echo "Installing OR-Tools ($OR_TOOLS_VER)..."
echo $EXT_DIR

cd $EXT_DIR
curl -LO https://github.com/google/or-tools/archive/v$OR_TOOLS_VER.tar.gz
tar -zxf v$OR_TOOLS_VER.tar.gz
rm v$OR_TOOLS_VER.tar.gz
cd or-tools-$OR_TOOLS_VER

# By default, configure a lean build including only free solvers.
# To enable support for additional solvers you have licensed, see:
#   https://github.com/google/or-tools/blob/v9.5/cmake/README.md
cmake -S . -B build -DBUILD_DEPS=ON -DBUILD_SAMPLES=OFF -DBUILD_EXAMPLES=OFF \
      -DBUILD_FLATZINC=OFF -DUSE_SCIP=OFF
cmake --build build --parallel $(nproc || sysctl -n hw.ncpu)
cmake --install build --prefix $EXT_DIR

cd ../
rm -rf or-tools-$OR_TOOLS_VER

echo "Done."
